from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import re
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET

# --- App setup ---
app = FastAPI(title="YouTube Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class SummarizeRequest(BaseModel):
    url: str
    style: str = "detailed"

class SummarizeResponse(BaseModel):
    title: str
    video_id: str
    summary: str
    transcript_length: int

# --- Helpers ---
def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def get_video_title(video_id: str) -> str:
    """Fetch video title via YouTube oEmbed (no API key needed)."""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(oembed_url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("title", f"Video {video_id}")
    except Exception:
        return f"Video {video_id}"

def get_transcript(video_id: str) -> str:
    """
    Fetch transcript using YouTube's internal timedtext API.
    This is the same endpoint youtube-transcript-api uses, called directly.
    """
    # Step 1: fetch the video page to get the caption track list
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(video_url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load video page: {str(e)}")

    # Step 2: extract captionTracks from the page JS
    match = re.search(r'"captionTracks":(\[.*?\])', html)
    if not match:
        raise HTTPException(status_code=400, detail="No captions found for this video. The video may not have subtitles.")

    try:
        caption_tracks = json.loads(match.group(1))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse caption data from video page.")

    if not caption_tracks:
        raise HTTPException(status_code=400, detail="No caption tracks available for this video.")

    # Step 3: prefer English, fall back to first available
    track = None
    for t in caption_tracks:
        lang = t.get("languageCode", "")
        if lang.startswith("en"):
            track = t
            break
    if not track:
        track = caption_tracks[0]  # fallback to first language

    base_url = track.get("baseUrl")
    if not base_url:
        raise HTTPException(status_code=400, detail="Caption track URL not found.")

    # Step 4: fetch the XML transcript
    try:
        req2 = urllib.request.Request(base_url, headers=headers)
        with urllib.request.urlopen(req2, timeout=15) as resp:
            xml_data = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript XML: {str(e)}")

    # Step 5: parse XML and extract plain text
    try:
        root = ET.fromstring(xml_data)
        texts = []
        for elem in root.iter("text"):
            t = (elem.text or "").strip()
            # Decode HTML entities
            t = t.replace("&#39;", "'").replace("&amp;", "&").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")
            if t:
                texts.append(t)
        transcript = " ".join(texts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse transcript: {str(e)}")

    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    return transcript

STYLE_PROMPTS = {
    "brief": "Summarize this YouTube video transcript in 3-4 sentences. Be concise.",
    "detailed": "Provide a detailed summary of this YouTube video transcript. Include the main topic, key points, and any important conclusions. Use 2-3 paragraphs.",
    "bullet": "Summarize this YouTube video transcript as a list of bullet points covering all the key ideas and takeaways.",
}

# --- Routes ---
@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube Summarizer API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/summarize", response_model=SummarizeResponse)
@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    model = get_gemini_model()

    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    title = get_video_title(video_id)
    transcript = get_transcript(video_id)

    max_chars = 100000
    trimmed = transcript[:max_chars]
    if len(transcript) > max_chars:
        trimmed += "\n\n[Transcript trimmed due to length]"

    style_prompt = STYLE_PROMPTS.get(req.style, STYLE_PROMPTS["detailed"])
    prompt = f"{style_prompt}\n\nHere is the transcript:\n\n{trimmed}"

    try:
        response = model.generate_content(prompt)
        summary_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    return SummarizeResponse(
        title=title,
        video_id=video_id,
        summary=summary_text,
        transcript_length=len(transcript),
    )