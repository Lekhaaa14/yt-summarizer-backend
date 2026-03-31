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

app = FastAPI(title="YouTube Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    url: str
    style: str = "detailed"

class SummarizeResponse(BaseModel):
    title: str
    video_id: str
    summary: str
    transcript_length: int

def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def get_video_info(video_id: str) -> dict:
    """Get video title and caption track list using YouTube Data API v3."""
    yt_api_key = os.environ.get("YOUTUBE_API_KEY")
    if not yt_api_key:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY not configured.")

    # Get video title
    params = urllib.parse.urlencode({
        "part": "snippet",
        "id": video_id,
        "key": yt_api_key
    })
    url = f"https://www.googleapis.com/youtube/v3/videos?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        title = data["items"][0]["snippet"]["title"]
    except Exception:
        title = f"Video {video_id}"

    return {"title": title}

def get_transcript(video_id: str) -> str:
    """
    Fetch captions using YouTube Data API v3:
    1. List caption tracks for the video
    2. Download the caption in plaintext format
    """
    yt_api_key = os.environ.get("YOUTUBE_API_KEY")
    if not yt_api_key:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY not configured.")

    # Step 1: list available caption tracks
    params = urllib.parse.urlencode({
        "part": "snippet",
        "videoId": video_id,
        "key": yt_api_key
    })
    list_url = f"https://www.googleapis.com/youtube/v3/captions?{params}"

    try:
        with urllib.request.urlopen(list_url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not list captions: {str(e)}")

    items = data.get("items", [])
    if not items:
        raise HTTPException(status_code=400, detail="No captions available for this video.")

    # Pick English track first, fallback to first available
    caption_id = None
    for item in items:
        lang = item["snippet"].get("language", "")
        if lang.startswith("en"):
            caption_id = item["id"]
            break
    if not caption_id:
        caption_id = items[0]["id"]

    # Step 2: download the caption track (requires OAuth for private videos,
    # but works with API key for auto-generated/public captions via tfmt=vtt)
    dl_params = urllib.parse.urlencode({
        "tfmt": "vtt",
        "key": yt_api_key
    })
    dl_url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}?{dl_params}"

    try:
        with urllib.request.urlopen(dl_url, timeout=10) as resp:
            vtt_content = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        # Caption download via API requires OAuth - fall back to timedtext
        return get_transcript_timedtext(video_id)
    except Exception as e:
        return get_transcript_timedtext(video_id)

    # Parse VTT format
    lines = []
    for line in vtt_content.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        # Remove VTT tags like <00:00:00.000><c>text</c>
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)

    transcript = " ".join(lines).strip()
    if not transcript:
        return get_transcript_timedtext(video_id)
    return transcript


def get_transcript_timedtext(video_id: str) -> str:
    """
    Fallback: use YouTube's internal timedtext API with the YouTube Data API key
    to get the caption URL, bypassing bot detection.
    """
    yt_api_key = os.environ.get("YOUTUBE_API_KEY")

    # Use YouTube's timedtext endpoint directly with known params
    # Try auto-generated English captions first
    for lang in ["en", "en-US", "en-GB", "a.en"]:
        params = urllib.parse.urlencode({
            "v": video_id,
            "lang": lang,
            "fmt": "json3",
            "xorb": "2",
            "xobt": "3",
            "xovt": "3",
        })
        tt_url = f"https://www.youtube.com/api/timedtext?{params}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            req = urllib.request.Request(tt_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="replace")
                if not content.strip():
                    continue
                data = json.loads(content)
                lines = []
                for event in data.get("events", []):
                    for seg in event.get("segs", []):
                        text = seg.get("utf8", "").strip()
                        if text and text != "\n":
                            lines.append(text)
                transcript = " ".join(lines).strip()
                if transcript:
                    return transcript
        except Exception:
            continue

    raise HTTPException(
        status_code=400,
        detail="Could not fetch transcript. The video may not have English captions, or captions are disabled."
    )


STYLE_PROMPTS = {
    "brief": "Summarize this YouTube video transcript in 3-4 sentences. Be concise.",
    "detailed": "Provide a detailed summary of this YouTube video transcript. Include the main topic, key points, and any important conclusions. Use 2-3 paragraphs.",
    "bullet": "Summarize this YouTube video transcript as a list of bullet points covering all the key ideas and takeaways.",
}

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

    info = get_video_info(video_id)
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
        title=info["title"],
        video_id=video_id,
        summary=summary_text,
        transcript_length=len(transcript),
    )