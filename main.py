from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import yt_dlp
import os
import re

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
    style: str = "detailed"  # "brief", "detailed", "bullet"

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

def get_transcript(video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "ttml",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Try manual captions first, then auto-generated
            subtitles = info.get("subtitles", {})
            auto_captions = info.get("automatic_captions", {})

            captions = subtitles.get("en") or auto_captions.get("en")
            if not captions:
                # Try other English variants like en-US, en-GB
                for key in list(subtitles.keys()) + list(auto_captions.keys()):
                    if key.startswith("en"):
                        captions = subtitles.get(key) or auto_captions.get(key)
                        break

            if not captions:
                raise HTTPException(status_code=400, detail="No English transcript available for this video.")

            # Get the JSON3 format which is easiest to parse
            json3_url = next((c["url"] for c in captions if c.get("ext") == "json3"), None)
            if not json3_url:
                # Fall back to any available format url
                json3_url = captions[0]["url"]

            import urllib.request, json
            with urllib.request.urlopen(json3_url) as resp:
                data = json.loads(resp.read().decode())

            # Extract plain text from json3 format
            lines = []
            for event in data.get("events", []):
                for seg in event.get("segs", []):
                    text = seg.get("utf8", "").strip()
                    if text and text != "\n":
                        lines.append(text)

            transcript = " ".join(lines).strip()
            if not transcript:
                raise HTTPException(status_code=400, detail="Transcript is empty for this video.")
            return transcript

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript: {str(e)}")

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
        title=f"Video {video_id}",
        video_id=video_id,
        summary=summary_text,
        transcript_length=len(transcript),
    )