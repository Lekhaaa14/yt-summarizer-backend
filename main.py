from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import re
import json
import urllib.request
import urllib.error
import time
import random

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
    keyPoints: list[str]
    timestamps: list[str]
    transcript_length: int

# Rotate user agents to avoid rate limiting
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

def fetch_url(url: str, retries: int = 3, timeout: int = 15) -> str:
    """Fetch a URL with retry + backoff on 429."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=get_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                # Handle gzip if needed
                try:
                    import gzip
                    return gzip.decompress(raw).decode("utf-8", errors="replace")
                except Exception:
                    return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (2 ** attempt) + random.uniform(1, 3)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise HTTPException(status_code=429, detail="YouTube is rate-limiting this server. Please wait a minute and try again.")

def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=3000,
            response_mime_type="application/json",
        )
    )

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def fetch_transcript(video_id: str) -> tuple[str, str]:
    """Fetch transcript + title from YouTube page source."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        html = fetch_url(video_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load video page: {str(e)}")

    # Extract title
    title = f"Video {video_id}"
    for pattern in [r'<title>([^<]+)</title>', r'"title":"([^"]{5,100})"']:
        m = re.search(pattern, html)
        if m:
            title = m.group(1).replace(" - YouTube", "").strip()
            try:
                title = title.encode().decode("unicode_escape", errors="replace")
            except Exception:
                pass
            break

    # Extract caption tracks
    caption_match = re.search(r'"captionTracks":(\[.*?\])', html)
    if not caption_match:
        raise HTTPException(
            status_code=400,
            detail="No captions available for this video. It may not have subtitles enabled."
        )

    try:
        caption_tracks = json.loads(caption_match.group(1))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse caption data.")

    if not caption_tracks:
        raise HTTPException(status_code=400, detail="No caption tracks found for this video.")

    # Pick best track: manual English > auto English > any English > first available
    selected = None
    for kind_filter in [
        lambda t: t.get("languageCode", "").startswith("en") and t.get("kind") != "asr",
        lambda t: t.get("languageCode", "").startswith("en"),
        lambda t: True,
    ]:
        for track in caption_tracks:
            if kind_filter(track):
                selected = track
                break
        if selected:
            break

    base_url = selected.get("baseUrl", "")
    if not base_url:
        raise HTTPException(status_code=400, detail="Caption URL not found.")

    # Request JSON3 format
    if "fmt=" in base_url:
        base_url = re.sub(r"fmt=[^&]+", "fmt=json3", base_url)
    else:
        base_url += "&fmt=json3"

    try:
        content = fetch_url(base_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript: {str(e)}")

    try:
        data = json.loads(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse transcript JSON.")

    lines = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs", 0)
        text = "".join(s.get("utf8", "") for s in event.get("segs", [])).strip()
        if text and text != "\n":
            total_secs = int(start_ms / 1000)
            mins, secs = divmod(total_secs, 60)
            lines.append(f"[{mins:02d}:{secs:02d}] {text}")

    transcript = "\n".join(lines)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty for this video.")

    return transcript, title


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

    transcript, video_title = fetch_transcript(video_id)

    max_chars = 80000
    trimmed = transcript[:max_chars]
    if len(transcript) > max_chars:
        trimmed += "\n[Transcript trimmed]"

    prompt = f"""You are an expert content analyst. Below is the real timestamped transcript of a YouTube video titled "{video_title}".

Analyze it carefully and return a JSON summary.

TRANSCRIPT:
{trimmed}

Rules:
- Base summary ONLY on the transcript — do not hallucinate
- Write 4-5 complete paragraphs covering the full video
- Never cut off mid-sentence
- keyPoints must be specific insights from the actual content
- Use [MM:SS] markers from transcript for accurate timestamps

Return this JSON:
{{
  "title": "{video_title}",
  "summary": "4-5 complete paragraphs summarizing the full video",
  "keyPoints": ["point 1", "point 2", "point 3", "point 4", "point 5", "point 6"],
  "timestamps": ["00:00 - Intro", "MM:SS - Topic", "MM:SS - Topic", "MM:SS - Conclusion"]
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)

    try:
        data = json.loads(raw.strip())
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group()) if m else {
            "title": video_title, "summary": raw, "keyPoints": [], "timestamps": []
        }

    return SummarizeResponse(
        title=data.get("title", video_title),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )