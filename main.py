from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import re
import json
import urllib.request
import urllib.parse

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
    """
    Fetch transcript using YouTube's Innertube API.
    Returns (transcript_text, video_title).
    Works without OAuth or API key.
    """
    # Step 1: Get video metadata + caption tracks via Innertube
    innertube_url = "https://www.youtube.com/youtubei/v1/get_transcript"
    
    # First get the video page to extract the serialized share entity
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    req = urllib.request.Request(video_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load video page: {str(e)}")

    # Extract video title
    title_match = re.search(r'"title":"([^"]+)"', html)
    title = title_match.group(1) if title_match else f"Video {video_id}"
    # Clean unicode escapes
    title = title.encode().decode("unicode_escape", errors="replace")

    # Extract caption tracks from page source
    caption_match = re.search(r'"captionTracks":(\[.*?\])', html)
    if not caption_match:
        raise HTTPException(
            status_code=400,
            detail="No captions/transcript available for this video. The video may not have subtitles enabled."
        )

    try:
        caption_tracks = json.loads(caption_match.group(1))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse caption data.")

    if not caption_tracks:
        raise HTTPException(status_code=400, detail="No caption tracks found for this video.")

    # Pick best track: prefer English, then English auto-generated, then first available
    selected = None
    for track in caption_tracks:
        lang = track.get("languageCode", "")
        kind = track.get("kind", "")
        if lang == "en" and kind != "asr":  # manual English captions
            selected = track
            break
    if not selected:
        for track in caption_tracks:
            if track.get("languageCode", "").startswith("en"):  # auto-generated English
                selected = track
                break
    if not selected:
        selected = caption_tracks[0]  # any language as fallback

    base_url = selected.get("baseUrl", "")
    if not base_url:
        raise HTTPException(status_code=400, detail="Caption URL not found.")

    # Add fmt=json3 for easy parsing
    if "fmt=" not in base_url:
        base_url += "&fmt=json3"
    else:
        base_url = re.sub(r"fmt=[^&]+", "fmt=json3", base_url)

    # Fetch the transcript
    req2 = urllib.request.Request(base_url, headers=headers)
    try:
        with urllib.request.urlopen(req2, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript data: {str(e)}")

    try:
        data = json.loads(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse transcript data.")

    # Build transcript with timestamps
    lines = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs", 0)
        segs = event.get("segs", [])
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if text and text != "\n":
            # Convert ms to MM:SS
            total_secs = int(start_ms / 1000)
            mins, secs = divmod(total_secs, 60)
            lines.append(f"[{mins:02d}:{secs:02d}] {text}")

    transcript = "\n".join(lines)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

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

    # Fetch real transcript from YouTube
    transcript, video_title = fetch_transcript(video_id)

    # Trim if too long (keep ~80k chars which is ~20k tokens)
    max_chars = 80000
    trimmed_transcript = transcript[:max_chars]
    if len(transcript) > max_chars:
        trimmed_transcript += "\n[Transcript trimmed due to length]"

    prompt = f"""You are an expert content analyst. Below is the real timestamped transcript of a YouTube video titled "{video_title}".

Analyze the transcript carefully and return a JSON object summarizing the video accurately.

TRANSCRIPT:
{trimmed_transcript}

Rules:
- Base your summary ONLY on the transcript above — do not guess or hallucinate
- Summary must be 4-5 complete paragraphs, covering the full video from start to finish
- Every sentence must be complete — never cut off mid-sentence
- keyPoints must be specific, standalone insights from the actual content
- For timestamps, use the [MM:SS] markers from the transcript to identify real topic changes

Return this exact JSON:
{{
  "title": "{video_title}",
  "summary": "4-5 complete paragraphs summarizing the full video accurately based on the transcript",
  "keyPoints": [
    "Specific insight 1 from the video",
    "Specific insight 2 from the video",
    "Specific insight 3 from the video",
    "Specific insight 4 from the video",
    "Specific insight 5 from the video",
    "Specific insight 6 from the video"
  ],
  "timestamps": [
    "00:00 - Introduction",
    "MM:SS - Topic",
    "MM:SS - Topic",
    "MM:SS - Topic",
    "MM:SS - Conclusion"
  ]
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except Exception:
                data = {"title": video_title, "summary": raw, "keyPoints": [], "timestamps": []}
        else:
            data = {"title": video_title, "summary": raw, "keyPoints": [], "timestamps": []}

    return SummarizeResponse(
        title=data.get("title", video_title),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )