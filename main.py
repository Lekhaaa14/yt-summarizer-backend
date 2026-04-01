from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import re
import json
import urllib.request

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

def fetch_transcript(video_url: str) -> str:
    """Fetch transcript via Supadata API — works without bot detection issues."""
    api_key = os.environ.get("SUPADATA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SUPADATA_API_KEY not configured.")

    import urllib.parse
    encoded_url = urllib.parse.quote(video_url, safe="")
    endpoint = f"https://api.supadata.ai/v1/youtube/transcript?url={encoded_url}&text=true"

    req = urllib.request.Request(
        endpoint,
        headers={"x-api-key": api_key}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise HTTPException(status_code=400, detail=f"Transcript fetch failed: {body}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcript fetch error: {str(e)}")

    # text=true returns content as a plain string
    content = data.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="No transcript available for this video.")

    return content

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

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    # Get real transcript via Supadata
    transcript = fetch_transcript(youtube_url)

    # Trim if too long
    max_chars = 80000
    trimmed = transcript[:max_chars]
    if len(transcript) > max_chars:
        trimmed += "\n[Transcript trimmed]"

    prompt = f"""You are an expert content analyst. Below is the real transcript of a YouTube video.
Analyze it carefully and return an accurate JSON summary.

TRANSCRIPT:
{trimmed}

Rules:
- Base your answer ONLY on the transcript — no hallucination
- Write 4-5 complete paragraphs covering the full video from start to finish
- Never cut off mid-sentence
- keyPoints must be specific, standalone insights from the actual content
- For timestamps use approximate positions based on the transcript flow

Return this exact JSON:
{{
  "title": "infer the video title from the transcript content",
  "summary": "4-5 complete paragraphs summarizing the full video accurately",
  "keyPoints": [
    "Specific insight 1",
    "Specific insight 2",
    "Specific insight 3",
    "Specific insight 4",
    "Specific insight 5",
    "Specific insight 6"
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

    try:
        data = json.loads(raw.strip())
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group()) if m else {
            "title": f"Video {video_id}", "summary": raw, "keyPoints": [], "timestamps": []
        }

    return SummarizeResponse(
        title=data.get("title", f"Video {video_id}"),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )