import os
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import re
import json
import httpx

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
    transcript: str = ""
    style: str = "detailed"

class SummarizeResponse(BaseModel):
    title: str
    video_id: str
    summary: str
    keyPoints: list[str]
    timestamps: list[str]
    transcript_length: int

# ✅ Use only ONE model (avoid multiple API calls)
MODEL = "gemini-1.5-flash"

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    return genai.Client(api_key=api_key)

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID")

def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except:
        return {}

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest, authorization: str = Header(default=None)):
    client = get_client()

    try:
        video_id = extract_video_id(req.url)
    except:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcript = (req.transcript or "").strip()
    has_transcript = len(transcript) > 50

    # ✅ Reduce token usage (IMPORTANT)
    trimmed = transcript[:2000]

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=512,  # ✅ reduced
    )

    try:
        if has_transcript:
            prompt = f"""
Summarize this YouTube transcript in simple terms.

Return ONLY valid JSON:
{{
  "title": "Video title",
  "summary": "Short summary (2 paragraphs)",
  "keyPoints": ["point1", "point2", "point3"],
  "timestamps": ["00:00 - intro", "00:30 - topic"]
}}

Transcript:
{trimmed}
"""
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config
            )
        else:
            prompt = """
Summarize this YouTube video.

Return ONLY valid JSON:
{
  "title": "Video title",
  "summary": "Short summary",
  "keyPoints": ["point1", "point2", "point3"],
  "timestamps": ["00:00 - intro"]
}
"""
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config
            )

    except Exception as e:
        raise HTTPException(status_code=429, detail="Quota exceeded. Try later.")

    raw = response.text.strip()
    data = extract_json(raw)

    if not data.get("summary"):
        data = {
            "title": f"Video {video_id}",
            "summary": raw,
            "keyPoints": [],
            "timestamps": []
        }

    return SummarizeResponse(
        title=data.get("title", ""),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )