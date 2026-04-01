from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import re
import json

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

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    return genai.Client(api_key=api_key)

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {}

JSON_SCHEMA = '''{
  "title": "video title",
  "summary": "5 complete paragraphs covering full video",
  "keyPoints": ["point 1", "point 2", "point 3", "point 4", "point 5", "point 6"],
  "timestamps": ["00:00 - Intro", "MM:SS - Topic", "MM:SS - End"]
}'''

@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube Summarizer API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/summarize", response_model=SummarizeResponse)
@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    client = get_client()
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    transcript = (req.transcript or "").strip()
    has_transcript = len(transcript) > 50

    try:
        if has_transcript:
            # Transcript-based — accurate for spoken content
            trimmed = transcript[:80000]
            prompt = f"""Analyze this YouTube video transcript and return a JSON summary.

TRANSCRIPT:
{trimmed}

Return ONLY valid JSON, no markdown:
{{
  "title": "infer from transcript",
  "summary": "5 complete paragraphs covering the full video from start to finish. Never truncate.",
  "keyPoints": ["insight 1", "insight 2", "insight 3", "insight 4", "insight 5", "insight 6", "insight 7", "insight 8"],
  "timestamps": ["00:00 - Introduction", "MM:SS - Topic", "MM:SS - Topic", "MM:SS - Conclusion"]
}}"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=4096,
                )
            )
        else:
            # No transcript — watch video visually using Part.from_uri
            prompt = """Watch this entire YouTube video carefully and return a JSON summary of exactly what you see and hear.

Describe real visual events, actions, scenes, people, objects, text on screen, and any audio/speech.
Be specific — mention actual things that happen, not generic descriptions.

Return ONLY valid JSON, no markdown:
{
  "title": "the actual video title",
  "summary": "3-4 paragraphs describing exactly what happens visually and audibly from start to finish. Mention specific moments, actions, and details.",
  "keyPoints": ["specific event/detail 1", "specific event/detail 2", "specific event/detail 3", "specific event/detail 4", "specific event/detail 5"],
  "timestamps": ["00:00 - Start", "MM:SS - Event", "MM:SS - Event", "MM:SS - End"]
}"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_uri(
                        file_uri=youtube_url,
                        mime_type="video/mp4"
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=4096,
                )
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    raw = response.text.strip()
    data = extract_json(raw)

    if not data.get("summary"):
        clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        clean = re.sub(r"\s*```\s*$", "", clean, flags=re.MULTILINE).strip()
        data = {"title": f"Video {video_id}", "summary": clean, "keyPoints": [], "timestamps": []}

    return SummarizeResponse(
        title=data.get("title", f"Video {video_id}"),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )