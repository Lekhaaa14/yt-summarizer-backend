from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
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

def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=4096,
        )
    )

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

JSON_FORMAT = '''{
  "title": "video title",
  "summary": "detailed summary",
  "keyPoints": ["point 1", "point 2", "point 3", "point 4", "point 5", "point 6"],
  "timestamps": ["00:00 - Intro", "MM:SS - Topic", "MM:SS - Topic", "MM:SS - End"]
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
    model = get_gemini_model()

    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    transcript = (req.transcript or "").strip()
    has_transcript = len(transcript) > 50

    if has_transcript:
        # Use real transcript — most accurate
        trimmed = transcript[:80000]
        prompt = f"""You are an expert content analyst. Analyze this YouTube video transcript and return a JSON summary.

TRANSCRIPT:
{trimmed}

Return ONLY valid JSON, no markdown fences, no extra text:
{{
  "title": "infer the video title from the transcript",
  "summary": "Write 5 complete paragraphs: (1) what the video is about and its goal, (2) first major section with specific details, (3) middle sections in detail, (4) later/advanced topics, (5) conclusions and key takeaways. Never truncate mid-sentence.",
  "keyPoints": [
    "Specific insight 1 from the video",
    "Specific insight 2 from the video",
    "Specific insight 3 from the video",
    "Specific insight 4 from the video",
    "Specific insight 5 from the video",
    "Specific insight 6 from the video",
    "Specific insight 7 from the video",
    "Specific insight 8 from the video"
  ],
  "timestamps": ["00:00 - Introduction", "MM:SS - Topic", "MM:SS - Topic", "MM:SS - Topic", "MM:SS - Conclusion"]
}}"""
    else:
        # No transcript — Gemini watches the video visually
        prompt = f"""You are an expert video analyst. Watch this YouTube video carefully and return a JSON summary of what you see and hear.

Video URL: {youtube_url}

Watch the FULL video. Describe exactly what happens — actions, scenes, people, objects, dialogue, humor, or key moments. Be specific, not generic.

Return ONLY valid JSON, no markdown fences, no extra text:
{{
  "title": "the actual video title",
  "summary": "Write 3-4 paragraphs describing exactly what happens in the video from start to finish. Mention specific visual details, actions, and moments.",
  "keyPoints": [
    "Specific thing that happens 1",
    "Specific thing that happens 2",
    "Specific thing that happens 3",
    "Specific thing that happens 4"
  ],
  "timestamps": ["00:00 - Start", "MM:SS - Event", "MM:SS - Event", "MM:SS - End"]
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    data = extract_json(raw)

    if not data.get("summary"):
        clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        clean = re.sub(r"\s*```\s*$", "", clean, flags=re.MULTILINE).strip()
        data = {
            "title": f"Video {video_id}",
            "summary": clean,
            "keyPoints": [],
            "timestamps": []
        }

    return SummarizeResponse(
        title=data.get("title", f"Video {video_id}"),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )