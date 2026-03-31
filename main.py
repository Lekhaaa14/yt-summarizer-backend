from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    return genai.GenerativeModel("gemini-2.5-flash")

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")

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

    prompt = f"""You are an expert video analyst. Analyze this YouTube video thoroughly and respond ONLY with a valid JSON object — no markdown, no backticks, no explanation outside the JSON.

Video URL: {youtube_url}

IMPORTANT INSTRUCTIONS:
- Watch/analyze the FULL video content
- If the video is not in English, still summarize everything in English
- The summary must be detailed, insightful, and capture the essence of the video
- keyPoints must be the most important takeaways a viewer would want to know
- timestamps should reflect the actual structure of the video (use real time markers if possible)
- Be specific — avoid vague statements like "the video discusses X", instead say what was actually said about X

Return exactly this JSON structure:
{{
  "title": "the actual video title",
  "summary": "A thorough 3-4 paragraph summary covering the main topic, key arguments, examples used, and conclusions drawn. Be specific and informative.",
  "keyPoints": [
    "Specific key point 1",
    "Specific key point 2", 
    "Specific key point 3",
    "Specific key point 4",
    "Specific key point 5",
    "Specific key point 6"
  ],
  "timestamps": [
    "00:00 - Introduction / overview of what the video covers",
    "MM:SS - Section title",
    "MM:SS - Section title",
    "MM:SS - Section title",
    "MM:SS - Conclusion"
  ]
}}"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,  # lower = more focused, faster
                max_output_tokens=2048,
            )
        )
        raw = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: return raw as summary if JSON parse fails
        data = {
            "title": f"Video {video_id}",
            "summary": raw,
            "keyPoints": [],
            "timestamps": []
        }

    return SummarizeResponse(
        title=data.get("title", f"Video {video_id}"),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(raw),
    )