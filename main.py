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
            response_mime_type="application/json",  # force JSON output
        )
    )

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

    prompt = f"""Analyze this YouTube video completely and return a JSON object.

Video URL: {youtube_url}

Rules:
- Watch the FULL video before summarizing
- If the video language is not English, translate and summarize in English
- The summary must be COMPLETE — do not cut off mid-sentence. Cover the entire video from start to finish.
- Be specific and informative — mention actual concepts, tools, techniques, or arguments from the video
- keyPoints must be standalone insights a viewer would find valuable even without watching
- Timestamps must reflect the actual video structure with accurate time markers

Return this exact JSON schema:
{{
  "title": "exact video title",
  "summary": "Write 4-5 full paragraphs covering: (1) what the video is about and its goal, (2) the main content and techniques covered in the first half, (3) the main content covered in the second half, (4) specific examples, demos, or case studies shown, (5) final conclusions and recommendations from the creator. Every sentence must be complete. Do not truncate.",
  "keyPoints": [
    "Complete, specific key point 1",
    "Complete, specific key point 2",
    "Complete, specific key point 3",
    "Complete, specific key point 4",
    "Complete, specific key point 5",
    "Complete, specific key point 6"
  ],
  "timestamps": [
    "00:00 - Introduction",
    "MM:SS - Topic name",
    "MM:SS - Topic name",
    "MM:SS - Topic name",
    "MM:SS - Conclusion"
  ]
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    # Clean up any markdown fences just in case
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object from the response if there's extra text
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except Exception:
                data = {"title": f"Video {video_id}", "summary": raw, "keyPoints": [], "timestamps": []}
        else:
            data = {"title": f"Video {video_id}", "summary": raw, "keyPoints": [], "timestamps": []}

    return SummarizeResponse(
        title=data.get("title", f"Video {video_id}"),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(raw),
    )