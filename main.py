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
    transcript: str
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
    # No response_mime_type — let Gemini output freely, we parse manually
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
    """Robustly extract JSON from Gemini response, handling all fence/leak cases."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find from first { to last } (handles trailing text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {}

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

    if not req.transcript or len(req.transcript.strip()) < 50:
        raise HTTPException(status_code=400, detail="Transcript is empty or too short.")

    max_chars = 80000
    trimmed = req.transcript[:max_chars]

    prompt = f"""You are an expert content analyst. Analyze this YouTube video transcript and return a structured JSON summary.

TRANSCRIPT:
{trimmed}

INSTRUCTIONS:
- Base your response ONLY on the transcript content above
- Be specific — mention actual tools, concepts, techniques, examples from the video
- The summary must cover the ENTIRE video from start to finish in detail
- Every sentence must be complete — never truncate mid-sentence
- keyPoints should be organized thematically (group related points together)
- Timestamps should reflect real topic transitions in the video

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "title": "exact or inferred video title",
  "summary": "Write a thorough 5-paragraph summary:\n- Para 1: What the video is about, who it's for, and its main goal\n- Para 2: First major section covered (specific topics, tools, examples)\n- Para 3: Middle sections covered in detail\n- Para 4: Later sections and advanced topics\n- Para 5: Final conclusions, recommendations, and overall takeaways",
  "keyPoints": [
    "Topic category 1 — specific detail from video",
    "Topic category 2 — specific detail from video",
    "Topic category 3 — specific detail from video",
    "Topic category 4 — specific detail from video",
    "Topic category 5 — specific detail from video",
    "Topic category 6 — specific detail from video",
    "Topic category 7 — specific detail from video",
    "Topic category 8 — specific detail from video"
  ],
  "timestamps": [
    "00:00 - Introduction and overview",
    "MM:SS - Topic name",
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

    data = extract_json(raw)

    if not data.get("summary"):
        # JSON extraction failed — try one more time with a simpler approach
        try:
            # Sometimes Gemini wraps in extra text before/after JSON
            json_match = re.search(r'\{.*?"summary".*?\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
        except Exception:
            pass
        # If still no summary, use raw text but clean it up
        if not data.get("summary"):
            clean = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            clean = re.sub(r'\s*```\s*$', '', clean, flags=re.MULTILINE).strip()
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
        transcript_length=len(req.transcript),
    )