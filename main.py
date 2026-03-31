from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import re

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

STYLE_PROMPTS = {
    "brief": "in 3-4 sentences, be concise",
    "detailed": "in detail — cover the main topic, key points, and conclusions in 2-3 paragraphs",
    "bullet": "as a bullet point list of all key ideas and takeaways",
}

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
    style = STYLE_PROMPTS.get(req.style, STYLE_PROMPTS["detailed"])

    prompt = f"""Please watch this YouTube video and summarize it {style}.

Video URL: {youtube_url}

At the very start of your response, write the video title on its own line in this format:
TITLE: <title here>

Then write the summary on the next line."""

    try:
        response = model.generate_content(prompt)
        full_text = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    # Parse title out of response
    title = f"Video {video_id}"
    summary_text = full_text
    for line in full_text.splitlines():
        if line.strip().upper().startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
            summary_text = full_text.replace(line, "").strip()
            break

    return SummarizeResponse(
        title=title,
        video_id=video_id,
        summary=summary_text,
        transcript_length=len(full_text),
    )