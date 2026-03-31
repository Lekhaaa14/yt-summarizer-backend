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
    return genai.GenerativeModel("gemini-2.0-flash")

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")

STYLE_PROMPTS = {
    "brief": "Summarize this YouTube video in 3-4 sentences. Be concise.",
    "detailed": "Provide a detailed summary of this YouTube video. Include the main topic, key points, and any important conclusions. Use 2-3 paragraphs.",
    "bullet": "Summarize this YouTube video as a list of bullet points covering all the key ideas and takeaways.",
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

    # Normalize URL
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    style_prompt = STYLE_PROMPTS.get(req.style, STYLE_PROMPTS["detailed"])

    prompt = f"""{style_prompt}

Also provide the video title on the very first line in this exact format:
TITLE: <title here>

Then on a new line, give the summary."""

    try:
        # Gemini 1.5 natively understands YouTube URLs — no transcript needed
        response = model.generate_content([
            {"role": "user", "parts": [
                {"text": prompt},
                {"file_data": {"mime_type": "video/mp4", "file_uri": youtube_url}}
            ]}
        ])
        full_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    # Extract title from response
    title = f"Video {video_id}"
    summary_text = full_text
    if full_text.startswith("TITLE:"):
        lines = full_text.split("\n", 2)
        title = lines[0].replace("TITLE:", "").strip()
        summary_text = "\n".join(lines[1:]).strip()

    return SummarizeResponse(
        title=title,
        video_id=video_id,
        summary=summary_text,
        transcript_length=len(full_text),
    )