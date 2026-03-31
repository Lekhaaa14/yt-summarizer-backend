from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import os
import re

# --- App setup ---
app = FastAPI(title="YouTube Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini setup ---
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")


# --- Schemas ---
class SummarizeRequest(BaseModel):
    url: str
    style: str = "detailed"  # "brief", "detailed", "bullet"


class SummarizeResponse(BaseModel):
    title: str
    video_id: str
    summary: str
    transcript_length: int


# --- Helpers ---
def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")


def get_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=400, detail="No transcript found for this video.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript: {str(e)}")


STYLE_PROMPTS = {
    "brief": "Summarize this YouTube video transcript in 3-4 sentences. Be concise.",
    "detailed": "Provide a detailed summary of this YouTube video transcript. Include the main topic, key points, and any important conclusions. Use 2-3 paragraphs.",
    "bullet": "Summarize this YouTube video transcript as a list of bullet points covering all the key ideas and takeaways.",
}


# --- Routes ---
@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube Summarizer API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# Both /api/summarize (v0 frontend default) and /summarize (legacy)
@app.post("/api/summarize", response_model=SummarizeResponse)
@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    transcript = get_transcript(video_id)

    max_chars = 100000
    trimmed = transcript[:max_chars]
    if len(transcript) > max_chars:
        trimmed += "\n\n[Transcript trimmed due to length]"

    style_prompt = STYLE_PROMPTS.get(req.style, STYLE_PROMPTS["detailed"])
    prompt = f"{style_prompt}\n\nHere is the transcript:\n\n{trimmed}"

    try:
        response = model.generate_content(prompt)
        summary_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    return SummarizeResponse(
        title=f"Video {video_id}",
        video_id=video_id,
        summary=summary_text,
        transcript_length=len(transcript),
    )