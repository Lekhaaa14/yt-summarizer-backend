from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import re
import json

app = FastAPI(title="YouTube Summarizer API")

# Setup CORS for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your Vercel URL
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
    
    # Using 1.5-flash as it is more stable for JSON tasks than 2.0-experimental/flash in some regions
    return genai.GenerativeModel(
        "gemini-1.5-flash", 
        generation_config={
            "temperature": 0.1,  # Lower temperature reduces hallucination
            "max_output_tokens": 4096,
            "response_mime_type": "application/json", # Forces Gemini to return valid JSON
        }
    )

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is live"}

@app.post("/api/summarize", response_model=SummarizeResponse)
@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    model = get_gemini_model()

    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    transcript = (req.transcript or "").strip()
    
    # --- CRITICAL FIX: GATEKEEPER ---
    # If no transcript is provided, we stop here to prevent hallucination and save quota.
    if len(transcript) < 100:
        raise HTTPException(
            status_code=400, 
            detail="No valid transcript found. This video cannot be summarized without text data."
        )

    # Clean and trim transcript to fit context window
    trimmed_transcript = transcript[:100000] 

    prompt = f"""You are an expert content analyst. Analyze the following YouTube transcript.
    
    RULES:
    1. Base the summary ONLY on the text provided.
    2. If the transcript is mostly music, gibberish, or 'thank you', state that in the summary.
    3. Return a valid JSON object.

    TRANSCRIPT:
    {trimmed_transcript}

    JSON STRUCTURE:
    {{
      "title": "Actual title or inferred subject",
      "summary": "5 detailed paragraphs explaining the goal, major sections, and conclusions.",
      "keyPoints": ["At least 7 specific insights"],
      "timestamps": ["00:00 - Intro", "MM:SS - Topic", "MM:SS - Conclusion"]
    }}"""

    try:
        response = model.generate_content(prompt)
        # Because we used response_mime_type: "application/json", we can load directly
        data = json.loads(response.text)
    except Exception as e:
        # Check specifically for quota errors
        if "429" in str(e) or "quota" in str(e).lower():
             raise HTTPException(status_code=429, detail="Gemini API daily quota reached. Please try again later.")
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

    return SummarizeResponse(
        title=data.get("title", f"Video {video_id}"),
        video_id=video_id,
        summary=data.get("summary", "No summary generated."),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )