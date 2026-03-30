from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os, re, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reads GEMINI_API_KEY from environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # free and fast

def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

@app.get("/")
def root():
    return {"status": "YouTube Summarizer API is running"}

@app.post("/api/summarize")
async def summarize(body: dict):
    url = body.get("url", "")

    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Fetch transcript
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Transcript not available: {str(e)}")

    # Format transcript with timestamps
    full_text = "\n".join([
        f"[{int(t['start'])}s] {t['text']}"
        for t in transcript_data
    ])

    # Send to Gemini
    prompt = f"""
You are a YouTube video summarizer. Analyze this transcript and respond ONLY with valid JSON, no extra text.

{{
  "summary": "3-5 sentence summary of the entire video",
  "keyPoints": [
    "key point 1",
    "key point 2",
    "key point 3",
    "key point 4",
    "key point 5"
  ],
  "timestamps": [
    {{"time": "0:32", "seconds": 32, "label": "Brief topic label"}},
    {{"time": "1:45", "seconds": 105, "label": "Brief topic label"}},
    {{"time": "3:20", "seconds": 200, "label": "Brief topic label"}}
  ]
}}

Transcript:
{full_text[:12000]}
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Clean up markdown code fences if Gemini adds them
        raw = re.sub(r"```json|```", "", raw).strip()

        # Validate it's proper JSON
        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
