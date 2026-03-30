from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os, re, json

app = FastAPI()

# ✅ CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Gemini API setup
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Extract YouTube video ID
def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# ✅ Root check
@app.get("/")
def root():
    return {"status": "YouTube Summarizer API is running"}

# ✅ Summarize endpoint
@app.post("/api/summarize")
async def summarize(body: dict):
    url = body.get("url", "")

    # Validate URL
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # ✅ FIXED: Fetch transcript (NEW API)
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id)

        transcript_data = [
            {"text": t.text, "start": t.start}
            for t in transcript_list
        ]

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not available: {str(e)}"
        )

    # Format transcript
    full_text = "\n".join([
        f"[{int(t['start'])}s] {t['text']}"
        for t in transcript_data
    ])

    # Gemini prompt
    prompt = f"""
You are a YouTube video summarizer. Analyze this transcript and respond ONLY with valid JSON.

{{
  "summary": "3-5 sentence summary",
  "keyPoints": [
    "point 1",
    "point 2",
    "point 3",
    "point 4",
    "point 5"
  ],
  "timestamps": [
    "0:32 - Topic",
    "1:45 - Topic",
    "3:20 - Topic"
  ]
}}

Transcript:
{full_text[:12000]}
"""

    # Generate summary
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Remove markdown formatting if present
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")