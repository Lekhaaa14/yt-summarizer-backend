from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os, re, json, subprocess, glob

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Extract video ID
def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# ✅ NEW: Get transcript using yt-dlp
def get_transcript(video_url):
    try:
        command = [
            "yt-dlp",
            "--write-auto-subs",
            "--skip-download",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "-o", "subtitles",
            video_url
        ]

        subprocess.run(command, check=True)

        files = glob.glob("subtitles*.json3")

        if not files:
            raise Exception("No subtitles found")

        with open(files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        transcript = []
        for event in data.get("events", []):
            if "segs" in event:
                text = "".join([seg.get("utf8", "") for seg in event["segs"]])
                transcript.append({
                    "text": text,
                    "start": event.get("tStartMs", 0) / 1000
                })

        return transcript

    except Exception as e:
        raise Exception(f"Transcript fetch failed: {str(e)}")

# ✅ Root
@app.get("/")
def root():
    return {"status": "YouTube Summarizer API is running"}

# ✅ Summarize
@app.post("/api/summarize")
async def summarize(body: dict):
    url = body.get("url", "")

    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # ✅ Use yt-dlp instead of blocked API
    try:
        transcript_data = get_transcript(url)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

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

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Clean markdown
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")