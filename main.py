from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os, json, re

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

@app.get("/")
def root():
    return {"status": "YouTube Summarizer API is running"}

# ✅ NEW: Summarize using transcript text (NOT URL)
@app.post("/api/summarize")
async def summarize(body: dict):
    text = body.get("text", "")

    if not text:
        raise HTTPException(status_code=400, detail="No transcript provided")

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
{text[:12000]}
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")