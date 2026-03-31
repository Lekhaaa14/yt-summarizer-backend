from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os, json, re

app = FastAPI()

# ✅ CORS (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Gemini API setup
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Root route
@app.get("/")
def root():
    return {"status": "YouTube Summarizer API is running"}

# ✅ Summarize API (FIXED)
@app.post("/api/summarize")
async def summarize(body: dict):
    url = body.get("url", "")

    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    # 👉 TEMP: Use URL directly (no transcript extraction yet)
    prompt = f"""
You are a YouTube video summarizer.

Summarize the content of this YouTube video:
{url}

Return ONLY valid JSON in this format:

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
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # ✅ Clean response
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)
        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")