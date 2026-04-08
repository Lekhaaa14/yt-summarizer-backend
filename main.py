import os
print("DEBUG API KEY:", os.getenv("GEMINI_API_KEY"))
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import re
import json
import httpx
#change

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
    transcript: str = ""
    style: str = "detailed"

class SummarizeResponse(BaseModel):
    title: str
    video_id: str
    summary: str
    keyPoints: list[str]
    timestamps: list[str]
    transcript_length: int

MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    return genai.Client(api_key=api_key)

def get_supabase_headers():
    return {
        "apikey": os.environ.get("SUPABASE_SERVICE_KEY", ""),
        "Authorization": f"Bearer {os.environ.get('SUPABASE_SERVICE_KEY', '')}",
        "Content-Type": "application/json"
    }

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {}

def verify_token(authorization: str) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    supabase_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supabase_url or not service_key:
        return None
    try:
        response = httpx.get(
            f"{supabase_url}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": service_key},
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("id")
    except Exception:
        pass
    return None

def save_summary_to_supabase(user_id: str, video_id: str, video_url: str, data: dict):
    supabase_url = os.environ.get("SUPABASE_URL")
    if not supabase_url or not user_id:
        return
    try:
        httpx.post(
            f"{supabase_url}/rest/v1/summaries",
            headers=get_supabase_headers(),
            json={
                "user_id": user_id,
                "video_id": video_id,
                "video_url": video_url,
                "title": data.get("title", ""),
                "summary": data.get("summary", ""),
                "key_points": data.get("keyPoints", []),
                "timestamps": data.get("timestamps", []),
            },
            timeout=5
        )
    except Exception:
        pass

def generate_with_fallback(client, contents, config):
    """Try each model in order until one works."""
    last_error = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "429" in err or "exhausted" in err or "rate" in err:
                last_error = e
                continue  # try next model
            raise e  # non-quota error, raise immediately
    raise HTTPException(status_code=429, detail="All Gemini models quota exhausted. Please try again tomorrow or add billing at console.cloud.google.com")

@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube Summarizer API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/summarize", response_model=SummarizeResponse)
@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest, authorization: str = Header(default=None)):
    user_id = verify_token(authorization)
    client = get_client()

    try:
        video_id = extract_video_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    transcript = (req.transcript or "").strip()
    has_transcript = len(transcript) > 50

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2048,
    )

    try:
        if has_transcript:
            trimmed = transcript[:60000]
            prompt = f"""Summarize this YouTube video transcript. Return ONLY valid JSON:
{{
  "title": "infer from transcript",
  "summary": "3 concise paragraphs covering the full video",
  "keyPoints": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "timestamps": ["00:00 - Introduction", "MM:SS - Topic", "MM:SS - Conclusion"]
}}

TRANSCRIPT:
{trimmed}"""
            response = generate_with_fallback(client, prompt, config)
        else:
            prompt = """Summarize this YouTube video. Return ONLY valid JSON:
{
  "title": "the actual video title",
  "summary": "3 concise paragraphs describing what happens in the video",
  "keyPoints": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "timestamps": ["00:00 - Start", "MM:SS - Topic", "MM:SS - End"]
}"""
            response = generate_with_fallback(
                client,
                [types.Part.from_uri(file_uri=youtube_url, mime_type="video/mp4"), prompt],
                config
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    raw = response.text.strip()
    data = extract_json(raw)

    if not data.get("summary"):
        clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        clean = re.sub(r"\s*```\s*$", "", clean, flags=re.MULTILINE).strip()
        data = {"title": f"Video {video_id}", "summary": clean, "keyPoints": [], "timestamps": []}

    if user_id:
        save_summary_to_supabase(user_id, video_id, req.url, data)

    return SummarizeResponse(
        title=data.get("title", f"Video {video_id}"),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )
