from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import re
import json
import urllib.request
import urllib.parse
import urllib.error
import time

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
    keyPoints: list[str]
    timestamps: list[str]
    transcript_length: int

def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=3000,
            response_mime_type="application/json",
        )
    )

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def supadata_request(endpoint: str, api_key: str) -> dict:
    """Make a request to Supadata API and return parsed JSON."""
    req = urllib.request.Request(
        endpoint,
        headers={
            "x-api-key": api_key,
            "Accept": "application/json",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise HTTPException(status_code=400, detail=f"Supadata error {e.code}: {body}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcript fetch error: {str(e)}")

def fetch_transcript(video_url: str) -> str:
    """Fetch transcript via Supadata API with async job polling support."""
    api_key = os.environ.get("SUPADATA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SUPADATA_API_KEY not configured.")

    # Properly encode the video URL as a query parameter
    encoded_url = urllib.parse.quote(video_url, safe="")
    endpoint = f"https://api.supadata.ai/v1/transcript?url={encoded_url}&text=true&lang=en&mode=native"

    req = urllib.request.Request(
        endpoint,
        headers={
            "x-api-key": api_key,
            "Accept": "application/json",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        # Try to parse error details
        try:
            err_data = json.loads(err_body)
            detail = err_data.get("message") or err_data.get("error") or err_body
        except Exception:
            detail = err_body
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript: {detail}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcript fetch error: {str(e)}")

    # Handle async job (202) — large videos
    if status == 202:
        job_id = body.get("jobId")
        if not job_id:
            raise HTTPException(status_code=400, detail="No job ID returned from Supadata.")

        # Poll for job completion (max 90 seconds)
        poll_url = f"https://api.supadata.ai/v1/transcript/{job_id}"
        for _ in range(90):
            time.sleep(1)
            poll_req = urllib.request.Request(
                poll_url,
                headers={"x-api-key": api_key, "Accept": "application/json"}
            )
            try:
                with urllib.request.urlopen(poll_req, timeout=10) as poll_resp:
                    poll_data = json.loads(poll_resp.read().decode())
            except Exception:
                continue

            poll_status = poll_data.get("status")
            if poll_status == "completed":
                content = poll_data.get("content", "")
                if not content:
                    raise HTTPException(status_code=400, detail="Transcript job completed but content is empty.")
                return content
            elif poll_status == "failed":
                raise HTTPException(status_code=400, detail=f"Transcript job failed: {poll_data.get('error', 'Unknown error')}")

        raise HTTPException(status_code=408, detail="Transcript job timed out. Please try again.")

    # Handle direct response (200)
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="No transcript available for this video.")

    return content


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
    transcript = fetch_transcript(youtube_url)

    max_chars = 80000
    trimmed = transcript[:max_chars]
    if len(transcript) > max_chars:
        trimmed += "\n[Transcript trimmed]"

    prompt = f"""You are an expert content analyst. Below is the real transcript of a YouTube video.
Analyze it and return an accurate JSON summary based ONLY on this transcript.

TRANSCRIPT:
{trimmed}

Rules:
- Base your answer ONLY on the transcript — no hallucination
- Write 4-5 complete paragraphs covering the full video
- Never cut off mid-sentence
- keyPoints must be specific insights from the actual content
- Infer approximate timestamps from transcript position

Return this exact JSON:
{{
  "title": "infer the video title from the transcript content",
  "summary": "4-5 complete paragraphs summarizing the full video accurately",
  "keyPoints": [
    "Specific insight 1",
    "Specific insight 2",
    "Specific insight 3",
    "Specific insight 4",
    "Specific insight 5",
    "Specific insight 6"
  ],
  "timestamps": [
    "00:00 - Introduction",
    "MM:SS - Topic",
    "MM:SS - Topic",
    "MM:SS - Topic",
    "MM:SS - Conclusion"
  ]
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)

    try:
        data = json.loads(raw.strip())
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group()) if m else {
            "title": f"Video {video_id}", "summary": raw, "keyPoints": [], "timestamps": []
        }

    return SummarizeResponse(
        title=data.get("title", f"Video {video_id}"),
        video_id=video_id,
        summary=data.get("summary", ""),
        keyPoints=data.get("keyPoints", []),
        timestamps=data.get("timestamps", []),
        transcript_length=len(transcript),
    )