from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url):
    import re
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


@app.post("/api/summarize")
async def summarize(body: dict):
    url = body.get("url", "")

    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    try:
        # ✅ Extract video ID
        video_id = extract_video_id(url)

        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # ✅ Get transcript (SERVER SIDE → no CORS)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])

        prompt = f"""
Summarize this YouTube transcript.

Return ONLY JSON:

{{
  "summary": "3-5 sentence summary",
  "keyPoints": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "timestamps": ["0:32 - Topic", "1:45 - Topic", "3:20 - Topic"]
}}

Transcript:
{full_text[:12000]}
"""

        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)
        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))