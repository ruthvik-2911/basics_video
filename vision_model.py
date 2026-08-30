# vision_model.py
"""
The vision-model plug-in point. Sends the retrieved transcript/OCR context
+ the exact-timestamp frame to a vision-capable Azure OpenAI deployment
(e.g. gpt-4o) and gets back a grounded answer.
"""

import base64
import mimetypes
import os

from openai import AzureOpenAI

_client = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            timeout=60.0,
            max_retries=3,
        )
    return _client


def call_vision_model(context_text: str, frame_paths: list[str], question: str) -> str:
    is_multi_frame = len(frame_paths) > 1

    if is_multi_frame:
        prompt = (
            "You are providing a step-by-step overview of a video using "
            f"{len(frame_paths)} keyframe snapshots (ordered chronologically as Image 1, Image 2, etc.) "
            "and transcript context.\n\n"
            f"Transcript + OCR context:\n{context_text}\n\n"
            f"User question: {question}\n\n"
            "Return a JSON object with this EXACT structure:\n"
            "{\n"
            '  "summary": "Brief overall summary of the video",\n'
            '  "steps": [\n'
            '    {\n'
            '      "step_number": 1,\n'
            '      "title": "Title for Step 1",\n'
            '      "description": "Explanation for Step 1 based on Image 1 and transcript"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Ensure the 'steps' array contains exactly {len(frame_paths)} step items corresponding to Image 1 through Image {len(frame_paths)}."
        )
    else:
        prompt = (
            "You are answering a question about a video, using an exact frame snapshot "
            "grabbed at a relevant moment plus the transcript/on-screen text (OCR) from around that moment.\n\n"
            f"Transcript + OCR context:\n{context_text}\n\n"
            f"User question: {question}\n\n"
            "Answer clearly and directly. Ground your answer in what's visible in the frame "
            "and what's said/shown in the context. If the context doesn't actually answer the question, say so."
        )

    content = [{"type": "text", "text": prompt}]
    
    # Load and encode each frame
    for path in frame_paths:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            media_type = mimetypes.guess_type(path)[0] or "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_b64}",
                    "detail": "low"  # low detail saves tokens and keeps it fast
                },
            })

    kwargs = {
        "model": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": content}],
    }
    if is_multi_frame:
        kwargs["response_format"] = {"type": "json_object"}

    # Execute with automatic retry on intermittent SSL drops
    last_err = None
    for attempt in range(3):
        try:
            response = _get_client().chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            import time
            time.sleep(1)

    raise last_err
