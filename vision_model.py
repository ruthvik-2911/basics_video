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


def call_vision_model(context_text: str, frame_paths: list[str], question: str, language: str = "English") -> str:
    is_multi_frame = len(frame_paths) > 1

    if is_multi_frame:
        lang_instruction = ""
        if language and language.lower() != "english":
            lang_instruction = (
                f"\n\nCRITICAL MULTILINGUAL INSTRUCTION: You MUST write the 'summary', all step 'title's, and all step 'description's in fluent, natural {language}. "
                "Keep the JSON keys in English ('summary', 'steps', 'step_number', 'title', 'description'), but translate all value contents into rich, fluent {language}."
            )

        prompt = (
            "You are providing a step-by-step overview of a video or document collection using "
            f"{len(frame_paths)} keyframe snapshots/images (ordered chronologically as Image 1, Image 2, etc.) "
            "and transcript/document context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User question: {question}\n\n"
            "Return a JSON object with this EXACT structure:\n"
            "{\n"
            '  "summary": "Brief overall summary in markdown",\n'
            '  "steps": [\n'
            '    {\n'
            '      "step_number": 1,\n'
            '      "title": "Title for Step 1",\n'
            '      "description": "Clear explanation for Step 1 based on Image 1 and context"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Ensure the 'steps' array contains exactly {len(frame_paths)} step items corresponding to Image 1 through Image {len(frame_paths)}."
            f"{lang_instruction}"
        )
    else:
        citation_instruction = (
            "\n\nINLINE CITATION REQUIREMENTS:\n"
            "- You MUST place an inline citation badge immediately beside each paragraph, bullet point, or answer statement that comes from a context snippet.\n"
            "- Format inline citations strictly as `[Source: File Name | Location]` using the exact file name and location provided in the snippet headers (e.g., `[Source: Azure Free Account Walkthrough | 00:30]` or `[Source: Document.pdf | Page 2]`).\n"
            "- Do NOT gather or list citations at the end of your response. Insert each citation badge inline right next to its corresponding sentence or paragraph."
        )

        lang_instruction = ""
        if language and language.lower() != "english":
            lang_instruction = (
                f"\n\nCRITICAL LANGUAGE & FORMATTING RULES:\n"
                f"1. Language: Answer entirely in natural, fluent {language}.\n"
                "2. NO JSON: Do NOT output JSON, JSON keys, quotes around the answer, or curly braces. Output clean Markdown directly.\n"
                "3. Beautiful Structure: Structure your response cleanly using rich Markdown. Start with a direct introductory summary sentence, use bold highlights (e.g. **important point**), and use clean bullet points or numbered lists.\n"
                "4. Inline Citations: At the end of each translated paragraph or key point, include the exact inline citation badge `[Source: File Name | Location]` corresponding to the snippet source."
            )
        else:
            lang_instruction = (
                "\n\nFORMATTING RULES:\n"
                "- Structure your response cleanly using rich Markdown: start with a direct concise statement, use bold highlights and clear bullet points or numbered lists where appropriate for readability.\n"
                "- Do NOT output JSON or curly braces."
            )

        prompt = (
            "You are an expert AI assistant answering a question about a video, document, or knowledge library, using an exact frame snapshot "
            "or document page context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User question: {question}\n\n"
            "Answer clearly, thoroughly, and directly. Ground your answer in what's visible in the frame "
            "and what's said/shown in the context. If the context doesn't actually answer the question, say so."
            f"{citation_instruction}"
            f"{lang_instruction}"
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
