"""Small improvements applied on top of agent.py without editing that file.

app.py imports this module once. It does two things:
1. Adds formatting rules so replies come with headings, bold key terms and bullets.
2. Replaces the image generator with a free-first one and recognises more
   "draw / generate an image" requests.
"""
import base64
import os
import random
import re
from urllib.parse import quote

import requests
from requests.exceptions import RequestException

import agent

# ---------------------------------------------------------------------------
# 1. Formatting rules for replies
# ---------------------------------------------------------------------------
FORMAT_RULES = (
    "Formatting rules: write answers in clean Markdown. Start research answers "
    "with a one-sentence summary. When the answer has two or more parts, use "
    "'##' headings. Use short paragraphs and bullet lists for lists of points. "
    "Put **bold** on key terms and important facts, never on whole sentences. "
    "Use a table only to compare things. Keep casual chat replies short and "
    "plain, without headings."
)
if FORMAT_RULES not in agent.BASE_SYSTEM_PROMPT:
    agent.BASE_SYSTEM_PROMPT = agent.BASE_SYSTEM_PROMPT + " " + FORMAT_RULES

# ---------------------------------------------------------------------------
# 2. Image requests
# ---------------------------------------------------------------------------
_IMAGE_REQUEST = re.compile(
    r"^(?:(?:please|pls|can you|could you|would you|i want you to|help me)\s+)?"
    r"(?:draw\b|paint\b|sketch\b|"
    r"(?:generate|create|make|produce|design|render)(?:\s+me)?\s+"
    r"(?:an?\s+|the\s+|some\s+)?(?:[\w-]+\s+){0,3}?"
    r"(?:image|images|picture|pictures|pic|illustration|drawing|photo|"
    r"photograph|artwork|painting|wallpaper|logo|poster)\b)",
    re.IGNORECASE,
)


def _is_image_creation_request(topic: str) -> bool:
    return bool(_IMAGE_REQUEST.match(re.sub(r"\s+", " ", topic.strip())))


agent._is_image_creation_request = _is_image_creation_request

# ---------------------------------------------------------------------------
# 3. Image generation: free providers first
#    Order can be changed with the IMAGE_PROVIDERS environment variable,
#    for example "pollinations,huggingface,gemini".
# ---------------------------------------------------------------------------
IMAGE_TIMEOUT = 45
_HEADERS = {"User-Agent": "Mozilla/5.0 (VYOM image generator)"}


def _result(content: bytes, mime_type: str) -> dict[str, str]:
    return {
        "base64": base64.b64encode(content).decode("ascii"),
        "mime_type": mime_type,
        "description": "Generated image",
    }


def _from_pollinations(prompt: str) -> dict[str, str]:
    url = (
        "https://image.pollinations.ai/prompt/" + quote(prompt[:900], safe="")
        + f"?width=1024&height=1024&nologo=true&model=flux&seed={random.randint(1, 10**6)}"
    )
    response = requests.get(url, headers=_HEADERS, timeout=IMAGE_TIMEOUT)
    response.raise_for_status()
    mime = response.headers.get("content-type", "").split(";", 1)[0].strip()
    if not mime.startswith("image/") or not response.content:
        raise ValueError("Pollinations did not return an image.")
    return _result(response.content, mime)


def _from_huggingface(prompt: str) -> dict[str, str]:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("HF_TOKEN is not set.")
    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
        headers={"Authorization": f"Bearer {token}", **_HEADERS},
        json={"inputs": prompt[:900]},
        timeout=IMAGE_TIMEOUT,
    )
    response.raise_for_status()
    mime = response.headers.get("content-type", "").split(";", 1)[0].strip()
    if not mime.startswith("image/") or not response.content:
        raise ValueError("Hugging Face did not return an image.")
    return _result(response.content, mime)


def _from_gemini(prompt: str) -> dict[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    response = requests.post(
        agent.IMAGE_GENERATION_URL,
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        },
        timeout=IMAGE_TIMEOUT,
    )
    response.raise_for_status()
    for part in response.json()["candidates"][0]["content"]["parts"]:
        data = part.get("inlineData") or part.get("inline_data")
        if isinstance(data, dict) and data.get("data"):
            mime = data.get("mimeType") or data.get("mime_type") or "image/png"
            return {"base64": data["data"], "mime_type": str(mime), "description": "Generated image"}
    raise ValueError("Gemini returned no image data.")


_PROVIDERS = {
    "pollinations": _from_pollinations,
    "huggingface": _from_huggingface,
    "gemini": _from_gemini,
}


def _generate_image(prompt: str) -> dict[str, str]:
    order = [
        name.strip().lower()
        for name in os.getenv("IMAGE_PROVIDERS", "pollinations,huggingface,gemini").split(",")
        if name.strip().lower() in _PROVIDERS
    ]
    for name in order:
        try:
            return _PROVIDERS[name](prompt)
        except (RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            agent.LOGGER.warning("Image provider %s failed: %s", name, error)
    raise RuntimeError("All image providers failed.")


agent._generate_image = _generate_image