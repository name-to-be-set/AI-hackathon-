
"""
VYOM patches
-------------
Adds response-formatting instructions and overrides the image-generation
functions in agent.py.

Import this module once from app.py:
    import patches
"""

import base64
import os
import random
import re
from urllib.parse import quote

import requests
from requests.exceptions import RequestException

import agent


# ============================================================
# 1. RESPONSE FORMATTING
# ============================================================

FORMAT_RULES = (
    "Formatting rules: write answers in clean Markdown. "
    "Start research answers with a one-sentence summary. "
    "When an answer has two or more parts, use ## headings. "
    "Use short paragraphs and bullet lists for lists of points. "
    "Put **bold** on key terms and important facts, never on whole sentences. "
    "Use tables only when comparing things. "
    "Keep casual chat replies short and plain, without headings."
)

if FORMAT_RULES not in agent.BASE_SYSTEM_PROMPT:
    agent.BASE_SYSTEM_PROMPT = (
        agent.BASE_SYSTEM_PROMPT + " " + FORMAT_RULES
    )


# ============================================================
# 2. IMAGE REQUEST DETECTION
# ============================================================

_IMAGE_REQUEST = re.compile(
    r"^(?:"
    r"please\s+|pls\s+|"
    r"can\s+you\s+|could\s+you\s+|would\s+you\s+|"
    r"i\s+want\s+you\s+to\s+|"
    r"help\s+me\s+|"
    r"now\s+|"
    r")*"
    r"(?:"
    r"draw\b|paint\b|sketch\b|"
    r"illustrate\b|"
    r"generate\b|create\b|make\b|produce\b|design\b|render\b"
    r")",
    re.IGNORECASE,
)

_IMAGE_OBJECT = re.compile(
    r"\b(?:"
    r"image|images|picture|pictures|pic|"
    r"illustration|illustrations|drawing|drawings|"
    r"photo|photos|photograph|photographs|"
    r"artwork|painting|paintings|wallpaper|"
    r"logo|poster|banner|icon|thumbnail"
    r")\b",
    re.IGNORECASE,
)

_IMAGE_FOLLOWUP = re.compile(
    r"^(?:"
    r"now\s+)?"
    r"(?:"
    r"make|create|generate|draw|paint|"
    r"another|one\s+more|"
    r"try"
    r")\b",
    re.IGNORECASE,
)


def _is_image_creation_request(topic: str) -> bool:
    """
    Recognise common image requests, including some short follow-ups.

    Examples:
        Generate an image of a futuristic city
        Can you draw a cat?
        Make me a logo
        Design a poster
        Draw another one
    """

    if not isinstance(topic, str):
        return False

    normalized = re.sub(r"\s+", " ", topic.strip())

    if not normalized:
        return False

    if not _IMAGE_REQUEST.match(normalized):
        return False

    # Normal requests should contain an image-related object.
    if _IMAGE_OBJECT.search(normalized):
        return True

    # Short follow-ups may refer to an image already discussed.
    if _IMAGE_FOLLOWUP.match(normalized):
        return bool(
            re.search(
                r"\b(?:another|one|more|again|same|different)\b",
                normalized,
                re.IGNORECASE,
            )
        )

    # Drawing verbs can also be used without the word "image".
    if re.match(
        r"^(?:(?:please|pls|can you|could you|would you|"
        r"i want you to|help me)\s+)*(?:draw|paint|sketch|illustrate)\b",
        normalized,
        re.IGNORECASE,
    ):
        return True

    return False


# Override the original function in agent.py.
agent._is_image_creation_request = _is_image_creation_request


# ============================================================
# 3. IMAGE PROVIDER CONFIGURATION
# ============================================================

IMAGE_TIMEOUT = 45

HEADERS = {
    "User-Agent": "Mozilla/5.0 (VYOM image generator)"
}

DEFAULT_IMAGE_PROVIDERS = (
    "pollinations",
    "huggingface",
    "gemini",
)


# ============================================================
# 4. COMMON IMAGE RESULT FORMAT
# ============================================================

def _result(content: bytes, mime_type: str) -> dict[str, str]:
    """Convert image bytes into the format expected by app.py."""

    if not content:
        raise ValueError("The image provider returned empty data.")

    if not mime_type.startswith("image/"):
        raise ValueError("The provider did not return a valid image.")

    return {
        "base64": base64.b64encode(content).decode("ascii"),
        "mime_type": mime_type,
        "description": "Generated image",
    }


def _validate_response(response: requests.Response) -> str:
    """Check whether a provider actually returned image data."""

    response.raise_for_status()

    mime_type = (
        response.headers.get("content-type", "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )

    if not mime_type.startswith("image/"):
        raise ValueError(
            f"Provider returned an unexpected content type: {mime_type}"
        )

    if not response.content:
        raise ValueError("Provider returned an empty image.")

    return mime_type


# ============================================================
# 5. POLLINATIONS — FREE-FIRST PROVIDER
# ============================================================

def _from_pollinations(prompt: str) -> dict[str, str]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Image prompt is empty.")

    encoded_prompt = quote(prompt[:900], safe="")

    seed = random.randint(1, 10**6)

    url = (
        "https://image.pollinations.ai/prompt/"
        + encoded_prompt
        + f"?width=1024&height=1024"
        + f"&nologo=true&model=flux&seed={seed}"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=IMAGE_TIMEOUT,
    )

    mime_type = _validate_response(response)

    return _result(response.content, mime_type)


# ============================================================
# 6. HUGGING FACE — OPTIONAL TOKEN
# ============================================================

def _from_huggingface(prompt: str) -> dict[str, str]:
    token = os.getenv("HF_TOKEN", "").strip()

    if not token:
        raise ValueError("HF_TOKEN is not configured.")

    response = requests.post(
        (
            "https://router.huggingface.co/"
            "hf-inference/models/black-forest-labs/FLUX.1-schnell"
        ),
        headers={
            "Authorization": f"Bearer {token}",
            **HEADERS,
        },
        json={
            "inputs": prompt[:900],
        },
        timeout=IMAGE_TIMEOUT,
    )

    mime_type = _validate_response(response)

    return _result(response.content, mime_type)


# ============================================================
# 7. GEMINI — API KEY PROVIDER
# ============================================================

def _from_gemini(prompt: str) -> dict[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    response = requests.post(
        agent.IMAGE_GENERATION_URL,
        params={"key": api_key},
        json={
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"]
            },
        },
        timeout=IMAGE_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    candidates = payload.get("candidates") or []

    if not candidates:
        raise ValueError("Gemini returned no image candidates.")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []

    for part in parts:
        if not isinstance(part, dict):
            continue

        image_data = (
            part.get("inlineData")
            or part.get("inline_data")
        )

        if not isinstance(image_data, dict):
            continue

        encoded = image_data.get("data")

        if not isinstance(encoded, str) or not encoded:
            continue

        mime_type = (
            image_data.get("mimeType")
            or image_data.get("mime_type")
            or "image/png"
        )

        # Ensure the returned data is valid Base64.
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            continue

        return _result(decoded, str(mime_type).lower())

    raise ValueError("Gemini returned no usable image data.")


# ============================================================
# 8. PROVIDER REGISTRY
# ============================================================

_PROVIDERS = {
    "pollinations": _from_pollinations,
    "huggingface": _from_huggingface,
    "gemini": _from_gemini,
}


def _provider_order() -> list[str]:
    """
    Read provider order from environment.

    Example:
        IMAGE_PROVIDERS=pollinations,huggingface,gemini
    """

    configured = os.getenv(
        "IMAGE_PROVIDERS",
        ",".join(DEFAULT_IMAGE_PROVIDERS),
    )

    order = []
    seen = set()

    for item in configured.split(","):
        name = item.strip().lower()

        if name in _PROVIDERS and name not in seen:
            order.append(name)
            seen.add(name)

    # If the environment variable is invalid or empty,
    # use the default provider order.
    if not order:
        order = list(DEFAULT_IMAGE_PROVIDERS)

    return order


# ============================================================
# 9. MAIN IMAGE GENERATOR
# ============================================================

def _generate_image(prompt: str) -> dict[str, str]:
    """
    Try image providers in order.

    A provider failure moves to the next provider.
    The first valid generated image is returned.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Cannot generate an image from an empty prompt.")

    failures = []

    for provider_name in _provider_order():
        provider = _PROVIDERS[provider_name]

        try:
            result = provider(prompt.strip())

            if not isinstance(result, dict):
                raise ValueError("Provider returned an invalid result.")

            if not result.get("base64"):
                raise ValueError("Provider returned no image data.")

            agent.LOGGER.info(
                "Image successfully generated using %s.",
                provider_name,
            )

            return result

        except (
            RequestException,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:

            agent.LOGGER.warning(
                "Image provider %s failed: %s",
                provider_name,
                error,
            )

            failures.append(provider_name)

    agent.LOGGER.error(
        "All configured image providers failed: %s",
        ", ".join(failures),
    )

    raise RuntimeError(
        "All image-generation providers failed. Please try again later."
    )


# Override the original image generator in agent.py.
agent._generate_image = _generate_image
