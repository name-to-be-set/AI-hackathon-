import base64
import hmac
import html
import json
import mimetypes
import os
import random
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
import streamlit as st
from openai import OpenAIError
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from streamlit.errors import StreamlitAPIException, StreamlitSecretNotFoundError

from agent import run_research_stream, transcribe_audio

try:
    from agent import LimitReached
except ImportError:  # older agent.py without the limit message
    class LimitReached(Exception):
        pass
from attachments import run_attachment_stream
import patches  # noqa: F401


def _load_streamlit_secrets_into_environment():
    for name in (
        "TAVILY_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "BIFROST_BASE_URL",
        "BIFROST_API_KEY",
        "BIFROST_MODELS",
        "GROQ_MODELS",
        "GEMINI_MODELS",
        "ADMIN_PASSWORD",
    ):
        if os.getenv(name, "").strip():
            continue
        try:
            value = st.secrets.get(name)
        except StreamlitSecretNotFoundError:
            continue
        if value is not None and str(value).strip():
            os.environ[name] = str(value)


_load_streamlit_secrets_into_environment()

APP_DIR = Path(__file__).resolve().parent

DEFAULT_APP_URL = "https://ai-research-assistant-sw7p.onrender.com"

CHAT_PLACEHOLDERS = (
    "Ask me anything...",
    "What's on your mind?",
    "Type or tap the mic to speak...",
    "What would you like to explore?",
)

TIME_GREETING_LINES = {
    "morning": (
        "Good morning! Ready to start the day?",
        "Good morning! How was breakfast? What would you like to work on?",
        "Good morning! What would you like to explore today?",
    ),
    "afternoon": (
        "Good afternoon! Hope lunch went well. What can I help with?",
        "Good afternoon! Ready to make a little progress today?",
        "Good afternoon! What would you like to tackle today?",
    ),
    "evening": (
        "Good evening! How was your day?",
        "Good evening! What would you like to work on this evening?",
        "Good evening! How can I help you wrap up the day?",
    ),
    "night": (
        "Good night! You're up late—what can I help you with?",
        "It's getting late! What would you like to work through?",
        "Good night! What would you like to finish before the day ends?",
    ),
}

FILE_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "txt", "md", "csv"]

THEME_OPTIONS = ("System", "Light", "Dark")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def _initial_theme() -> str:
    """Starting theme from the URL (?theme=light|dark), otherwise System."""
    raw = (
        st.query_params.get("theme")
        or st.query_params.get("renderer_theme")
        or ""
    )
    normalized = str(raw).strip().lower()
    if normalized == "light":
        return "Light"
    if normalized == "dark":
        return "Dark"
    return "System"


def _resolve_renderer_theme() -> str:
    """The user's explicit choice always wins."""
    value = st.session_state.get("user_theme_override")
    return value if value in THEME_OPTIONS else _initial_theme()


_THEME_SCRIPT = """
<script>
(() => {
    // The newest mode always wins: store it on window so listeners
    // registered earlier never use a stale value.
    window.__vyomMode = __MODE__;

    const resolve = () => {
        const mode = window.__vyomMode || "System";
        if (mode === "Light") return "light";
        if (mode === "Dark") return "dark";

        const external = String(
            window.__VYOM_RENDERER_THEME__ ||
            document.documentElement.dataset.rendererTheme ||
            ""
        ).trim().toLowerCase();
        if (external === "light" || external === "dark") return external;

        return window.matchMedia("(prefers-color-scheme: light)").matches
            ? "light"
            : "dark";
    };

    const sync = () => {
        document.documentElement.dataset.vyomTheme = resolve();
    };

    window.__vyomSyncTheme = sync;
    sync();

    if (!window.__vyomThemeBound) {
        window.__vyomThemeBound = true;
        const mq = window.matchMedia("(prefers-color-scheme: light)");
        mq.addEventListener?.("change", () => window.__vyomSyncTheme());
        window.addEventListener("message", (event) => {
            const data = event?.data;
            if (!data || typeof data !== "object") return;
            const candidate = String(data.theme || data.rendererTheme || "").toLowerCase();
            if (candidate === "light" || candidate === "dark" || candidate === "system") {
                window.__VYOM_RENDERER_THEME__ = candidate;
                window.__vyomSyncTheme();
            }
        });
    }
})();
</script>
"""


def _apply_renderer_theme_script(renderer_mode: str) -> str:
    return _THEME_SCRIPT.replace("__MODE__", json.dumps(renderer_mode))


@st.cache_resource(show_spinner=False)
def _background_data_uri(theme: str):
    """Find <theme>_bg.(jpg|png|webp) and embed it as a small data URI.

    Looks in static/, assets/, images/ and next to app.py.
    """
    for folder in ("static", "assets", "images", ""):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = APP_DIR / folder / f"{theme}_bg{ext}"
            if not path.is_file():
                continue

            try:
                from PIL import Image

                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((1920, 1920))
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG", quality=80, optimize=True)
                data, mime = buffer.getvalue(), "image/jpeg"
            except Exception:
                data = path.read_bytes()
                mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"

            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    return None


def _background_css() -> str:
    rules = []

    for theme, rgb in (("dark", "8, 10, 19"), ("light", "243, 246, 251")):
        uri = _background_data_uri(theme)
        if not uri:
            continue

        layer = (
            f"linear-gradient(rgba({rgb}, .45), rgba({rgb}, .45)), "
            f'url("{uri}")'
        )
        rules.append(
            f'html[data-vyom-theme="{theme}"] .stApp {{ background-image: {layer} !important; }}'
        )
        if theme == "dark":
            rules.append(
                f"html:not([data-vyom-theme]) .stApp {{ background-image: {layer} !important; }}"
            )

    return "<style>" + "\n".join(rules) + "</style>" if rules else ""


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def _logo_path():
    """Find logo.jpg / .jpeg / .webp / .png in static/, assets/, images/ or next to app.py."""
    for folder in ("static", "assets", "images", ""):
        for extension in ("jpg", "jpeg", "webp", "png"):
            path = APP_DIR / folder / f"logo.{extension}"
            if path.is_file():
                return path
    return None


@st.cache_resource(show_spinner=False)
def _logo_bytes():
    """Crop white margins, shrink to <=360px, round the corners smoothly.

    Returns small PNG bytes, or None (the app then shows the text title)."""
    path = _logo_path()
    if path is None:
        return None

    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageOps

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")

        # flatten on white (JPG / transparent backgrounds)
        flat = Image.new("RGB", image.size, (255, 255, 255))
        flat.paste(image, mask=image.split()[3])

        # crop the empty white margins
        pixels = np.asarray(flat).astype(int)
        ys, xs = np.where(pixels.min(axis=2) < 225)
        if len(xs):
            pad = int(max(flat.size) * 0.03)
            flat = flat.crop(
                (
                    max(int(xs.min()) - pad, 0),
                    max(int(ys.min()) - pad, 0),
                    min(int(xs.max()) + pad + 1, flat.width),
                    min(int(ys.max()) + pad + 1, flat.height),
                )
            )

        flat.thumbnail((360, 360), Image.LANCZOS)
        width, height = flat.size

        # smooth rounded corners (mask drawn 8x bigger, then shrunk)
        scale = 8
        mask = Image.new("L", (width * scale, height * scale), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, width * scale - 1, height * scale - 1),
            radius=int(width * 0.14 * scale),
            fill=255,
        )
        card = flat.convert("RGBA")
        card.putalpha(mask.resize((width, height), Image.LANCZOS))

        buffer = BytesIO()
        card.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    except Exception as error:
        print(f"VYOM: could not prepare the logo: {error}")
        return None


def _logo_data_uri():
    data = _logo_bytes()
    if not data:
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


@st.cache_resource
def _feedback_store():
    return {"like": {}, "dislike": {}}


def _set_feedback(message, choice, question, reply):
    store = _feedback_store()
    message_id = message.setdefault("id", uuid.uuid4().hex)
    new_value = "none" if message.get("feedback") == choice else choice
    message["feedback"] = new_value

    store["like"].pop(message_id, None)
    store["dislike"].pop(message_id, None)

    if new_value in ("like", "dislike"):
        store[new_value][message_id] = {
            "time": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M"),
            "question": question[:300],
            "reply": reply[:500],
        }


def _render_admin_panel():
    if st.query_params.get("admin") != "1":
        return

    with st.expander("Admin", expanded=False):
        expected = os.getenv("ADMIN_PASSWORD", "").strip()
        if not expected:
            st.caption("Set ADMIN_PASSWORD to enable this panel.")
            return

        entered = st.text_input("Password", type="password", key="admin_password")
        if not hmac.compare_digest(entered.encode(), expected.encode()):
            return

        store = _feedback_store()
        for kind, title in (("like", "Likes"), ("dislike", "Dislikes")):
            rows = list(store[kind].values())
            st.markdown(f"**{title} ({len(rows)})**")
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Chat management
# ---------------------------------------------------------------------------

def _new_chat():
    return {"id": uuid.uuid4().hex, "title": "New chat", "messages": []}


def _init_chats():
    if "chats" not in st.session_state:
        chat = _new_chat()
        st.session_state["chats"] = [chat]
        st.session_state["active_chat"] = chat["id"]


def _active_chat():
    chats = st.session_state["chats"]
    for chat in chats:
        if chat["id"] == st.session_state.get("active_chat"):
            return chat
    return chats[0]


def _active_messages():
    return _active_chat()["messages"]


def _start_new_chat():
    if _active_messages():
        chat = _new_chat()
        st.session_state["chats"].append(chat)
        st.session_state["active_chat"] = chat["id"]

    st.session_state["greeting"] = _time_greeting()
    st.session_state["generation_active"] = False
    st.session_state.pop("share_message", None)


def _switch_chat(chat_id):
    st.session_state["active_chat"] = chat_id
    st.session_state["generation_active"] = False
    st.session_state.pop("share_message", None)


def _regenerate():
    messages = _active_messages()

    while messages and messages[-1].get("role") == "assistant":
        messages.pop()

    if messages and messages[-1].get("role") == "user":
        user_message = messages.pop()
        st.session_state["pending_submission"] = {
            "topic": user_message["content"],
            "files": user_message.get("files", []),
        }


def _stop_generation():
    """Mark the response as stopped when Streamlit processes the button event."""
    st.session_state["generation_active"] = False

    for message in reversed(_active_messages()):
        if message.get("role") == "assistant" and not message.get("result"):
            message.pop("partial_report", None)
            message["generation_stopped"] = True
            break


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _user_text(content, files):
    names = [name for name, _, _ in files or []]
    if not names:
        return content
    return content + "\n\n" + "  \n".join(f"📎 `{name}`" for name in names)


def _loader_html(stage="Thinking..."):
    return (
        '<div class="vyom-loader"><span class="vyom-dots">'
        "<span></span><span></span><span></span></span>"
        f"<span>{html.escape(stage)}</span></div>"
    )


def _time_greeting(hour=None):
    if hour is None:
        hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour

    if 5 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 17:
        period = "afternoon"
    elif 17 <= hour < 21:
        period = "evening"
    else:
        period = "night"

    return random.choice(TIME_GREETING_LINES[period])


def _visible_report(report):
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", report)


def _html_inline(body):
    """Render HTML sized to its content (so it doesn't stretch a row)."""
    try:
        st.html(body, width="content")
    except TypeError:
        st.html(body)


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _format_inline_pdf(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    return escaped


@st.cache_data(show_spinner=False, max_entries=64)
def _build_pdf(report, sources):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="AI Research Assistant response",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            leading=15,
            spaceAfter=7,
            alignment=TA_LEFT,
        )
    )

    story = []

    for line in _visible_report(report).splitlines():
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 5))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            style = styles["Heading1"] if len(heading.group(1)) < 3 else styles["Heading2"]
            story.append(Paragraph(_format_inline_pdf(heading.group(2)), style))
            continue

        bullet = re.match(r"^(?:[-*])\s+(.+)$", stripped)
        text = f"- {bullet.group(1)}" if bullet else stripped
        story.append(Paragraph(_format_inline_pdf(text), styles["ReportBody"]))

    if sources:
        story.append(Paragraph("Sources", styles["Heading2"]))
        for source in sources:
            title = html.escape(str(source.get("title", "Source")))
            url = html.escape(str(source.get("url", "")), quote=True)
            story.append(
                Paragraph(
                    f'{title}<br/><link href="{url}">{url}</link>',
                    styles["ReportBody"],
                )
            )

    document.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_image_bytes(url):
    try:
        response = requests.get(
            url,
            timeout=6,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8",
            },
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()

    if (
        not content_type.startswith("image/")
        or content_type == "image/svg+xml"
        or not response.content
        or len(response.content) > 5_000_000
    ):
        return None

    return response.content


def _render_images(images):
    if not isinstance(images, list):
        return

    pending = []

    for image in images[:4]:
        if not isinstance(image, dict):
            continue

        description = image.get("description") or None
        url = image.get("url")
        encoded = image.get("base64")

        if isinstance(url, str) and url.startswith(("https://", "http://")):
            pending.append((url, None, description))
        elif isinstance(encoded, str) and encoded.strip():
            try:
                decoded = base64.b64decode(encoded, validate=True)
                if decoded:
                    pending.append((None, decoded, description))
            except (ValueError, base64.binascii.Error):
                continue

    if not pending:
        return

    urls = [url for url, _, _ in pending if url]
    fetched = {}

    if urls:
        with ThreadPoolExecutor(max_workers=4) as executor:
            fetched = dict(zip(urls, executor.map(_fetch_image_bytes, urls)))

    valid_images = []

    for url, data, description in pending:
        image_data = data if url is None else fetched.get(url)
        if image_data:
            valid_images.append((image_data, description))

    if not valid_images:
        return

    columns = st.columns(len(valid_images), gap="small")

    for column, (image_data, description) in zip(columns, valid_images):
        try:
            with column:
                st.image(image_data, caption=description, width="stretch")
        except (StreamlitAPIException, ValueError, OSError):
            continue


# ---------------------------------------------------------------------------
# Copy and sharing (static markup; behaviour is wired by the global script)
# ---------------------------------------------------------------------------

_COPY_SVG = (
    '<svg class="vyom-copy-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="9" width="13" height="13" rx="2"></rect>'
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>'
)

_DONE_SVG = (
    '<svg class="vyom-done" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 6L9 17l-5-5"></path></svg>'
)


def _copy_icon_html(report):
    encoded = base64.b64encode(report.encode("utf-8")).decode("ascii")

    return (
        '<button type="button" class="vyom-copy" aria-label="Copy reply" '
        f'title="Copy" data-text="{encoded}"></button>'
    )


def _render_copy_button(report):
    """Copy button that carries its own script, so it does not depend on the
    page-level script. Falls back to the old markup if components are missing."""
    mode = st.session_state.get("user_theme_override", "System")
    encoded = base64.b64encode(report.encode("utf-8")).decode("ascii")

    if mode == "Light":
        fg, border, hover, accent = "#1c2b42", "rgba(65,100,143,.30)", "rgba(205,220,240,.7)", "#1f86a6"
    else:
        fg, border, hover, accent = "#edf1fa", "rgba(160,177,205,.28)", "rgba(130,151,190,.14)", "#48b6ce"

    media = ""
    if mode == "System":
        media = (
            "@media (prefers-color-scheme: light){"
            "button{border-color:rgba(65,100,143,.30);color:#1c2b42}"
            "button:hover{background:rgba(205,220,240,.7);border-color:#1f86a6}"
            "button.ok{color:#1f86a6}}"
        )

    document = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;padding:0;background:transparent;overflow:hidden}}
body{{display:flex;align-items:center;justify-content:flex-start;height:34px}}
button{{box-sizing:border-box;width:32px;height:32px;border-radius:50%;border:1px solid {border};
background:transparent;color:{fg};cursor:pointer;padding:0;display:flex;align-items:center;
justify-content:center;transition:background .2s,border-color .2s,color .2s}}
button:hover{{background:{hover};border-color:{accent}}}
button.ok{{color:{accent}}}
svg{{width:18px;height:18px;pointer-events:none}}
.d{{display:none}} .ok .c{{display:none}} .ok .d{{display:block}}
{media}
</style></head><body>
<button id="b" type="button" title="Copy" aria-label="Copy reply">
<svg class="c" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
<svg class="d" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
</button>
<script>
const text = new TextDecoder().decode(Uint8Array.from(atob("{encoded}"), c => c.charCodeAt(0)));
const b = document.getElementById("b");
b.addEventListener("click", async () => {{
  let ok = false;
  try {{
    await navigator.clipboard.writeText(text);
    ok = true;
  }} catch (e) {{
    const t = document.createElement("textarea");
    t.value = text;
    t.style.position = "fixed";
    t.style.opacity = "0";
    document.body.appendChild(t);
    t.select();
    try {{ ok = document.execCommand("copy"); }} catch (e2) {{}}
    t.remove();
  }}
  if (ok) {{
    b.classList.add("ok");
    setTimeout(() => b.classList.remove("ok"), 1200);
  }}
}});
</script></body></html>"""

    try:
        import streamlit.components.v1 as components

        components.html(document, height=34, width=34)
    except Exception:
        _html_inline(_copy_icon_html(report))


def _native_share_html(report):
    encoded = base64.b64encode(report.encode("utf-8")).decode("ascii")

    return (
        '<button type="button" class="vyom-native-share" '
        f'data-text="{encoded}" '
        'style="display:none;width:100%;padding:0.5rem 0.8rem;'
        "border-radius:0.5rem;border:1px solid var(--vyom-border);"
        'background:transparent;color:inherit;font:inherit;cursor:pointer;text-align:left;">'
        "Share with your phone apps…</button>"
    )


def _share_links(report):
    app_url = os.getenv("APP_PUBLIC_URL", DEFAULT_APP_URL)
    short = report if len(report) <= 1500 else report[:1500].rsplit(" ", 1)[0] + "…"
    tweet = report if len(report) <= 240 else report[:237].rsplit(" ", 1)[0] + "…"

    return [
        ("WhatsApp", f"https://wa.me/?text={quote(short, safe='')}"),
        ("Telegram", f"https://t.me/share/url?url={quote(app_url, safe='')}&text={quote(short, safe='')}"),
        ("X (Twitter)", f"https://twitter.com/intent/tweet?text={quote(tweet, safe='')}"),
        ("Facebook (shares the app link)", f"https://www.facebook.com/sharer/sharer.php?u={quote(app_url, safe='')}"),
        ("Email", f"mailto:?subject={quote('VYOM reply', safe='')}&body={quote(short, safe='')}"),
    ]


# ---------------------------------------------------------------------------
# Reply actions
# ---------------------------------------------------------------------------

def _render_reply_actions(message, key_prefix, question="", is_last=False):
    result = message.get("result")
    report = result.get("report") if isinstance(result, dict) else message.get("error")

    if not isinstance(report, str) or not report.strip():
        return

    report = _visible_report(report)
    feedback = message.get("feedback")

    valid_sources = []
    if isinstance(result, dict):
        sources = result.get("sources")
        if isinstance(sources, list):
            valid_sources = [
                source for source in sources
                if isinstance(source, dict)
                and isinstance(source.get("url"), str)
                and source["url"].startswith(("https://", "http://"))
            ]

    with st.container(
        key=f"{key_prefix}_actions",
        horizontal=True,
        vertical_alignment="center",
        gap="small",
    ):
        st.button(
            "\u200b",
            icon=":material/thumb_up:" if feedback == "like" else ":material/thumb_up_off_alt:",
            key=f"{key_prefix}_like",
            help="Good reply",
            type="primary" if feedback == "like" else "tertiary",
            on_click=_set_feedback,
            args=(message, "like", question, report),
        )

        st.button(
            "\u200b",
            icon=":material/thumb_down:" if feedback == "dislike" else ":material/thumb_down_off_alt:",
            key=f"{key_prefix}_dislike",
            help="Bad reply",
            type="primary" if feedback == "dislike" else "tertiary",
            on_click=_set_feedback,
            args=(message, "dislike", question, report),
        )

        if is_last:
            st.button(
                "\u200b",
                icon=":material/refresh:",
                key=f"{key_prefix}_regen",
                help="Regenerate",
                type="tertiary",
                on_click=_regenerate,
            )

        _render_copy_button(report)

        with st.popover(
            "\u200b",
            icon=":material/share:",
            help="Share this reply",
            type="tertiary",
            key=f"{key_prefix}_sharepop",
        ):
            st.html(_native_share_html(report))

            for label, url in _share_links(report):
                st.link_button(label, url, width="stretch")

            st.download_button(
                "Save as .txt (Notepad)",
                data=report,
                file_name="vyom-reply.txt",
                mime="text/plain",
                key=f"{key_prefix}_txt",
                icon=":material/description:",
                width="stretch",
            )

        if isinstance(result, dict):
            st.download_button(
                "\u200b",
                data=_build_pdf(report, valid_sources),
                file_name="response.pdf",
                mime="application/pdf",
                key=f"{key_prefix}_pdf",
                icon=":material/picture_as_pdf:",
                help="Download this reply as a PDF",
                type="tertiary",
            )


# ---------------------------------------------------------------------------
# Conversation history and rendering
# ---------------------------------------------------------------------------

def _conversation_history(messages):
    history = []

    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            content = message["content"]

            if context := message.get("attachment_context"):
                content += "\n\n[Attached content]\n" + str(context)[:6000]

            history.append({"role": "user", "content": content})

        elif message.get("role") == "assistant":
            result = message.get("result")

            if isinstance(result, dict) and isinstance(result.get("report"), str):
                history.append({"role": "assistant", "content": result["report"]})
            elif isinstance(message.get("error"), str):
                history.append({"role": "assistant", "content": message["error"]})

    return history[-10:]


def _render_result(result, key_prefix="result"):
    if result.get("error"):
        st.error(str(result["error"]))
        return

    report = result.get("report")

    if not isinstance(report, str) or not report.strip():
        st.error("The research agent did not return a report.")
        return

    report = _visible_report(report)

    st.markdown(report)

    if result.get("kind") == "chat":
        if result.get("images"):
            _render_images(result.get("images"))
        return

    _render_images(result.get("images"))

    sources = result.get("sources")

    if not isinstance(sources, list):
        st.error("The research agent returned an invalid source list.")
        return

    valid_sources = [
        source for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("title"), str)
        and isinstance(source.get("url"), str)
        and source["url"].startswith(("https://", "http://"))
    ]

    if valid_sources:
        with st.expander(
            "Sources",
            expanded=False,
            key=f"{key_prefix}_sources",
            type="compact",
        ):
            for index, source in enumerate(valid_sources, start=1):
                st.link_button(
                    f"{index}. {source['title']}",
                    source["url"],
                    key=f"{key_prefix}_source_{index}",
                )
    else:
        st.info("No source links were found for this topic.")

    download_content = report

    if valid_sources:
        source_lines = [
            f"- [{source['title']}]({source['url']})"
            for source in valid_sources
        ]
        download_content += "\n\n## Sources\n\n" + "\n".join(source_lines)

    st.download_button(
        "Download report (.md)",
        data=download_content,
        file_name="research-report.md",
        mime="text/markdown",
        key=f"{key_prefix}_download",
        type="tertiary",
    )


def _render_conversation(messages):
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            with st.container(key=f"user_msg_{index}"):
                st.markdown(_user_text(message["content"], message.get("files")))
            continue

        question = ""
        if index > 0 and messages[index - 1].get("role") == "user":
            question = str(messages[index - 1].get("content", ""))

        is_last = index == len(messages) - 1

        with st.container(key=f"assistant_msg_{index}"):
            if error := message.get("error"):
                st.error(error)
                if detail := message.get("error_detail"):
                    st.caption(detail)

            elif result := message.get("result"):
                _render_result(result, key_prefix=f"chat_{index}")

            elif message.get("generation_stopped"):
                st.caption("You stopped this response.")
                if is_last:
                    st.button(
                        "\u200b",
                        icon=":material/refresh:",
                        key=f"chat_{index}_regen",
                        help="Try again",
                        type="tertiary",
                        on_click=_regenerate,
                    )

            _render_reply_actions(
                message,
                key_prefix=f"chat_{index}",
                question=question,
                is_last=is_last,
            )


# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VYOM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_CSS = """
<style>
@import url("https://googleapis.com");

:root {
    --fs-body:1rem;
    --fs-h1:1.6rem;
    --fs-h2:1.3rem;
    --fs-h3:1.12rem;
    --fs-small:.85rem;
    --fs-table:.85rem;
    --fs-title:2.1rem;
    --fs-greeting:3rem;
}

.stApp {font-family:"DM Sans","Segoe UI",system-ui,sans-serif;}

html,body,.stApp,[data-testid="stAppViewContainer"],
[data-testid="stMain"],section.main {
    max-width:100vw;
    overflow-x:hidden;
}

[data-testid="stMainBlockContainer"],section.main>div {
    width:100%;
    max-width:none;
    margin-inline:auto;
    padding:3.5rem 1.25rem 9rem;
}

[data-testid="stMarkdownContainer"] {
    overflow-wrap:anywhere;
    word-break:break-word;
}

[data-testid="stMarkdownContainer"] pre {white-space:pre-wrap;}
[data-testid="stMarkdownContainer"] img,
[data-testid="stImage"] img {max-width:100%;height:auto;}

[data-testid="stMarkdownContainer"] table {
    display:block;
    width:max-content;
    max-width:100%;
    overflow-x:auto;
    -webkit-overflow-scrolling:touch;
    border-collapse:collapse;
    font-size:var(--fs-table);
}

[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] td {
    padding:.3rem .55rem!important;
    min-width:5.5rem;
    overflow-wrap:normal;
    word-break:normal;
    vertical-align:top;
}

header[data-testid="stHeader"] {background:transparent;}

.st-key-app_title {
    position:fixed;
    top:.4rem;
    left:4.4rem;
    z-index:1000002;
    pointer-events:none;
}

.st-key-app_title div[data-testid="stMarkdownContainer"] h1 {
    margin:0;
    padding:0;
    font-family:"Cormorant Garamond",Georgia,serif;
    font-size:var(--fs-title);
    font-style:italic;
    font-weight:700;
    letter-spacing:.18em;
    line-height:1;
    text-transform:uppercase;
    white-space:nowrap;
}

[data-testid^="stChatMessageAvatar"] {display:none!important;}

div[class*="st-key-user_msg_"] {
    align-items:flex-end;
    margin-bottom:.4rem;
}

div[class*="st-key-user_msg_"] [data-testid="stElementContainer"] {
    width:fit-content;
    max-width:85%;
    margin-left:auto;
    padding:.65rem .95rem;
    border:1px solid var(--vyom-border);
    border-radius:1.1rem 1.1rem .25rem 1.1rem;
    display:flex;
    align-items:center;
}

div[class*="st-key-user_msg_"] [data-testid="stMarkdown"],
div[class*="st-key-user_msg_"] [data-testid="stMarkdownContainer"] {
    margin:0!important;
    padding:0!important;
}

div[class*="st-key-user_msg_"] [data-testid="stMarkdownContainer"] p {
    margin:0!important;
    line-height:1.45;
}

div[class*="st-key-assistant_msg_"] {
    width:100%;
    min-width:0;
    margin-bottom:.8rem;
}

.greeting-text {
    font-size:clamp(2rem,5vw,var(--fs-greeting));
    font-weight:600;
    line-height:1.2;
}

/* Centers the greeting and icon horizontally and moves them to the vertical middle */
div:has(> .greeting-text) {
    display:flex;
    justify-content:center;
    align-items:center;
    text-align:center;
    margin-top:28vh;
    width:100%;
}

.jasmine-flower {
    flex:0 0 auto;
    color:var(--vyom-strong);
    font-size:clamp(3rem,7vw,5rem);
    line-height:1;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size:var(--fs-body);
    line-height:1.7;
}

[data-testid="stMarkdownContainer"] h1 {
    font-size:var(--fs-h1);
    font-weight:700;
    line-height:1.25;
    margin:1.1rem 0 .5rem;
}

[data-testid="stMarkdownContainer"] h2 {
    font-size:var(--fs-h2);
    font-weight:700;
    line-height:1.3;
    margin:1.1rem 0 .45rem;
    padding-bottom:.25rem;
    border-bottom:1px solid var(--vyom-border);
}

[data-testid="stMarkdownContainer"] h3 {
    font-size:var(--fs-h3);
    font-weight:600;
    line-height:1.35;
    margin:.9rem 0 .35rem;
}

[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol {
    padding-left:1.4rem;
    margin:.4rem 0 .8rem;
}

[data-testid="stMarkdownContainer"] li {margin:.25rem 0;}

[data-testid="stMarkdownContainer"] strong {
    font-weight:700;
    background:var(--vyom-hl);
    border-radius:.25rem;
    padding:0 .2em;
}

[data-testid="stMarkdownContainer"] :not(pre)>code {
    background:var(--vyom-soft);
    padding:.1em .35em;
    border-radius:.3rem;
    font-size:.9em;
}

[data-testid="stCaptionContainer"] {font-size:var(--fs-small);}

.vyom-loader {
    display:flex;
    align-items:center;
    gap:.6rem;
    color:var(--vyom-dim);
    font-size:.9rem;
    padding:.3rem 0;
}

.vyom-dots {display:inline-flex;gap:4px;}

.vyom-dots span {
    width:7px;
    height:7px;
    border-radius:50%;
    background:currentColor;
    animation:vyom-bounce 1.2s infinite ease-in-out;
}

.vyom-dots span:nth-child(2) {animation-delay:.15s;}
.vyom-dots span:nth-child(3) {animation-delay:.3s;}

@keyframes vyom-bounce {
    0%,80%,100% {opacity:.3;transform:translateY(0);}
    40% {opacity:1;transform:translateY(-4px);}
}

@media(prefers-reduced-motion:reduce) {
    .vyom-dots span {animation:none;opacity:.7;}
}

/* The real Stop button stays in the DOM (the pause overlay clicks it) */
.st-key-stop_generation {
    position:absolute!important;
    width:1px;
    height:1px;
    overflow:hidden;
    opacity:0;
    pointer-events:none;
}

#vyom-pause {
    position:fixed;
    z-index:1000003;
    display:none;
    align-items:center;
    justify-content:center;
    width:38px;
    height:38px;
    padding:0;
    border-radius:50%;
    cursor:pointer;
}

#vyom-pause.show {display:flex;}

div[data-testid="stBottomBlockContainer"] {
    max-width:none!important;
    padding-inline:1.25rem!important;
}

div[data-testid="stChatInput"] {
    width:100%!important;
    max-width:none!important;
    margin:0!important;
    padding:0!important;
}

.st-key-chat_input button {
    width:32px;
    height:32px;
    min-height:32px;
    padding:0;
    border-radius:50%;
}

div[data-testid="stChatInput"]>div {
    border-radius:16px!important;
    outline:none!important;
}

div[data-testid="stChatInput"] textarea {
    min-height:40px!important;
    max-height:4.5rem;
    font-size:.95rem;
    font-family:inherit;
    caret-color:currentColor!important;
    cursor:text;
    border-color:transparent!important;
    outline:none!important;
    box-shadow:none!important;
}

#vyom-scroll-down {
    position:fixed;
    left:50%;
    bottom:6rem;
    transform:translateX(-50%);
    z-index:1000;
    display:none;
    align-items:center;
    justify-content:center;
    width:36px;
    height:36px;
    padding:0;
    border-radius:50%;
    box-shadow:0 4px 16px rgba(0,0,0,.35);
    cursor:pointer;
}

#vyom-scroll-down.show {display:flex;}

section[data-testid="stSidebar"] div[data-testid="stButton"]>button {
    justify-content:flex-start;
    text-align:left;
    min-height:2rem;
    padding:.35rem .6rem;
    border-radius:.55rem;
}

section[data-testid="stSidebar"] div[data-testid="stButton"]>button>div {
    justify-content:flex-start!important;
}

.st-key-new_chat_top button {border-radius:10px;}

@media(min-width:641px) {
    [data-testid="stMainBlockContainer"],
    section.main>div {max-width:820px;}

    div[data-testid="stBottomBlockContainer"] {
        max-width:820px!important;
        margin-inline:auto;
    }
}

@media(max-width:640px) {
    [data-testid="stMainBlockContainer"],
    section.main>div {padding:3.5rem .5rem 9rem;}

    div[data-testid="stBottomBlockContainer"] {padding-inline:.5rem!important;}

    div[class*="st-key-user_msg_"] [data-testid="stElementContainer"] {
        max-width:92%!important;
    }

    [data-testid="stChatInput"] textarea {font-size:16px!important;}
}
</style>
"""


st.markdown(BASE_CSS, unsafe_allow_html=True)

if "user_theme_override" not in st.session_state:
    st.session_state["user_theme_override"] = _initial_theme()

_renderer_theme = _resolve_renderer_theme()
st.session_state["theme_mode"] = _renderer_theme

_THEME_CSS_PATH = APP_DIR / "vyom_theme.css"

try:
    st.markdown(
        f"<style>{_THEME_CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
except OSError:
    st.warning("VYOM theme stylesheet was not found; using the default styling.")

if _bg_css := _background_css():
    st.markdown(_bg_css, unsafe_allow_html=True)

st.html(
    _apply_renderer_theme_script(_renderer_theme),
    unsafe_allow_javascript=True,
)

# ---------------------------------------------------------------------------
# Intro animation (same look as index.html). Plays once per session and is
# skipped when the visitor already saw index.html (it adds ?from=splash).
# The HTML part is kept on one line on purpose: markdown would turn
# indented / blank-line-separated HTML into a code block.
# ---------------------------------------------------------------------------

_INTRO_HTML = (
    "<style>"
    "#vyom-intro{position:fixed;inset:0;z-index:2000000;display:flex;align-items:center;"
    "justify-content:center;padding:1.25rem;"
    "background:radial-gradient(60rem 40rem at 50% -10%,rgba(160,170,255,.22),transparent 60%),var(--vyom-bg,#0e0f13);"
    "color:var(--vyom-fg,#e8eaed);animation:vyom-intro-out .6s ease 2.4s forwards;}"
    "#vyom-intro .vi-box{width:100%;max-width:30rem;text-align:center;}"
    "#vyom-intro .vyom-search-icon{width:3.4rem;height:3.4rem;margin:0 auto .6rem;animation:vi-rise .7s ease both;}"
    "#vyom-intro .vi-title{font-family:'Cormorant Garamond',Georgia,serif;font-style:italic;font-weight:700;"
    "font-size:clamp(2.6rem,10vw,3.6rem);letter-spacing:.18em;text-transform:uppercase;margin:0 0 .4rem;"
    "background:linear-gradient(100deg,#e77abf,#9b83e8 55%,#53b9d4);-webkit-background-clip:text;"
    "background-clip:text;color:transparent;animation:vi-rise .8s ease both;}"
    "#vyom-intro .vi-logo{display:block;width:min(13rem,58vw);height:auto;margin:0 auto 1rem;"
    "box-shadow:0 10px 40px rgba(72,182,206,.25);border-radius:1.4rem;animation:vi-rise .8s ease both;}"
    "#vyom-intro .vi-sub{color:var(--vyom-dim,#aab5c9);margin:0 0 1.6rem;animation:vi-rise .8s .1s ease both;}"
    "#vyom-intro .vi-chat{text-align:left;display:grid;gap:.7rem;margin-bottom:1.6rem;}"
    "#vyom-intro .vi-me{justify-self:end;max-width:80%;padding:.6rem .9rem;border:1px solid var(--vyom-border,rgba(255,255,255,.14));"
    "background:var(--vyom-soft,rgba(255,255,255,.07));border-radius:1.1rem 1.1rem .25rem 1.1rem;font-size:.95rem;"
    "animation:vi-rise .7s .3s ease both;}"
    "#vyom-intro .vi-bot{display:grid;gap:.45rem;animation:vi-rise .7s .7s ease both;}"
    "#vyom-intro .vi-bot i{display:block;height:.7rem;border-radius:.4rem;width:100%;"
    "background:linear-gradient(90deg,var(--vyom-soft,rgba(255,255,255,.07)) 20%,var(--vyom-border,rgba(255,255,255,.14)) 50%,var(--vyom-soft,rgba(255,255,255,.07)) 80%);"
    "background-size:200% 100%;animation:vi-shimmer 1.6s linear infinite;}"
    "#vyom-intro .vi-bot i:nth-child(2){width:88%;}#vyom-intro .vi-bot i:nth-child(3){width:64%;}"
    "#vyom-intro .vi-status{display:flex;align-items:center;justify-content:center;gap:.6rem;"
    "color:var(--vyom-dim,#aab5c9);font-size:.95rem;}"
    "#vyom-intro .vi-dots{display:inline-flex;gap:4px;}"
    "#vyom-intro .vi-dots span{width:7px;height:7px;border-radius:50%;background:currentColor;"
    "animation:vi-bounce 1.2s infinite ease-in-out;}"
    "#vyom-intro .vi-dots span:nth-child(2){animation-delay:.15s;}"
    "#vyom-intro .vi-dots span:nth-child(3){animation-delay:.3s;}"
    "#vyom-intro .vi-bar{height:3px;border-radius:3px;background:var(--vyom-soft,rgba(255,255,255,.07));"
    "margin:1.1rem auto 0;overflow:hidden;max-width:16rem;}"
    "#vyom-intro .vi-bar b{display:block;height:100%;width:40%;background:var(--vyom-fg,#e8eaed);"
    "border-radius:3px;animation:vi-slide 1.6s ease-in-out infinite;}"
    "@keyframes vyom-intro-out{to{opacity:0;visibility:hidden;pointer-events:none;}}"
    "@keyframes vi-rise{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}"
    "@keyframes vi-shimmer{to{background-position:-200% 0;}}"
    "@keyframes vi-bounce{0%,80%,100%{opacity:.3;transform:translateY(0);}40%{opacity:1;transform:translateY(-4px);}}"
    "@keyframes vi-slide{0%{transform:translateX(-120%);}100%{transform:translateX(300%);}}"
    "@media(prefers-reduced-motion:reduce){#vyom-intro *{animation:none!important;}"
    "#vyom-intro{animation-duration:.01s!important;animation-delay:.6s!important;}}"
    "</style>"
    '<div id="vyom-intro" aria-hidden="true"><div class="vi-box">'
    "<!--LOGO-->"
    '<p class="vi-sub">Your AI research assistant</p>'
    '<div class="vi-status"><span class="vi-dots"><span></span><span></span><span></span></span>'
    "<span>Getting things ready…</span></div>"
    '<div class="vi-bar"><b></b></div>'
    "</div></div>"
)

def _intro_html():
    uri = _logo_data_uri()
    if uri:
        block = f'<img class="vi-logo" src="{uri}" alt="VYOM logo">'
    else:
        block = (
            '<div class="vyom-search-icon"></div>'
            '<div class="vi-title">Vyom</div>'
        )
    return _INTRO_HTML.replace("<!--LOGO-->", block)


if "intro_done" not in st.session_state:
    st.session_state["intro_done"] = True
    if st.query_params.get("from") != "splash":
        st.markdown(_intro_html(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session initialization
# ---------------------------------------------------------------------------

if "greeting" not in st.session_state:
    st.session_state["greeting"] = _time_greeting()

if "chat_placeholder" not in st.session_state:
    st.session_state["chat_placeholder"] = random.choice(CHAT_PLACEHOLDERS)

st.session_state.setdefault("generation_active", False)

_init_chats()
messages = _active_messages()

with st.container(key="app_title"):
    _header_logo = _logo_bytes()
    if _header_logo:
        st.image(_header_logo, width=42)
    else:
        st.title("VYOM")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Chats")

    st.button(
        "New chat",
        key="new_chat_top",
        icon=":material/edit:",
        on_click=_start_new_chat,
    )

    for chat in reversed(st.session_state["chats"]):
        if not chat["messages"]:
            continue

        st.button(
            chat["title"],
            key=f"chat_btn_{chat['id']}",
            type="primary" if chat["id"] == st.session_state["active_chat"] else "secondary",
            width="stretch",
            on_click=_switch_chat,
            args=(chat["id"],),
        )

    st.divider()

    st.radio(
        "Theme",
        THEME_OPTIONS,
        key="user_theme_override",
        horizontal=True,
    )

    missing_backgrounds = [
        name for name in ("dark", "light") if not _background_data_uri(name)
    ]
    if missing_backgrounds:
        st.caption(
            "Background image not found: "
            + ", ".join(f"{name}_bg.jpg" for name in missing_backgrounds)
            + " (put it in the static/ folder next to app.py)."
        )

    _render_admin_panel()

# ---------------------------------------------------------------------------
# Main conversation
# ---------------------------------------------------------------------------

if messages:
    _render_conversation(messages)
else:
    with st.container(key="assistant_msg_greeting"):
        st.markdown(
            '<div class="vyom-greeting-row">'
            '<div class="vyom-search-icon" role="img" aria-label="Search"></div>'
            f'<div class="greeting-text">{html.escape(st.session_state["greeting"])}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Global script: scroll-down button, pause button, copy + native share
# ---------------------------------------------------------------------------

st.html(
    """
    <script>
    (() => {
        const SCROLL_ID = "vyom-scroll-down";
        const PAUSE_ID = "vyom-pause";

        // ---- find the element that actually scrolls the conversation ----
        const scroller = () => {
            let el = document.querySelector('[data-testid="stMainBlockContainer"]');
            while (el && el !== document.body) {
                const oy = getComputedStyle(el).overflowY;
                if ((oy === "auto" || oy === "scroll") && el.scrollHeight > el.clientHeight + 1) {
                    return el;
                }
                el = el.parentElement;
            }
            return document.scrollingElement || document.documentElement;
        };

        const findSend = () => {
            const box = document.querySelector('[data-testid="stChatInput"]');
            if (!box) return null;

            const direct = box.querySelector('[data-testid="stChatInputSubmitButton"]');
            if (direct) return direct;

            const skip = /attach|upload|file|add|mic|record|audio|voice/i;
            const buttons = [...box.querySelectorAll("button")].filter(button =>
                !skip.test(
                    (button.getAttribute("aria-label") || "") + " " +
                    (button.getAttribute("title") || "")
                )
            );
            return buttons[buttons.length - 1] || null;
        };

        let hiddenSend = null;

        const ensureButtons = () => {
            let down = document.getElementById(SCROLL_ID);
            if (!down) {
                down = document.createElement("button");
                down.id = SCROLL_ID;
                down.type = "button";
                down.setAttribute("aria-label", "Scroll to latest message");
                down.innerHTML =
                    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>';
                down.addEventListener("click", () => {
                    const el = scroller();
                    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
                });
                document.body.appendChild(down);
            }

            let pause = document.getElementById(PAUSE_ID);
            if (!pause) {
                pause = document.createElement("button");
                pause.id = PAUSE_ID;
                pause.type = "button";
                pause.setAttribute("aria-label", "Stop response");
                pause.innerHTML =
                    '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="4" width="5" height="16" rx="1.2"/><rect x="14" y="4" width="5" height="16" rx="1.2"/></svg>';
                pause.addEventListener("click", () => {
                    const source = document.querySelector(".st-key-stop_generation button");
                    if (source) source.click();
                });
                document.body.appendChild(pause);
            }
            return { down, pause };
        };

        const tick = () => {
            const { down, pause } = ensureButtons();

            // scroll-down button
            const el = scroller();
            const away = el.scrollHeight - el.scrollTop - el.clientHeight;
            down.classList.toggle("show", away > 200);

            // pause button sits on top of the send button while generating
            const source = document.querySelector(".st-key-stop_generation button");
            const send = findSend();
            const box = document.querySelector('[data-testid="stChatInput"]');

            if (source) {
                let cx = window.innerWidth - 40;
                let cy = window.innerHeight - 50;
                const rect = send ? send.getBoundingClientRect() : null;

                if (rect && rect.width) {
                    cx = rect.left + rect.width / 2;
                    cy = rect.top + rect.height / 2;
                    send.style.visibility = "hidden";
                    hiddenSend = send;
                } else if (box) {
                    const b = box.getBoundingClientRect();
                    if (b.width) {
                        cx = b.right - 30;
                        cy = b.top + b.height / 2;
                    }
                }

                pause.style.left = (cx - 19) + "px";
                pause.style.top = (cy - 19) + "px";
                pause.classList.add("show");
            } else {
                pause.classList.remove("show");
                if (hiddenSend) {
                    hiddenSend.style.visibility = "";
                    hiddenSend = null;
                }
            }

            // when loaded inside the splash page, say "ready" once the chat box exists
            if (window.parent !== window && !window.__vyomReadySent &&
                document.querySelector('[data-testid="stChatInput"]')) {
                window.__vyomReadySent = true;
                try { window.parent.postMessage({ type: "vyom-ready" }, "*"); } catch (e) {}
            }

            // reveal the native "share with phone apps" button where supported
            if (navigator.share) {
                document.querySelectorAll(".vyom-native-share").forEach(b => {
                    b.style.display = "block";
                });
            }
        };

        const decode = (b64) => {
            try {
                return new TextDecoder().decode(
                    Uint8Array.from(atob(b64), c => c.charCodeAt(0))
                );
            } catch (e) {
                return "";
            }
        };

        // ---- one delegated click handler for copy + native share ----
        if (!window.__vyomClickBound) {
            window.__vyomClickBound = true;

            document.addEventListener("click", async (event) => {
                const target = event.target;
                if (!target || !target.closest) return;

                const copyBtn = target.closest(".vyom-copy");
                if (copyBtn) {
                    const text = decode(copyBtn.getAttribute("data-text") || "");
                    if (!text) return;

                    let ok = false;
                    try {
                        await navigator.clipboard.writeText(text);
                        ok = true;
                    } catch (e) {
                        const input = document.createElement("textarea");
                        input.value = text;
                        input.style.position = "fixed";
                        input.style.opacity = "0";
                        document.body.appendChild(input);
                        input.select();
                        try { ok = document.execCommand("copy"); } catch (e2) {}
                        input.remove();
                    }

                    if (ok) {
                        copyBtn.classList.add("copied");
                        copyBtn.setAttribute("aria-label", "Copied");
                        clearTimeout(copyBtn.__vyomTimer);
                        copyBtn.__vyomTimer = setTimeout(() => {
                            copyBtn.classList.remove("copied");
                            copyBtn.setAttribute("aria-label", "Copy reply");
                        }, 1200);
                    }
                    return;
                }

                const shareBtn = target.closest(".vyom-native-share");
                if (shareBtn && navigator.share) {
                    const text = decode(shareBtn.getAttribute("data-text") || "");
                    navigator.share({ title: "VYOM reply", text }).catch(() => {});
                }
            });
        }

        // ---- run now, and keep running (Streamlit re-renders constantly) ----
        clearInterval(window.__vyomTimer);
        window.__vyomObserver && window.__vyomObserver.disconnect();

        if (!window.__vyomScrollBound) {
            window.__vyomScrollBound = true;
            document.addEventListener("scroll", () => window.__vyomTick && window.__vyomTick(), true);
            window.addEventListener("resize", () => window.__vyomTick && window.__vyomTick());
        }

        window.__vyomTick = tick;
        window.__vyomObserver = new MutationObserver(tick);
        window.__vyomObserver.observe(document.body, { childList: true, subtree: true });
        window.__vyomTimer = setInterval(tick, 300);
        tick();
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

prompt = st.chat_input(
    st.session_state["chat_placeholder"],
    key="chat_input",
    accept_file="multiple",
    file_type=FILE_TYPES,
    accept_audio=True,
    width="stretch",
)

pending = st.session_state.pop("pending_submission", None)

topic = ""
files = []
audio = None
has_input = False

if prompt:
    has_input = True

    if isinstance(prompt, str):
        topic = prompt.strip()
    else:
        topic = (getattr(prompt, "text", "") or "").strip()
        audio = getattr(prompt, "audio", None)
        files = [
            (file.name, file.getvalue(), file.type or "application/octet-stream")
            for file in (getattr(prompt, "files", None) or [])
        ]

elif pending:
    has_input = True
    topic = pending["topic"]
    files = pending.get("files", [])

# ---------------------------------------------------------------------------
# Request processing
# ---------------------------------------------------------------------------

if has_input:
    transcription_failed = False

    if audio is not None:
        try:
            with st.spinner("Transcribing your voice..."):
                transcript = transcribe_audio(
                    (
                        audio.name,
                        audio.getvalue(),
                        audio.type or "audio/wav",
                    )
                )

            topic = "\n".join(part for part in (topic, transcript) if part)

        except (OpenAIError, RuntimeError, TimeoutError, ValueError) as error:
            transcription_failed = True
            st.error(
                "Voice transcription failed. Please try again or type your message."
            )
            st.caption(str(error))

    if files and not topic:
        topic = "Please summarize the attached file(s)."

    if topic:
        conversation_history = _conversation_history(messages)

        user_message = {
            "role": "user",
            "content": topic,
            "files": files,
        }

        messages.append(user_message)

        chat = _active_chat()

        if chat["title"] == "New chat":
            chat["title"] = topic[:32] + ("…" if len(topic) > 32 else "")

        with st.container(key=f"user_msg_{len(messages) - 1}"):
            st.markdown(_user_text(topic, files))

        progress_steps = []
        assistant_message = {
            "role": "assistant",
            "progress": progress_steps,
        }

        messages.append(assistant_message)
        st.session_state["generation_active"] = True

        # Hidden real button; the floating pause overlay clicks this one.
        st.button(
            "Stop",
            key="stop_generation",
            help="Stop the current response",
            on_click=_stop_generation,
        )

        with st.container(key=f"assistant_msg_{len(messages) - 1}"):
            # No "Thinking" panel: only a small loader line with the live stage.
            loader = st.empty()
            loader_state = {"active": True}

            loader.markdown(
                _loader_html("Thinking..."),
                unsafe_allow_html=True,
            )

            def update_progress(stage):
                progress_steps.append(stage)
                if loader_state["active"]:
                    loader.markdown(
                        _loader_html(stage),
                        unsafe_allow_html=True,
                    )

            result_container = {}

            def tracked_stream():
                if files:
                    stream = run_attachment_stream(
                        topic,
                        files,
                        result_container,
                        progress_callback=update_progress,
                        conversation_history=conversation_history,
                    )
                else:
                    stream = run_research_stream(
                        topic,
                        result_container,
                        progress_callback=update_progress,
                        conversation_history=conversation_history,
                    )

                for chunk in stream:
                    if loader_state["active"]:
                        loader_state["active"] = False
                        loader.empty()

                    assistant_message["partial_report"] = (
                        assistant_message.get("partial_report", "") + chunk
                    )

                    yield chunk

            try:
                st.write_stream(tracked_stream())
                result = result_container.get("result")
                st.session_state["generation_active"] = False

            except (
                OpenAIError,
                RuntimeError,
                TimeoutError,
                ValueError,
            ) as error:
                st.session_state["generation_active"] = False
                loader_state["active"] = False
                loader.empty()
                update_progress("The response could not be completed.")

                if isinstance(error, LimitReached):
                    assistant_message["error"] = str(error)
                else:
                    assistant_message["error"] = (
                        "The request could not be completed. "
                        "Check the error details below."
                    )
                    assistant_message["error_detail"] = str(error)

            else:
                loader_state["active"] = False
                loader.empty()

                if context := result_container.get("attachment_context"):
                    user_message["attachment_context"] = context

                if not isinstance(result, dict):
                    update_progress("The service returned an unexpected response.")
                    assistant_message["error"] = (
                        "The research service returned an unexpected response."
                    )

                elif result.get("error"):
                    update_progress("The response could not be completed.")
                    assistant_message["error"] = str(result["error"])

                else:
                    update_progress("Response ready.")
                    assistant_message.pop("partial_report", None)
                    assistant_message["result"] = result

        st.rerun()

    elif not transcription_failed:
        st.warning("Write a message before sending.")