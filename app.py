
import base64
import hmac
import html
import json
import os
import random
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
import streamlit as st
from openai import OpenAIError
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from streamlit.errors import StreamlitAPIException, StreamlitSecretNotFoundError

from agent import run_research_stream, transcribe_audio
from attachments import run_attachment_stream
import patches  # noqa: F401


def _load_streamlit_secrets_into_environment():
    for name in (
        "TAVILY_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
        "BIFROST_BASE_URL", "BIFROST_API_KEY", "BIFROST_MODELS",
        "GROQ_MODELS", "GEMINI_MODELS", "ADMIN_PASSWORD",
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


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

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
    return {"id": uuid.uuid4().hex[:8], "title": "New chat", "messages": []}


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


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

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
            story.append(Paragraph(html.escape(heading.group(2)), style))
            continue

        bullet = re.match(r"^(?:[-*])\s+(.+)$", stripped)
        text = f"- {bullet.group(1)}" if bullet else stripped
        story.append(Paragraph(html.escape(text), styles["ReportBody"]))

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
        elif isinstance(encoded, str):
            try:
                pending.append(
                    (None, base64.b64decode(encoded, validate=True), description)
                )
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
# Copy and sharing
# ---------------------------------------------------------------------------

def _copy_icon_html(key_prefix, report):
    encoded = base64.b64encode(report.encode("utf-8")).decode("ascii")

    return f"""
    <button id="copy-reply-{key_prefix}" type="button"
        aria-label="Copy reply" title="Copy"
        style="display:flex;align-items:center;justify-content:center;
        width:38px;height:38px;background:var(--vyom-panel);
        border:1px solid var(--vyom-border);color:var(--vyom-fg);
        border-radius:9px;cursor:pointer;padding:0 9px;min-width:38px;gap:5px;">
    </button>
    <script>
    (() => {{
        const button = document.getElementById("copy-reply-{key_prefix}");
        if (!button) return;

        const COPY = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
        const DONE = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';

        button.innerHTML = COPY;

        const text = new TextDecoder().decode(
            Uint8Array.from(atob("{encoded}"), c => c.charCodeAt(0))
        );

        let timer;

        const done = () => {{
            button.innerHTML = DONE + '<span class="vyom-copy-label">Copied</span>';
            button.setAttribute("aria-label", "Copied");
            clearTimeout(timer);
            timer = setTimeout(() => {{
                button.innerHTML = COPY;
                button.setAttribute("aria-label", "Copy reply");
            }}, 1000);
        }};

        button.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(text);
                done();
            }} catch {{
                const input = document.createElement("textarea");
                input.value = text;
                document.body.appendChild(input);
                input.select();
                const ok = document.execCommand("copy");
                input.remove();
                if (ok) done();
            }}
        }});
    }})();
    </script>
    """


def _native_share_html(key_prefix, report):
    encoded = base64.b64encode(report.encode("utf-8")).decode("ascii")

    return f"""
    <button id="native-share-{key_prefix}" type="button"
        style="display:none;width:100%;padding:0.5rem 0.8rem;
        border-radius:0.5rem;border:1px solid var(--vyom-border);
        background:transparent;color:inherit;font:inherit;
        cursor:pointer;text-align:left;">
        Share with your phone apps…
    </button>
    <script>
    (() => {{
        const button = document.getElementById("native-share-{key_prefix}");
        if (!button || !navigator.share) return;

        button.style.display = "block";

        const text = new TextDecoder().decode(
            Uint8Array.from(atob("{encoded}"), c => c.charCodeAt(0))
        );

        button.addEventListener("click", () => {{
            navigator.share({{ title: "VYOM reply", text }}).catch(() => {{}});
        }});
    }})();
    </script>
    """


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

        st.html(_copy_icon_html(key_prefix, report), unsafe_allow_javascript=True)

        with st.popover(
            "\u200b",
            icon=":material/share:",
            help="Share this reply",
            type="tertiary",
            key=f"{key_prefix}_sharepop",
        ):
            st.html(_native_share_html(key_prefix, report), unsafe_allow_javascript=True)

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
        sources = result.get("sources")
        valid_sources = [
            source for source in sources
            if isinstance(source, dict)
            and isinstance(source.get("url"), str)
            and source["url"].startswith(("https://", "http://"))
        ] if isinstance(sources, list) else []

        st.download_button(
            "Download PDF",
            data=_build_pdf(report, valid_sources),
            file_name="response.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
            help="Download this reply as a PDF",
            width="content",
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

    _, top_copy_right = st.columns([0.90, 0.10], gap="small")
    with top_copy_right:
        st.html(
            _copy_icon_html(f"{key_prefix}_top", report),
            unsafe_allow_javascript=True,
        )

    st.markdown(report)

    if result.get("kind") == "chat":
        images = result.get("images")
        generated_images = [
            image for image in images
            if isinstance(image, dict)
            and image.get("description") == "Generated image"
        ] if isinstance(images, list) else []

        if generated_images:
            _render_images(generated_images)
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
        "Download report",
        data=download_content,
        file_name="research-report.md",
        mime="text/markdown",
        key=f"{key_prefix}_download",
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
            if progress := message.get("progress"):
                with st.expander(
                    "Thinking",
                    expanded=False,
                    key=f"thinking_{index}",
                    type="compact",
                ):
                    for step in progress:
                        st.caption(step)

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
@import url("https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,600;1,700&family=DM+Sans:wght@400;500;600;700&display=swap");

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
    padding:3.5rem 1.25rem 2rem;
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
    background:var(--vyom-soft)!important;
    display:flex;
    align-items:center;
    color:var(--vyom-fg);
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

div[class*="_actions"] {
    flex-wrap:nowrap!important;
    gap:.1rem!important;
    align-items:center;
}

div[class*="_actions"] button,
div[class*="_regen"] button {
    min-width:2.3rem!important;
    padding-inline:.35rem!important;
    background:transparent!important;
    border-color:transparent!important;
    box-shadow:none!important;
}

div[class*="_actions"] button span[data-testid="stIconMaterial"],
div[class*="_regen"] button span[data-testid="stIconMaterial"] {
    color:var(--vyom-dim)!important;
}

div[class*="_actions"] button:hover span[data-testid="stIconMaterial"] {
    color:var(--vyom-strong)!important;
}

div[class*="_like"] button[data-testid="stBaseButton-primary"] span[data-testid="stIconMaterial"],
div[class*="_dislike"] button[data-testid="stBaseButton-primary"] span[data-testid="stIconMaterial"] {
    color:var(--vyom-strong)!important;
}

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
    width:32px;
    height:32px;
    padding:0;
    border-radius:50%;
    cursor:pointer;
    border:1px solid var(--vyom-border);
    background:var(--vyom-strong);
    color:var(--vyom-invert);
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
    border:1px solid color-mix(in srgb,currentColor 18%,transparent);
    color:currentColor;
    background:color-mix(in srgb,currentColor 5%,transparent);
}

div[data-testid="stChatInput"]>div {
    border:1px solid var(--vyom-border)!important;
    border-radius:16px!important;
    outline:none!important;
    background:var(--vyom-panel)!important;
    box-shadow:0 6px 24px color-mix(in srgb,currentColor 7%,transparent)!important;
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

div[data-testid="stChatInput"] textarea::placeholder {
    color:var(--vyom-dim)!important;
    opacity:1!important;
}

#vyom-scroll-down {
    position:fixed;
    left:50%;
    bottom:5.5rem;
    transform:translateX(-50%);
    z-index:1000;
    display:none;
    align-items:center;
    justify-content:center;
    width:36px;
    height:36px;
    padding:0;
    border-radius:50%;
    border:1px solid var(--vyom-border);
    color:var(--vyom-strong);
    background:var(--vyom-pop);
    box-shadow:0 4px 16px rgba(0,0,0,.35);
    cursor:pointer;
}

#vyom-scroll-down.show {display:flex;}

section[data-testid="stSidebar"] div[data-testid="stButton"]>button {
    justify-content:flex-start;
    text-align:left;
    min-height:2rem;
    padding:.35rem .6rem;
    border-color:transparent;
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
    section.main>div {padding:3.5rem .5rem 5rem;}

    div[data-testid="stBottomBlockContainer"] {padding-inline:.5rem!important;}

    div[class*="st-key-user_msg_"] [data-testid="stElementContainer"] {
        max-width:92%!important;
    }

    [data-testid="stChatInput"] textarea {font-size:16px!important;}
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# The external stylesheet controls both themes.
st.session_state.setdefault("theme_mode", "System")
_THEME_CSS_PATH = os.path.join(os.path.dirname(__file__), "vyom_theme.css")

try:
    with open(_THEME_CSS_PATH, "r", encoding="utf-8") as theme_file:
        st.markdown(f"<style>{theme_file.read()}</style>", unsafe_allow_html=True)
except OSError:
    st.warning("VYOM theme stylesheet was not found; using the default styling.")

_theme_mode = st.session_state.get("theme_mode", "System")

st.html(
    f"""
    <script>
    (() => {{
        const requested = {json.dumps(_theme_mode)};
        const key = "vyom-theme-mode";
        let saved = null;

        try {{ saved = localStorage.getItem(key); }} catch (e) {{}}

        const mode = requested === "System" ? (saved || "System") : requested;

        const apply = value => {{
            const resolved = value === "System"
                ? (window.matchMedia("(prefers-color-scheme: light)").matches ? "Light" : "Dark")
                : value;

            document.documentElement.dataset.vyomTheme = resolved.toLowerCase();
        }};

        apply(mode);

        if (requested !== "System") {{
            try {{ localStorage.setItem(key, requested); }} catch (e) {{}}
        }}

        if (!window.__vyomThemeListener) {{
            window.__vyomThemeListener = true;

            window.matchMedia("(prefers-color-scheme: light)").addEventListener?.("change", () => {{
                let current = "System";

                try {{ current = localStorage.getItem(key) || "System"; }} catch (e) {{}}

                if (current === "System") apply("System");
            }});
        }}
    }})();
    </script>
    """,
    unsafe_allow_javascript=True,
)

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

    st.selectbox(
        "Appearance",
        options=["System", "Dark", "Light"],
        key="theme_mode",
        help="Choose a theme or follow your device appearance.",
    )

    _render_admin_panel()

# ---------------------------------------------------------------------------
# Main conversation
# ---------------------------------------------------------------------------

if messages:
    _render_conversation(messages)
else:
    with st.container(key="assistant_msg_greeting"):
        greeting_column, flower_column = st.columns([0.88, 0.12])

        with greeting_column:
            st.markdown(
                f'<div class="greeting-text">{html.escape(st.session_state["greeting"])}</div>',
                unsafe_allow_html=True,
            )

        with flower_column:
            st.markdown(
                '<div class="jasmine-flower" role="img" '
                'aria-label="White jasmine flower">❀</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Floating scroll and stop controls
# ---------------------------------------------------------------------------

st.html(
    """
    <script>
    (() => {
        if (window.__vyomInit) return;
        window.__vyomInit = true;

        const down = document.createElement("button");
        down.id = "vyom-scroll-down";
        down.type = "button";
        down.setAttribute("aria-label", "Scroll to latest message");
        down.innerHTML =
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>';
        document.body.appendChild(down);

        const pause = document.createElement("button");
        pause.id = "vyom-pause";
        pause.type = "button";
        pause.setAttribute("aria-label", "Stop response");
        pause.innerHTML =
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="4" width="5" height="16" rx="1.2"/><rect x="14" y="4" width="5" height="16" rx="1.2"/></svg>';
        document.body.appendChild(pause);

        const scroller = () =>
            document.querySelector('[data-testid="stMain"]') ||
            document.querySelector("section.main") ||
            document.scrollingElement;

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

        const syncPause = () => {
            const source = document.querySelector(".st-key-stop_generation button");
            const send = findSend();

            if (source && send) {
                const rect = send.getBoundingClientRect();

                if (rect.width) {
                    pause.style.left = (rect.left + rect.width / 2 - 16) + "px";
                    pause.style.top = (rect.top + rect.height / 2 - 16) + "px";
                    pause.classList.add("show");
                    send.style.visibility = "hidden";
                    hiddenSend = send;
                    return;
                }
            }

            pause.classList.remove("show");

            if (hiddenSend) {
                hiddenSend.style.visibility = "";
                hiddenSend = null;
            }
        };

        const syncDown = () => {
            const element = scroller();
            if (!element) return;

            const away = element.scrollHeight - element.scrollTop - element.clientHeight;
            down.classList.toggle("show", away > 300);
        };

        pause.addEventListener("click", () => {
            const source = document.querySelector(".st-key-stop_generation button");
            if (source) source.click();
        });

        down.addEventListener("click", () => {
            const element = scroller();
            if (element) {
                element.scrollTo({
                    top: element.scrollHeight,
                    behavior: "smooth"
                });
            }
        });

        document.addEventListener("scroll", syncDown, true);
        window.addEventListener("resize", syncPause);

        const observer = new MutationObserver(() => {
            syncDown();
            syncPause();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        setInterval(() => {
            syncDown();
            syncPause();
        }, 350);
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
    if not topic and audio is None and not files:
        st.warning("Write a message before sending.")

    else:
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

            # This button is retained for the existing stop-control UI.
            # Reliable cancellation requires a non-blocking streaming architecture.
            st.button(
                "Stop",
                key="stop_generation",
                help="Stop the current response",
                on_click=_stop_generation,
            )

            with st.container(key=f"assistant_msg_{len(messages) - 1}"):
                with st.expander(
                    "Thinking",
                    expanded=False,
                    key=f"thinking_current_{len(messages)}",
                    type="compact",
                ):
                    progress_line = st.empty()

                loader = st.empty()
                loader_state = {"active": True}

                # Keep this label constant. Detailed progress stays inside Thinking.
                loader.markdown(
                    _loader_html("Thinking..."),
                    unsafe_allow_html=True,
                )

                def update_progress(stage):
                    progress_steps.append(stage)
                    progress_line.caption(stage)

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
                    loader.empty()
                    update_progress("The response could not be completed.")

                    assistant_message["error"] = (
                        "The request could not be completed. "
                        "Check the error details below."
                    )
                    assistant_message["error_detail"] = str(error)

                else:
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
