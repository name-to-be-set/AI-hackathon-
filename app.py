
import base64
import html
import os
import random
import re
from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo

import streamlit as st
from openai import OpenAIError
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from streamlit.errors import StreamlitAPIException, StreamlitSecretNotFoundError

from agent import run_research_stream, transcribe_audio


def _load_streamlit_secrets_into_environment() -> None:
    for name in (
        "TAVILY_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
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

CHAT_PLACEHOLDERS = (
    "Ask me anything...",
    "What's on your mind?",
    "Type or tap the mic to speak...",
    "What would you like to explore?",
)
MAX_REQUESTS_PER_SESSION = 10
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

def _start_new_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["greeting"] = _time_greeting()
    st.session_state["generation_active"] = False
    st.session_state.pop("share_message", None)


def _stop_generation() -> None:
    st.session_state["generation_active"] = False
    for existing_message in reversed(st.session_state["messages"]):
        if existing_message.get("role") == "assistant":
            existing_message["generation_stopped"] = True
            break


def _time_greeting(hour: int | None = None) -> str:
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


def _visible_report(report: str) -> str:
    return re.sub(r"\s*\[(?:\d+(?:\s*,\s*\d+)*)\]", "", report)


def _build_pdf(report: str, sources: list[dict]) -> bytes:
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
            style = styles[
                "Heading1" if len(heading.group(1)) < 3 else "Heading2"
            ]
            story.append(Paragraph(html.escape(heading.group(2)), style))
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
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


def _render_images(images: object) -> None:
    if not isinstance(images, list):
        return
    valid_images = []
    for image in images[:4]:
        if not isinstance(image, dict):
            continue
        description = image.get("description") or None
        url = image.get("url")
        encoded = image.get("base64")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            valid_images.append((url, description))
        elif isinstance(encoded, str):
            try:
                valid_images.append(
                    (base64.b64decode(encoded, validate=True), description)
                )
            except ValueError:
                continue

    if not valid_images:
        return
    columns = st.columns(len(valid_images), gap="small")
    for column, (image_data, description) in zip(columns, valid_images):
        try:
            with column:
                st.image(image_data, caption=description, width="stretch")
        except (StreamlitAPIException, ValueError, OSError):
            continue


def _render_reply_actions(
    message: dict,
    key_prefix: str,
) -> None:
    result = message.get("result")
    report = result.get("report") if isinstance(result, dict) else message.get("error")
    if not isinstance(report, str) or not report.strip():
        return
    report = _visible_report(report)

    with st.container(key=f"{key_prefix}_actions"):
        (
            like_column,
            dislike_column,
            share_column,
            copy_column,
        ) = st.columns([1, 1, 1, 1], gap="small")
        with like_column:
            if st.button(
                "Like",
                icon=":material/thumb_up_off_alt:",
                key=f"{key_prefix}_like",
                help="Mark this as a good reply",
                type=(
                    "primary"
                    if message.get("feedback") == "like"
                    else "tertiary"
                ),
                use_container_width=True,
            ):
                message["feedback"] = (
                    "none" if message.get("feedback") == "like" else "like"
                )
                st.rerun()
        with dislike_column:
            if st.button(
                "Dislike",
                icon=":material/thumb_down_off_alt:",
                key=f"{key_prefix}_dislike",
                help="Mark this as a bad reply",
                type=(
                    "primary"
                    if message.get("feedback") == "dislike"
                    else "tertiary"
                ),
                use_container_width=True,
            ):
                message["feedback"] = (
                    "none" if message.get("feedback") == "dislike" else "dislike"
                )
                st.rerun()
        with share_column:
            if st.button(
                "Share",
                key=f"{key_prefix}_share",
                type="tertiary",
                use_container_width=True,
            ):
                st.session_state["share_message"] = key_prefix
        with copy_column:
            encoded_report = base64.b64encode(report.encode("utf-8")).decode("ascii")
            st.html(
                f"""
                <button id="copy-reply-{key_prefix}" type="button" aria-label="Copy reply"
                    style="display:flex;align-items:center;justify-content:center;
                    width:100%;height:40px;box-sizing:border-box;background:transparent;
                    border:0;color:#e5e7eb;font:inherit;padding:0 8px;cursor:pointer;
                    white-space:nowrap;">
                    Copy
                </button>
                <script>
                (() => {{
                    const button = document.getElementById("copy-reply-{key_prefix}");
                    const text = new TextDecoder().decode(
                        Uint8Array.from(atob("{encoded_report}"), character =>
                            character.charCodeAt(0))
                    );
                    let resetTimer;
                    const showCopied = () => {{
                        button.textContent = "Copied";
                        clearTimeout(resetTimer);
                        resetTimer = setTimeout(() => {{
                            button.textContent = "Copy";
                        }}, 1500);
                    }};
                    button.addEventListener("click", async () => {{
                        try {{
                            await navigator.clipboard.writeText(text);
                            showCopied();
                        }} catch {{
                            const input = document.createElement("textarea");
                            input.value = text;
                            document.body.appendChild(input);
                            input.select();
                            const copied = document.execCommand("copy");
                            input.remove();
                            if (copied) {{
                                showCopied();
                            }} else {{
                                button.textContent = "Copy unavailable";
                            }}
                        }}
                    }});
                }})();
                </script>
                """,
                unsafe_allow_javascript=True,
            )

    if isinstance(result, dict):
        sources = result.get("sources")
        valid_sources = []
        if isinstance(sources, list):
            valid_sources = [
                source
                for source in sources
                if isinstance(source, dict)
                and isinstance(source.get("url"), str)
                and source["url"].startswith(("https://", "http://"))
            ]
        st.download_button(
            "Download PDF",
            data=_build_pdf(report, valid_sources),
            file_name="response.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
            help="Download this reply as a PDF",
            width="content",
        )

    if st.session_state.get("share_message") == key_prefix:
        with st.expander(
            "Share this reply",
            expanded=True,
            key=f"{key_prefix}_share_text",
        ):
            st.caption("Copy the text below to share this reply.")
            st.code(report, language=None, wrap_lines=True)


def _conversation_history(messages: list[dict]) -> list[dict[str, str]]:
    history = []
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            history.append({"role": "user", "content": message["content"]})
        elif message.get("role") == "assistant":
            result = message.get("result")
            if isinstance(result, dict) and isinstance(result.get("report"), str):
                history.append({"role": "assistant", "content": result["report"]})
            elif isinstance(message.get("error"), str):
                history.append({"role": "assistant", "content": message["error"]})
    return history[-10:]


def _render_result(result: dict, key_prefix: str = "result") -> None:
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
        images = result.get("images")
        if any(
            isinstance(image, dict)
            and image.get("description") == "Generated image"
            for image in (images if isinstance(images, list) else [])
        ):
            _render_images(images)
        return

    _render_images(result.get("images"))

    sources = result.get("sources")
    if not isinstance(sources, list):
        st.error("The research agent returned an invalid source list.")
        return
    valid_sources = [
        source
        for source in sources
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


def _render_conversation(messages: list[dict]) -> None:
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
            continue

        with st.chat_message("assistant"):
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
            elif partial_report := message.get("partial_report"):
                st.markdown(_visible_report(partial_report))
                if message.get("generation_stopped"):
                    st.caption("Response stopped.")
            _render_reply_actions(
                message,
                key_prefix=f"chat_{index}",
            )


st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.markdown(
    """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap");

    .stApp {
        font-family: "DM Sans", "Segoe UI", system-ui, sans-serif;
    }

    section.main > div {
        width: 100%;
        max-width: none;
        margin-inline: auto;
        padding-inline: 1.25rem;
    }

    div[class*="_actions"] div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 0.15rem !important;
    }

    div[class*="_actions"] div[data-testid="stColumn"] {
        width: auto !important;
        flex: 0 0 auto !important;
        min-width: 0 !important;
        padding-inline: 0 !important;
    }

    div[class*="_actions"] button {
        min-width: 0 !important;
        white-space: nowrap !important;
        padding-inline: 0.35rem !important;
        font-size: 0.78rem !important;
    }

    div[class*="_actions"] div[data-testid="stHorizontalBlock"] > div:nth-child(-n+2) button {
        min-width: 2.4rem !important;
        padding-inline: 0.35rem !important;
        color: rgba(255, 255, 255, 0.78);
    }

    div[class*="_actions"] div[data-testid="stHorizontalBlock"] > div:nth-child(-n+2) button[kind="primary"] {
        color: #ffffff !important;
        background: transparent !important;
        border-color: transparent !important;
    }

    div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
        display: none !important;
    }

    div[data-testid="stChatMessage"] {
        width: 100% !important;
        padding-inline: 0 !important;
    }

    div[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        width: 100% !important;
        max-width: none !important;
        min-width: 0 !important;
    }

    div[data-testid="stChatMessage"][aria-label*="user"] [data-testid="stChatMessageContent"] {
        width: fit-content !important;
        max-width: min(90%, 900px) !important;
        margin-right: auto !important;
        padding: 0.65rem 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 1.1rem 1.1rem 1.1rem 0.25rem;
        background: rgba(255, 255, 255, 0.08);
    }

    .greeting-text {
        font-size: clamp(2rem, 5vw, 3.4rem);
        font-weight: 600;
        line-height: 1.2;
    }

    .jasmine-flower {
        flex: 0 0 auto;
        color: #ffffff;
        font-size: clamp(3rem, 7vw, 5rem);
        line-height: 1;
        filter: drop-shadow(0 0 0.8rem rgba(255, 255, 255, 0.28));
    }

    .st-key-stop_generation,
    .st-key-stop_generation_streaming {
        position: fixed;
        z-index: 1001;
        right: max(calc((100vw - 520px) / 2 + 80px), 5.5rem);
        bottom: 1.05rem;
    }

    .st-key-stop_generation button,
    .st-key-stop_generation_streaming button {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        min-height: 32px;
        padding: 0;
        border-radius: 50%;
        color: #ffffff;
        background: transparent;
        border-color: transparent;
    }

    div[data-testid="stMarkdownContainer"] h1 {
        font-size: 1.65rem;
        font-weight: 700;
    }

    .st-key-app_title h1 {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    div[data-testid="stMarkdownContainer"] h2 {
        font-size: 1.35rem;
        font-weight: 700;
    }

    div[data-testid="stMarkdownContainer"] h3 {
        font-size: 1.12rem;
        font-weight: 600;
    }

    div[data-testid="stCheckbox"] input {
        accent-color: #ffffff;
    }

    div[data-testid="stCheckbox"] input:focus-visible {
        outline: 2px solid #ffffff;
        outline-offset: 2px;
    }

    .st-key-chat_input button {
        width: 32px;
        height: 32px;
        min-height: 32px;
        padding: 0;
        border-radius: 50%;
    }

    div[data-testid="stChatInput"] {
        width: min(100%, 520px) !important;
        max-width: 520px !important;
        margin-inline: auto !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 0;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] > div {
        border: 1px solid rgba(100, 116, 139, 0.5) !important;
        border-radius: 10px !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] textarea {
        min-height: 40px !important;
        max-height: 4.5rem;
        font-size: 0.95rem;
        border-radius: 8px;
        font-family: inherit;
        caret-color: #ffffff !important;
        cursor: text;
        border-color: transparent !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border-color: transparent !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        opacity: 0.48 !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border: none !important;
    }

    div[data-testid="stChatInput"]:focus-within > div {
        border-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        justify-content: flex-start;
        text-align: left;
        min-height: 2rem;
        padding: 0.35rem 0.6rem;
        border-color: transparent;
        border-radius: 0.55rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button > div {
        justify-content: flex-start !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }

    .st-key-new_chat_top button {
        min-width: 38px;
        border-radius: 10px;
        font-size: 1.2rem;
    }

    .st-key-new_chat_top {
        margin-top: -1rem;
    }

    div[data-testid="stChatMessage"] div[data-testid="stButton"] button {
        border-color: transparent;
        padding-inline: 0.4rem;
    }

    @media (max-width: 640px) {
        section.main > div {
            width: 100%;
            max-width: 100%;
            padding: 0.75rem 0.5rem 5rem;
        }

        div[data-testid="stChatInput"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        .st-key-app_title h1 {
            font-size: 1.45rem;
        }

        div[class*="_actions"] button {
            padding-inline: 0.15rem !important;
            font-size: 0.72rem !important;
        }

        .st-key-stop_generation,
        .st-key-stop_generation_streaming {
            right: 5.5rem;
            bottom: 1rem;
        }

        .st-key-new_chat_top {
            margin-top: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "greeting" not in st.session_state:
    st.session_state["greeting"] = _time_greeting()
if "chat_placeholder" not in st.session_state:
    st.session_state["chat_placeholder"] = random.choice(CHAT_PLACEHOLDERS)
st.session_state.setdefault("demo_requests_used", 0)
st.session_state.setdefault("generation_active", False)

with st.sidebar:
    st.markdown("### This session")
    st.caption(
        f"{MAX_REQUESTS_PER_SESSION - st.session_state['demo_requests_used']} "
        "demo requests remaining"
    )

title_column, new_chat_column = st.columns([0.95, 0.05])
with title_column:
    with st.container(key="app_title"):
        st.title("AI RESEARCH ASSISTANT")
with new_chat_column:
    if st.button("✎", key="new_chat_top", help="New chat"):
        _start_new_chat()
        st.rerun()

st.session_state.setdefault("messages", [])
messages = st.session_state["messages"]
if messages:
    _render_conversation(messages)
else:
    with st.chat_message("assistant"):
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

if st.session_state["generation_active"]:
    st.button(
        "⏸",
        key="stop_generation",
        help="Stop the current response",
        on_click=_stop_generation,
    )

if prompt := st.chat_input(
    st.session_state["chat_placeholder"],
    key="chat_input",
    accept_audio=True,
    width=520,
):
    if isinstance(prompt, str):
        topic = prompt.strip()
        audio = None
    else:
        topic = prompt.text.strip()
        audio = prompt.audio

    if not topic and audio is None:
        st.warning("Write a message before sending.")
    elif st.session_state["demo_requests_used"] >= MAX_REQUESTS_PER_SESSION:
        st.warning("You've reached the 10-request demo limit for this session.")
    else:
        st.session_state["demo_requests_used"] += 1
        if audio is not None:
            try:
                with st.spinner("Transcribing your voice..."):
                    transcript = transcribe_audio(
                        (audio.name, audio.getvalue(), audio.type or "audio/wav")
                    )
                topic = "\n".join(part for part in (topic, transcript) if part)
            except (OpenAIError, RuntimeError, TimeoutError, ValueError) as error:
                st.error("Voice transcription failed. Please try again or type your message.")
                st.caption(str(error))

        if topic:
            conversation_history = _conversation_history(messages)
            messages.append({"role": "user", "content": topic})
            with st.chat_message("user"):
                st.markdown(topic)

            progress_steps = []
            assistant_message = {"role": "assistant", "progress": progress_steps}
            messages.append(assistant_message)
            st.session_state["generation_active"] = True
            with st.chat_message("assistant"):
                with st.expander(
                    "Thinking",
                    expanded=False,
                    key=f"thinking_current_{len(messages)}",
                    type="compact",
                ):
                    progress_line = st.empty()

                st.button(
                    "⏸",
                    key="stop_generation_streaming",
                    help="Stop the current response",
                    on_click=_stop_generation,
                )

                def update_progress(stage: str) -> None:
                    progress_steps.append(stage)
                    progress_line.caption(stage)

                result_container = {}

                def tracked_stream():
                    for chunk in run_research_stream(
                        topic,
                        result_container,
                        progress_callback=update_progress,
                        conversation_history=conversation_history,
                    ):
                        assistant_message["partial_report"] = (
                            assistant_message.get("partial_report", "") + chunk
                        )
                        yield chunk

                try:
                    st.write_stream(tracked_stream())
                    result = result_container.get("result")
                    st.session_state["generation_active"] = False
                except (OpenAIError, RuntimeError, TimeoutError, ValueError) as error:
                    st.session_state["generation_active"] = False
                    update_progress("The response could not be completed.")
                    assistant_message["error"] = (
                        "The request could not be completed. Check the error details below."
                    )
                    assistant_message["error_detail"] = str(error)
                else:
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
