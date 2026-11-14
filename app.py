import streamlit as st
from openai import OpenAIError

from agent import run_research

EXAMPLE_PROMPTS = (
    (
        "Market research",
        "Assess the current market for AI note-taking apps: target users, leading products, differentiators, and growth drivers.",
    ),
    (
        "Technical comparison",
        "Compare LFP and NMC batteries for grid-scale storage across safety, cost, lifespan, and recyclability.",
    ),
    (
        "Recent developments",
        "What are the most significant developments in open-source large language models in the past six months?",
    ),
    (
        "Policy and regulation",
        "Summarize current regulatory approaches to generative AI transparency in the EU and United States.",
    ),
    (
        "Case studies",
        "Find examples of small businesses using AI customer-support agents and summarize outcomes and pitfalls.",
    ),
    (
        "Evidence review",
        "What does recent research say about how blue-light exposure affects sleep, and what are the main limitations?",
    ),
)


def _choose_prompt(prompt: str) -> None:
    st.session_state["topic"] = prompt
    st.session_state["next_topic"] = ""
    st.session_state.pop("research_result", None)
    st.session_state.pop("research_error", None)
    st.session_state.pop("research_error_detail", None)
    st.session_state.pop("search_requested", None)


def _reset_research() -> None:
    st.session_state["topic"] = ""
    st.session_state["next_topic"] = ""
    st.session_state.pop("research_result", None)
    st.session_state.pop("research_error", None)
    st.session_state.pop("research_error_detail", None)
    st.session_state.pop("search_requested", None)


def _submit_followup() -> None:
    st.session_state["topic"] = st.session_state.get("next_topic", "")
    st.session_state["search_requested"] = True


def _render_result(result: dict) -> None:
    if result.get("error"):
        st.error(str(result["error"]))
        return

    report = result.get("report")
    sources = result.get("sources")
    if not isinstance(report, str) or not report.strip():
        st.error("The research agent did not return a report.")
        return
    if not isinstance(sources, list):
        st.error("The research agent returned an invalid source list.")
        return

    st.markdown(report)

    valid_sources = [
        source
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("title"), str)
        and isinstance(source.get("url"), str)
        and source["url"].startswith(("https://", "http://"))
    ]
    if valid_sources:
        st.subheader("Sources")
        for index, source in enumerate(valid_sources, start=1):
            with st.container(border=True):
                title_column, link_column = st.columns([5, 1])
                with title_column:
                    st.write(f"{index}. {source['title']}")
                with link_column:
                    st.link_button(
                        "Open source",
                        source["url"],
                        key=f"source_{index}",
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
    )


st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7ff 0%, #eef8ff 40%, #f8f9ff 100%);
    }
    .stApp [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.25);
    }
    div[data-testid="stTextInput"] > div {
        border-radius: 999px;
        box-shadow: 0 14px 30px rgba(79, 70, 229, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.18);
        background: rgba(255, 255, 255, 0.95);
        overflow: hidden;
    }
    div[data-testid="stTextInput"] input {
        font-size: 1.08rem;
        padding: 1rem 1.1rem;
        border: none;
        background: transparent;
        color: #1f2937;
        min-height: 3rem;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #6b7280;
    }
    div[data-testid="stTextInput"] label {
        font-weight: 700;
        color: #312e81;
        letter-spacing: 0.02em;
    }
    div[data-testid="stButton"] > button {
        border-radius: 14px;
        font-weight: 700;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.18);
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        border: none;
        color: white;
        min-height: 3.1rem;
        border-radius: 50%;
        width: 3.1rem;
        min-width: 3.1rem;
        height: 3.1rem;
        padding: 0;
        font-size: 1.3rem;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(99,102,241,0.15);
        color: #4338ca;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding: 2rem 0 1rem;">
      <div style="font-size: 0.9rem; font-weight: 700; color: #6366f1;
                  text-transform: uppercase; letter-spacing: 0.12em;">
        Your research assistant
      </div>
      <h1 style="margin: 0.4rem 0; color: #1f2450;">
        Hello! What would you like to explore?
      </h1>
      <p style="font-size: 1.1rem; color: #5b6178; margin-bottom: 0;">
        Ask a question and get a clear report backed by web sources.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Example prompts")
    for index, (label, prompt) in enumerate(EXAMPLE_PROMPTS):
        st.button(
            label,
            key=f"example_{index}",
            on_click=_choose_prompt,
            args=(prompt,),
            use_container_width=True,
        )

st.caption("What would you like to research?")
search_column, submit_column, reset_column = st.columns([12, 1, 1])
with search_column:
    topic = st.text_input(
        "Research search",
        key="topic",
        placeholder="Search a topic, question, or idea...",
        label_visibility="collapsed",
    )
with submit_column:
    start_clicked = st.button(
        "➜",
        key="start_research",
        type="primary",
        help="Search",
        use_container_width=True,
    )
with reset_column:
    st.button(
        "Reset",
        key="reset_research",
        on_click=_reset_research,
        use_container_width=True,
    )

followup_clicked = st.session_state.pop("search_requested", False)
start_clicked = start_clicked or followup_clicked

if start_clicked:
    st.session_state.pop("research_result", None)
    st.session_state.pop("research_error", None)
    st.session_state.pop("research_error_detail", None)

    if not topic.strip():
        st.session_state["research_error"] = (
            "Enter a research topic or choose an example prompt."
        )
    else:
        with st.status("Planning queries...", expanded=True) as status:
            try:
                result = run_research(topic, progress_callback=status.write)
            except (OpenAIError, RuntimeError, TimeoutError, ValueError) as error:
                status.update(label="Research could not be completed", state="error")
                st.session_state["research_error"] = (
                    "Research could not be completed. Check that Bifrost is running "
                    "and your API keys are configured."
                )
                st.session_state["research_error_detail"] = str(error)
            else:
                if not isinstance(result, dict):
                    status.update(label="Research returned an invalid response", state="error")
                    st.session_state["research_error"] = (
                        "The research service returned an unexpected response."
                    )
                elif result.get("error"):
                    status.update(label="Research could not be completed", state="error")
                    st.session_state["research_error"] = str(result["error"])
                else:
                    status.update(label="Research complete", state="complete")
                    st.session_state["research_result"] = result

if error := st.session_state.get("research_error"):
    st.error(error)
    if detail := st.session_state.get("research_error_detail"):
        st.caption(detail)

if result := st.session_state.get("research_result"):
    _render_result(result)
    st.divider()
    st.subheader("What would you like to research next?")
    next_search_column, next_submit_column = st.columns([12, 1])
    with next_search_column:
        st.text_input(
            "Next research search",
            key="next_topic",
            placeholder="Ask another question...",
            label_visibility="collapsed",
        )
    with next_submit_column:
        st.button(
            "➜",
            key="next_research",
            type="primary",
            help="Search again",
            use_container_width=True,
            on_click=_submit_followup,
        )

st.caption("Built with Streamlit, Tavily, and Bifrost.")
