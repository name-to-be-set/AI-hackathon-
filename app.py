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
    st.session_state.pop("research_result", None)
    st.session_state.pop("research_error", None)
    st.session_state.pop("research_error_detail", None)


def _reset_research() -> None:
    st.session_state["topic"] = ""
    st.session_state.pop("research_result", None)
    st.session_state.pop("research_error", None)
    st.session_state.pop("research_error_detail", None)


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
st.title("AI Research Assistant")
st.write("Turn a topic into a sourced research report.")

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

topic = st.text_input(
    "What would you like to research?",
    key="topic",
    placeholder="For example: Compare the leading approaches to grid-scale energy storage",
)

start_column, reset_column = st.columns([3, 1])
with start_column:
    start_clicked = st.button(
        "Start Research",
        key="start_research",
        type="primary",
        use_container_width=True,
    )
with reset_column:
    st.button(
        "Reset",
        key="reset_research",
        on_click=_reset_research,
        use_container_width=True,
    )

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

st.caption("Built with Streamlit, Tavily, and Bifrost.")
