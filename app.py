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


def _archive_latest_search() -> None:
    topic = st.session_state.get("last_search_topic")
    result = st.session_state.get("research_result")
    error = st.session_state.get("research_error")
    if topic and (result or error):
        st.session_state.setdefault("research_history", []).insert(
            0,
            {
                "topic": topic,
                "result": result,
                "error": error,
                "error_detail": st.session_state.get("research_error_detail"),
            },
        )


def _render_result(result: dict, key_prefix: str = "result") -> None:
    if result.get("error"):
        st.error(str(result["error"]))
        return

    report = result.get("report")
    if not isinstance(report, str) or not report.strip():
        st.error("The research agent did not return a report.")
        return
    st.markdown(report)
    if result.get("kind") == "chat":
        return

    sources = result.get("sources")
    if not isinstance(sources, list):
        st.error("The research agent returned an invalid source list.")
        return

    images = result.get("images")
    valid_images = []
    if isinstance(images, list):
        valid_images = [
            image
            for image in images
            if isinstance(image, dict)
            and isinstance(image.get("url"), str)
            and image["url"].startswith(("https://", "http://"))
        ]
    if valid_images:
        st.subheader("Images")
        image_columns = st.columns(min(len(valid_images), 3))
        for index, image in enumerate(valid_images):
            with image_columns[index % len(image_columns)]:
                st.image(
                    image["url"],
                    caption=image.get("description") or None,
                    width="stretch",
                )

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


st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")
st.write("Ask a quick question or turn a topic into a sourced research report.")

st.session_state.setdefault("research_history", [])

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
    "What would you like to ask or research?",
    key="topic",
    placeholder="For example: Compare the leading approaches to grid-scale energy storage",
)

start_clicked = st.button(
    "Search",
    key="start_research",
    type="primary",
    use_container_width=True,
)

if start_clicked:
    _archive_latest_search()
    st.session_state.pop("research_result", None)
    st.session_state.pop("research_error", None)
    st.session_state.pop("research_error_detail", None)
    st.session_state["last_search_topic"] = topic.strip()

    if not topic.strip():
        st.session_state["research_error"] = (
            "Enter a question or choose an example prompt."
        )
    else:
        with st.status("Preparing your response...", expanded=True) as status:
            try:
                result = run_research(topic, progress_callback=status.write)
            except (OpenAIError, RuntimeError, TimeoutError, ValueError) as error:
                status.update(label="Research could not be completed", state="error")
                st.session_state["research_error"] = (
                    "The request could not be completed. Check the error details below."
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
                elif result.get("kind") == "chat":
                    status.update(label="Reply complete", state="complete")
                    st.session_state["research_result"] = result
                else:
                    status.update(label="Research complete", state="complete")
                    st.session_state["research_result"] = result

if error := st.session_state.get("research_error"):
    st.error(error)
    if detail := st.session_state.get("research_error_detail"):
        st.caption(detail)

if result := st.session_state.get("research_result"):
    if latest_topic := st.session_state.get("last_search_topic"):
        st.caption(f"Latest response for: {latest_topic}")
    _render_result(result)

history = st.session_state["research_history"]
if history:
    with st.expander(f"Previous searches ({len(history)})"):
        for index, entry in enumerate(history):
            with st.expander(entry["topic"]):
                if entry.get("error"):
                    st.error(entry["error"])
                    if entry.get("error_detail"):
                        st.caption(entry["error_detail"])
                elif entry.get("result"):
                    _render_result(entry["result"], key_prefix=f"history_{index}")

st.caption("Built with Streamlit, Tavily, and Bifrost. Search history lasts for this session.")
