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


def _render_result(result: dict, key: str) -> None:
    if result.get("error"):
        st.error(str(result["error"]))
        return

    report = result.get("report")
    sources = result.get("sources")
    if not isinstance(report, str) or not report.strip():
        st.error("The research agent did not return a report.")
        return
    if not isinstance(sources, list):
        sources = []

    images = [
        url
        for url in result.get("images", [])
        if isinstance(url, str) and url.startswith(("https://", "http://"))
    ]
    if images:
        columns = st.columns(3)
        for index, url in enumerate(images[:6]):
            columns[index % 3].image(url, use_container_width=True)

    valid_sources = [
        source
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("title"), str)
        and isinstance(source.get("url"), str)
        and source["url"].startswith(("https://", "http://"))
    ]

    st.caption(f"{len(report.split())} words · {len(valid_sources)} sources")
    st.markdown(report)

    if valid_sources:
        st.subheader("Sources")
        for index, source in enumerate(valid_sources, start=1):
            with st.container(border=True):
                title_column, link_column = st.columns([5, 1])
                with title_column:
                    st.write(f"{index}. {source['title']}")
                with link_column:
                    st.link_button(
                        "Open",
                        source["url"],
                        key=f"{key}_src_{index}",
                    )
    else:
        st.info("No source links were found for this topic.")

    download_content = report
    if valid_sources:
        source_lines = [
            f"- [{source['title']}]({source['url']})" for source in valid_sources
        ]
        download_content += "\n\n## Sources\n\n" + "\n".join(source_lines)

    st.download_button(
        "Download report",
        data=download_content,
        file_name="research-report.md",
        mime="text/markdown",
        key=f"{key}_download",
    )


def _queue_prompt(prompt: str) -> None:
    st.session_state["pending"] = prompt


def _new_chat() -> None:
    st.session_state["history"] = []
    st.session_state.pop("pending", None)


st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")
st.write("Turn a topic into a sourced research report.")

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.header("Example prompts")
    for index, (label, prompt_text) in enumerate(EXAMPLE_PROMPTS):
        st.button(
            label,
            key=f"example_{index}",
            on_click=_queue_prompt,
            args=(prompt_text,),
            use_container_width=True,
        )
    st.divider()
    st.button(
        "New chat",
        key="new_chat",
        on_click=_new_chat,
        use_container_width=True,
    )

# Show all previous questions and reports, one after another.
for index, entry in enumerate(st.session_state["history"]):
    with st.chat_message("user"):
        st.write(entry["topic"])
    with st.chat_message("assistant"):
        if entry.get("error"):
            st.error(entry["error"])
            if entry.get("detail"):
                st.caption(entry["detail"])
        else:
            _render_result(entry["result"], key=f"h{index}")

prompt = st.chat_input("What would you like to research?") or st.session_state.pop(
    "pending", None
)

if prompt and prompt.strip():
    prompt = prompt.strip()
    new_index = len(st.session_state["history"])
    entry = {"topic": prompt}

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.status("Planning queries...", expanded=True) as status:
            try:
                result = run_research(prompt, progress_callback=status.write)
            except (OpenAIError, RuntimeError, TimeoutError, ValueError) as error:
                status.update(label="Research could not be completed", state="error")
                entry["error"] = (
                    "Research could not be completed. Check that Bifrost is running "
                    "and your API keys are configured."
                )
                entry["detail"] = str(error)
            else:
                if not isinstance(result, dict):
                    status.update(
                        label="Research returned an invalid response", state="error"
                    )
                    entry["error"] = (
                        "The research service returned an unexpected response."
                    )
                elif result.get("error"):
                    status.update(
                        label="Research could not be completed", state="error"
                    )
                    entry["error"] = str(result["error"])
                else:
                    status.update(
                        label="Research complete", state="complete", expanded=False
                    )
                    entry["result"] = result

        if entry.get("error"):
            st.error(entry["error"])
            if entry.get("detail"):
                st.caption(entry["detail"])
        else:
            _render_result(entry["result"], key=f"h{new_index}")

    st.session_state["history"].append(entry)

st.caption("Built with Streamlit, Tavily, and Bifrost.")
