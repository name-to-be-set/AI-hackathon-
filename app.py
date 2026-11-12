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

st.set_page_config(page_title="AI Research Assistant")
st.title("AI Research Assistant")
st.write("Turn a topic into a sourced research report.")

with st.sidebar:
    st.header("Example prompts")
    for index, (label, prompt) in enumerate(EXAMPLE_PROMPTS):
        if st.button(label, key=f"example_{index}", use_container_width=True):
            st.session_state["topic"] = prompt
            st.rerun()

topic = st.text_input(
    "What would you like to research?",
    key="topic",
    placeholder="For example: Compare the leading approaches to grid-scale energy storage",
)

if st.button("Start Research", type="primary"):
    if not topic.strip():
        st.warning("Enter a research topic or choose an example prompt.")
    else:
        result = None
        with st.status("Planning queries...", expanded=True) as status:
            try:
                result = run_research(topic, progress_callback=status.write)
            except (OpenAIError, RuntimeError, ValueError) as error:
                status.update(label="Research could not be completed", state="error")
                st.error(
                    "Research could not be completed. Check that Bifrost is running "
                    "and your API keys are configured."
                )
                st.caption(str(error))
            else:
                status.update(label="Research complete", state="complete")

        if isinstance(result, dict):
            if result.get("error"):
                st.error(str(result["error"]))
            else:
                report = result.get("report")
                sources = result.get("sources")
                if not isinstance(report, str) or not report.strip():
                    st.error("The research agent did not return a report.")
                elif not isinstance(sources, list):
                    st.error("The research agent returned an invalid source list.")
                else:
                    st.markdown(report)
                    if sources:
                        st.subheader("Sources")
                        for index, source in enumerate(sources, start=1):
                            if (
                                isinstance(source, dict)
                                and source.get("title")
                                and source.get("url")
                            ):
                                st.markdown(
                                    f"{index}. [{source['title']}]({source['url']})"
                                )
                    else:
                        st.info("No source links were found for this topic.")

                    download_content = report
                    if sources:
                        source_lines = [
                            f"- [{source['title']}]({source['url']})"
                            for source in sources
                            if isinstance(source, dict)
                            and source.get("title")
                            and source.get("url")
                        ]
                        if source_lines:
                            download_content += "\n\n## Sources\n\n"
                            download_content += "\n".join(source_lines)

                    st.download_button(
                        "Download report",
                        data=download_content,
                        file_name="research-report.md",
                        mime="text/markdown",
                    )
        elif result is not None:
            st.error("The research agent returned an unexpected response.")
