import streamlit as st

st.set_page_config(page_title="Research Assistant")
st.title("Research Assistant")
st.write("Enter a topic to see a sample research result.")

query = st.text_input("Research topic")

if st.button("Start Research"):
    if not query.strip():
        st.warning("Enter a research topic to get started.")
    else:
        results = {
            "summary": f"This is a sample summary for: {query}",
            "sources": [
                {
                    "title": "Example source",
                    "url": "https://example.com",
                }
            ],
        }

        st.subheader("Results")
        st.write(results["summary"])
        st.markdown("**Sources**")
        for source in results["sources"]:
            st.markdown(f"- [{source['title']}]({source['url']})")
