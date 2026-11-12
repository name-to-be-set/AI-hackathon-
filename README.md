# AI Research Assistant

Turn a research topic into a structured Markdown report with citations and
clickable source links. The Streamlit UI calls `run_research(topic)` in
`agent.py`; the backend plans focused searches, gathers web results, and writes
the report.

## Stack and flow

- **Streamlit** provides the search UI, progress tracker, source cards, and
  report download.
- **Bifrost** provides an OpenAI-compatible gateway for the language models.
- **Tavily** searches the web and returns source titles, URLs, and useful text.
- The backend uses Groq as its primary model and retries with Gemini through
  Bifrost if the primary model request fails.

`run_research(topic)` returns:

```python
{
    "report": "Markdown report with [1], [2] citations",
    "sources": [{"title": "Source title", "url": "https://example.com"}],
}
```

Source numbers in the report match the order of the returned `sources` list.

## Setup

1. Install Python 3.10 or later and Node.js.
2. Clone the repository and enter its directory:

   ```bash
   git clone <repo-url>
   cd AI-hackathon-
   ```

3. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   On Windows, activate with `.venv\Scripts\activate`.

4. Start Bifrost:

   ```bash
   npx -y @maximhq/bifrost
   ```

   Open `http://localhost:8080`, add your Gemini and Groq provider credentials,
   and create a virtual key that permits both configured models.

5. Create a local `.env` file in the repository directory:

   ```dotenv
   TAVILY_API_KEY=your_tavily_api_key
   BIFROST_API_KEY=your_bifrost_virtual_key
   ```

   Keep the real values private. `.env` is ignored by Git; verify with
   `git check-ignore -v .env`. Never commit API keys or virtual keys.

## Run

With Bifrost running, start the app from the repository directory:

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal (usually
`http://localhost:8501`). Enter a topic or select an example prompt, then click
**Start Research**. The app shows planning, searching, and report-writing
progress. Use **Reset** to clear the topic and result, or **Download report**
to save the Markdown report and source links.

## Tests

Run the backend and Streamlit UI tests:

```bash
python -m unittest -v test_agent test_app
```

The six varied research prompts are in [test_prompts.md](./test_prompts.md).
The live prompt run and observed latency notes are in
[day3_test_results.md](./day3_test_results.md). The slide-by-slide pitch draft
is in [pitch_deck.md](./pitch_deck.md), with an exported
[PowerPoint deck](./pitch_deck.pptx).
