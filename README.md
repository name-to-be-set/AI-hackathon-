# AI Research Assistant

Chat naturally about everyday tasks, or turn a research topic into a structured
Markdown report with source links. The Streamlit UI
calls `run_research(...)` in `agent.py`; the backend routes each message to chat
or research, then searches and writes a report only when sources are needed.

## Stack and flow

- **Streamlit** provides the chat UI, progress tracker, source cards, and
  report download.
- **Bifrost** provides an OpenAI-compatible gateway for the language models.
- **Tavily** searches the web and returns source titles, URLs, and useful text.
- The backend uses Groq as its primary model and retries with Gemini through
  Bifrost if the primary model request fails.
- The model classifies each message before query generation. Chat and general
  knowledge requests skip Tavily; research requests generate 3-4 focused
  queries. Up to 10 previous user/assistant messages are used for follow-ups.
- Invalid routing JSON defaults to chat. Invalid query JSON is retried once,
  then the original user message is used as a single search query.

`run_research(topic)` returns:

```python
{
    "report": "Markdown report",
    "sources": [{"title": "Source title", "url": "https://example.com"}],
}
```

Research source links are returned separately and displayed under **Sources**.
The result may include one image URL for a visual or person-focused request:

```python
{
    "report": "Markdown report or a short conversational reply",
    "sources": [{"title": "Source title", "url": "https://example.com"}],
    "images": [{"url": "https://example.com/image.jpg", "description": "Caption"}],
    "kind": "chat",  # only present for casual chat replies
}
```

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
   BIFROST_BASE_URL=http://localhost:8080/v1
   # Optional; set a Bifrost image-generation model to enable generated images
   BIFROST_IMAGE_MODEL=
   # Optional; the default is groq/openai/whisper-large-v3-turbo
   BIFROST_TRANSCRIPTION_MODEL=groq/openai/whisper-large-v3-turbo
   ```

   Keep the real values private. `.env` is ignored by Git; verify with
   `git check-ignore -v .env`. Never commit API keys or virtual keys.
   Enable the configured Whisper transcription model for the Bifrost virtual key.

## Run

With Bifrost running, start the app from the repository directory:

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal (usually
`http://localhost:8501`). Send a message using the compact chat composer; press
Enter or use its up-arrow button. A brief, time-aware greeting appears before
the first message. Casual conversation gets a natural reply; factual questions
about people, events, news, and other real-world topics are answered using web
search results. If the results do not contain the answer, the assistant says so
rather than guessing. The assistant acts as a practical, supportive mentor and
may offer a concise suggestion when useful, with a compact **Thinking**
disclosure for progress updates.
The composer includes microphone dictation; recordings are sent to the configured
Bifrost transcription model and inserted as text before the message is handled.
Chats are saved locally in `~/.ai_research_assistant/chats.sqlite3` and appear
in the expandable chat-history sidebar. Set `CHAT_DB_PATH` to choose another
database location. Replies include horizontal like/dislike, share, and copy
controls after the first response; feedback is saved with each chat. Replies
can also be downloaded as PDFs. Research reports have a Markdown download and
a **Sources** section; inline source numbers are omitted from the displayed
report.

Search results show at most one image, and only for image- or person-focused
questions. To generate images directly, set `BIFROST_IMAGE_MODEL` to an image
generation model enabled for your Bifrost virtual key. Without it, image requests
return a ready-to-use generation prompt instead.

If Bifrost reports an authentication error, verify that `BIFROST_API_KEY` is a
valid Bifrost virtual key and that it allows both configured models. The app
cannot repair a missing or invalid provider key.

## Deploy for a hackathon (Render)

The repository includes a Render Blueprint in `render.yaml`. Push the project
to GitHub, create a new Blueprint in Render from that repository, then set the
requested environment variables:

- `TAVILY_API_KEY`: your Tavily API key.
- `BIFROST_API_KEY`: a Bifrost virtual key with access to the configured models.
- `BIFROST_BASE_URL`: the reachable Bifrost endpoint ending in `/v1`.

The app and Bifrost must both be reachable from Render. Do not use
`http://localhost:8080/v1` for a separately hosted Bifrost instance; `localhost`
would refer to the Render app container. Keep provider credentials in Render's
environment settings, not in Git or the repository's `.env` file. The
`BIFROST_IMAGE_MODEL` and `BIFROST_TRANSCRIPTION_MODEL` settings are optional;
configure them only if those features are enabled for your Bifrost key.

The Blueprint runs Streamlit on Render's assigned port. The free instance may
sleep when idle, so allow a short startup delay when opening the demo. Local
chat history uses SQLite on the app's filesystem and is not durable across
instance replacement or redeploy; that is usually acceptable for a hackathon
demo.

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
