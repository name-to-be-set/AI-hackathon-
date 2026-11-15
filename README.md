# AI Research Assistant

Chat naturally about everyday tasks, or turn a research topic into a structured
Markdown report with source links. The Streamlit UI
calls `run_research(...)` in `agent.py`; the backend routes each message to chat
or research, then searches and writes a report only when sources are needed.

## Stack and flow

- **Streamlit** provides the chat UI, progress tracker, source cards, and
  report download.
- **Bifrost** is an optional OpenAI-compatible gateway for the language models.
- **Tavily** searches the web and returns source titles, URLs, and useful text.
- When `BIFROST_HOST` is set, requests use Bifrost. Otherwise, the backend calls
  Groq directly and falls back to Gemini if Groq fails.
- A fast model routes messages and plans up to three focused queries; the larger
  model streams the final research report. Tavily queries run concurrently,
  request at most three results each, and cache results for one hour.
- Casual greetings skip routing and search and use one direct LLM call. Other
  chat and general knowledge requests skip Tavily. Up to 10 previous
  user/assistant messages are used for follow-ups.
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
Research results may include up to four image URLs and descriptions:

```python
{
    "report": "Markdown report or a short conversational reply",
    "sources": [{"title": "Source title", "url": "https://example.com"}],
    "images": [
        {"url": "https://example.com/image.jpg", "description": "Caption"},
    ],
    "kind": "chat",  # only present for casual chat replies
}
```

## Setup

1. Install Python 3.10 or later.
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

4. Create a local `.env` file in the repository directory:

   ```dotenv
   TAVILY_API_KEY=your_tavily_api_key
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   # Optional: set both values to route requests through Bifrost instead.
   BIFROST_HOST=
   BIFROST_API_KEY=
   # Optional; the default is groq/openai/whisper-large-v3-turbo
   BIFROST_TRANSCRIPTION_MODEL=groq/openai/whisper-large-v3-turbo
   ```

   Keep the real values private. `.env` is ignored by Git; verify with
   `git check-ignore -v .env`. Never commit API keys or virtual keys.
   For local Bifrost use, start a Bifrost gateway and set `BIFROST_HOST` and
   `BIFROST_API_KEY` to its URL and virtual key.

## Run

With provider credentials configured, start the app from the repository
directory. Bifrost is only needed when `BIFROST_HOST` is set:

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
The composer includes microphone dictation; when Bifrost is configured,
recordings are sent to its transcription model and inserted as text before the
message is handled.
Chats are kept only in the active Streamlit session and are not saved to a
local database or restored in a new session. Database files created by older
versions are left untouched and are no longer accessed. Replies include
horizontal like/dislike, share, and copy controls after the first response;
feedback stays with the reply for the active session. Replies can also be
downloaded as PDFs. Research reports have a Markdown download and a **Sources**
section; inline source numbers are omitted from the displayed report.

Research answers may show up to four Tavily images in a responsive row. Requests
starting with image-generation commands such as “draw,” “generate image,” “create
a picture,” or “make an image” use Gemini's `gemini-2.5-flash-image` model and
fall back to Pollinations if Gemini is unavailable. The app uses `GEMINI_API_KEY`
from `.env` or the deployment environment. The initial greeting uses India time
and appears once at the top of an empty chat. The layout adapts to mobile and
desktop screen widths.

If both direct providers fail, the app displays a friendly unavailable message.
When using Bifrost, verify that `BIFROST_API_KEY` is valid and permits the
configured models.

## Deploy for a hackathon (Render)

The Render Blueprint in `render.yaml` defines the Streamlit app and an optional
Bifrost web service. The app can call Groq and Gemini directly; leave
`BIFROST_HOST` unset to use that mode. Push the repository to GitHub and create
or update the Blueprint in Render. Provider credentials and virtual-key values
are supplied through Render environment settings, not Git.

- `TAVILY_API_KEY`: your Tavily API key.
- `GROQ_API_KEY` and `GEMINI_API_KEY`: direct provider credentials for the app.
- `BIFROST_HOST` and `BIFROST_API_KEY`: optional settings for Bifrost mode.

The free Bifrost service has a public URL, so inference authentication and
dashboard authentication are enabled. The Blueprint shares the virtual key
through a Render environment-variable group; set a fresh `sk-bf-` key there
and do not reuse any key exposed elsewhere. Render generates the Bifrost admin
password, setup token, and encryption key. Keep all provider credentials in
Render's environment settings, never in Git or the repository's `.env` file.
Set `BIFROST_HOST` only when you want the app to route requests through the
optional gateway; the app adds the OpenAI-compatible `/v1` path.

Free Render services sleep after 15 minutes without traffic and can take about
a minute to wake. Bifrost's file-only configuration is restored at startup, but
its dashboard configuration is read-only; change providers or virtual keys in
`bifrost/config.json` and redeploy. The `BIFROST_TRANSCRIPTION_MODEL` setting is
optional; configure it only if transcription is enabled for your Bifrost key.

The Blueprint runs Streamlit on Render's assigned port. The free instance may
sleep when idle, so allow a short startup delay when opening the demo. The app
allows up to 10 submitted messages per Streamlit session. This helps limit
accidental demo usage, but a new session resets the count and it is not a
provider billing limit; configure provider-side quotas separately.

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
