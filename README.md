# VYOM

Chat naturally about everyday tasks, or turn a research topic into a structured
Markdown report with source links. The Streamlit UI
calls `run_research(...)` in `agent.py`; the backend routes each message to chat
or research, then searches and writes a report only when sources are needed.

## Stack and flow

- **Streamlit** provides the chat UI, progress tracker, source cards, and
  report download.
- **Tavily** searches the web and returns source titles, URLs, and useful text.
- The backend calls Groq directly and falls back to Gemini if Groq fails.
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
   ```

   Keep the real values private. `.env` is ignored by Git; verify with
   `git check-ignore -v .env`. Never commit API keys or virtual keys.

## Run

With provider credentials configured, start the app from the repository
directory:

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
The composer includes microphone dictation; recordings are transcribed with
Groq and inserted as text before the message is handled.
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

## Deploy for a hackathon (Render)

The Render Blueprint in `render.yaml` defines the Streamlit app. Push the
repository to GitHub and create or update the Blueprint in Render. Provider
credentials are supplied through Render environment settings, not Git.

- `TAVILY_API_KEY`: your Tavily API key.
- `GROQ_API_KEY` and `GEMINI_API_KEY`: direct provider credentials for the app.
Keep all provider credentials in Render's environment settings, never in Git
or the repository's `.env` file.

Free Render services sleep after 15 minutes without traffic and can take about
a minute to wake.

The Blueprint runs Streamlit on Render's assigned port. The free instance may
sleep when idle, so allow a short startup delay when opening the demo. The app
allows up to 10 submitted messages per Streamlit session. This helps limit
accidental demo usage, but a new session resets the count and it is not a
provider billing limit; configure provider-side quotas separately.
The backend also caps prompts at 12,000 characters, keeps conversation history
within a 20,000-character budget, and limits each process to five active
requests, 60 requests per minute overall, and 10 per session per minute. Requests
wait in a FIFO queue for up to 15 seconds before receiving a clear busy message.
These limits are in-process only and are not shared across multiple app instances.
Provider calls retry transient connection, timeout, rate-limit, and server errors
up to two times before trying the fallback provider. Provider tokenization differs,
so the input budget is character-based rather than an exact token count.

## Tests

Run the backend and Streamlit UI tests:

```bash
python -m unittest -v test_agent test_app test_request_limiter
```

These tests mock external AI and search providers. They cover prompt edge cases,
empty input, search-result prompt-injection boundaries, provider retries,
request limiting, and 20 simultaneous backend conversations with separate
histories. They do not measure live model quality or deployment capacity; the
backend concurrency test is not a substitute for load testing the deployed
Streamlit app.

The six varied research prompts are in [test_prompts.md](./test_prompts.md).
The live prompt run and observed latency notes are in
[day3_test_results.md](./day3_test_results.md). The slide-by-slide pitch draft
is in [pitch_deck.md](./pitch_deck.md), with an exported
[PowerPoint deck](./pitch_deck.pptx).
