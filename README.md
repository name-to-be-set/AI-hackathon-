# VYOM — AI Research Assistant

Chat naturally about everyday tasks, or turn any research topic into a
structured report with source links. VYOM decides on its own whether a message
needs a web search, so casual chat stays fast and factual answers are backed by
real sources.

## Live demo

- **Open VYOM:** https://fluffy-pixie-f01d00.netlify.app/
  (shows a short loading animation, then opens the app automatically)
- **Direct app link (backup):** https://ai-research-assistant-sw7p.onrender.com

The app runs on a free Render instance. It is kept awake with a regular uptime
ping, but if the first load is slow, give it a few seconds.

## What it can do

- **Smart routing:** casual chat and general knowledge get a direct reply;
  questions about people, events, news and other real-world topics are
  researched on the web first. If the sources do not contain the answer, VYOM
  says so instead of guessing.
- **Research reports:** a structured Markdown report with a **Sources** section
  (title and link for every source) and up to four relevant images.
- **Voice input:** tap the microphone in the composer; the recording is
  transcribed with Groq and handled like typed text.
- **File attachments:** attach PDF, image, text, Markdown or CSV files and ask
  questions about them.
- **Image generation:** requests such as "draw…", "generate image…" or
  "make an image…" use Gemini (`gemini-2.5-flash-image`), with Pollinations as
  a fallback.
- **Reply controls:** like / dislike, regenerate, copy, share (WhatsApp,
  Telegram, X, Facebook, email, plain text) and download as PDF. Research
  reports also download as Markdown.
- **Stop button:** a pause button appears inside the composer while a reply is
  being generated.
- **Themes:** System, Light and Dark, switchable from the sidebar, with
  separate background images for light and dark.
- **Multiple chats** in the sidebar and a time-aware greeting (India time).
- **Mobile and desktop** layouts.

## Stack and flow

- **Streamlit** provides the chat UI, source cards and downloads.
- **Tavily** searches the web and returns source titles, URLs and useful text.
- The backend calls **Groq** directly and falls back to **Gemini** if Groq
  fails.
- A fast model routes each message and plans up to three focused search
  queries; a larger model streams the final report. Tavily queries run
  concurrently, request at most three results each, and are cached for one
  hour.
- Casual greetings skip routing and search and use one direct LLM call. Other
  chat and general-knowledge requests skip Tavily. Up to 10 previous
  user/assistant messages are used for follow-ups.
- Invalid routing JSON defaults to chat. Invalid query JSON is retried once,
  then the original message is used as a single search query.
- If both providers fail, the app shows a friendly unavailable message.

`run_research(topic)` returns:

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

Source links are returned separately and displayed under **Sources**; inline
source numbers are removed from the displayed report.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI (chat, themes, reply controls, intro animation) |
| `agent.py` | Routing, search, report writing, transcription |
| `attachments.py` | Answers questions about attached files |
| `patches.py` | Runtime compatibility patches |
| `vyom_theme.css` | Light / dark theme styles |
| `static/` | `dark_bg.jpg` and `light_bg.jpg` background images |
| `index.html` | Loading page hosted on Netlify; opens the app when ready |
| `render.yaml` | Render Blueprint for deployment |

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
   `git check-ignore -v .env`. Never commit API keys.

## Run

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).
Type a message, attach a file, or use the microphone; press Enter or the
up-arrow button to send. Chats live only in the active Streamlit session and
are not saved to a database.

## Deploy (Render + Netlify)

The Render Blueprint in `render.yaml` defines the Streamlit app. Push the
repository to GitHub and create or update the Blueprint in Render. Provider
credentials are supplied through Render environment settings, never Git:

- `TAVILY_API_KEY`
- `GROQ_API_KEY` and `GEMINI_API_KEY`

Free Render services sleep after 15 minutes without traffic. To avoid slow
first loads, an uptime monitor pings
`https://ai-research-assistant-sw7p.onrender.com/_stcore/health` every
5 minutes. `index.html` is hosted separately on Netlify: it shows the loading
animation instantly, warms the app in the background and then opens it.

The app allows up to 10 submitted messages per Streamlit session to limit
accidental demo usage (a new session resets the count; configure provider-side
quotas separately). The backend also caps prompts at 12,000 characters, keeps
conversation history within a 20,000-character budget, and limits each process
to five active requests, 60 requests per minute overall and 10 per session per
minute. Requests wait in a FIFO queue for up to 15 seconds before receiving a
clear busy message. These limits are in-process only and are not shared across
multiple app instances. Provider calls retry transient connection, timeout,
rate-limit and server errors up to two times before trying the fallback
provider. The input budget is character-based, not an exact token count.

## Tests

```bash
python -m unittest -v test_agent test_app test_request_limiter
```

The tests mock external AI and search providers. They cover prompt edge cases,
empty input, search-result prompt-injection boundaries, provider retries,
request limiting, and 20 simultaneous backend conversations with separate
histories. They do not measure live model quality or deployment capacity.

The six varied research prompts are in [test_prompts.md](./test_prompts.md).
The live prompt run and latency notes are in
[day3_test_results.md](./day3_test_results.md). The pitch draft is in
[pitch_deck.md](./pitch_deck.md), with an exported
[PowerPoint deck](./pitch_deck.pptx).