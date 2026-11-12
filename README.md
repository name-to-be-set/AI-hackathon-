# AI-hackathon-
for the hackathon global by hack culture

## Research backend

`agent.py` provides `run_research(topic)`, returning a Markdown `report` and a
`sources` list of `{title, url}` objects. It uses Tavily for web search and
Bifrost for the AI calls.

Set `TAVILY_API_KEY` and `BIFROST_API_KEY` in the local, Git-ignored `.env`
file, then start Bifrost at `http://localhost:8080`.

## Run the app

From this directory, run `streamlit run app.py`. The app uses the backend
progress callback to display its planning, searching, and report-writing
stages. Run the backend unit tests with `python -m unittest -v test_agent`.
