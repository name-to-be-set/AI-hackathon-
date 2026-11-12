import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise SystemExit("Add TAVILY_API_KEY to your local .env file.")

response = TavilyClient(api_key=api_key).search(
    query="AI hackathon resources",
    max_results=5,
)
for result in response.get("results", []):
    print(f"{result.get('title', '')}: {result.get('url', '')}")
