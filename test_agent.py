import unittest
from unittest.mock import patch

import agent


class AgentTests(unittest.TestCase):
    def test_empty_topic_returns_contract_without_api_calls(self):
        result = agent.run_research("  ")

        self.assertEqual(set(result), {"report", "sources"})
        self.assertIn("non-empty topic", result["report"])
        self.assertEqual(result["sources"], [])

    def test_query_parser_accepts_json_and_numbered_lines(self):
        json_queries = agent._parse_queries(
            '["first topic query", "second topic query", '
            '"third topic query", "fourth topic query"]'
        )
        numbered_queries = agent._parse_queries(
            "1. first topic query\n2. second topic query\n"
            "3. third topic query\n4. fourth topic query"
        )

        self.assertEqual(len(json_queries), 4)
        self.assertEqual(len(numbered_queries), 4)

    def test_search_continues_after_failure_and_removes_duplicate_urls(self):
        class FakeTavilyClient:
            def search(self, query, **kwargs):
                if query == "unavailable":
                    raise TimeoutError("request timed out")
                return {
                    "results": [
                        {
                            "title": "Source A",
                            "url": "https://example.com/a",
                            "content": "Useful text",
                        },
                        {
                            "title": "Duplicate Source A",
                            "url": "https://example.com/a",
                            "content": "Duplicate text",
                        },
                    ]
                }

        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch("agent.TavilyClient", return_value=FakeTavilyClient()),
        ):
            result = agent.search_web(["unavailable", "available"])

        self.assertEqual(result["sources"], [
            {"title": "Source A", "url": "https://example.com/a"}
        ])
        self.assertIn("[1] Source A", result["text"])
        self.assertEqual(len(result["errors"]), 1)

    def test_run_research_reports_progress_and_preserves_contract(self):
        progress = []
        search_results = {
            "text": "[1] Source A (https://example.com/a)\nUseful text",
            "sources": [{"title": "Source A", "url": "https://example.com/a"}],
            "errors": [],
        }

        with (
            patch("agent.generate_queries", return_value=["one", "two", "three"]),
            patch("agent.search_web", return_value=search_results),
            patch("agent.write_report", return_value="# Report"),
        ):
            result = agent.run_research("A research topic", progress.append)

        self.assertEqual(
            progress,
            ["Planning queries...", "Searching web...", "Writing report..."],
        )
        self.assertEqual(
            result,
            {
                "report": "# Report",
                "sources": [{"title": "Source A", "url": "https://example.com/a"}],
            },
        )


if __name__ == "__main__":
    unittest.main()
