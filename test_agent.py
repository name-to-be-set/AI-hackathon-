import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent
from openai import OpenAIError


class AgentTests(unittest.TestCase):
    def test_empty_topic_returns_contract_without_api_calls(self):
        result = agent.run_research("  ")

        self.assertEqual(set(result), {"report", "sources", "images"})
        self.assertIn("non-empty topic", result["report"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["images"], [])

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
        search_calls = []

        class FakeTavilyClient:
            def search(self, query, **kwargs):
                search_calls.append(kwargs)
                if query == "unavailable":
                    raise TimeoutError("request timed out")
                return {
                    "images": [
                        {
                            "url": "https://example.com/image.jpg",
                            "description": "Example image",
                        }
                    ],
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
            result = agent.search_web(["unavailable", "available", "second available"])

        self.assertEqual(result["sources"], [
            {"title": "Source A", "url": "https://example.com/a"}
        ])
        self.assertEqual(
            result["images"],
            [{"url": "https://example.com/image.jpg", "description": "Example image"}],
        )
        self.assertIn("[1] Source A", result["text"])
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(len(result["images"]), 1)
        self.assertTrue(all(call["include_images"] for call in search_calls))

    def test_run_research_reports_progress_and_preserves_contract(self):
        progress = []
        search_results = {
            "text": "[1] Source A (https://example.com/a)\nUseful text",
            "sources": [{"title": "Source A", "url": "https://example.com/a"}],
            "images": [],
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
                "images": [],
            },
        )

    def test_casual_message_skips_search_and_returns_chat_reply(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!"))]
        )
        with (
            patch("agent._chat_completion", return_value=response) as completion,
            patch("agent.search_web") as search,
        ):
            result = agent.run_research("How are you?")

        completion.assert_called_once()
        search.assert_not_called()
        self.assertEqual(
            result,
            {"kind": "chat", "report": "Hello!", "sources": [], "images": []},
        )

    def test_chat_completion_retries_with_backup_model(self):
        class FakeCompletions:
            def __init__(self):
                self.models = []

            def create(self, model, messages):
                self.models.append(model)
                if model == agent.MODEL:
                    raise OpenAIError("primary unavailable")
                return "backup response"

        completions = FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        with patch("agent._get_llm_client", return_value=client):
            response = agent._chat_completion([{"role": "user", "content": "test"}])

        self.assertEqual(response, "backup response")
        self.assertEqual(completions.models, [agent.MODEL, agent.FALLBACK_MODEL])

    def test_chat_completion_does_not_retry_authentication_failure(self):
        class FakeAuthenticationError(OpenAIError):
            pass

        class FakeCompletions:
            def create(self, model, messages):
                raise FakeAuthenticationError("unauthorized")

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        with (
            patch("agent.AuthenticationError", FakeAuthenticationError),
            patch("agent._get_llm_client", return_value=client),
        ):
            with self.assertRaisesRegex(RuntimeError, "Bifrost rejected BIFROST_API_KEY"):
                agent._chat_completion([{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
