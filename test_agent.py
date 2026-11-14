import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import agent
from openai import OpenAIError


class AgentTests(unittest.TestCase):
    @staticmethod
    def _completion(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def test_llm_client_uses_configured_bifrost_url(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "BIFROST_API_KEY": "test-key",
                    "BIFROST_BASE_URL": "https://gateway.example/v1",
                },
                clear=True,
            ),
            patch("agent.OpenAI") as openai,
        ):
            agent._get_llm_client()

        openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://gateway.example/v1",
        )

    def test_llm_client_defaults_to_local_bifrost_url(self):
        with (
            patch.dict("os.environ", {"BIFROST_API_KEY": "test-key"}, clear=True),
            patch("agent.OpenAI") as openai,
        ):
            agent._get_llm_client()

        openai.assert_called_once_with(
            api_key="test-key",
            base_url="http://localhost:8080/v1",
        )

    def test_empty_topic_returns_contract_without_api_calls(self):
        result = agent.run_research("  ")

        self.assertEqual(set(result), {"report", "sources", "images"})
        self.assertIn("non-empty topic", result["report"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["images"], [])

    def test_query_parser_accepts_fenced_json_and_json_after_prose(self):
        json_queries = agent._parse_queries(
            '```json\n{"queries":["first topic query", "second topic query", '
            '"third topic query", "fourth topic query"]}\n```'
        )
        queries_with_prefix = agent._parse_queries(
            'Queries: ["first topic query", "second topic query", '
            '"third topic query"]'
        )

        self.assertEqual(len(json_queries), 4)
        self.assertEqual(len(queries_with_prefix), 3)

    def test_model_output_normalizes_html_breaks_and_bullet_characters(self):
        output = (
            "• Task completion <br>• Tool‑call accuracy <br>"
            "• Faithfulness (truthfulness) <br>• Safety"
        )
        self.assertEqual(
            agent._normalize_model_output(output),
            "- Task completion\n- Tool-call accuracy\n"
            "- Faithfulness (truthfulness)\n- Safety",
        )

    def test_only_visual_or_person_focused_research_shows_search_images(self):
        self.assertTrue(agent._should_display_image("Who is Ada Lovelace?"))
        self.assertTrue(agent._should_display_image("Show me a photo of a red panda"))
        self.assertTrue(agent._should_display_image("Tell me about Taylor Swift"))
        self.assertFalse(agent._should_display_image("Compare current solar prices"))

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
            result = agent.search_web(
                ["unavailable", "available", "second available"],
                include_images=True,
            )

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
            patch("agent.route_intent", return_value="research"),
            patch("agent.generate_queries", return_value=["one", "two", "three"]),
            patch("agent.search_web", return_value=search_results),
            patch("agent.write_report", return_value="# Report"),
        ):
            result = agent.run_research("A research topic", progress.append)

        self.assertEqual(
            progress,
            [
                "Understanding your question...",
                "Preparing search queries...",
                "Looking for useful sources...",
                "Putting together your response...",
            ],
        )
        self.assertEqual(
            result,
            {
                "report": "# Report",
                "sources": [{"title": "Source A", "url": "https://example.com/a"}],
                "images": [],
            },
        )

    def test_person_research_keeps_at_most_one_image(self):
        search_results = {
            "text": "Biography details",
            "sources": [{"title": "Biography", "url": "https://example.com"}],
            "images": [
                {"url": "https://example.com/person.jpg", "description": "Portrait"},
                {"url": "https://example.com/other.jpg", "description": "Other image"},
            ],
            "errors": [],
        }
        with (
            patch("agent.route_intent", return_value="research"),
            patch("agent.generate_queries", return_value=["Ada Lovelace biography"]),
            patch("agent.search_web", return_value=search_results) as search,
            patch("agent.write_report", return_value="A short biography."),
        ):
            result = agent.run_research("Who is Ada Lovelace?")

        search.assert_called_once_with(
            ["Ada Lovelace biography"],
            include_images=True,
        )
        self.assertEqual(result["images"], search_results["images"][:1])

    def test_chat_route_skips_search_and_asks_about_vague_requests(self):
        history = [{"role": "user", "content": "Hi there"}]
        with (
            patch(
                "agent._chat_completion",
                side_effect=[
                    self._completion('{"intent":"chat"}'),
                    self._completion(
                        "Sure! What would you like me to do? I can help write, "
                        "explain, brainstorm, or research something."
                    ),
                ],
            ) as completion,
            patch("agent.search_web") as search,
        ):
            result = agent.run_research(
                "Can you do something?",
                conversation_history=history,
            )

        search.assert_not_called()
        self.assertEqual(completion.call_count, 2)
        messages = completion.call_args_list[1].args[0]
        self.assertIn(agent.BASE_SYSTEM_PROMPT, messages[0]["content"])
        self.assertIn("suggest examples", messages[0]["content"])
        self.assertEqual(messages[1], history[0])
        self.assertEqual(
            result,
            {
                "kind": "chat",
                "report": "Sure! What would you like me to do? I can help write, "
                "explain, brainstorm, or research something.",
                "sources": [],
                "images": [],
            },
        )

    def test_route_intent_retries_invalid_json_and_uses_conversation_history(self):
        history = [
            {"role": "user", "content": "What is photosynthesis?"},
            {"role": "assistant", "content": "It is how plants make food."},
        ]
        with patch(
            "agent._chat_completion",
            side_effect=[
                self._completion("intent: research"),
                self._completion('```json\n{"intent":"research"}\n```'),
            ],
        ) as completion:
            intent = agent.route_intent("Tell me more", history)

        self.assertEqual(intent, "research")
        self.assertEqual(completion.call_count, 2)
        messages = completion.call_args_list[0].args[0]
        self.assertIn(agent.BASE_SYSTEM_PROMPT, messages[0]["content"])
        self.assertEqual(messages[1:3], history)
        self.assertEqual(messages[-1]["content"], "Tell me more")
        self.assertIn(
            "intent: research",
            completion.call_args_list[1].args[0][-2]["content"],
        )

    def test_route_intent_defaults_to_research_after_two_invalid_responses(self):
        with patch(
            "agent._chat_completion",
            return_value=self._completion("not JSON"),
        ) as completion:
            intent = agent.route_intent("Can you do something?")

        self.assertEqual(intent, "research")
        self.assertEqual(completion.call_count, 2)

    def test_factual_questions_are_routed_to_web_research(self):
        with patch(
            "agent._chat_completion",
            return_value=self._completion('{"intent":"research"}'),
        ) as completion:
            intent = agent.route_intent("Who was Ada Lovelace?")

        self.assertEqual(intent, "research")
        route_prompt = completion.call_args.args[0][0]["content"]
        self.assertIn("Never answer those factual questions from memory", route_prompt)

    def test_simple_hello_never_triggers_web_search(self):
        with (
            patch("agent.route_intent", return_value="research") as route,
            patch(
                "agent._chat_completion",
                return_value=self._completion("Hello! What can I help with?"),
            ) as completion,
            patch("agent.search_web") as search,
        ):
            result = agent.run_research("Hello!")

        route.assert_called_once()
        search.assert_not_called()
        self.assertEqual(completion.call_count, 1)
        self.assertIn(
            "one brief sentence",
            completion.call_args.args[0][0]["content"],
        )
        self.assertEqual(result["kind"], "chat")
        self.assertEqual(result["report"], "Hello! What can I help with?")

    def test_query_generation_retries_then_falls_back_to_original_topic(self):
        with patch(
            "agent._chat_completion",
            side_effect=[
                self._completion("not JSON"),
                self._completion('{"queries":["only one query"]}'),
            ],
        ) as completion:
            queries = agent.generate_queries("Original research question")

        self.assertEqual(queries, ["Original research question"])
        self.assertEqual(completion.call_count, 2)

    def test_query_generation_uses_valid_queries_after_retry(self):
        queries = [
            "first focused query",
            "second focused query",
            "third focused query",
        ]
        history = [{"role": "user", "content": "I am looking at solar energy."}]
        with patch(
            "agent._chat_completion",
            side_effect=[
                self._completion("not JSON"),
                self._completion('{"queries":' + json.dumps(queries) + "}"),
            ],
        ) as completion:
            result = agent.generate_queries("What about costs?", history)

        self.assertEqual(result, queries)
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(completion.call_args_list[0].args[0][1], history[0])

    def test_run_research_uses_original_message_if_query_generation_raises(self):
        search_results = {
            "text": "",
            "sources": [],
            "images": [],
            "errors": [],
        }
        with (
            patch("agent.route_intent", return_value="research"),
            patch(
                "agent.generate_queries",
                side_effect=RuntimeError("query service unavailable"),
            ),
            patch("agent.search_web", return_value=search_results) as search,
            patch("agent.write_report", return_value="No sources found."),
        ):
            result = agent.run_research("Find current solar energy prices.")

        search.assert_called_once_with(
            ["Find current solar energy prices."],
            include_images=False,
        )
        self.assertEqual(result["report"], "No sources found.")

    def test_creative_writing_is_done_directly_without_research(self):
        with (
            patch(
                "agent._chat_completion",
                side_effect=[
                    self._completion('{"intent":"chat"}'),
                    self._completion("Kindness makes a hard day softer."),
                ],
            ) as completion,
            patch("agent.search_web") as search,
        ):
            result = agent.run_research("Write a paragraph about kindness.")

        search.assert_not_called()
        system_prompt = completion.call_args_list[1].args[0][0]["content"]
        self.assertIn("Do the requested writing directly", system_prompt)
        self.assertTrue(result["report"].endswith("Would you like any changes?"))
        self.assertEqual(
            result["report"],
            "Kindness makes a hard day softer.\n\nWould you like any changes?",
        )

    def test_image_creation_request_returns_a_prompt_honestly(self):
        with (
            patch(
                "agent._chat_completion",
                side_effect=[
                    self._completion('{"intent":"chat"}'),
                    self._completion("A cozy watercolor cat..."),
                ],
            ) as completion,
            patch("agent.search_web") as search,
        ):
            result = agent.run_research("Create an image of a cat reading a book.")

        search.assert_not_called()
        system_prompt = completion.call_args_list[1].args[0][0]["content"]
        self.assertIn("image-generation prompt", system_prompt)
        self.assertEqual(result["kind"], "chat")
        self.assertTrue(
            result["report"].startswith("Direct image generation is not configured")
        )
        self.assertTrue(result["report"].endswith("Would you like any changes?"))

    def test_configured_image_model_returns_generated_image_data(self):
        generate = Mock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(b64_json="aW1hZ2U=")]
            )
        )
        client = SimpleNamespace(
            images=SimpleNamespace(generate=generate),
        )
        with (
            patch("agent.IMAGE_MODEL", "test-image-model"),
            patch("agent._get_llm_client", return_value=client),
            patch(
                "agent._chat_completion",
                side_effect=[
                    self._completion('{"intent":"chat"}'),
                    self._completion("A watercolor cat reading a book."),
                ],
            ),
        ):
            result = agent.run_research("Create an image of a cat reading a book.")

        self.assertEqual(result["images"][0]["base64"], "aW1hZ2U=")
        generate.assert_called_once_with(
            model="test-image-model",
            prompt="A watercolor cat reading a book.",
            size="1024x1024",
        )

    def test_report_prompt_is_supportive_and_answers_the_specific_question(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="# Report"))]
        )
        search_results = {
            "sources": [{"title": "Source", "url": "https://example.com"}],
            "text": "[1] Source (https://example.com)\nUseful information",
            "errors": [],
        }

        with patch("agent._chat_completion", return_value=response) as completion:
            agent.write_report("I'm worried about this issue", search_results)

        system_prompt = completion.call_args.args[0][0]["content"]
        self.assertIn(agent.BASE_SYSTEM_PROMPT, system_prompt)
        self.assertIn("directly answers the user's specific question", system_prompt)
        self.assertIn("use only the supplied search results", system_prompt)
        self.assertIn("I could not find this in the search results.", system_prompt)
        self.assertIn("source links separately", system_prompt)

    def test_report_without_sources_returns_required_not_found_message(self):
        with patch("agent._chat_completion") as completion:
            report = agent.write_report(
                "What happened?",
                {"sources": [], "text": "", "errors": []},
            )

        self.assertEqual(report, "I could not find this in the search results.")
        completion.assert_not_called()

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

    def test_transcribe_audio_uses_bifrost_and_returns_recognized_text(self):
        create = Mock(return_value=SimpleNamespace(text="Hello there"))
        client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=create),
            )
        )
        audio_file = ("speech.wav", b"wav-audio", "audio/wav")

        with patch("agent._get_llm_client", return_value=client):
            transcript = agent.transcribe_audio(audio_file)

        self.assertEqual(transcript, "Hello there")
        create.assert_called_once_with(
            model=agent.TRANSCRIPTION_MODEL,
            file=audio_file,
            response_format="json",
        )

    def test_transcribe_audio_reports_silence(self):
        client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=Mock(return_value=SimpleNamespace(text="  ")),
                )
            )
        )

        with (
            patch("agent._get_llm_client", return_value=client),
            self.assertRaisesRegex(RuntimeError, "No speech was recognized"),
        ):
            agent.transcribe_audio(("silence.wav", b"wav-audio", "audio/wav"))

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
