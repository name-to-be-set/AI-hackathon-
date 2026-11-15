import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import agent
from openai import OpenAIError
from requests.exceptions import RequestException


class AgentTests(unittest.TestCase):
    @staticmethod
    def _completion(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
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

        self.assertEqual(len(json_queries), 3)
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

    def test_supported_image_creation_phrases_are_detected(self):
        self.assertTrue(agent._is_image_creation_request("Draw a red panda"))
        self.assertTrue(agent._is_image_creation_request("Generate an image of a city"))
        self.assertTrue(agent._is_image_creation_request("Create a picture of a cat"))
        self.assertTrue(agent._is_image_creation_request("Make an image of the moon"))
        self.assertFalse(
            agent._is_image_creation_request("Tell me about an image model")
        )

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
        self.assertTrue(
            all(call["include_image_descriptions"] for call in search_calls)
        )
        self.assertTrue(all(call["max_results"] == 3 for call in search_calls))
        self.assertTrue(all(call["search_depth"] == "basic" for call in search_calls))

    def test_search_trims_result_text_to_six_hundred_characters(self):
        response = {
            "images": [],
            "results": [
                {
                    "title": "Long source",
                    "url": "https://example.com/long",
                    "content": "x" * 900,
                }
            ],
        }
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch("agent._cached_tavily_search", return_value=response),
        ):
            result = agent.search_web(["one query"])

        self.assertEqual(len(result["text"].split("\n", 1)[1]), 600)

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

    def test_research_answers_include_web_images(self):
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
            patch("agent.generate_queries", return_value=["Compare solar prices"]),
            patch("agent.search_web", return_value=search_results) as search,
            patch("agent.write_report", return_value="A short biography."),
        ):
            result = agent.run_research("Compare current solar prices.")

        search.assert_called_once_with(
            ["Compare solar prices"],
            include_images=True,
        )
        self.assertEqual(result["images"], search_results["images"])

    def test_chat_route_skips_search_and_asks_about_vague_requests(self):
        history = [{"role": "user", "content": "Hi there"}]
        with (
            patch(
                "agent.call_llm",
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
            "agent.call_llm",
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
            "agent.call_llm",
            return_value=self._completion("not JSON"),
        ) as completion:
            intent = agent.route_intent("Can you do something?")

        self.assertEqual(intent, "research")
        self.assertEqual(completion.call_count, 2)

    def test_factual_questions_are_routed_to_web_research(self):
        with patch(
            "agent.call_llm",
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
                "agent.call_llm",
                return_value=self._completion("Hello! What can I help with?"),
            ) as completion,
            patch("agent.search_web") as search,
        ):
            result = agent.run_research("Hello!")

        route.assert_not_called()
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
            "agent.call_llm",
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
            "agent.call_llm",
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
            include_images=True,
        )
        self.assertEqual(result["report"], "No sources found.")

    def test_creative_writing_is_done_directly_without_research(self):
        with (
            patch(
                "agent.call_llm",
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
                "agent.call_llm",
                return_value=self._completion("A cozy watercolor cat..."),
            ) as completion,
            patch(
                "agent._generate_image",
                return_value={
                    "base64": "aW1hZ2U=",
                    "mime_type": "image/png",
                    "description": "Generated image",
                },
            ),
            patch("agent.search_web") as search,
        ):
            result = agent.run_research("Create an image of a cat reading a book.")

        search.assert_not_called()
        system_prompt = completion.call_args.args[0][0]["content"]
        self.assertIn("image-generation prompt", system_prompt)
        self.assertEqual(result["kind"], "chat")
        self.assertEqual(result["report"], "Here's the image you requested.")
        self.assertEqual(completion.call_count, 1)

    def test_configured_image_model_returns_generated_image_data(self):
        response = Mock()
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "data": "aW1hZ2U=",
                                    "mimeType": "image/png",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-test"}),
            patch("agent.requests.post", return_value=response) as post,
            patch(
                "agent.call_llm",
                return_value=self._completion("A watercolor cat reading a book."),
            ),
        ):
            result = agent.run_research("Create an image of a cat reading a book.")

        self.assertEqual(result["images"][0]["base64"], "aW1hZ2U=")
        post.assert_called_once()
        self.assertIn("gemini-2.5-flash-image", post.call_args.args[0])

    def test_image_generation_falls_back_to_pollinations(self):
        response = Mock()
        response.headers = {"content-type": "image/png"}
        response.content = b"pollinations-image"
        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-test"}),
            patch(
                "agent.requests.post",
                side_effect=RequestException("Gemini quota exhausted"),
            ),
            patch("agent.requests.get", return_value=response) as pollinations,
        ):
            image = agent._generate_image("A small blue bird")

        self.assertEqual(image["mime_type"], "image/png")
        self.assertEqual(
            base64.b64decode(image["base64"]),
            b"pollinations-image",
        )
        self.assertIn(
            "A%20small%20blue%20bird",
            pollinations.call_args.args[0],
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

        with patch("agent.call_llm", return_value=response) as completion:
            agent.write_report("I'm worried about this issue", search_results)

        system_prompt = completion.call_args.args[0][0]["content"]
        self.assertIn(agent.BASE_SYSTEM_PROMPT, system_prompt)
        self.assertIn("directly answers the user's specific question", system_prompt)
        self.assertIn("use only the supplied search results", system_prompt)
        self.assertIn("I could not find this in the search results.", system_prompt)
        self.assertIn("source links separately", system_prompt)
        self.assertIn("same language as the user's latest message", system_prompt)

    def test_report_without_sources_returns_required_not_found_message(self):
        with patch("agent.call_llm") as completion:
            report = agent.write_report(
                "What happened?",
                {"sources": [], "text": "", "errors": []},
            )

        self.assertEqual(report, "I could not find this in the search results.")
        completion.assert_not_called()

    def test_call_llm_uses_groq_with_thirty_second_timeout(self):
        create = Mock(return_value="Groq response")
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        messages = [{"role": "user", "content": "test"}]

        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "groq-test"}, clear=True),
            patch("agent.OpenAI", return_value=client) as openai,
        ):
            response = agent.call_llm(messages)

        self.assertEqual(response, "Groq response")
        openai.assert_called_once_with(
            api_key="groq-test",
            base_url=agent.GROQ_BASE_URL,
            timeout=30,
        )
        create.assert_called_once_with(model=agent.MODEL, messages=messages)

    def test_call_llm_falls_back_to_gemini_when_groq_fails(self):
        messages = [{"role": "user", "content": "test"}]
        groq_create = Mock(side_effect=OpenAIError("Groq unavailable"))
        gemini_create = Mock(return_value="Gemini response")
        groq_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=groq_create)
            )
        )
        gemini_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=gemini_create)
            )
        )

        with (
            patch.dict(
                "os.environ",
                {"GROQ_API_KEY": "groq-test", "GEMINI_API_KEY": "gemini-test"},
                clear=True,
            ),
            patch("agent.OpenAI", side_effect=[groq_client, gemini_client]) as openai,
        ):
            response = agent.call_llm(messages)

        self.assertEqual(response, "Gemini response")
        self.assertEqual(
            openai.call_args_list,
            [
                unittest.mock.call(
                    api_key="groq-test",
                    base_url=agent.GROQ_BASE_URL,
                    timeout=30,
                ),
                unittest.mock.call(
                    api_key="gemini-test",
                    base_url=agent.GEMINI_BASE_URL,
                    timeout=30,
                ),
            ],
        )
        gemini_create.assert_called_once_with(
            model=agent.GEMINI_MODEL,
            messages=messages,
        )

    def test_stream_llm_uses_groq_streaming_and_falls_back_to_gemini(self):
        messages = [{"role": "user", "content": "test"}]
        event = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Gemini chunk"))]
        )
        groq_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(side_effect=OpenAIError("Groq unavailable"))
                )
            )
        )
        gemini_create = Mock(return_value=iter([event]))
        gemini_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=gemini_create)
            )
        )

        with (
            patch.dict(
                "os.environ",
                {"GROQ_API_KEY": "groq-test", "GEMINI_API_KEY": "gemini-test"},
                clear=True,
            ),
            patch("agent.OpenAI", side_effect=[groq_client, gemini_client]),
        ):
            chunks = list(agent.stream_llm(messages))

        self.assertEqual(chunks, ["Gemini chunk"])
        self.assertEqual(
            groq_client.chat.completions.create.call_args.kwargs,
            {"model": agent.MODEL, "messages": messages, "stream": True},
        )
        gemini_create.assert_called_once_with(
            model=agent.GEMINI_MODEL,
            messages=messages,
            stream=True,
        )

    def test_run_research_stream_returns_the_existing_result_shape(self):
        results = {
            "sources": [{"title": "Source", "url": "https://example.com"}],
            "images": [],
        }
        prepared = {
            "topic": "A research topic",
            "results": results,
            "history": [],
        }
        result_container = {}
        with (
            patch("agent._prepare_response", return_value=prepared),
            patch("agent.stream_report", return_value=iter(["A streamed ", "report."])),
        ):
            streamed = "".join(
                agent.run_research_stream("A research topic", result_container)
            )

        self.assertEqual(streamed, "A streamed report.")
        self.assertEqual(
            result_container["result"],
            {
                "report": "A streamed report.",
                "sources": results["sources"],
                "images": [],
            },
        )

    def test_call_llm_reports_a_friendly_error_when_both_providers_fail(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(side_effect=OpenAIError("private provider details"))
                )
            )
        )
        with self.assertRaisesRegex(
            RuntimeError,
            agent.LLM_UNAVAILABLE_MESSAGE,
        ) as error:
            with (
                patch.dict(
                    "os.environ",
                    {
                        "GROQ_API_KEY": "groq-test",
                        "GEMINI_API_KEY": "gemini-test",
                    },
                    clear=True,
                ),
                patch("agent.OpenAI", return_value=client),
            ):
                agent.call_llm([{"role": "user", "content": "test"}])

        self.assertNotIn("private provider details", str(error.exception))

    def test_call_llm_ignores_stale_bifrost_settings(self):
        create = Mock(return_value="Groq response")
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        messages = [{"role": "user", "content": "test"}]

        with (
            patch.dict(
                "os.environ",
                {
                    "GROQ_API_KEY": "groq-test",
                    "BIFROST_HOST": "bifrost.example",
                    "BIFROST_API_KEY": "bifrost-test",
                },
                clear=True,
            ),
            patch("agent.OpenAI", return_value=client) as openai,
        ):
            response = agent.call_llm(messages)

        self.assertEqual(response, "Groq response")
        openai.assert_called_once_with(
            api_key="groq-test",
            base_url=agent.GROQ_BASE_URL,
            timeout=30,
        )
        create.assert_called_once_with(
            model=agent.MODEL,
            messages=messages,
        )

    def test_transcribe_audio_uses_groq_and_returns_recognized_text(self):
        create = Mock(return_value=SimpleNamespace(text="Hello there"))
        client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=create),
            )
        )
        audio_file = ("speech.wav", b"wav-audio", "audio/wav")

        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "groq-test"}),
            patch("agent.OpenAI", return_value=client) as openai,
        ):
            transcript = agent.transcribe_audio(audio_file)

        self.assertEqual(transcript, "Hello there")
        openai.assert_called_once_with(
            api_key="groq-test",
            base_url=agent.GROQ_BASE_URL,
            timeout=30,
        )
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
            patch.dict("os.environ", {"GROQ_API_KEY": "groq-test"}),
            patch("agent.OpenAI", return_value=client),
            self.assertRaisesRegex(RuntimeError, "No speech was recognized"),
        ):
            agent.transcribe_audio(("silence.wav", b"wav-audio", "audio/wav"))

    def test_stream_llm_ignores_stale_bifrost_settings(self):
        event = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Groq chunk"))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=iter([event])))
            )
        )
        with (
            patch.dict(
                "os.environ",
                {
                    "GROQ_API_KEY": "groq-test",
                    "BIFROST_HOST": "bifrost.example",
                    "BIFROST_API_KEY": "bifrost-test",
                },
                clear=True,
            ),
            patch("agent.OpenAI", return_value=client) as openai,
        ):
            chunks = list(agent.stream_llm([{"role": "user", "content": "hello"}]))

        self.assertEqual(chunks, ["Groq chunk"])
        openai.assert_called_once_with(
            api_key="groq-test",
            base_url=agent.GROQ_BASE_URL,
            timeout=30,
        )


if __name__ == "__main__":
    unittest.main()
