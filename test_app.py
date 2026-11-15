import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).with_name("app.py"))


def _send_message(app: AppTest, message: str) -> None:
    app.chat_input[0].set_value(message).run()


def _stream_response(result):
    def stream(
        topic,
        result_container,
        progress_callback=None,
        conversation_history=None,
    ):
        result_container["result"] = result
        yield result["report"]

    return stream


class AppTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        env_patch = patch.dict(
            os.environ,
            {"CHAT_DB_PATH": str(Path(temporary_directory.name) / "chats.sqlite3")},
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_chat_renders_responses_and_hides_greeting_after_first_message(self):
        result = {
            "report": "# Sample report\nA finding [1].",
            "sources": [{"title": "Example source", "url": "https://example.com"}],
            "images": [
                {
                    "url": "https://example.com/image.jpg",
                    "description": "Example illustration",
                },
                {
                    "url": "https://example.com/image-2.jpg",
                    "description": "Second illustration",
                },
                {
                    "url": "https://example.com/image-3.jpg",
                    "description": "Third illustration",
                },
            ],
        }
        stages = []
        histories = []

        def fake_research(
            topic,
            result_container,
            progress_callback=None,
            conversation_history=None,
        ):
            histories.append(conversation_history)
            for stage in (
                "Understanding your question...",
                "Preparing search queries...",
                "Looking for useful sources...",
                "Putting together your response...",
            ):
                progress_callback(stage)
                stages.append(stage)
            result_container["result"] = result
            yield result["report"]

        with patch("agent.run_research_stream", side_effect=fake_research) as research:
            app = AppTest.from_file(APP_FILE).run()
            greeting = app.session_state["greeting"]
            self.assertTrue(
                any(greeting in element.value for element in app.markdown)
            )
            _send_message(
                app,
                "Assess the current market for AI note-taking apps.",
            )

            self.assertFalse(app.exception)
            self.assertTrue(research.called)
            self.assertEqual(histories[0], [])
            self.assertEqual(
                [message["role"] for message in app.session_state["messages"]],
                ["user", "assistant"],
            )
            self.assertFalse(
                any(greeting in element.value for element in app.markdown)
            )
            self.assertEqual(len(app.download_button), 2)
            self.assertTrue(
                any(expander.label == "Sources" for expander in app.expander)
            )
            self.assertTrue(
                any(expander.label == "Thinking" for expander in app.expander)
            )
            self.assertTrue(
                any("Sample report" in element.value for element in app.markdown)
            )
            self.assertTrue(
                any(
                    element.url == "https://example.com"
                    for element in app.link_button
                )
            )
            self.assertEqual(len(app.image), 3)
            self.assertTrue(
                all("[1]" not in element.value for element in app.markdown)
            )
            self.assertTrue(
                any(
                    button.label == "Download PDF"
                    for button in app.download_button
                )
            )
            self.assertTrue(any(button.label == "Like" for button in app.button))
            self.assertTrue(any(button.label == "Dislike" for button in app.button))
            self.assertTrue(any(button.label == "Share" for button in app.button))
            self.assertEqual(
                stages,
                [
                    "Understanding your question...",
                    "Preparing search queries...",
                    "Looking for useful sources...",
                    "Putting together your response...",
                ],
            )

            _send_message(app, "A second topic")
            self.assertEqual(
                [message["role"] for message in app.session_state["messages"]],
                ["user", "assistant", "user", "assistant"],
            )
            self.assertEqual(research.call_args.args[0], "A second topic")
            self.assertEqual(
                histories[1],
                [
                    {
                        "role": "user",
                        "content": "Assess the current market for AI note-taking apps.",
                    },
                    {"role": "assistant", "content": result["report"]},
                ],
            )

    def test_timeout_shows_friendly_error_without_traceback(self):
        with patch(
            "agent.run_research_stream",
            side_effect=TimeoutError("simulated timeout"),
        ):
            app = AppTest.from_file(APP_FILE).run()
            _send_message(app, "test topic")

        self.assertFalse(app.exception)
        self.assertTrue(
            any("request could not be completed" in item.value for item in app.error)
        )

    def test_blank_message_is_rejected_without_using_demo_request(self):
        with patch("agent.run_research_stream") as research:
            app = AppTest.from_file(APP_FILE).run()
            requests_before = app.session_state["demo_requests_used"]
            app.chat_input[0].set_value("   ").run()

        self.assertFalse(app.exception)
        research.assert_not_called()
        self.assertEqual(app.session_state["demo_requests_used"], requests_before)
        self.assertTrue(
            any("Write a message before sending." in item.value for item in app.warning)
        )

    def test_shared_rate_limit_is_shown_and_does_not_use_demo_request(self):
        from request_limiter import RequestLimitExceeded

        with (
            patch(
                "request_limiter.FairRequestLimiter.request_slot",
                side_effect=RequestLimitExceeded("service is busy"),
            ),
            patch("agent.run_research_stream") as research,
        ):
            app = AppTest.from_file(APP_FILE).run()
            app.chat_input[0].set_value("please respond").run()

        self.assertFalse(app.exception)
        research.assert_not_called()
        self.assertEqual(app.session_state["demo_requests_used"], 0)
        self.assertTrue(
            any("service is busy" in item.value for item in app.error)
        )

    def test_edge_case_prompts_reach_the_backend_without_truncation(self):
        prompts = [
            "x",
            "?? 🤖",
            "Explain this, yaar — Hindi + English please.",
            "Format this code/table/link: `x = 1` | https://example.com",
            "long " * 2500,
        ]
        received = []

        def fake_stream(topic, result_container, **_kwargs):
            received.append(topic)
            result = {
                "kind": "chat",
                "report": "Received",
                "sources": [],
                "images": [],
            }
            result_container["result"] = result
            yield result["report"]

        with patch("agent.run_research_stream", side_effect=fake_stream):
            app = AppTest.from_file(APP_FILE).run()
            for prompt in prompts:
                _send_message(app, prompt)
                self.assertFalse(app.exception)

        self.assertEqual(received, [prompt.strip() for prompt in prompts])

    def test_demo_request_limit_blocks_another_research_call(self):
        with patch("agent.run_research_stream") as research:
            app = AppTest.from_file(APP_FILE).run()
            app.session_state["demo_requests_used"] = 10
            _send_message(app, "one more request")

        self.assertFalse(app.exception)
        research.assert_not_called()
        self.assertTrue(
            any("10-request demo limit" in warning.value for warning in app.warning)
        )

    def test_chat_is_only_available_in_the_current_session(self):
        with patch(
            "agent.run_research_stream",
            side_effect=_stream_response({
                "kind": "chat",
                "report": "Saved response",
                "sources": [],
                "images": [],
            }),
        ):
            first_app = AppTest.from_file(APP_FILE).run()
            _send_message(first_app, "Remember this conversation")
            self.assertFalse(first_app.exception)
            self.assertEqual(
                [message["role"] for message in first_app.session_state["messages"]],
                ["user", "assistant"],
            )
            like_button = next(
                button for button in first_app.button if button.label == "Like"
            )
            like_button.click().run()
            dislike_button = next(
                button for button in first_app.button if button.label == "Dislike"
            )
            dislike_button.click().run()
            self.assertEqual(
                first_app.session_state["messages"][1]["feedback"],
                "dislike",
            )

            reopened_app = AppTest.from_file(APP_FILE).run()

        self.assertFalse(reopened_app.exception)
        self.assertEqual(reopened_app.session_state["messages"], [])
        self.assertFalse(
            any(
                button.label == "Remember this conversation"
                for button in reopened_app.button
            )
        )
        self.assertFalse(Path(os.environ["CHAT_DB_PATH"]).exists())

    def test_initial_greeting_is_shown_before_first_message(self):
        app = AppTest.from_file(APP_FILE).run()

        self.assertFalse(app.exception)
        self.assertFalse(
            any(button.label == "Market research" for button in app.button)
        )
        self.assertEqual(app.session_state["messages"], [])
        self.assertTrue(
            any(
                app.session_state["greeting"] in element.value
                for element in app.markdown
            )
        )
        self.assertTrue(
            any("White jasmine flower" in element.value for element in app.markdown)
        )
        self.assertFalse(
            any(
                button.label in {"Like", "Dislike", "Share"}
                for button in app.button
            )
        )
        self.assertTrue(
            app.session_state["greeting"].startswith(("Good ", "It's getting late!"))
        )
        self.assertFalse(
            any(
                phrase in element.value
                for element in app.markdown
                for phrase in (
                    "Saved on this device",
                    "thoughtful companion",
                    "Tap the mic",
                    "recordings are sent",
                )
            )
        )

    def test_mobile_layout_has_a_small_screen_breakpoint(self):
        app = AppTest.from_file(APP_FILE).run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any("@media (max-width: 640px)" in element.value for element in app.markdown)
        )

    def test_stop_button_marks_an_in_progress_response_as_stopped(self):
        app = AppTest.from_file(APP_FILE).run()
        app.session_state["messages"] = [
            {"role": "user", "content": "A research question"},
            {"role": "assistant", "partial_report": "Partial answer"},
        ]
        app.session_state["generation_active"] = True
        app.run()

        self.assertFalse(app.exception)
        stop_button = next(button for button in app.button if button.label == "⏸")
        stop_button.click().run()

        self.assertFalse(app.exception)
        self.assertFalse(app.session_state["generation_active"])
        self.assertTrue(app.session_state["messages"][-1]["generation_stopped"])

    def test_pause_control_matches_composer_button_size(self):
        app = AppTest.from_file(APP_FILE).run()

        styles = "\n".join(element.value for element in app.markdown)
        self.assertIn(".st-key-stop_generation button", styles)
        self.assertIn(
            "right: max(calc((100vw - 520px) / 2 + 80px), 5.5rem);",
            styles,
        )
        self.assertIn("width: 32px;", styles)
        self.assertIn("height: 32px;", styles)


if __name__ == "__main__":
    unittest.main()
