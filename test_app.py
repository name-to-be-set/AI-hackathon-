import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).with_name("app.py"))


def _send_message(app: AppTest, message: str) -> None:
    app.chat_input[0].set_value(message).run()


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
            return result

        with patch("agent.run_research", side_effect=fake_research) as research:
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
            self.assertEqual(len(app.image), 1)
            self.assertTrue(
                all("[1]" not in element.value for element in app.markdown)
            )
            self.assertTrue(
                any(
                    button.label == "Download PDF"
                    for button in app.download_button
                )
            )
            self.assertTrue(any(button.label == "👍" for button in app.button))
            self.assertTrue(any(button.label == "👎" for button in app.button))
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
        with patch("agent.run_research", side_effect=TimeoutError("simulated timeout")):
            app = AppTest.from_file(APP_FILE).run()
            _send_message(app, "test topic")

        self.assertFalse(app.exception)
        self.assertTrue(
            any("request could not be completed" in item.value for item in app.error)
        )

    def test_demo_request_limit_blocks_another_research_call(self):
        with patch("agent.run_research") as research:
            app = AppTest.from_file(APP_FILE).run()
            app.session_state["demo_requests_used"] = 10
            _send_message(app, "one more request")

        self.assertFalse(app.exception)
        research.assert_not_called()
        self.assertTrue(
            any("10-request demo limit" in warning.value for warning in app.warning)
        )

    def test_chat_history_is_saved_and_can_be_reopened(self):
        with patch(
            "agent.run_research",
            return_value={
                "kind": "chat",
                "report": "Saved response",
                "sources": [],
                "images": [],
            },
        ):
            first_app = AppTest.from_file(APP_FILE).run()
            _send_message(first_app, "Remember this conversation")
            self.assertFalse(first_app.exception)
            like_button = next(
                button for button in first_app.button if button.label == "👍"
            )
            like_button.click().run()
            dislike_button = next(
                button for button in first_app.button if button.label == "👎"
            )
            dislike_button.click().run()

            reopened_app = AppTest.from_file(APP_FILE).run()
            history_button = next(
                button
                for button in reopened_app.button
                if button.label == "Remember this conversation"
            )
            history_button.click().run()

        self.assertFalse(reopened_app.exception)
        self.assertEqual(
            [message["role"] for message in reopened_app.session_state["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(
            reopened_app.session_state["messages"][1]["feedback"],
            "dislike",
        )
        self.assertTrue(
            any("Saved response" in element.value for element in reopened_app.markdown)
        )

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
        self.assertFalse(
            any(
                button.label in {"👍", "👎", "Share"}
                for button in app.button
            )
        )
        self.assertTrue(
            any(
                phrase in app.session_state["greeting"]
                for phrase in ("breakfast", "lunch", "day", "late")
            )
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


if __name__ == "__main__":
    unittest.main()
