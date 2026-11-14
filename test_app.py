import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).with_name("app.py"))


class AppTests(unittest.TestCase):
    def test_examples_render_report_sources_and_reset(self):
        result = {
            "report": "# Sample report\nA finding [1].",
            "sources": [{"title": "Example source", "url": "https://example.com"}],
        }
        stages = []

        def fake_research(topic, progress_callback=None):
            for stage in (
                "Planning queries...",
                "Searching web...",
                "Writing report...",
            ):
                progress_callback(stage)
                stages.append(stage)
            return result

        with patch("agent.run_research", side_effect=fake_research) as research:
            app = AppTest.from_file(APP_FILE).run()
            app.button[2].click().run()
            self.assertIn("AI note-taking apps", app.text_input[0].value)
            app.button[0].click().run()

            self.assertFalse(app.exception)
            self.assertTrue(research.called)
            self.assertEqual(len(app.download_button), 1)
            self.assertTrue(
                any("Sample report" in element.value for element in app.markdown)
            )
            self.assertTrue(
                any(
                    element.url == "https://example.com"
                    for element in app.link_button
                )
            )
            self.assertEqual(
                stages,
                ["Planning queries...", "Searching web...", "Writing report..."],
            )

            app.button[1].click().run()
            self.assertEqual(app.text_input[0].value, "")
            self.assertFalse(
                any("Sample report" in element.value for element in app.markdown)
            )

    def test_timeout_shows_friendly_error_without_traceback(self):
        with patch("agent.run_research", side_effect=TimeoutError("simulated timeout")):
            app = AppTest.from_file(APP_FILE).run()
            app.text_input[0].set_value("test topic").run()
            app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any("Research could not be completed" in item.value for item in app.error)
        )

    def test_empty_topic_shows_prompt(self):
        app = AppTest.from_file(APP_FILE).run()
        app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any("Enter a research topic" in item.value for item in app.error)
        )


if __name__ == "__main__":
    unittest.main()
