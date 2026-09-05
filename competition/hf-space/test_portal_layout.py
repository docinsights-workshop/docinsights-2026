import json
from pathlib import Path
import re
import unittest

from fastapi.testclient import TestClient

from app import PORTAL_CSS, demo


class PortalLayoutTests(unittest.TestCase):
    def test_space_metadata_pins_the_proven_gradio_sdk(self):
        readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

        self.assertIn("sdk: gradio\n", readme)
        self.assertIn("sdk_version: 4.42.0\n", readme)

    def test_root_config_and_api_info_routes_generate_without_schema_errors(self):
        client = TestClient(demo.app, raise_server_exceptions=False)

        root = client.get("/")
        config = client.get("/config")
        demo.app.api_info = None
        api_info = client.get("/info")

        self.assertEqual(
            (root.status_code, config.status_code, api_info.status_code),
            (200, 200, 200),
            {
                "root": root.text[:200],
                "config": config.text[:200],
                "api_info": api_info.text[:200],
            },
        )
        self.assertIn("text/html", root.headers["content-type"])
        self.assertEqual(config.json()["version"], "4.42.0")
        self.assertIn("/submit_predictions", api_info.json()["named_endpoints"])
        self.assertIn("/my_test_submissions", api_info.json()["named_endpoints"])

    def test_portal_exposes_validation_default_and_optional_test_workflow(self):
        config = demo.get_config_file()
        serialized = json.dumps(config)

        self.assertIn("Validation (development)", serialized)
        self.assertIn("Test (final)", serialized)
        self.assertIn("Sign in with Hugging Face", serialized)
        self.assertIn("My test submissions", serialized)

        split_selectors = [
            component
            for component in config["components"]
            if component["type"] == "dropdown"
            and component["props"].get("label") == "Evaluation split"
        ]
        self.assertEqual(len(split_selectors), 1)
        self.assertEqual(
            split_selectors[0]["props"]["value"], "Validation (development)"
        )

    def test_initial_public_config_does_not_serialize_private_test_state(self):
        serialized = json.dumps(demo.get_config_file()).casefold()

        for private_value in (
            "server-oauth-subject",
            "server@example.org",
            "sealed/gold.jsonl",
            "projections/test/organizer_leaderboard.json",
            "test rank",
            "test score",
        ):
            with self.subTest(private_value=private_value):
                self.assertNotIn(private_value, serialized)

    def test_gradio_container_owns_vertical_scroll_in_embedded_space(self):
        document_rule = re.search(r"html,\s*body\s*\{(?P<body>.*?)\}", PORTAL_CSS, re.S)
        container_rule = re.search(r"\.gradio-container\s*\{(?P<body>.*?)\}", PORTAL_CSS, re.S)

        self.assertIsNotNone(document_rule)
        self.assertIsNotNone(container_rule)
        document_declarations = document_rule.group("body")
        declarations = container_rule.group("body")
        self.assertIn("height: 100%", document_declarations)
        self.assertIn("overflow: hidden !important", document_declarations)
        self.assertIn("height: 100vh !important", declarations)
        self.assertIn("height: 100dvh !important", declarations)
        self.assertIn("max-height: 100% !important", declarations)
        self.assertIn("overflow-y: auto !important", declarations)
        self.assertIn("-webkit-overflow-scrolling: touch", declarations)

    def test_portal_reports_the_latest_private_rescore_without_leaking_details(self):
        config = json.dumps(demo.get_config_file())

        self.assertIn("Three organizer-only validation labels", config)
        self.assertIn("most recently on September 3, 2026", config)
        self.assertIn("All existing submissions were rescored", config)
        self.assertIn("Leaderboard refreshed September 3, 2026", config)
        self.assertNotIn("Two organizer-only validation labels", config)
        self.assertNotRegex(config, r"task_\d{6}")


if __name__ == "__main__":
    unittest.main()
