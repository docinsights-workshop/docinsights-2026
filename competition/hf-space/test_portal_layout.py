import json
from pathlib import Path
import re
import unittest
import datetime as dt

import gradio as gr
from fastapi.testclient import TestClient

import app
from app import DATASET_BIBTEX, PORTAL_CSS, PORTAL_HEAD, demo
from submission_service import TrustedTestConfig


def trusted_layout_config(policy):
    return TrustedTestConfig(
        policy=policy,
        labels=[],
        scoring_gold_sha256=policy.gold_sha256,
        private_revision="d" * 40,
        public_revision="e" * 40,
        public_repo_id="public/docsem",
        task_manifest_path="test/tasks.jsonl",
    )


class PortalLayoutTests(unittest.TestCase):
    def test_space_metadata_pins_the_proven_gradio_sdk(self):
        readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

        self.assertIn("sdk: gradio\n", readme)
        self.assertIn("sdk_version: 4.42.0\n", readme)
        self.assertIn('python_version: "3.12"\n', readme)

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

    def test_full_width_gradio_container_owns_edge_scrollbar_while_main_is_centered(
        self,
    ):
        document_rule = re.search(r"html,\s*body\s*\{(?P<body>.*?)\}", PORTAL_CSS, re.S)
        container_rule = re.search(
            r"\.gradio-container\s*\{(?P<body>.*?)\}", PORTAL_CSS, re.S
        )
        main_rule = re.search(
            r"\.gradio-container\s*>\s*\.main\s*\{(?P<body>.*?)\}",
            PORTAL_CSS,
            re.S,
        )

        self.assertIsNotNone(document_rule)
        self.assertIsNotNone(container_rule)
        self.assertIsNotNone(main_rule)
        document_declarations = document_rule.group("body")
        container_declarations = container_rule.group("body")
        main_declarations = main_rule.group("body")
        self.assertIn("height: 100%", document_declarations)
        self.assertIn("overflow: hidden !important", document_declarations)
        self.assertIn("width: 100% !important", container_declarations)
        self.assertIn("max-width: none !important", container_declarations)
        self.assertIn("height: 100vh !important", container_declarations)
        self.assertIn("height: 100dvh !important", container_declarations)
        self.assertIn("overflow-y: auto !important", container_declarations)
        self.assertIn("overflow-x: hidden !important", container_declarations)
        self.assertIn("-webkit-overflow-scrolling: touch", container_declarations)
        self.assertIn("width: 100%", main_declarations)
        self.assertIn("max-width: 1540px", main_declarations)
        self.assertIn("margin: 0 auto", main_declarations)

    def test_uploader_and_mobile_overflow_are_scoped_to_the_portal(self):
        uploader_rule = re.search(
            r"#submission-file\s+\.file-preview-holder,\s*"
            r"#submission-file\s+\.file-preview\s*\{(?P<body>.*?)\}",
            PORTAL_CSS,
            re.S,
        )
        mobile_rule = re.search(
            r"@media\s*\(max-width:\s*760px\)\s*\{(?P<body>.*)\}\s*\Z",
            PORTAL_CSS,
            re.S,
        )

        self.assertIsNotNone(uploader_rule)
        self.assertIn("max-height: none !important", uploader_rule.group("body"))
        self.assertIn("overflow: visible !important", uploader_rule.group("body"))
        self.assertIsNotNone(mobile_rule)
        self.assertRegex(
            mobile_rule.group("body"),
            r"\.gradio-container\s*>\s*\.main\s*\{[^}]*max-width:\s*100%",
        )
        self.assertRegex(
            mobile_rule.group("body"),
            r"#split-controls,\s*#submission-fields,\s*#submission-actions,\s*"
            r"#leaderboard-heading\s*\{[^}]*min-width:\s*0",
        )
        self.assertRegex(
            mobile_rule.group("body"),
            r"#portal-header\s+\.portal-links\s*\{[^}]*flex-wrap:\s*wrap",
        )

    def test_css_targets_the_pinned_gradio_dom_and_uploader_has_no_fixed_height(self):
        frontend = Path(gr.__file__).parent / "_frontend_code"
        embed = (frontend / "core" / "src" / "Embed.svelte").read_text(encoding="utf-8")
        file_preview = (frontend / "file" / "shared" / "FilePreview.svelte").read_text(
            encoding="utf-8"
        )
        uploader = next(
            component
            for component in demo.get_config_file()["components"]
            if component["props"].get("elem_id") == "submission-file"
        )

        self.assertRegex(
            embed,
            r'class="gradio-container gradio-container-\{version\}"[\s\S]*?'
            r'<div class="main">\s*<slot',
        )
        self.assertIn('class="file-preview-holder"', file_preview)
        self.assertIn('class="file-preview"', file_preview)
        self.assertNotIn("height", uploader["props"])

    def test_space_card_records_the_required_post_deploy_safari_smoke(self):
        readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

        self.assertIn("## Safari post-deployment smoke check", readme)
        self.assertIn("scrollbar stays at the browser edge", readme)
        self.assertIn("uploader has no nested vertical scrollbar", readme)
        self.assertIn("no page-level horizontal scrollbar", readme)

    def test_citation_accordion_is_collapsed_labeled_and_exact(self):
        expected = """@article{singh2026gsmsem,
  title={GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations},
  author={Jyotika Singh and Fang Tu and Aziza Mirsaidova and Amit Agarwal and Hitesh Laxmichand Patel and Sandip Ghoshal and Miguel Ballesteros and Karan Dua and Yassine Benajiba and Weiyi Sun and Tao Sheng and Graham Horwood and Sujith Ravi and Dan Roth},
  year={2026},
  eprint={2605.07053},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.07053}
}"""
        config = demo.get_config_file()
        accordions = [
            component
            for component in config["components"]
            if component["type"] == "accordion"
            and component["props"].get("label") == "Cite this dataset"
        ]
        citation_codes = [
            component
            for component in config["components"]
            if component["type"] == "code"
            and component["props"].get("label") == "BibTeX citation"
        ]

        self.assertEqual(DATASET_BIBTEX, expected)
        self.assertEqual(len(accordions), 1)
        self.assertFalse(accordions[0]["props"]["open"])
        self.assertEqual(len(citation_codes), 1)
        self.assertEqual(citation_codes[0]["props"]["value"], expected)
        self.assertFalse(citation_codes[0]["props"]["interactive"])
        serialized = json.dumps(config)
        self.assertIn("https://arxiv.org/abs/2605.07053", serialized)

    def test_space_card_describes_released_inputs_and_attempt_feedback_policy(self):
        readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

        self.assertIn("The held-out test inputs have been released", readme)
        self.assertIn("3 accepted test submissions per Hugging Face account", readme)
        self.assertIn("Joint Accuracy, Answer Accuracy, and Evidence F1", readme)
        self.assertIn("private to that signed-in account", readme)
        self.assertIn("provisional public ranks use only attempt 1", readme)
        self.assertIn("display no metrics", readme)
        self.assertIn("best of all 3 eligible attempts", readme)
        self.assertIn("TEST_CLOSE_AT=2026-09-11T12:00:00Z", readme)
        self.assertNotIn("No public test scores or ranks are displayed", readme)

    def test_countdown_markup_is_accessible_and_client_script_uses_text_only(self):
        configured_notice = app._test_release_notice_html(
            dt.datetime(2026, 9, 6, tzinfo=dt.timezone.utc),
            deployment=app.TestDeploymentConfig(
                submissions_enabled=True,
                public_leaderboard_enabled=False,
                release_id="docsem-test-2026",
                task_manifest_sha256="a" * 64,
                gold_sha256="b" * 64,
                open_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
                close_at=dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc),
                release_config_path="private/test_release.json",
                gold_config_path="private/test_labels.jsonl",
            ),
            submissions_enabled=True,
            write_token="server-token",
            authoritative_loader=lambda now: trusted_layout_config(
                app.TestReleasePolicy(
                    release_id="docsem-test-2026",
                    task_manifest_sha256="a" * 64,
                    gold_sha256="b" * 64,
                    open_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc),
                    close_at=dt.datetime(2026, 9, 11, 12, tzinfo=dt.timezone.utc),
                    enabled=True,
                )
            ),
        )

        self.assertIn('role="timer"', configured_notice)
        self.assertIn(
            'data-docsem-submission-status role="status" aria-live="polite" '
            'aria-atomic="true"',
            configured_notice,
        )
        self.assertIn(
            'role="timer" aria-live="off" '
            'aria-label="Time remaining until test submissions close: 5 days, 12 hours, '
            '0 minutes, 0 seconds remaining"',
            configured_notice,
        )
        self.assertNotRegex(
            configured_notice,
            r'role="timer"[^>]*aria-live="polite"',
        )
        self.assertIn(
            '<time datetime="2026-09-10T23:59:59-12:00">'
            "September 10, 2026 at 11:59:59 PM Anywhere on Earth</time>",
            configured_notice,
        )
        self.assertIn(
            '<time datetime="2026-09-11T12:00:00Z">'
            "September 11, 2026 at 12:00:00 UTC</time>",
            configured_notice,
        )
        self.assertIn("data-docsem-close-at", PORTAL_HEAD)
        self.assertIn("[data-docsem-submission-status]", PORTAL_HEAD)
        self.assertIn("textContent", PORTAL_HEAD)
        self.assertIn('setAttribute("aria-label"', PORTAL_HEAD)
        self.assertNotIn("innerHTML", PORTAL_HEAD)

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
