import re
import unittest

from app import PORTAL_CSS


class PortalLayoutTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
