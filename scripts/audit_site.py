#!/usr/bin/env python3
from html.parser import HTMLParser
from pathlib import Path
import sys


ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_site")
BASE = ROOT / "docinsights-2026"
REQUIRED_PAGES = {
    "home": BASE / "index.html",
    "cfp": BASE / "call-for-papers" / "index.html",
    "dates": BASE / "dates" / "index.html",
    "program": BASE / "program" / "index.html",
    "shared task": BASE / "shared-task" / "index.html",
    "speakers": BASE / "speakers" / "index.html",
    "organizers": BASE / "organizers" / "index.html",
    "faq": BASE / "faq" / "index.html",
}


class SiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.h1_count = 0
        self.links = []
        self.images = []
        self.ids = set()
        self.classes = set()
        self.text_parts = []
        self._capture_text = True

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h1":
            self.h1_count += 1
        if tag == "a":
            self.links.append(attrs)
        if tag == "img":
            self.images.append(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if "class" in attrs:
            self.classes.update(attrs["class"].split())
        if tag in {"script", "style"}:
            self._capture_text = False

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self._capture_text = True

    def handle_data(self, data):
        if self._capture_text:
            self.text_parts.append(data)

    @property
    def text(self):
        return " ".join(part.strip() for part in self.text_parts if part.strip())


def parse(path):
    parser = SiteParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    for label, path in REQUIRED_PAGES.items():
        assert_true(path.exists(), f"missing {label} page: {path}")

    home = parse(REQUIRED_PAGES["home"])
    assert_true(home.h1_count == 1, "home page must have exactly one h1")
    for text in [
        "Beyond Plain Text",
        "Submit Direct Paper",
        "Commit ARR Paper",
        "Research Themes",
        "Program Preview",
        "Shared Task",
    ]:
        assert_true(text in home.text, f"home page missing text: {text}")
    assert_true("skip-link" in home.classes, "layout must include skip-link class")
    assert_true("main-content" in home.ids, "main content must have id='main-content'")

    for label in [
        "Home",
        "CFP",
        "Dates",
        "Program",
        "Shared Task",
        "Speakers",
        "Organizers",
        "FAQ",
    ]:
        assert_true(label in home.text, f"navigation missing {label}")

    for label, path in REQUIRED_PAGES.items():
        parser = parse(path)
        for link in parser.links:
            if link.get("target") == "_blank":
                rel = link.get("rel", "")
                assert_true("noopener" in rel, f"{label} has target=_blank without noopener")
        for img in parser.images:
            assert_true(img.get("alt", "") != "", f"{label} has image without alt text")

    cfp = parse(REQUIRED_PAGES["cfp"])
    for text in [
        "direct archival",
        "direct non-archival",
        "ARR Commitment",
        "at least 3 reviews",
    ]:
        assert_true(text in cfp.text, f"CFP missing required detail: {text}")

    program = parse(REQUIRED_PAGES["program"])
    for text in ["Invited talks", "Poster", "Panel", "Exact schedule pending"]:
        assert_true(text in program.text, f"program missing required detail: {text}")

    shared_task = parse(REQUIRED_PAGES["shared task"])
    for text in [
        "RUST-BENCH",
        "structure-aware tabular reasoning",
        "details will be announced",
    ]:
        assert_true(text in shared_task.text, f"shared task missing required detail: {text}")

    print("site audit passed")


if __name__ == "__main__":
    main()
