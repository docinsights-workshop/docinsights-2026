# DocInsights Website Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the DocInsights 2026 Jekyll site into a modern, accessible workshop hub with verified public facts, clear submission paths, new informational pages, and a local preview.

**Architecture:** Keep the static Jekyll site and split reusable content into YAML data and Liquid includes. Pages consume shared data for navigation, dates, news, themes, program preview, submission CTAs, and footer links to prevent content drift.

**Tech Stack:** Jekyll/GitHub Pages, Markdown, Liquid, YAML data files, CSS, Ruby/Jekyll build, Python static audit script.

---

## File Structure

- Create `_data/navigation.yml` for header and footer navigation.
- Create `_data/dates.yml` for direct, ARR, notification, camera-ready, and workshop schedule date entries.
- Create `_data/news.yml` for reverse-chronological site updates.
- Create `_data/themes.yml` for homepage research theme cards.
- Create `_data/program_preview.yml` for provisional program format cards.
- Create `_data/submission_links.yml` for direct submission, ARR commitment, and CFP action links.
- Create `_includes/cta-row.html` to render submission action buttons.
- Create `_includes/date-list.html` to render date cards.
- Create `_includes/news-list.html` to render news items.
- Create `_includes/hero.html` for the semantic homepage hero.
- Create `_includes/theme-grid.html` for homepage research themes.
- Create `_includes/program-preview.html` for provisional program sections.
- Modify `_layouts/default.html` to use data-driven nav/footer, skip link, nav expanded state, and secure external links.
- Modify `index.md`, `call-for-papers.md`, `program.md`, `organizers.md`, and `assets/css/style.css`.
- Create `dates.md`, `shared-task.md`, `speakers.md`, and `faq.md`.
- Create `scripts/audit_site.py` to validate built HTML for key redesign requirements.

## Task 1: Static Audit

**Files:**
- Create: `scripts/audit_site.py`

- [ ] **Step 1: Write the failing audit script**

Create `scripts/audit_site.py`:

```python
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

    nav_text = home.text
    for label in ["Home", "CFP", "Dates", "Program", "Shared Task", "Speakers", "Organizers", "FAQ"]:
        assert_true(label in nav_text, f"navigation missing {label}")

    for label, path in REQUIRED_PAGES.items():
        parser = parse(path)
        for link in parser.links:
            if link.get("target") == "_blank":
                rel = link.get("rel", "")
                assert_true("noopener" in rel, f"{label} has target=_blank without noopener")
        for img in parser.images:
            assert_true(img.get("alt", "") != "", f"{label} has image without alt text")

    cfp = parse(REQUIRED_PAGES["cfp"])
    for text in ["direct archival", "direct non-archival", "ARR Commitment", "at least 3 reviews"]:
        assert_true(text in cfp.text, f"CFP missing required detail: {text}")

    program = parse(REQUIRED_PAGES["program"])
    for text in ["Invited talks", "Poster", "Panel", "Exact schedule pending"]:
        assert_true(text in program.text, f"program missing required detail: {text}")

    shared_task = parse(REQUIRED_PAGES["shared task"])
    for text in ["RUST-BENCH", "structure-aware tabular reasoning", "details will be announced"]:
        assert_true(text in shared_task.text, f"shared task missing required detail: {text}")

    print("site audit passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build and run audit to verify it fails**

Run:

```bash
BUNDLE_GEMFILE=/Users/aamita/Oracle/amitbcp/grail.github.io/Gemfile bundle exec jekyll build --source /Users/aamita/Oracle/amitbcp/docinsights-2026 --destination /private/tmp/docinsights-redesign-audit/docinsights-2026
/Users/aamita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/audit_site.py /private/tmp/docinsights-redesign-audit
```

Expected: audit fails because `dates`, `shared-task`, `speakers`, and `faq` pages do not exist yet.

## Task 2: Data And Includes

**Files:**
- Create: `_data/navigation.yml`
- Create: `_data/dates.yml`
- Create: `_data/news.yml`
- Create: `_data/themes.yml`
- Create: `_data/program_preview.yml`
- Create: `_data/submission_links.yml`
- Create: `_includes/cta-row.html`
- Create: `_includes/date-list.html`
- Create: `_includes/news-list.html`
- Create: `_includes/hero.html`
- Create: `_includes/theme-grid.html`
- Create: `_includes/program-preview.html`

- [ ] **Step 1: Add YAML data**

Add structured data for nav links, verified dates, submission actions, news, themes, and program cards. Use OpenReview date strings for submission windows only as source notes; keep visible public dates as the current site's AoE dates.

- [ ] **Step 2: Add Liquid includes**

Add small includes that render one responsibility each: CTAs, date cards, news list, hero, theme grid, and program preview.

- [ ] **Step 3: Rebuild**

Run:

```bash
BUNDLE_GEMFILE=/Users/aamita/Oracle/amitbcp/grail.github.io/Gemfile bundle exec jekyll build --source /Users/aamita/Oracle/amitbcp/docinsights-2026 --destination /private/tmp/docinsights-redesign-audit/docinsights-2026
```

Expected: build exits 0.

## Task 3: Layout And Navigation

**Files:**
- Modify: `_layouts/default.html`

- [ ] **Step 1: Update layout**

Render primary nav from `_data/navigation.yml`, add a skip link, set `main` id to `main-content`, add `aria-expanded` toggling, render footer groups from data, and add `rel="noopener"` to external links.

- [ ] **Step 2: Rebuild**

Run the Jekyll build command from Task 2.

Expected: build exits 0.

## Task 4: Pages And Content

**Files:**
- Modify: `index.md`
- Modify: `call-for-papers.md`
- Modify: `program.md`
- Modify: `organizers.md`
- Create: `dates.md`
- Create: `shared-task.md`
- Create: `speakers.md`
- Create: `faq.md`

- [ ] **Step 1: Rebuild homepage**

Use the hero, CTA row, theme grid, program preview, dates list, news list, and contact copy. Keep title/date/location in HTML text.

- [ ] **Step 2: Reorganize CFP**

Add submission paths, contribution types, review process, ARR policy, ethics/policy links, and expanded proposal-grounded topics.

- [ ] **Step 3: Add new pages**

Add Dates, Shared Task, Speakers, and FAQ pages with cautious wording for unconfirmed details.

- [ ] **Step 4: Update Program and Organizers**

Replace the empty program message with a provisional full-day structure. Add concise organizer bios from the accepted proposal without adding unverified external profile links.

- [ ] **Step 5: Run audit**

Run:

```bash
BUNDLE_GEMFILE=/Users/aamita/Oracle/amitbcp/grail.github.io/Gemfile bundle exec jekyll build --source /Users/aamita/Oracle/amitbcp/docinsights-2026 --destination /private/tmp/docinsights-redesign-audit/docinsights-2026
/Users/aamita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/audit_site.py /private/tmp/docinsights-redesign-audit
```

Expected: audit passes or reports only CSS/visual issues not covered by text checks.

## Task 5: CSS Redesign

**Files:**
- Modify: `assets/css/style.css`

- [ ] **Step 1: Replace visual system**

Add responsive styles for semantic hero, CTA buttons, fact strip, cards, date cards, program preview, FAQ blocks, footer groups, focus states, and mobile nav.

- [ ] **Step 2: Rebuild and audit**

Run the build and audit commands from Task 4.

Expected: build exits 0 and audit passes.

## Task 6: Local Browser Verification

**Files:**
- No source changes unless verification finds a defect.

- [ ] **Step 1: Start local server**

Run:

```bash
ruby -run -e httpd /private/tmp/docinsights-redesign-audit -p 4012 -b 127.0.0.1
```

Expected: local preview at `http://127.0.0.1:4012/docinsights-2026/`.

- [ ] **Step 2: Browser checks**

Check desktop and mobile viewports for:

- Homepage first viewport readable and action-oriented.
- No horizontal overflow.
- Mobile nav opens and closes.
- New pages load.
- Images load.
- External blank links include `rel="noopener"`.

- [ ] **Step 3: Fix defects and rerun**

If any defect appears, patch the relevant file and rerun build, audit, and browser checks.
