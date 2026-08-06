#!/usr/bin/env python3
from html.parser import HTMLParser
from pathlib import Path
import re
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

    home_html = REQUIRED_PAGES["home"].read_text(encoding="utf-8")
    home = parse(REQUIRED_PAGES["home"])
    assert_true(home.h1_count == 1, "home page must have exactly one h1")
    for text in [
        "Beyond Plain Text",
        "Submit Direct Paper",
        "Commit ARR Paper",
        "August 2, 2026",
        "August 10, 2026 at 12:59 PM UTC",
        "Direct submission deadline extended to August 10",
        "The previous August 2 deadline has been superseded",
        "August 30, 2026",
        "Research Themes",
        "Program Preview",
        "Challenges",
        "DocSem",
        "Dr.DocBench",
        "August 3 through September 10, 2026",
        "USD 5,000+",
    ]:
        assert_true(text in home.text, f"home page missing text: {text}")
    assert_true("skip-link" in home.classes, "layout must include skip-link class")
    assert_true("main-content" in home.ids, "main content must have id='main-content'")
    assert_true('rel="icon"' in home_html and "circular_logo.png" in home_html, "layout must define a favicon")
    assert_true("hero-facts" not in home.classes, "home hero must not render a right-side facts rail")
    assert_true("sponsor-lockup--hero" in home.classes, "home hero must expose the workshop sponsors")
    assert_true("budapest_hero_clean.jpg" in home_html, "home hero must use the cleaned Budapest background")
    assert_true("banner_cover_photo.png" not in home_html, "home hero must not use text-bearing banner artwork")
    assert_true(home_html.count('class="btn-date ') >= 2, "home hero CTA buttons must show submission deadlines")
    assert_true(
        "<del>August 2, 2026</del>" in home_html,
        "home hero must show the superseded direct-submission deadline",
    )
    assert_true("Extended:" in home.text, "home hero must label the extended direct-submission deadline")

    for label in [
        "Home",
        "CFP",
        "Dates",
        "Program",
        "Challenges",
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
    assert_true("Tentative Program" in program.text, "program must use Tentative Program wording")
    assert_true("Provisional Flow" not in program.text, "program must not use Provisional Flow wording")

    shared_task_html = REQUIRED_PAGES["shared task"].read_text(encoding="utf-8")
    shared_task = parse(REQUIRED_PAGES["shared task"])
    shared_task_header = shared_task_html.split('<header class="page-header">', 1)[-1].split("</header>", 1)[0]
    assert_true(
        'class="sponsor-lockup sponsor-lockup--page"' in shared_task_header,
        "challenges page title band must expose sponsors",
    )
    assert_true("Sponsors" in shared_task_header, "challenges page title band must label the sponsors")
    assert_true(
        "DocInsights 2026" not in shared_task_header,
        "challenges page title band must not repeat the site name",
    )
    for text in [
        "Two challenges advancing document intelligence beyond plain text",
        "August 3–September 10, 2026",
        "USD 5,000+",
        "DocSem",
        "Dr.DocBench",
        "Resources live",
        "Participant release in preparation",
        "System papers and presentations",
        "Selected contributions",
    ]:
        assert_true(text in shared_task.text, f"shared task missing required detail: {text}")
    shared_task_links = [link.get("href", "") for link in shared_task.links]
    for href in [
        "https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data",
        "https://amitbcp-docsem-docinsights.hf.space/",
        "https://github.com/oracle-samples/gsm-sem/tree/main/docsem",
        "https://drdocbench-challenge.abaka-pages.com/",
    ]:
        assert_true(href in shared_task_links, f"challenges page missing public resource: {href}")
    assert_true(
        not any("eval.ai" in href for href in shared_task_links),
        "challenges page must not publish private Dr.DocBench EvalAI access",
    )
    for stale_domain in [
        "aaronluo00.github.io/drdocbench-challenge",
        "abaka-skills.github.io/drdocbench-challenge",
    ]:
        assert_true(
            not any(stale_domain in href for href in shared_task_links),
            f"challenges page must not publish non-canonical Dr.DocBench URL: {stale_domain}",
        )
    assert_true(
        "Shared task details will be announced soon" not in shared_task.text,
        "challenges page must not retain the stale shared-task placeholder",
    )
    for label, parser in [
        ("home", home),
        ("program", program),
        ("shared task", shared_task),
        ("faq", parse(REQUIRED_PAGES["faq"])),
    ]:
        assert_true("RUST-BENCH" not in parser.text, f"{label} must not visibly expose the hidden shared task name")
        assert_true(
            "structure-aware tabular reasoning" not in parser.text,
            f"{label} must not visibly expose hidden shared task details",
        )

    dates_html = REQUIRED_PAGES["dates"].read_text(encoding="utf-8")
    dates = parse(REQUIRED_PAGES["dates"])
    for text in [
        "August 2, 2026",
        "August 10, 2026",
        "12:59 PM UTC (UTC+00:00)",
        "Challenge Season",
        "August 3, 2026",
        "September 10, 2026",
    ]:
        assert_true(text in dates.text, f"dates page missing challenge milestone: {text}")
    assert_true(
        "<del>August 2, 2026</del>" in dates_html,
        "dates page must strike through the superseded direct-submission deadline",
    )

    faq = parse(REQUIRED_PAGES["faq"])
    for text in [
        "What challenges are running?",
        "When does the competition run?",
        "combined prize pool will exceed USD 5,000",
        "Is Dr.DocBench open for submissions?",
    ]:
        assert_true(text in faq.text, f"FAQ missing challenge detail: {text}")

    speakers = parse(REQUIRED_PAGES["speakers"])
    speakers_html = REQUIRED_PAGES["speakers"].read_text(encoding="utf-8")
    for text in [
        "Yunyao Li",
        "Adobe",
        "Shafiq R. Joty",
        "Salesforce Research",
        "NTU",
        "Julian Eisenschlos",
        "Google DeepMind",
        "Wenhu Chen",
        "University of Waterloo",
        "Talk titles and abstracts",
    ]:
        assert_true(text in speakers.text, f"speakers page missing required detail: {text}")
    assert_true("speaker-icon" in speakers_html, "speaker social buttons must use icon markup")
    assert_true(">WEB<" not in speakers_html and ">GS<" not in speakers_html, "speaker social buttons must not use text badges")
    speaker_links = [link.get("href", "") for link in speakers.links]
    for href in [
        "https://yunyaoli.github.io/",
        "https://www.linkedin.com/in/yunyao-li",
        "https://scholar.google.com/citations?hl=en&user=u5LmeasAAAAJ",
        "https://raihanjoty.github.io/",
        "https://www.linkedin.com/in/shafiq-joty-b1a80a122",
        "https://scholar.google.com/citations?hl=en&user=hR249csAAAAJ",
        "https://research.google/people/106772/",
        "https://ch.linkedin.com/in/eisenjulian",
        "https://scholar.google.com/citations?hl=en&user=2uAC2NQAAAAJ",
        "https://cs.uwaterloo.ca/~wenhuche/",
        "https://www.linkedin.com/in/wenhu-chen-ab59317b",
        "https://scholar.google.com/citations?hl=en&user=U8ShbhUAAAAJ",
    ]:
        assert_true(href in speaker_links, f"speakers page missing profile link: {href}")
    speaker_images = {img.get("alt", ""): img.get("src", "") for img in speakers.images}
    for name, filename in [
        ("Yunyao Li", "yunyao_li.jpg"),
        ("Shafiq R. Joty", "shafiq_joty.jpg"),
        ("Julian Eisenschlos", "julian_eisenschlos.jpg"),
        ("Wenhu Chen", "wenhu_chen.jpg"),
    ]:
        expected_src = f"/assets/images/speakers/{filename}"
        assert_true(name in speaker_images, f"speakers page missing photo alt text for {name}")
        assert_true(expected_src in speaker_images[name], f"speakers page missing local photo for {name}")
        assert_true((BASE / "assets" / "images" / "speakers" / filename).exists(), f"missing built speaker image file: {filename}")
    assert_true("speaker-card-header" in speakers.classes, "speaker cards must use a compact header wrapper")
    assert_true("speaker-meta-line" in speakers.classes, "speaker cards must group affiliation and profile links")

    css = (BASE / "assets" / "css" / "style.css").read_text(encoding="utf-8")
    max_width_match = re.search(r"--max-width:\s*(\d+)px", css)
    assert_true(max_width_match is not None, "CSS must define --max-width in px")
    assert_true(int(max_width_match.group(1)) >= 2040, "desktop max width must use fuller screens")
    assert_true("--content-gutter" in css, "CSS must define responsive content gutters")
    assert_true(
        "calc(100% - (var(--content-gutter) * 2))" in css,
        "wide containers must use viewport-relative gutters",
    )
    assert_true("white-space: nowrap" in css, "CSS must prevent compact action links from wrapping")
    assert_true(
        "grid-template-columns: repeat(auto-fit, minmax(520px, 1fr))" in css,
        "organizer grid must auto-fit across wide and tablet screens",
    )
    assert_true(
        "grid-template-columns: minmax(0, 1fr)" in css,
        "home hero must use a single full-width text column",
    )
    assert_true(
        ".btn-primary *::selection" in css and ".btn-secondary *::selection" in css,
        "CTA buttons must define readable selected-text colors",
    )
    assert_true(
        ".btn-primary:hover" in css and ".btn-primary:active" in css,
        "primary CTA must keep readable text on hover and active states",
    )
    assert_true(".btn:focus-visible" in css, "CTA buttons must define an explicit focus state")
    assert_true(
        "grid-template-columns: repeat(auto-fit, minmax(420px, 1fr))" in css,
        "FAQ list must use compact responsive columns instead of oversized full-width cards",
    )
    assert_true(
        re.search(r"\.faq-list h2\s*{[^}]*font-family:\s*var\(--font-sans\)[^}]*font-size:\s*1rem", css, re.S) is not None,
        "FAQ question headings must use compact utility-card typography",
    )
    assert_true(
        re.search(r"\.faq-list article\s*{[^}]*padding:\s*0\.95rem 1rem", css, re.S) is not None,
        "FAQ cards must use tighter card padding",
    )
    assert_true(
        re.search(r"@media \(max-width: 620px\).*?\.faq-list,\s*\.footer-grid\s*{[^}]*grid-template-columns:\s*1fr", css, re.S) is not None,
        "FAQ list must collapse to one fluid column on mobile",
    )
    for selector in [
        ".challenge-season",
        ".challenge-feature",
        ".challenge-status",
        ".challenge-actions",
        ".challenge-timeline",
    ]:
        assert_true(selector in css, f"CSS missing challenge layout selector: {selector}")
    assert_true(
        re.search(r"@media \(max-width: 620px\).*?\.challenge-timeline\s*{[^}]*grid-template-columns:\s*1fr", css, re.S) is not None,
        "challenge timeline must collapse to one column on mobile",
    )
    assert_true(
        ".speaker-card-header" in css and "grid-template-columns: minmax(0, 1fr) auto" in css,
        "speaker headers must keep identity and social links in one compact row on wide cards",
    )
    assert_true(
        ".speaker-meta-line" in css and "display: flex" in css,
        "speaker affiliation and social links must be compactly grouped",
    )
    assert_true(
        re.search(r"\.speaker-card\s*{[^}]*display:\s*flow-root", css, re.S) is not None,
        "speaker cards must let bio text flow around the photo",
    )
    assert_true(
        re.search(r"\.speaker-photo\s*{[^}]*float:\s*left", css, re.S) is not None,
        "speaker photos must be floated so text can use space below them",
    )
    section_intro_match = re.search(r"\.section-intro\s*{[^}]*max-width:\s*(\d+)px", css, re.S)
    assert_true(section_intro_match is not None, "CSS must define section intro max width")
    assert_true(int(section_intro_match.group(1)) >= 1500, "section intro copy must use fuller desktop width")
    hero_copy_match = re.search(r"\.hero-copy\s*{[^}]*max-width:\s*(\d+)px", css, re.S)
    assert_true(hero_copy_match is not None, "CSS must define hero copy max width")
    assert_true(int(hero_copy_match.group(1)) >= 1500, "home hero copy must use fuller desktop width")
    cta_row_match = re.search(r"\.cta-row\s*{[^}]*max-width:\s*(\d+)px", css, re.S)
    assert_true(cta_row_match is not None, "CSS must define CTA row max width")
    assert_true(int(cta_row_match.group(1)) >= 1450, "home CTA row must use fuller desktop width")
    organizers = parse(REQUIRED_PAGES["organizers"])
    organizers_html = REQUIRED_PAGES["organizers"].read_text(encoding="utf-8")
    assert_true("person-card-header" in organizers.classes, "organizer cards must use a compact header wrapper")
    assert_true("person-meta-line" in organizers.classes, "organizer name and affiliation must share a compact line")
    assert_true("person-icon" in organizers_html, "organizer social buttons must use icon markup")
    assert_true("person-link-mark" not in organizers_html, "organizer social buttons must not use text badge markup")
    assert_true("Amazon, Seattle, USA" not in organizers.text, "Santosh affiliation must not include Seattle")
    assert_true("Amazon, USA" in organizers.text, "Santosh affiliation must stay visible without Seattle")
    assert_true(
        "Workshop Challenge Organizers" in organizers.text,
        "organizers page must include the Workshop Challenge Organizers section",
    )
    assert_true("Program Committee" in organizers.text, "organizers page must include the Program Committee section")
    challenge_html = organizers_html.split(
        '<h2 id="workshop-challenge-organizers">Workshop Challenge Organizers</h2>', 1
    )[-1].split('<h2 id="program-committee">Program Committee</h2>', 1)[0]
    challenge_names = [
        "Jyotika Singh",
        "Hitesh Patel",
        "Rahul Suresh",
        "Xiaolong Luo",
        "Alexandra Bezea-Tudor",
        "Ruby Zhang",
    ]
    challenge_positions = [challenge_html.find(name) for name in challenge_names]
    assert_true(
        all(position >= 0 for position in challenge_positions),
        "challenge organizer section must include all six confirmed organizers",
    )
    assert_true(
        challenge_positions == sorted(challenge_positions),
        "challenge organizers must preserve the confirmed display order",
    )
    assert_true("Oracle" in challenge_html, "challenge organizers must identify Oracle")
    assert_true("Abaka AI" in challenge_html, "challenge organizers must identify Abaka AI")
    assert_true(
        "person-bio" not in challenge_html,
        "challenge organizer cards must not include unsourced bios",
    )
    assert_true(
        "challenge-organizer-grid" in organizers.classes,
        "challenge organizers must use the dedicated compact grid",
    )
    committee_html = organizers_html.split(
        '<h2 id="program-committee">Program Committee</h2>', 1
    )[-1].split('<h2 id="contact">Contact</h2>', 1)[0]
    assert_true("person-affiliation" not in committee_html, "program committee cards must not show affiliations")
    for name in ["Tampu Ravi Kumar", "Hansa Meghwani", "Karan Dua"]:
        assert_true(name in organizers.text, f"program committee missing {name}")
    for name in ["Hitesh Patel", "Jyotika Singh"]:
        assert_true(name not in committee_html, f"{name} must move out of the program committee")
    assert_true("committee-grid" in organizers.classes, "program committee must use the compact committee-grid layout")
    assert_true("assets/images/committee/" in organizers_html, "program committee must use local committee headshots")
    for filename in [
        "tampu_ravi_kumar.jpeg",
        "hansa_meghwani.jpg",
        "hitesh_patel.jpg",
        "jyotika_singh.jpg",
        "karan_dua.jpeg",
    ]:
        assert_true((BASE / "assets" / "images" / "committee" / filename).exists(), f"missing committee headshot: {filename}")
    assert_true(
        (BASE / "assets" / "images" / "challenge-organizers" / "rahul_suresh.png").exists(),
        "missing Rahul Suresh challenge-organizer headshot",
    )
    for filename in [
        "xiaolong_luo.jpg",
        "alexandra_bezea_tudor.jpg",
        "ruby_zhang.jpg",
    ]:
        assert_true(
            (BASE / "assets" / "images" / "challenge-organizers" / filename).exists(),
            f"missing challenge-organizer headshot: {filename}",
        )
    assert_true("footer-sponsors" in organizers.classes, "site footer must include the sponsors block")
    assert_true("nav-sponsors" not in organizers.classes, "primary navigation must not contain sponsor branding")
    assert_true("sponsor-lockup--page" in organizers.classes, "inner page title bands must expose sponsors")
    page_header_html = organizers_html.split('<header class="page-header">', 1)[-1].split("</header>", 1)[0]
    assert_true("DocInsights 2026" not in page_header_html, "inner page title band must not repeat the site name")
    assert_true("Sponsors" in page_header_html, "inner page title band must label the sponsor group")
    assert_true("Oracle logo" in organizers_html, "site footer must show the Oracle sponsor logo")
    assert_true("Abaka AI logo" in organizers_html, "site footer must show the Abaka AI sponsor logo")
    for filename in ["oracle.svg", "abaka-ai.png"]:
        assert_true((BASE / "assets" / "sponsors" / filename).exists(), f"missing sponsor logo: {filename}")
    oracle_svg = (BASE / "assets" / "sponsors" / "oracle.svg").read_text(encoding="utf-8")
    assert_true(
        'width="231"' in oracle_svg and 'height="30"' in oracle_svg,
        "Oracle sponsor logo must include intrinsic dimensions",
    )
    assert_true(
        re.search(r"\.person-card\s*{[^}]*display:\s*flow-root", css, re.S) is not None,
        "organizer cards must let bio text flow around the photo",
    )
    assert_true(
        re.search(r"\.person-photo\s*{[^}]*float:\s*left", css, re.S) is not None,
        "organizer photos must be floated so bios can use space below them",
    )
    assert_true(
        re.search(r"\.committee-grid\s*{[^}]*grid-template-columns:\s*repeat\(auto-fit,\s*minmax\(420px,\s*1fr\)\)", css, re.S) is not None,
        "program committee grid must use denser wide-screen columns",
    )
    assert_true(
        re.search(r"\.people-grid\s*{[^}]*align-items:\s*start", css, re.S) is not None,
        "people grids must keep cards content-height instead of stretching white boxes",
    )
    assert_true(
        re.search(
            r"\.challenge-organizer-grid\s*{[^}]*grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)",
            css,
            re.S,
        )
        is not None,
        "challenge organizer grid must use three equal desktop columns",
    )

    print("site audit passed")


if __name__ == "__main__":
    main()
