# DocInsights 2026 Website Redesign Design

## Purpose

Revamp the DocInsights 2026 website into a polished workshop hub for authors, attendees, reviewers, speakers, and organizers. The site should make the workshop's identity clear within the first viewport, guide visitors quickly to submission and logistics information, and communicate the accepted proposal's strongest themes without publishing unconfirmed private proposal details as confirmed public facts.

## Source Material

The redesign is grounded in four inputs:

- Current Jekyll site in this repository.
- Accepted workshop proposal at `/Users/aamita/Downloads/GRAIL_EMNLP.pdf`.
- Official public facts from EMNLP 2026, OpenReview, and ACL/ARR policy pages.
- Pattern research from recent strong NLP workshop sites, especially SURGeLLM ACL 2026, Table Representation Learning ACL 2025, BlackboxNLP 2025, NewSumm 2025, WMT, and AI & Scientific Discovery.

## Goals

- Make DocInsights read as a serious EMNLP workshop rather than a minimal CFP page.
- Put the workshop thesis up front: document intelligence beyond plain text, bridging NLP and Document AI through structure-aware modeling, grounding, retrieval, reasoning, evaluation, and deployment.
- Help authors act quickly through clear submission CTAs, dates, tracks, and policies.
- Help attendees understand what the workshop will contain before the final schedule is public.
- Keep the site maintainable for collaborators by using small Jekyll pages and structured data instead of a monolithic HTML page.
- Improve mobile experience, accessibility, and link hygiene.

## Non-Goals

- Do not rewrite the site in a JavaScript framework.
- Do not publish proposed speaker names as confirmed unless the user explicitly confirms they are public-ready.
- Do not claim a finalized workshop day if EMNLP's central program still lists workshop dates as pending. The public site may mention OpenReview metadata separately only if the wording makes the source and uncertainty clear.
- Do not add search, proceedings, paper lists, or registration flows until there is real content to power them.

## Public Content Rules

Public copy must separate confirmed facts from proposal or planning details.

- Confirmed: DocInsights is accepted as an EMNLP 2026 workshop, officially listed as "Workshop on Document Intelligence and Understanding"; EMNLP 2026 is October 24-29, 2026 in Budapest, Hungary; OpenReview has direct submission and ARR commitment groups.
- Public but cautious: use "at EMNLP 2026, October 24-29, 2026; exact workshop schedule pending EMNLP program" until the official workshop day is confirmed for publication.
- Proposal-derived and public-ready: workshop themes, expected format categories, review process, inclusion commitments, and planned shared task framing.
- Proposal-derived but confirmation-gated: invited speaker names, exact shared-task operational details, travel or registration support, awards, sponsors, and final room or hybrid details.

## Information Architecture

The redesigned navigation should be:

- Home
- CFP
- Dates
- Program
- Shared Task
- Speakers
- Organizers
- FAQ

Footer navigation should group links by action:

- Submit: Direct OpenReview, ARR Commitment, CFP.
- Attend: EMNLP 2026, Venue, Program, FAQ.
- Policies: ACL Code of Ethics, ACL Anti-Harassment Policy, ARR policy, contact.

## Homepage Design

The homepage should become the main conversion surface.

First viewport:

- Semantic text `h1`: "DocInsights 2026".
- Subtitle: "Workshop on Document Intelligence and Understanding".
- Short theme line: "Beyond Plain Text: Bridging NLP and Document AI".
- Event context: "Co-located with EMNLP 2026, Budapest, Hungary".
- CTAs: "Submit Direct Paper", "Commit ARR Paper", "Read CFP".
- Compact fact strip: full-day workshop, EMNLP 2026, Budapest, direct submission deadline, ARR commitment deadline.

Below first viewport:

- "Why DocInsights" section that explains the gap: documents are structured visual-textual evidence, not plain text.
- "Research Themes" cards: structure-aware modeling; reasoning and grounding; multimodal and cross-document understanding; knowledge integration; evaluation and interaction; applications and deployment.
- "Program Preview" cards: invited talks, oral presentations, posters, system/demo session, panel, early-career spotlights.
- "Shared Task" teaser: planned RUST-BENCH structure-aware tabular reasoning track, with details pending organizer confirmation.
- "Important Dates" timeline with direct and ARR deadlines.
- "Latest News" reverse-chronological updates.
- Contact block.

The current banner image may remain as visual texture, but it must not be the only source of the workshop title, dates, or location. On mobile, the hero must present readable text and CTAs before or alongside the image.

## CFP Page

The CFP page should be reorganized for author decisions:

- Submission paths: direct archival, direct non-archival, ARR commitment.
- Contribution types: long papers, short papers, position/theory papers, benchmark/dataset papers, survey papers, system/demo papers, reproducibility studies, negative or diagnostic analyses.
- Topics: expand current topics with proposal language around evidence granularity, OCR/layout assumptions, robustness to document noise, provenance, privacy, governance, and human-centered deployments.
- Requirements: ACL format, anonymization, page limits, PDF submission, self-contained appendices.
- Review process: double-blind, at least 3 reviews plus meta-review, OpenReview, conflict handling, relevance and fit.
- Policies: multiple submission policy, archival/non-archival policy, ARR commitment policy, ethics and responsible NLP links.
- Submission links: direct OpenReview and ARR commitment OpenReview.

## Dates Page

The dates page should be a shareable reference and should duplicate the important dates on the homepage and CFP page.

Use mobile-friendly date cards rather than relying only on wide tables. Each date item should include:

- Event label.
- Date.
- Deadline time and timezone when relevant.
- Action link when available.
- Notes when the date is source-sensitive, such as "exact workshop schedule pending EMNLP program."

## Program Page

Replace the current empty program message with a credible provisional program structure.

Public program sections:

- Workshop format.
- Expected full-day flow: opening, invited talks, oral session, poster/session demos, shared task spotlight, panel, early-career spotlights, closing.
- Time zone and exact room note: pending EMNLP program release.
- Presentation modes note: only claim hybrid/virtual details once confirmed by EMNLP or organizers.

The page must avoid listing speaker names, talk titles, paper titles, awards, or room details until confirmed.

## Shared Task Page

Add a shared task page that can start as a high-level announcement.

Public-ready content:

- Planned task theme: structure-aware tabular reasoning grounded in tabular evidence.
- Motivation: real-world tables are large, heterogeneous, domain-specific, and require multi-hop reasoning.
- Expected participation flow: task overview, held-out evaluation, leaderboard, short system descriptions, workshop presentations.
- Status: details and participation instructions will be announced after organizer confirmation.

Confirmation-gated content:

- CodaLab URL.
- Exact dataset release dates.
- Leaderboard link.
- Awards.
- Submission format details.

## Speakers Page

Add a speakers page with a clean holding state:

- Explain that invited speakers will be announced after confirmation.
- Reserve structure for future speaker cards: name, affiliation, photo, talk title, abstract, bio, external links.
- Do not publish proposal speaker names without explicit confirmation.

## Organizers Page

Keep the current organizer cards but improve them:

- Add concise role-oriented bios from the accepted proposal where appropriate.
- Add optional website or profile links only when verified.
- Preserve image alt text.
- Consider reducing heavy image sizes during implementation if page weight becomes a problem.

## FAQ Page

Add an FAQ page for common author and attendee questions:

- What is DocInsights?
- What submission types are accepted?
- Can I submit non-archival work?
- How do ARR commitments work?
- Are submissions double-blind?
- What are the page limits?
- What is the shared task?
- Will the workshop support remote participation?
- When will speakers and the program be announced?
- Who should I contact?

Answers should be short, policy-aligned, and link to official sources where possible.

## Visual Direction

The site should feel modern, focused, and academic without becoming a marketing page.

- Use a readable editorial hierarchy with a clear hero, compact facts, and strong CTAs.
- Use restrained color anchored in the existing DocInsights blue, with secondary accent colors from the logo/banner rather than a one-note palette.
- Use cards sparingly for repeated items and action clusters, not nested cards.
- Replace wide tables on mobile with stacked cards or timeline rows.
- Keep typography readable and avoid scaling font size with viewport width.
- Use icons only where they clarify actions or categories.
- Avoid decorative orbs, generic gradients, and oversized empty hero space.

## Accessibility And UX Requirements

- Add a skip-to-content link.
- Ensure the homepage has exactly one meaningful `h1`.
- Keep keyboard focus visible.
- Add `rel="noopener"` to external links that open in a new tab.
- Ensure nav toggle exposes expanded state.
- Keep all important title/date/location text in HTML, not image-only text.
- Preserve descriptive alt text for meaningful images.
- Avoid horizontal overflow on mobile.
- Maintain adequate color contrast for buttons, badges, links, and footer text.

## Architecture

Keep the site as Jekyll with Markdown pages, Liquid includes, and YAML data files.

Recommended structure:

- `_data/navigation.yml`: primary and footer navigation links.
- `_data/dates.yml`: dates used by homepage, CFP, and dates page.
- `_data/news.yml`: reverse-chronological news items.
- `_data/themes.yml`: homepage research theme cards.
- `_data/program_preview.yml`: program structure cards.
- `_includes/hero.html`: homepage hero.
- `_includes/date-list.html`: reusable dates block.
- `_includes/news-list.html`: reusable news block.
- `_includes/cta-row.html`: reusable submission CTAs.
- Pages: `index.md`, `call-for-papers.md`, `dates.md`, `program.md`, `shared-task.md`, `speakers.md`, `organizers.md`, `faq.md`.

This keeps content edits local and avoids duplicating dates or links across pages.

## Data Flow

Jekyll reads YAML data from `_data/` and renders it through includes into Markdown pages. The homepage, CFP page, and dates page should consume the same dates data to prevent deadline drift. Navigation should be rendered from a single data file so adding pages does not require repeated edits in the layout and footer.

## Error Handling And Content Drift

Static site "errors" are mainly stale or overclaimed information.

- Dates must include a source note when they depend on OpenReview or EMNLP central schedule status.
- Unconfirmed sections should use deliberate language such as "details will be announced after confirmation" rather than vague filler.
- External links should be verified during implementation.
- Build validation must catch missing pages, missing images, and Liquid errors.

## Validation

Implementation is complete only after:

- Jekyll build succeeds.
- Browser smoke check passes for home, CFP, dates, program, shared task, speakers, organizers, and FAQ.
- Desktop and mobile screenshots show no horizontal overflow, unreadable hero text, or broken navigation.
- Mobile nav opens and closes.
- All local images load.
- External `target="_blank"` links include `rel="noopener"`.
- No source files contain accidental incomplete-work markers.

## Rollout Sequence

1. Create structured data and includes.
2. Redesign layout navigation, footer, and accessibility scaffolding.
3. Rebuild homepage around semantic hero, CTAs, themes, program preview, dates, news, and contact.
4. Reorganize CFP content.
5. Add Dates, Shared Task, Speakers, and FAQ pages.
6. Replace the empty Program page with provisional program structure.
7. Polish CSS for desktop and mobile.
8. Run build and browser validation.
