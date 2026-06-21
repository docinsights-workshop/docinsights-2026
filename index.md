---
layout: default
title: Home
---

<section class="section-intro">
  <p class="section-kicker">Why DocInsights</p>
  <h2>Documents are structured visual-textual evidence, not just plain text.</h2>
  <p>Real-world documents in science, healthcare, law, finance, government, and enterprise settings combine natural language with tables, forms, figures, charts, lists, layout cues, and cross-page relationships. Reliable document intelligence requires models and systems that reason across these elements while staying grounded in document evidence.</p>
  <p><strong>DocInsights 2026</strong> brings together researchers and practitioners across NLP, Document AI, multimodal learning, information retrieval, knowledge representation, and human-centered systems to advance trustworthy, scalable document understanding.</p>
</section>

<section>
  <div class="section-heading">
    <p class="section-kicker">Scope</p>
    <h2>Research Themes</h2>
  </div>
  {% include theme-grid.html %}
</section>

<section>
  <div class="section-heading">
    <p class="section-kicker">Workshop Format</p>
    <h2>Program Preview</h2>
    <p>Exact timing, room, accepted papers, and confirmed speakers will be announced after the EMNLP workshop program and decisions are finalized.</p>
  </div>
  {% include program-preview.html %}
</section>

<section class="split-section">
  <div>
    {% comment %}
    Hidden until shared task details are public-ready:
    <h2>RUST-BENCH: structure-aware tabular reasoning</h2>
    <p>The workshop plans a shared task around table-centric question answering and reasoning grounded in tabular evidence. The task highlights real-world challenges such as scale, heterogeneity, domain specificity, and multi-hop inference.</p>
    {% endcomment %}
    <p class="section-kicker">Shared Task</p>
    <h2>Shared task details will be announced soon</h2>
    <p>We are finalizing the shared task scope, timeline, participation instructions, and evaluation details. The dedicated page will be updated once the announcement is ready.</p>
    <a class="inline-action" href="{{ '/shared-task/' | relative_url }}">View shared task announcement</a>
  </div>
  <div class="callout-panel">
    <h3>Submission Tracks</h3>
    <p>DocInsights welcomes direct archival and direct non-archival submissions, plus eligible ARR commitments through a separate OpenReview group.</p>
    <a class="inline-action" href="{{ '/call-for-papers/' | relative_url }}">View author guidance</a>
  </div>
</section>

<section>
  <div class="section-heading">
    <p class="section-kicker">Timeline</p>
    <h2>Important Dates</h2>
    <p>{{ site.data.dates.timezone_note }}</p>
  </div>
  {% include date-list.html %}
</section>

<section>
  <div class="section-heading">
    <p class="section-kicker">Updates</p>
    <h2>Latest News</h2>
  </div>
  {% include news-list.html %}
</section>

<section class="contact-band">
  <div>
    <p class="section-kicker">Contact</p>
    <h2>Questions about the workshop?</h2>
    <p>Reach the organizing team by email, or follow DocInsights for updates as speakers, program details, and shared task instructions are announced.</p>
  </div>
  <div class="social-links">
    <a href="mailto:docinsights-workshop-chairs@googlegroups.com" class="social-link">Email</a>
    <a href="https://x.com/DocInsights_26" target="_blank" rel="noopener" class="social-link">X</a>
    <a href="https://bsky.app/profile/docinsights.bsky.social" target="_blank" rel="noopener" class="social-link">Bluesky</a>
  </div>
</section>
