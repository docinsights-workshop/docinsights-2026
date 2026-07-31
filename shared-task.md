---
layout: default
title: Challenges
permalink: /shared-task/
---

<section class="section-intro">
  <p class="section-kicker">DocInsights 2026 Challenges</p>
  <h2>Two challenges advancing document intelligence beyond plain text</h2>
  <p>DocInsights 2026 hosts two complementary challenge tracks: <strong>DocSem</strong> for document-grounded quantitative reasoning with evidence attribution, and <strong>Dr.DocBench</strong> for expert-level document parsing across complex visual and structural content.</p>
</section>

<section class="challenge-season" aria-labelledby="challenge-season-title">
  <div class="challenge-season-copy">
    <p class="section-kicker">Competition Season</p>
    <h2 id="challenge-season-title">August 3–September 10, 2026</h2>
    <p>Teams may participate in either or both challenges. Final challenge-specific rules, eligibility requirements, and prize allocations will be published through the official challenge resources.</p>
  </div>
  <div class="challenge-season-stats">
    <div class="challenge-stat">
      <span class="challenge-stat-value">USD 5,000+</span>
      <span class="challenge-stat-label">Total prize pool</span>
    </div>
    <div class="challenge-stat">
      <span class="challenge-stat-value">2 tracks</span>
      <span class="challenge-stat-label">Distinct challenge tasks</span>
    </div>
    <div class="challenge-stat">
      <span class="challenge-stat-value">Workshop pathway</span>
      <span class="challenge-stat-label">System papers and presentations</span>
    </div>
  </div>
</section>

<section class="challenge-list" aria-label="DocInsights challenge tracks">
  <article class="challenge-feature challenge-feature-docsem">
    <header class="challenge-feature-header">
      <div>
        <p class="challenge-number">Challenge 1</p>
        <h2>DocSem</h2>
        <p class="challenge-tagline">Document-grounded quantitative reasoning with evidence attribution</p>
      </div>
      <span class="challenge-status challenge-status-ready">Resources live</span>
    </header>

    <div class="challenge-feature-body">
      <div class="challenge-description">
        <p>Participants receive a PDF document and a paraphrased query. Systems must identify the relevant quantitative passage, derive the requested answer from the supplied document, and return the visible PDF block IDs that support the prediction.</p>
        <div class="challenge-actions">
          <a class="challenge-action challenge-action-primary" href="https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data" target="_blank" rel="noopener">Dataset and guide</a>
          <a class="challenge-action" href="https://amitbcp-docsem-docinsights.hf.space/" target="_blank" rel="noopener">Submission portal</a>
          <a class="challenge-action" href="https://github.com/oracle-samples/gsm-sem/tree/main/docsem" target="_blank" rel="noopener">Canonical source</a>
        </div>
      </div>
      <dl class="challenge-facts">
        <div>
          <dt>Development data</dt>
          <dd>908 labelled training tasks with PDFs, answers, and evidence block IDs.</dd>
        </div>
        <div>
          <dt>Evaluation data</dt>
          <dd>217 validation tasks with organizer-held labels and a public leaderboard.</dd>
        </div>
        <div>
          <dt>Submission</dt>
          <dd>One complete JSONL file containing an answer and supporting block IDs for every validation task.</dd>
        </div>
      </dl>
    </div>
  </article>

  <article class="challenge-feature challenge-feature-drdoc">
    <header class="challenge-feature-header">
      <div>
        <p class="challenge-number">Challenge 2</p>
        <h2>Dr.DocBench</h2>
        <p class="challenge-tagline">Expert-level parsing of complex, real-world documents</p>
      </div>
      <span class="challenge-status challenge-status-preparing">Participant release in preparation</span>
    </header>

    <div class="challenge-feature-body">
      <div class="challenge-description">
        <p>Dr.DocBench evaluates document parsing across difficult text, tables, formulas, reading order, document structure, chemistry notation, and musical notation. The organizer-side evaluation pipeline has completed end-to-end testing.</p>
        <p>The image-only participant package, final submission specification, public evaluation access, and evaluation-integrity review are being finalized. The current public annotated benchmark is not the complete challenge input package.</p>
        <div class="challenge-actions">
          <a class="challenge-action challenge-action-primary" href="https://drdocbench-challenge.abaka-pages.com/" target="_blank" rel="noopener">Challenge website</a>
        </div>
      </div>
      <dl class="challenge-facts">
        <div>
          <dt>Task scope</dt>
          <dd>Convert complex document pages and multi-page windows into structurally faithful Markdown.</dd>
        </div>
        <div>
          <dt>Release status</dt>
          <dd>Participant package, validator, sample submission, and frozen evaluation contract are being prepared.</dd>
        </div>
        <div>
          <dt>Submission access</dt>
          <dd>The public evaluation link will be posted after the final participant-readiness and integrity gates pass.</dd>
        </div>
      </dl>
    </div>
  </article>
</section>

<section>
  <div class="section-heading">
    <p class="section-kicker">Competition Timeline</p>
    <h2>Shared milestones</h2>
    <p>Challenge-specific portal times and any readiness-dependent updates will be posted through the official challenge resources.</p>
  </div>
  <div class="challenge-timeline">
    {% for item in site.data.dates.challenge_items %}
    <article class="timeline-step">
      <span class="timeline-step-date">{{ item.date }}</span>
      <div>
        <h3>{{ item.label }}</h3>
        <p>{{ item.note }}</p>
      </div>
    </article>
    {% endfor %}
  </div>
</section>

<section>
  <div class="section-heading">
    <p class="section-kicker">From Competition to Workshop</p>
    <h2>Share systems, findings, and lessons learned</h2>
    <p>Challenge participants can submit concise system papers for workshop consideration. Selected contributions, including winning solutions, will be invited to present their approaches and findings at DocInsights 2026.</p>
  </div>
  <div class="info-grid">
    <article class="info-card">
      <h3>Compete in either track</h3>
      <p>Teams may enter DocSem, Dr.DocBench, or both challenges, subject to each track's final participation rules.</p>
    </article>
    <article class="info-card">
      <h3>Document the system</h3>
      <p>Participants should report models, data, tools, prompts, and evaluation choices needed to understand and reproduce their submission.</p>
    </article>
    <article class="info-card">
      <h3>Present selected work</h3>
      <p>Winning teams and selected participant contributions will have a pathway to share their work with the workshop community.</p>
    </article>
  </div>
</section>

<section class="note-section">
  <h2>Prize and rules notice</h2>
  <p>The total prize pool will exceed USD 5,000. Track-level allocations, eligibility, team limits, tie-breaking rules, and award conditions will be published before final ranking begins.</p>
</section>
