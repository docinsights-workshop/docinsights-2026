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
    <h2 id="challenge-season-title">August 3–October 10, 2026</h2>
    <p>Teams may participate in either or both challenges. DocSem closes on September 10; Dr.DocBench closes on October 10 at 12:59 PM UTC. Each official challenge portal is the system of record for submissions and final rules.</p>
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

<section class="paper-submission-callout" aria-labelledby="shared-task-paper-title">
  <div>
    <p class="section-kicker">System Paper Submissions</p>
    <h2 id="shared-task-paper-title">Shared-task paper submissions are open</h2>
    <p><strong>Archival or non-archival</strong> system papers are welcome for <strong>DocSem or Dr.DocBench</strong>. Describe the system, data, models, prompts, tools, evaluation choices, and lessons learned from your challenge participation.</p>
    <p>Selected contributions, including winning solutions, will be invited to present at DocInsights 2026.</p>
  </div>
  <div class="paper-submission-action">
    <span class="paper-submission-deadline">September 15, 2026 at 11:59 PM UTC</span>
    <a class="challenge-action challenge-action-primary" href="https://openreview.net/group?id=EMNLP/2026/Workshop/DocInsights_Shared_Task" target="_blank" rel="noopener">Submit on OpenReview</a>
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
      <span class="challenge-status challenge-status-ready">Dataset frozen Aug 5</span>
    </header>

    <div class="challenge-feature-body">
      <div class="challenge-description">
        <p>Participants receive a PDF document and a paraphrased query. Systems must identify the relevant quantitative passage, derive the requested answer from the supplied document, and return the visible PDF block IDs that support the prediction.</p>
        <aside class="challenge-update" aria-labelledby="docsem-evaluation-update">
          <p class="challenge-update-kicker">Participant notice</p>
          <h3 id="docsem-evaluation-update">Use the August 5 dataset release</h3>
          <p>If you downloaded the problem set before August 5, pull the latest version. The development data was updated on <strong>August 5, 2026</strong>, and is now frozen; there will be no further updates to it.</p>
          <p><strong>Final evaluation:</strong> A held-out test set will be released five days before the September 10, 2026 final submission deadline. Participants will be notified when it becomes available and asked to submit their test-set results. Performance on the held-out test set will determine the final leaderboard.</p>
        </aside>
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
          <dd>217 validation tasks with organizer-held labels and a provisional public validation leaderboard. Final rankings use a separate held-out test set.</dd>
        </div>
        <div>
          <dt>Submission</dt>
          <dd>Submit one complete validation JSONL now. Participants will be notified and asked for a separate test-set submission when final evaluation opens.</dd>
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
      <span class="challenge-status challenge-status-ready">Submissions open</span>
    </header>

    <div class="challenge-feature-body">
      <div class="challenge-description">
        <p>Dr.DocBench challenges systems to recover structured content from complex document pages, including text, tables, formulas, reading order, and specialized notation. Challenge predictions use a <strong>single-page</strong> unit and produce structurally faithful Markdown.</p>
        <aside class="challenge-update" aria-labelledby="drdoc-live-update">
          <p class="challenge-update-kicker">Challenge live</p>
          <h3 id="drdoc-live-update">Submit through EvalAI by October 10</h3>
          <p>The competition runs from <strong>August 10 through October 10, 2026</strong>. Submissions close October 10 at <strong>12:59 PM UTC</strong>; final evaluation runs October 11–23, and winners will be announced at DocInsights 2026.</p>
          <p><strong>Awards:</strong> Dr.DocBench offers a prize pool of up to USD 3,000. Official final rankings are determined on the Private Test phase.</p>
        </aside>
        <div class="challenge-actions">
          <a class="challenge-action challenge-action-primary" href="https://eval.ai/web/challenges/challenge-page/2717/overview" target="_blank" rel="noopener">Enter on EvalAI</a>
          <a class="challenge-action" href="https://drdocbench-challenge.abaka-pages.com/" target="_blank" rel="noopener">Challenge website</a>
          <a class="challenge-action" href="https://huggingface.co/datasets/2077AIDataFoundation/DrDocBench" target="_blank" rel="noopener">Dataset and guide</a>
          <a class="challenge-action" href="https://arxiv.org/abs/2606.01393" target="_blank" rel="noopener">Benchmark paper</a>
        </div>
      </div>
      <dl class="challenge-facts">
        <div>
          <dt>Evaluation</dt>
          <dd>Text Edit Distance, Table TEDS, Formula CDM, and Reading Order are normalized into an Overall score. Teams are ranked by Overall score.</dd>
        </div>
        <div>
          <dt>Submission</dt>
          <dd>Upload a validated submission ZIP containing <code>predictions.jsonl</code> or the canonical <code>mds/</code> tree. Strict checks reject missing, extra, duplicate, or invalid predictions.</dd>
        </div>
        <div>
          <dt>Participation</dt>
          <dd>Open worldwide with no team-size limit. Teams may submit up to 3 times per day and select up to 2 submissions for final evaluation.</dd>
        </div>
      </dl>
    </div>
  </article>
</section>

<section>
  <div class="section-heading">
    <p class="section-kicker">Competition Timeline</p>
    <h2>Track milestones</h2>
    <p>DocSem and Dr.DocBench have separate final submission deadlines. Use the official portal for each track's exact submission status.</p>
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
  <p>The combined prize pool exceeds USD 5,000, including up to USD 3,000 for Dr.DocBench. Consult each official challenge portal for track-level eligibility, team rules, ranking procedures, and award conditions.</p>
</section>
