---
layout: default
title: Shared Task
permalink: /shared-task/
---

<section class="section-intro">
  <p class="section-kicker">DocInsights 2026 Shared Task</p>
  <h2>DocSem: document-grounded quantitative reasoning with evidence attribution</h2>
  <p>Participants receive a PDF document and a paraphrased query. Systems must identify the relevant quantitative passage, derive the answer from the supplied document, and report the visible PDF block IDs that support the prediction.</p>
</section>

<section class="info-grid">
  <article class="info-card">
    <h3>Development data</h3>
    <p>The public dataset provides 908 labelled training tasks, PDFs, answers, and evidence block IDs for system development.</p>
  </article>
  <article class="info-card">
    <h3>Validation</h3>
    <p>Validation provides 217 PDF tasks and queries. Its labels remain organizer-only and are used for the public leaderboard.</p>
  </article>
  <article class="info-card">
    <h3>Submission</h3>
    <p>Upload one JSONL prediction file through the submission portal. Each row includes an answer and one or more visible evidence block IDs.</p>
  </article>
</section>

## Get started

<div class="info-grid">
  <article class="info-card">
    <h3>1. Download the data</h3>
    <p><a href="https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data">Open the public Hugging Face dataset</a> and follow the participant instructions.</p>
  </article>
  <article class="info-card">
    <h3>2. Build with training data</h3>
    <p>Use the labelled train split to develop document reading, quantitative reasoning, and evidence-selection methods.</p>
  </article>
  <article class="info-card">
    <h3>3. Submit validation predictions</h3>
    <p><a href="https://amitbcp-docsem-docinsights.hf.space/">Open the submission portal and leaderboard</a> to evaluate a complete validation JSONL file.</p>
  </article>
</div>

## Task format

Each task contains a PDF and a separate `user_query`. PDF content blocks begin with visible identifiers such as `b01:`. Submit one JSON object for every validation instance:

```json
{"instance_id":"task_000909","answer":"140","evidence":["b14"]}
```

The primary metric is normalized exact-match accuracy on the final answer. Evidence block-set match and evidence F1 are reported separately.

## Data and evaluation policy

Train answers and evidence labels are public. Validation PDFs and queries are public, while validation labels are kept private by the organizers and used only by the submission portal. Do not infer answers from filenames, metadata, or external source-question lookup; solve each task from its supplied PDF.
