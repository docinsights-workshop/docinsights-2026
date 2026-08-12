---
layout: default
title: Important Dates
permalink: /dates/
---

<section class="section-intro">
  <p class="section-kicker">Timeline</p>
  <h2>Submission and workshop milestones</h2>
  <p>{{ site.data.dates.timezone_note }}</p>
</section>

{% include date-list.html %}

<section>
  <div class="section-heading">
    <p class="section-kicker">Challenge Season</p>
    <h2>Competition milestones</h2>
    <p>{{ site.data.dates.challenge_timezone_note }}</p>
  </div>
  <div class="date-list challenge-date-list">
    {% for item in site.data.dates.challenge_items %}
    <article class="date-card date-card-highlight">
      <div class="date-card-meta">{{ item.type }}</div>
      <h3>{{ item.label }}</h3>
      <p class="date-card-date">{{ item.date }}</p>
      {% if item.time %}
      <p class="date-card-time">{{ item.time }}</p>
      {% endif %}
      <p class="date-card-note">{{ item.note }}</p>
      <a class="inline-action" href="{{ '/shared-task/' | relative_url }}">View challenges</a>
    </article>
    {% endfor %}
  </div>
</section>

<section class="note-section">
  <h2>Schedule Note</h2>
  <p>EMNLP 2026 is scheduled for October 24-29, 2026 in Budapest, Hungary. The central EMNLP program still lists workshop and tutorial timing as pending, so DocInsights will publish the exact workshop day, room, and participation details after they are confirmed.</p>
</section>
