---
pretty_name: DocSem Workshop Submissions and Hidden Validation Labels
license: other
tags:
- docsem
- docinsights-2026
- shared-task
- private-evaluation
---

# DocSem Workshop Submissions and Hidden Validation Labels

This repository is intended to be private.

It stores organizer-only evaluation assets:

- `private/val_labels.jsonl`: hidden validation answers and evidence.
- `submissions/`: raw participant submissions saved by the hosted portal, including private participant names and contact email.
- `leaderboard/leaderboard.json`: generated leaderboard state, including organizer-only participant metadata.

Legacy records may not include `participant_names`. New submissions require it. Repeated attempts retain the same team-plus-contact identity, and only the latest attempt for that identity is shown on the public leaderboard. The private repository retains every valid submission and its attempt count.

Do not make this repository public while it contains validation labels or raw participant submissions.
