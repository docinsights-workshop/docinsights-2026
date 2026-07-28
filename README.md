# DocInsights 2026

Source for the [DocInsights 2026 workshop website](https://docinsights-workshop.github.io/docinsights-2026/), co-located with EMNLP 2026.

## Shared Task: DocSem

The [DocSem shared task](https://docinsights-workshop.github.io/docinsights-2026/shared-task/) is document-grounded quantitative reasoning with evidence attribution. Participants receive a PDF and a paraphrased query, then submit a final answer and the visible PDF block IDs that support it.

- [Public development dataset](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data)
- [Submission portal and leaderboard](https://amitbcp-docsem-docinsights.hf.space/)
- [Participant instructions](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data/blob/main/INSTRUCTIONS.md)

The public dataset provides labelled training data and unlabelled validation inputs. Validation labels, participant submissions, and leaderboard write state remain in a separate private organizer repository.

## Repository scope

This repository hosts the workshop site and the source code/docs for the Hugging Face integration. Generated PDFs, public dataset payloads, organizer labels, and submissions are deliberately excluded; publish data directly to their dedicated Hugging Face repositories.
