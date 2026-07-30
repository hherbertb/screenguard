# Jira Case Study

This folder contains the data for the BPM case study reported in the paper
(Section 5.2, "Case Study"). The case study assesses the utility of ScreenGuard
redaction in a **process-mining / task-mining** setting.

## What the case study does

We recorded an interaction log for a **17-step Jira workflow** (logging in,
navigating to a project, creating and assigning an issue, and adding a new
person to the project). We applied ScreenGuard's **generic LLM-based** approach
to the log to produce a redacted version. For this case study we used
**substitution rather than blocking** (e.g., real email addresses replaced with
synthetic ones) so the visual context stays intact.

We then ran [ActivityGen](https://zenodo.org/records/13375065) on both the
original and the redacted logs to obtain translucent event logs, and discovered
a process model from each. The redacted log yields **more general process
models**, because user-name-laden activity labels that would otherwise inflate
the model are removed (paper Figures 11 and 12).

## Folder contents

| Folder | Description |
|--------|-------------|
| `JiraCelonis/` | Raw input: the 17 Jira screenshots of the recorded workflow (case study input). |
| `JiraUnredacted/` | Pipeline output on the **original** (unredacted) screenshots: OCR, detected UI components (`compo`), merged elements (`merge`), generated activity names (`activity_name`), and logs. |
| `Redacted/` | Pipeline output on the **redacted** screenshots: OCR, `compo`, `merge`, the final `redacted/` images, and `log.txt`. |

## Reproducing

From the project root, place the `JiraCelonis/` screenshots in `input/` and run
the generic LLM workflow (see the top-level `README.md`):

```bash
python redact/LLM_workflow.py
```

Then run ActivityGen on the original and redacted logs to reproduce the two
process models compared in the paper.
