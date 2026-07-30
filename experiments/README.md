# Experiments — Quantitative Evaluation

This folder contains the data, intermediate outputs, and per-image evaluation
results for the quantitative benchmark reported in the paper (Section 5.1,
"Quantitative Evaluation").

## What the experiment measures

We evaluate detection and classification quality of the three ScreenGuard
strategies against a ground-truth dataset. For each detected term we compare the
predicted sensitivity label against the annotation, yielding true/false
positives and negatives, and report **accuracy, precision, recall, and F1**.
Sensitive elements that OCR fails to detect are counted as false negatives.

### Dataset

No existing dataset combined GUI screenshots with annotated sensitive regions,
so we created our own. HTML templates were populated across **ten categories**
(banking, chats, e-commerce, emails, admin pages, employee directories, patient
directories, registration forms, login forms, user profiles) with synthetic
personal data (names, birthdates, email addresses, salaries, IDs) and varying
fonts and colors. The resulting dataset contains **188 images**. Information
linked to a person (e.g., name, IBAN) is treated as sensitive.

### Results (paper Table 2)

| Strategy | Accuracy | Precision | Recall | F1 |
|----------|:--------:|:---------:|:------:|:----:|
| Rule-based       | 0.830 | 0.797 | 0.200 | 0.319 |
| Generic prompt   | 0.933 | 0.906 | 0.743 | 0.817 |
| Specific prompt  | 0.938 | 0.853 | 0.833 | 0.843 |

The LLM-based approaches more than double the F1 of the rule-based baseline,
primarily through higher recall.

## Folder contents

| Folder / file | Description |
|---------------|-------------|
| `dataset/` | The 188 benchmark screenshots (input) with bounding-box / sensitivity annotations. |
| `RuleBasedClassification/` | Pipeline outputs for the **rule-based** strategy (`ocr`, `compo`, `merge`, `tables`, `redacted`). |
| `GeneralPromptClassification/` | Pipeline outputs for the **generic-prompt** LLM strategy. |
| `SpecificPromptClassification/` | Pipeline outputs for the **specific-prompt** LLM strategy. |
| `EvaluationRuleBased/` | Per-image evaluation JSONs (TP/FP/TN/FN) for the rule-based strategy. |
| `EvaluationGeneralPrompt/` | Per-image evaluation JSONs for the generic-prompt strategy. |
| `EvaluationSpecificPrompt/` | Per-image evaluation JSONs for the specific-prompt strategy. |

Each `Evaluation*` folder holds one JSON per image (188) plus a `_global.json`
with the aggregated confusion matrix and metrics for that strategy.

## Reproducing

From the project root, place the `dataset/` screenshots in `input/` and run each
workflow (see the top-level `README.md`):

```bash
python redact/Basic_workflow.py      # rule-based
python redact/LLM_workflow.py        # generic prompt
python redact/LLM_workflow.py        # specific prompt (main2.txt)
```

Then compare each strategy's classification output against the dataset
annotations to regenerate the `Evaluation*` JSONs and the Table 2 metrics.

## Error analysis

Notes and per-strategy metrics from the benchmark. Overall counts are aggregated
over all 188 dataset images; the `avg_*` values are macro-averages over
per-image scores.

### Generic Prompt (LLM classifier)

**Overall:** tp 2373, fp 247, tn 12577, fn 819

| Metric | Value | | Macro-avg | Value |
|--------|:-----:|-|-----------|:-----:|
| Accuracy | 0.9334 | | avg accuracy | 0.9174 |
| Precision | 0.9057 | | avg precision | 0.9201 |
| Recall | 0.7434 | | avg recall | 0.7441 |
| Specificity | 0.9807 | | avg specificity | 0.9805 |
| F1 | 0.8166 | | | |

**Failure cases:**
- Ambiguous terms flip true/false when the prompt does not define them —
  e.g. `banking10_x`: dates / IBANs / salaries.
- Credit-card numbers separated by spaces are missed when a group lacks context.
  e.g. `banking20_0`: *"The term \"1645\" itself is just a number and is not
  private information. The surrounding word \"7229\" does not suggest that the
  term itself is private."*
- Names work well most of the time; false negatives when a name is also a common
  noun — e.g. `Chat50_3`.
- Ambiguous, prompt-undefined cases — e.g. `ecommerce10_x`: IDs, addresses,
  countries.
- Medical values (not specified in the generic prompt) are not recognized —
  e.g. `health10_x`.
- Missing textual context leads to false negatives — e.g. `Login_v1638`.

### Rule-based classifier

**Overall:** tp 637, fp 162, tn 12661, fn 2554

| Metric | Value | | Macro-avg | Value |
|--------|:-----:|-|-----------|:-----:|
| Accuracy | 0.8304 | | avg accuracy | 0.8185 |
| Precision | 0.7972 | | avg precision | 0.8629 |
| Recall | 0.1996 | | avg recall | 0.2870 |
| Specificity | 0.9874 | | avg specificity | 0.9840 |
| F1 | 0.3193 | | | |

**Failure cases:**
- Names are a problem overall: only caught if added to the keyword list
  manually, which can cause false positives (e.g. surname "Price").
- If a detected column header contains an OCR error that spell-checking cannot
  fix, the column is not redacted — e.g. `banking10_4`.
- Keyword database is incomplete — e.g. `userprofile10_7`.
- Table detection often fails — e.g. `eladmin-master0_7`; IDs get redacted by
  regex instead.
- Unreliable text recognition — e.g. `email20_5`, emails are not recognized.

> These notes cover the generic-prompt and rule-based strategies only; the
> specific-prompt strategy is reported in Table 2 above but not analysed here.
