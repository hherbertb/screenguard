# ScreenGuard

ScreenGuard is a tool for automatically redacting sensitive (private) information from screenshots of web applications and desktop software.

Given a screenshot as input, ScreenGuard detects text and UI components, classifies each detected term for privacy sensitivity, and replaces private terms with a neutral placeholder block. Three classification strategies are supported: rule-based, LLM with a general prompt, and LLM with a specific prompt.

---

## Requirements

- Ubuntu (developed and tested on Ubuntu with Python 3.10)
- Python 3.10
- A CUDA-capable GPU is recommended

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Additional setup

**EasyOCR fine-tuned model** — ScreenGuard uses a fine-tuned EasyOCR recognition model (`best_accuracy`). Install it by following the custom model instructions at:  
https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md

Place `best_accuracy.pth` and `best_accuracy.yaml` in the EasyOCR model directory (typically `~/.EasyOCR/model/`). The model files (`best_accuracy.pth`, `best_accuracy.yaml`, `best_accuracy.py`) are included in this repository.

**ActivityGen component classifier** — The rule-based workflow requires the ResNet component classifier from ActivityGen. Download `classifier_model.pth` from:  
https://zenodo.org/records/13375065

Place it in `activitygen/models/classifier_model.pth`.

**API keys** — For the LLM-based workflows, copy `configs/api_keys.cfg.template` to `configs/api_keys.cfg` and fill in your keys:

```ini
[MAIN]
GEMINI_API_KEY = <your key here>
HUGGINGFACE_API_KEY = <your key here>
```

---

## Usage

Place the screenshots to be redacted in the `input/` folder. Then run one of the three workflows from the project root:

**Rule-based classifier:**
```bash
python redact/Basic_workflow.py
```

**LLM classifier (general prompt):**
```bash
python redact/LLM_workflow.py
```

**LLM classifier (specific prompt):**  
Edit `redact/LLM_workflow.py` to use `main2.txt` as the prompt, or use the dedicated specific-prompt variant.

**Local LLM classifier:**
```bash
python redact/LocalLLM_workflow.py
```

Output is written to a timestamped subfolder under `output/`, containing:
- `redacted/` — redacted screenshots
- `redacted/blocked/` — screenshots with blocked-out regions highlighted
- `redacted/jsons/` — per-image JSON files with classification results

---

## Prompts

The LLM prompts are provided in `configs/prompts/`:

- `main.txt` — general prompt (classifies any private term)
- `main2.txt` — specific prompt (extended guidelines covering names, IBANs, credit card numbers, addresses, medical values, etc.)

---

## Configuration

Detection and component analysis parameters can be adjusted in `configs/main.cfg`. Key settings include gradient thresholds, element area filters, and word/line gap parameters for OCR merging. See the comments in the file for guidance.

---

## Project Structure

```
screenguard/
├── redact/                        # Redaction pipeline
│   ├── Basic_workflow.py          # Entry point: rule-based classifier
│   ├── LLM_workflow.py            # Entry point: LLM classifier (general prompt)
│   ├── LocalLLM_workflow.py       # Entry point: local LLM classifier
│   ├── Redactor.py                # Redaction logic
│   ├── Utils.py
│   ├── classification/
│   │   ├── BasicPrivacyClassifier.py   # Rule-based classification
│   │   ├── LLMPrivacyClassifier.py     # Cloud LLM classification
│   │   ├── LocalLLMClassifier.py       # Local LLM classification
│   │   └── Context.py
│   ├── detection/
│   │   ├── TextDetector.py        # OCR-based text detection (EasyOCR)
│   │   ├── TextDetection.py
│   │   └── TableDetector.py       # Table detection (DETR-based)
│   └── logging/
│       └── ScreenLogger.py
├── activitygen/                   # UI component detection (from ActivityGen)
│   ├── compo_detector/            # Component detection
│   ├── compo_classifier/          # ResNet-based component classification
│   ├── merge/                     # Element merging
│   ├── config/
│   └── models/                    # Place classifier_model.pth here (see setup)
├── configs/
│   ├── main.cfg                   # Detection parameters
│   ├── api_keys.cfg               # API keys (not tracked in git)
│   ├── api_keys.cfg.template      # API key template (copy to api_keys.cfg)
│   ├── prompts/
│   │   ├── main.txt               # General LLM prompt
│   │   └── main2.txt              # Specific LLM prompt
│   └── fonts/
│       └── Arial.ttf
├── data/
│   ├── JiraCelonis/               # Raw Jira screenshots (case study input)
│   └── PII_keywords.txt           # PII keyword list for rule-based classifier
├── evaluation/                    # Evaluation data
│   ├── Quantitative Evaluation/   # Quantitative benchmark
│   │   ├── dataset/               # Screenshots + bounding box annotations
│   │   ├── RuleBasedClassification/
│   │   ├── GeneralPromptClassification/
│   │   ├── SpecificPromptClassification/
│   │   ├── EvaluationRuleBased/
│   │   ├── EvaluationGeneralPrompt/
│   │   └── EvaluationSpecificPrompt/
│   └── Jira/                      # Jira case study (unredacted + redacted)
├── input/                         # Place input screenshots here
├── output/                        # Redaction results (generated at runtime)
├── best_accuracy.pth              # Fine-tuned EasyOCR model weights
├── best_accuracy.yaml             # Fine-tuned EasyOCR model config
├── best_accuracy.py               # Model definition
└── requirements.txt
```

---

## Citation

If you use ScreenGuard in your research, please cite:

```bibtex
@inproceedings{beyel2025screenguard,
  title  = {Redacting Sensitive Information in Screenshots},
  author = {Beyel, Harry H. and van der Aalst, Wil M. P.},
}
```

---

## License

To be added.
