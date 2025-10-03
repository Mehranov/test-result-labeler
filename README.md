# test-result-labeler
Label free-text lab results (Positive/Negative/Indeterminate/Unknown/Test not performed) using openai API.
# Test Result Labeler (Malaria)

This script classifies **free-text lab result statements** into one of:  
**Positive**, **Negative**, **Indeterminate**, **Unknown**, or **Test not performed**,  
and extracts **parasitemia %** if explicitly stated.

It combines simple **local heuristics** (regex cues) with **LLM-based labeling** using the OpenAI API.

---

## Features
- Regex pre-filter for obvious non-malaria results (fast + local)
- Batched LLM classification with retries and mini-batch recovery
- Resume-safe: appends results to the output CSV and skips previously processed rows
- Extracts parasitemia % or ratios when present
- Clear JSON-based system prompt for auditable outputs

---

## Setup

1. **Clone repo & create environment**
   ```bash
   git clone https://github.com/<YOUR_USER>/<YOUR_REPO>.git
   cd <YOUR_REPO>
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt   # or install pandas openai
2. Set your API key
  # macOS/Linux
    export OPENAI_API_KEY="sk-..."
  # Windows PowerShell
    setx OPENAI_API_KEY "sk-..."
## Usage
1. Place your input CSV with a column named raw_result somewhere accessible.
2. Edit the script config section (INPUT_FILE, OUTPUT_FILE, MODEL, etc.).
3. Run:
   python test_result_labeler.py

## EXAMPLE:
Input Example (CSV)

raw_result

"Malaria antigen detected"

"Negative for Plasmodium"

"Specimen rejected: hemolyzed"

"Report to follow"

"Reactive lymphocytes noted"

...

Output Example (CSV)

raw_result | pos_neg | parasitemia_pct | interpretation

"Malaria antigen detected" ,Positive,,Detected malaria antigen

"Negative for Plasmodium",Negative,,Not detected

"Specimen rejected: hemolyzed",Test not performed,,Specimen rejected

"Report to follow",Indeterminate,,Smear pending

"Reactive lymphocytes noted",Unknown,,Not malaria-related

Configuration

Adjust at the top of the script:

Paths: INPUT_FILE, OUTPUT_FILE

Model: e.g., gpt-4o-mini ---> price per token varies for each engine. pick an engine that fits your need. read about each engine specialty in openai website

Batching: BATCH_SIZE, MAX_RETRIES, RATE_LIMIT_DELAY_SEC

Regex cues: extend MALARIA_CUES as needed

Notes
Use only synthetic or de-identified data in this repo.
Add real data paths to .gitignore to avoid accidental commits.
If a line cannot be processed after retries, it is marked could not be processed.
