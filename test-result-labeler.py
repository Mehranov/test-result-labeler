#import os
import re
import time
import json
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import os
# %%
# =========================
# CONFIG
# =========================
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

INPUT_FILE  = r"path to your input file. it should have a column named raw_result"
OUTPUT_FILE = r"path to where you want the output file to be created"

MODEL                = "gpt-4o-mini"
BATCH_SIZE           = 50              # main batch size
MINI_RETRY_SIZE      = 5               # mini-batch size for missing items
MAX_RETRIES          = 3               # retries for main batches
MINI_MAX_RETRIES     = 2               # retries for mini-batches
RATE_LIMIT_DELAY_SEC = 2
TIMEOUT_SEC          = 120
MAX_BATCHES_FOR_TEST = None            # set to small int to test; None for full run

if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in your environment.")
client = OpenAI(api_key=API_KEY)
# %%

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are an expert clinical laboratory data analyst. You will classify free-text lab RESULT statements that may or may not be about malaria.

OUTPUT FIELDS (per item)
- id: the same numeric id provided for the line
- raw_result: echo the exact input string
- pos_neg: one of "Positive", "Negative", "Indeterminate", or "Unknown"
- parasitemia_pct: numeric fraction between 0 and 1 if a % parasitemia is explicitly stated (e.g., 3% -> 0.03) or a clear ratio convertible to a fraction (e.g., 1/200 RBC -> 0.005); otherwise null
- interpretation: concise, standardized note (e.g., species, method, workflow status, or reason it’s not malaria-related)

GENERAL PRINCIPLES
A. Base the label on the MAIN diagnostic statement for MALARIA. Ignore generic comments, disclaimers, or advice text.
B. If the text appears to be a general hematology/clinical comment unrelated to malaria (e.g., anemia, blasts, pancytopenia, ITP, sepsis, reactive lymphocytes, marrow recovery) and contains no malaria terms, label:
   - pos_neg = "Unknown"; parasitemia_pct = null; interpretation = "Not malaria-related".
C. NEGATIVE language (extend beyond exact examples). Label as Negative when any clear negative phrasing is present, including variants such as:
   - “negative for malaria/plasmodium/malaria antigen”
   - “not detected”
   - “no (malaria parasites|malarial forms|haemoparasites) (seen|detected|identified|observed)”
   - method-specific: “LAMP negative”, “RDT/rapid test negative”, “thin smear negative”, “thick smear negative”
   - When Negative, parasitemia_pct = null; interpretation may include method (e.g., “LAMP”).
D. POSITIVE language (extend beyond exact examples). Label as Positive when malaria is indicated by phrases such as:
   - “(malaria|plasmodium) (detected|identified|present|seen)”
   - species names: “falciparum”, “vivax”, “malariae”, “ovale”, “knowlesi”, or “mixed infection”
   - method-specific: “LAMP positive”, “RDT/rapid test positive”, “smear positive”
   - references to parasitemia (%, rings/gametocytes/trophozoites present)
   - If LAMP is positive and other tests are pending, still label Positive; interpretation should note “LAMP positive; smear pending” (or similar).
E. INDETERMINATE / WORKFLOW language. Use Indeterminate when testing is referenced but outcome is unclear or pending, including:
   - “inconclusive”, “equivocal”, “invalid”
   - “repeat recommended”, “recollect”
   - “forwarded for LAMP”, “sent for confirmatory testing”
   - “smear pending / to follow”, “report to follow”
   - pre-analytical issues: “insufficient”, “hemolyzed”, “inadequate specimen”
   - administrative cancellations: e.g., “previous NEG <7 days — cancel” (no new result)
   - interpretation should briefly explain (e.g., “Awaiting LAMP”, “Smear pending”, “Repeat canceled (<7 d)”).
F. Ignore boilerplate disclaimers (e.g., “PCR can remain positive after treatment”) when a clear result is stated elsewhere.

VCH PATHWAY AWARENESS (to inform C/D/E above)
- Central sites (LAMP available): LAMP performed first.
  • “LAMP negative — no further testing” → Negative (interpretation “LAMP”).
  • “LAMP positive — smear/rapid to follow” → Positive (interpretation “LAMP positive; smear pending”).
- HUB sites (no LAMP): rapid + thin smear screen first.
  • “Screen negative; forwarded for LAMP” → Indeterminate until LAMP outcome appears.
  • “Screen positive; stain slide and report” → Positive if explicit positivity is stated; if only “to follow/pending” without an outcome, Indeterminate.
- Repeat policy examples:
  • “Previous NEG <7 days — cancel” → Indeterminate (no new result).
  • “Previous POS <7 days — smears only” → classify per actual smear wording; “smear pending” remains Indeterminate.

DECISION ORDER (apply first matching)
1) Not malaria-related (no malaria terms; general hematology) → Unknown / “Not malaria-related”
2) Explicit Negative (any negative pattern) → Negative
3) Explicit Positive (any positive pattern) → Positive
4) Workflow/pending/unclear → Indeterminate
5) Otherwise → Unknown

MALARIA TERMS (non-exhaustive, for context)
“malaria”, “plasmodium”, “parasitemia”, “thick smear”, “thin smear”, “LAMP”, “rapid test”, “RDT”, “antigen”, “gametocytes”, “trophozoites”, species (“falciparum”, “vivax”, “malariae”, “ovale”, “knowlesi”), “mixed infection”.

OUTPUT FORMAT
- Respond ONLY with a JSON array of objects whose length exactly equals the number of inputs.
- Each object MUST include: id, raw_result, pos_neg, parasitemia_pct, interpretation.
- Use the exact id provided for each line; do not add extra fields or commentary.

"""
# %%
# =========================
# Helpers
# =========================
MALARIA_CUES = re.compile(
    r"(malaria|plasmod|parasitemia|thin smear|thick smear|lamp|rdt|rapid test|gametocyte|trophozoite|falciparum|vivax|malariae|ovale|knowlesi)",
    re.I,
)

def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text

def build_user_prompt_with_ids(batch_with_ids):
    """
    batch_with_ids: list of dicts like {"id": <int>, "text": <raw_result>}
    """
    rows_text = "\n".join([f'{item["id"]}. {item["text"]}' for item in batch_with_ids])
    return {
        "role": "user",
        "content": (
            "You will receive numbered result lines. For EACH line, return ONE JSON object with keys "
            'id, raw_result, pos_neg, parasitemia_pct, interpretation. Use the same "id" you received.\n\n'
            "Results:\n" + rows_text
        ),
    }

def call_model(batch_with_ids, model, timeout_s):
    """Single API call → parse JSON → return list[dict] (may be short)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            build_user_prompt_with_ids(batch_with_ids),
        ],
        temperature=0,
        timeout=timeout_s,
    )
    content = clean_json_text(resp.choices[0].message.content)
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("Response is not a JSON array.")
    return parsed

def normalize_records_from_model(parsed, id_to_text, expected_ids):
    """Return dict id->record for all items present; ignore extras, coerce keys."""
    out_by_id = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        _id = item.get("id", None)
        if _id in expected_ids:
            out_by_id[_id] = {
                "raw_result": id_to_text[_id],  # force exact echo
                "pos_neg": item.get("pos_neg", "Unknown"),
                "parasitemia_pct": item.get("parasitemia_pct", None),
                "interpretation": item.get("interpretation", None),
            }
    return out_by_id

def classify_batch_with_auto_retry(batch_texts, model, timeout_s, max_retries, mini_retry_size, mini_max_retries):
    """
    Main batch (list[str]) with per-batch IDs; if some IDs are missing, retry only missing
    in tiny mini-batches. Returns list[dict] length == len(batch_texts), aligned by original order.
    """
    # Assign local IDs for this batch
    ids = list(range(1, len(batch_texts) + 1))
    id_to_text = {i: t for i, t in zip(ids, batch_texts)}
    expected_ids = set(ids)

    # 1) Try the full batch with retries
    out_by_id = {}
    for attempt in range(1, max_retries + 1):
        try:
            parsed = call_model([{"id": i, "text": id_to_text[i]} for i in ids], model, timeout_s)
            out_by_id = normalize_records_from_model(parsed, id_to_text, expected_ids)
            # If we got everything, stop
            if len(out_by_id) == len(ids):
                break
        except Exception as e:
            wait = min(2 ** attempt + random.uniform(0, 1), 15)
            print(f"Batch call failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    # 2) If missing, mini-retry only missing IDs in small chunks
    missing_ids = [i for i in ids if i not in out_by_id]
    if missing_ids:
        # chunk missing into mini-batches
        for k in range(0, len(missing_ids), mini_retry_size):
            sub_ids = missing_ids[k:k+mini_retry_size]
            sub_pack = [{"id": sid, "text": id_to_text[sid]} for sid in sub_ids]

            got_sub = False
            for attempt in range(1, mini_max_retries + 1):
                try:
                    parsed = call_model(sub_pack, model, timeout_s)
                    sub_map = normalize_records_from_model(parsed, id_to_text, set(sub_ids))
                    # Fill what we got
                    out_by_id.update(sub_map)
                    got_sub = True
                    # If still some of these sub_ids missing, keep retrying; else break
                    if len(sub_map) == len(sub_ids):
                        break
                except Exception as e:
                    wait = min(2 ** attempt + random.uniform(0, 1), 15)
                    print(f"Mini-batch failed (attempt {attempt}/{mini_max_retries}): {e}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)
            # After mini retries, mark any still missing as fallback
            still_missing = [sid for sid in sub_ids if sid not in out_by_id]
            for sid in still_missing:
                out_by_id[sid] = {
                    "raw_result": id_to_text[sid],
                    "pos_neg": "could not be processed",
                    "parasitemia_pct": None,
                    "interpretation": None,
                }

    # 3) Return in original order
    return [out_by_id[i] if i in out_by_id else {
                "raw_result": id_to_text[i],
                "pos_neg": "could not be processed",
                "parasitemia_pct": None,
                "interpretation": None,
            } for i in ids]
# %%
# =========================
# LOAD INPUT & RESUME
# =========================
df_input = pd.read_csv(INPUT_FILE)
if "raw_result" not in df_input.columns:
    raise ValueError("Input CSV must have a column named 'raw_result'")
df_input["raw_result"] = df_input["raw_result"].astype(str).str.strip()

# Resume: skip already-processed raw_result values
done_set = set()
if Path(OUTPUT_FILE).exists() and Path(OUTPUT_FILE).stat().st_size > 0:
    df_done_existing = pd.read_csv(OUTPUT_FILE)
    if "raw_result" in df_done_existing.columns:
        done_set = set(df_done_existing["raw_result"].astype(str))
        print(f"Resuming: {len(done_set)} already processed")

df_todo = df_input[~df_input["raw_result"].isin(done_set)].reset_index(drop=True)
print(f"To process now: {len(df_todo)} rows")
results = df_todo["raw_result"].dropna().tolist()
total_rows = len(results)
# %%
# =========================
# LOCAL PRE-FILTER
# =========================
local_outputs = []
api_candidates = []
for r in results:
    if MALARIA_CUES.search(r or ""):
        api_candidates.append(r)
    else:
        local_outputs.append({
            "raw_result": r,
            "pos_neg": "Unknown",
            "parasitemia_pct": None,
            "interpretation": "Not malaria-related",
        })

print(f"Local (non-malaria) labeled: {len(local_outputs)} | To API: {len(api_candidates)}")

# Save local outputs immediately (append or create)
if local_outputs:
    df_local = pd.DataFrame(local_outputs)
    write_header = not Path(OUTPUT_FILE).exists() or Path(OUTPUT_FILE).stat().st_size == 0
    df_local.to_csv(OUTPUT_FILE, index=False, mode="a", header=write_header)

# Remove any that were just written from api_candidates (safety)
already_written = done_set.union({x["raw_result"] for x in local_outputs})
api_todo = [r for r in api_candidates if r not in already_written]
# %%
# =========================
# OPTIONAL: preflight
# =========================
try:
    _ = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Respond ONLY with JSON array like [{\"id\":1}]."},
            {"role": "user", "content": "ping"},
        ],
        temperature=0,
        timeout=20,
    )
    print("Preflight OK.")
except Exception as e:
    print(f"Preflight failed (continuing): {e}")
# %%
# =========================
# MAIN LOOP with mini-retry for missing IDs
# =========================
idx = 0
start_time = datetime.now()
total_api_rows = len(api_todo)

if total_api_rows == 0:
    print("Nothing to send to API. All rows labeled locally or previously processed.")
else:
    print(f"Sending {total_api_rows} rows to API in batches of {BATCH_SIZE} (mini-retry {MINI_RETRY_SIZE}).")

batches_sent = 0
while idx < total_api_rows:
    if MAX_BATCHES_FOR_TEST is not None and batches_sent >= MAX_BATCHES_FOR_TEST:
        print(f"Stopping early after {batches_sent} API batches (test mode).")
        break

    batch = api_todo[idx : idx + BATCH_SIZE]
    if not batch:
        break

    out = classify_batch_with_auto_retry(
        batch_texts=batch,
        model=MODEL,
        timeout_s=TIMEOUT_SEC,
        max_retries=MAX_RETRIES,
        mini_retry_size=MINI_RETRY_SIZE,
        mini_max_retries=MINI_MAX_RETRIES,
    )

    # Save immediately (append)
    df_batch = pd.DataFrame(out)
    write_header = not Path(OUTPUT_FILE).exists() or Path(OUTPUT_FILE).stat().st_size == 0
    df_batch.to_csv(OUTPUT_FILE, index=False, mode="a", header=write_header)

    # Move pointer
    idx += len(batch)
    batches_sent += 1

    # Progress / ETA on API portion
    elapsed = (datetime.now() - start_time).total_seconds()
    done_api = idx
    percent_api = (done_api / total_api_rows) * 100 if total_api_rows else 100.0
    avg_batch_time = elapsed / max(1, batches_sent)
    remaining_batches = (total_api_rows - done_api + BATCH_SIZE - 1) // BATCH_SIZE
    eta_min = (remaining_batches * avg_batch_time) / 60
    print(f"API batch {batches_sent} — {done_api}/{total_api_rows} ({percent_api:.1f}%) "
          f"— wrote {len(out)} rows — ETA ~ {eta_min:.1f} min")

    time.sleep(RATE_LIMIT_DELAY_SEC)

print(f"✅ Done (this session). Output saved to: {OUTPUT_FILE}")
