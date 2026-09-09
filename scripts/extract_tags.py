"""
One script, three sources, one output: extracts project-level tags for
GTEx, TCGA, and SRA from recount3 metadata already downloaded by
download_metadata.py, and writes a single unified tags.jsonl.

    metadata/<data_source>/<project>/<data_source>.<name>.<project>.MD.gz

GTEx and TCGA use structured field lookups (no LLM). SRA uses an LLM,
since its sample attributes are free-text and inconsistent across studies.
Every row in the output shares one schema:

    {"project": ..., "data_source": "gtex"|"tcga"|"sra", "n_samples": ...,
     "tags": [{"category", "value", "confidence", "source"}, ...],
     "raw_evidence": "..."}

USAGE
-----
    # inspect actual column names on your downloaded data first
    python extract_tags.py --inspect gtex
    python extract_tags.py --inspect tcga

    # pilot the SRA/LLM step before a full run -- GTEx/TCGA still run in
    # full alongside it, since they're cheap (no LLM calls involved)
    python extract_tags.py --sra-limit 20 --provider anthropic

    # full run, all three sources, writes tags.jsonl
    python extract_tags.py --provider anthropic

    # re-run later: GTEx/TCGA recompute fresh (cheap), SRA resumes from
    # its cache and skips projects already tagged
    python extract_tags.py --provider anthropic
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from llm_providers import LLMProvider, DEFAULT_MODELS

METADATA_DIR = Path("metadata")
OUT_JSONL = Path("tags.jsonl")
SRA_CACHE_JSONL = Path("sra_project_tags.jsonl")  # resumable cache for the LLM step only

MAX_WORKERS = 8
MAX_SAMPLES_PER_PROJECT = 500
MAX_VALUES_PER_KEY = 10
MAX_KEYS_SHOWN = 40

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Shared metadata helpers
# ---------------------------------------------------------------------------
def load_md(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)


def sample_md_path(data_source: str, project: str) -> Path:
    return METADATA_DIR / data_source / project / f"{data_source}.{data_source}.{project}.MD.gz"


def mode_or_none(series: pd.Series):
    series = series.dropna()
    return None if series.empty else series.mode().iat[0]


def find_col(columns, *keywords, exclude=()):
    for c in columns:
        cl = c.lower()
        if all(k in cl for k in keywords) and not any(e in cl for e in exclude):
            return c
    return None


def inspect(data_source: str):
    base = METADATA_DIR / data_source
    if not base.exists():
        print(f"No downloaded metadata found at {base}")
        return
    projects = [p.name for p in base.iterdir() if p.is_dir()]
    if not projects:
        print(f"No project folders found under {base}")
        return
    project = projects[0]
    path = sample_md_path(data_source, project)
    if not path.exists():
        print(f"Expected sample metadata file not found: {path}")
        return

    df = load_md(path)
    print(f"Inspecting {data_source} project '{project}' ({len(df)} samples)\n")
    print(f"{'column':50s} example value")
    print("-" * 90)
    for c in df.columns:
        example = df[c].dropna().iloc[0] if df[c].notna().any() else ""
        print(f"{c[:50]:50s} {str(example)[:60]}")


# ---------------------------------------------------------------------------
# GTEx / TCGA: structured field lookups, no LLM.
# GTEx (gtex.smts) and TCGA (project code) are BOTH closed, controlled
# vocabularies -- ~30 and exactly 33 possible values respectively -- so
# tissue_category is classified via explicit tables, with a keyword
# fallback (flagged confidence="medium") only for unrecognized values.
# ---------------------------------------------------------------------------
GTEX_TISSUE_CATEGORY = {
    "blood": "blood",
    "bone marrow": "blood",
    "spleen": "lymphoid_immune",
}
TCGA_PROJECT_CATEGORY = {
    "LAML": "blood",
    "DLBC": "blood",
    "THYM": "lymphoid_immune",
}
FALLBACK_BLOOD_TERMS = [
    "leukemia", "leukaemia", "lymphoma", "myeloma", "myelodysplastic",
    "hematopoietic", "haematopoietic", "pbmc", "peripheral blood",
    "whole blood", "bone marrow",
]
FALLBACK_LYMPHOID_TERMS = ["lymph node", "thymus", "tonsil", "spleen"]


def classify_text_fallback(*texts):
    blob = " ".join(t.lower() for t in texts if isinstance(t, str))
    if any(term in blob for term in FALLBACK_BLOOD_TERMS):
        return "blood", True
    if any(term in blob for term in FALLBACK_LYMPHOID_TERMS):
        return "lymphoid_immune", True
    return "solid", True


def classify_gtex(tissue_major, tissue_detail):
    key = (tissue_major or "").strip().lower()
    if key in GTEX_TISSUE_CATEGORY:
        return GTEX_TISSUE_CATEGORY[key], False
    return classify_text_fallback(tissue_major, tissue_detail)


def classify_tcga(project_code, disease, site):
    key = project_code.strip().upper()
    if key in TCGA_PROJECT_CATEGORY:
        return TCGA_PROJECT_CATEGORY[key], False
    return classify_text_fallback(disease, site)


def summarize_gtex_project(project: str):
    path = sample_md_path("gtex", project)
    if not path.exists():
        return None
    df = load_md(path)

    col_smts = next((c for c in df.columns if c.lower().endswith("smts")), None)
    col_smtsd = next((c for c in df.columns if c.lower().endswith("smtsd")), None)
    tissue_major = mode_or_none(df[col_smts]) if col_smts else None
    tissue_detail = mode_or_none(df[col_smtsd]) if col_smtsd else None

    category, inferred = classify_gtex(tissue_major, tissue_detail)

    tags = []
    tissue_value = tissue_detail or tissue_major
    if tissue_value:
        tags.append({"category": "tissue", "value": tissue_value.lower(),
                     "confidence": "high", "source": "structured_field"})
    tags.append({"category": "tissue_category", "value": category,
                 "confidence": "medium" if inferred else "high",
                 "source": "structured_field"})

    return {
        "project": project, "data_source": "gtex", "n_samples": len(df),
        "tags": tags,
        "raw_evidence": f"gtex.smts={tissue_major}; gtex.smtsd={tissue_detail}",
    }


def summarize_tcga_project(project: str):
    path = sample_md_path("tcga", project)
    if not path.exists():
        return None
    df = load_md(path)
    cols = df.columns

    col_disease = (
        find_col(cols, "project", "name")
        or find_col(cols, "disease_type")
        or find_col(cols, "primary_diagnosis")
    )
    col_site = (
        find_col(cols, "primary_site")
        or find_col(cols, "tissue_source_site", "project")
        or find_col(cols, "tumor_tissue_site")
    )
    col_sample_type = find_col(cols, "sample_type")

    disease = mode_or_none(df[col_disease]) if col_disease else None
    site = mode_or_none(df[col_site]) if col_site else None
    sample_type_counts = df[col_sample_type].value_counts().to_dict() if col_sample_type else {}

    category, inferred = classify_tcga(project, disease, site)

    tags = []
    if disease:
        tags.append({"category": "disease", "value": disease.lower(),
                     "confidence": "high", "source": "structured_field"})
    if site:
        tags.append({"category": "tissue", "value": site.lower(),
                     "confidence": "high", "source": "structured_field"})
    tags.append({"category": "tissue_category", "value": category,
                 "confidence": "medium" if inferred else "high",
                 "source": "structured_field"})

    sample_type_str = "; ".join(f"{k}={v}" for k, v in sample_type_counts.items())
    return {
        "project": project, "data_source": "tcga", "n_samples": len(df),
        "tags": tags,
        "raw_evidence": (
            f"gdc_cases.project.name={disease}; primary_site={site}; "
            f"sample_type_counts=({sample_type_str})"
        ),
    }


def run_structured_source(data_source: str, fn):
    base = METADATA_DIR / data_source
    if not base.exists():
        print(f"Skipping {data_source}: {base} not found", file=sys.stderr)
        return []
    projects = sorted(p.name for p in base.iterdir() if p.is_dir())
    print(f"Summarizing {len(projects)} {data_source} projects...")
    rows = []
    for project in projects:
        try:
            row = fn(project)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"  Failed on {data_source}/{project}: {e}", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# SRA: LLM-based extraction (the only source that needs one).
# ---------------------------------------------------------------------------
ALLOWED_CATEGORIES = {
    "tissue", "disease", "pathogen", "specimen", "cell_line", "tissue_category", "other",
}
ALLOWED_TISSUE_CATEGORY_VALUES = {"blood", "lymphoid_immune", "solid", "unknown"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
MAX_OTHER_TAGS = 2

SYSTEM_PROMPT = """You are annotating an RNA-seq dataset (an SRA study) for a \
domain expert who will review your tags before they're used for filtering. \
Base every tag ONLY on the metadata provided -- never guess or use outside \
knowledge about what a study "probably" is about.

Return ONLY a JSON array (no prose, no markdown fences) of tag objects:
[{"category": "<one of: tissue, disease, pathogen, specimen, cell_line, tissue_category, other>",
  "value": "<short lowercase phrase>",
  "confidence": "<high|medium|low>"}]

Category definitions:
- "tissue": the organ/anatomical tissue sampled (e.g. "liver", "lung").
- "specimen": the physical sample material when it's not simply an organ \
name -- e.g. "sputum", "cerebrospinal fluid", "urine", "biopsy", "pbmc".
- "cell_line": the name of a cell line, if one is used (e.g. "k562"), \
regardless of what tissue_category you assign it.
- "disease": a stated disease, condition, or infection.
- "pathogen": an infectious organism, only if explicitly named.
- "tissue_category": value must be exactly one of "blood", "lymphoid_immune", \
"solid", or "unknown". Use "blood" for whole blood, PBMC, serum, plasma, bone \
marrow aspirate, or a cell line KNOWN to be blood/leukemia derived (e.g. K562, \
Jurkat, HL-60). Use "lymphoid_immune" for spleen, lymph node, thymus, tonsil. \
Use "solid" for any other organ/tissue. Use "unknown" if the metadata gives \
no tissue signal at all.
- "other": a specific, notable clinical or experimental detail that doesn't \
fit any category above but would matter to someone searching for this \
dataset -- e.g. drug-resistance status, co-infection, disease stage/subtype, \
a demographic tightly tied to the condition. Use sparingly (max 2 per \
project) and only for something concrete and traceable to the text, never a \
vague catch-all summary.

Rules:
- Include a "disease" tag only if the metadata states a disease, condition, \
or infection -- do not infer disease from a cell line name alone unless the \
metadata itself names a condition.
- Include a "pathogen" tag only if an infectious organism is explicitly named.
- confidence "high" = the metadata states it directly; "medium" = reasonably \
implied by close terminology; "low" = a weak or indirect signal worth a \
human double-check.
- Emit at most 8 tags total (at most 2 of them "other"). If the metadata \
gives no usable signal, return [].
- Never invent values not traceable to the provided text."""


def parse_sra_attributes(attr_string):
    if not isinstance(attr_string, str):
        return {}
    pairs = {}
    for chunk in re.split(r";;|\|", attr_string):
        if ":" in chunk:
            k, v = chunk.split(":", 1)
            k, v = k.strip().lower(), v.strip()
            if k and v:
                pairs[k] = v
    return pairs


def build_project_profile(project: str):
    spath = sample_md_path("sra", project)
    if not spath.exists():
        return None
    df = load_md(spath)
    n_samples = len(df)

    attr_col = find_col(df.columns, "attribute")
    key_values = {}
    if attr_col:
        for raw in df[attr_col].dropna().head(MAX_SAMPLES_PER_PROJECT):
            for k, v in parse_sra_attributes(raw).items():
                key_values.setdefault(k, set()).add(v)

    text_cols = [c for c in df.columns if any(k in c.lower() for k in
                 ("title", "summary", "design", "source_name"))]
    text_snippets = {}
    for c in text_cols:
        vals = df[c].dropna().unique().tolist()[:MAX_VALUES_PER_KEY]
        if vals:
            text_snippets[c] = vals

    lines = [f"Project: {project}", f"N samples: {n_samples}"]
    if text_snippets:
        lines.append("\nStudy-level text fields:")
        for k, vals in text_snippets.items():
            lines.append(f"- {k}: {' | '.join(str(v) for v in vals)}")
    if key_values:
        lines.append("\nSample attribute keys (unique values seen):")
        for k, vals in list(key_values.items())[:MAX_KEYS_SHOWN]:
            lines.append(f"- {k}: {', '.join(list(vals)[:MAX_VALUES_PER_KEY])}")

    return "\n".join(lines), n_samples


def call_llm(client: LLMProvider, profile_text: str):
    text = client.complete(SYSTEM_PROMPT, profile_text, max_tokens=500)
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    tags = json.loads(text)

    cleaned = []
    other_count = 0
    for t in tags:
        cat = str(t.get("category", "")).strip().lower()
        val = str(t.get("value", "")).strip().lower()
        conf = str(t.get("confidence", "")).strip().lower()
        if cat not in ALLOWED_CATEGORIES or not val or conf not in ALLOWED_CONFIDENCE:
            continue
        if cat == "tissue_category" and val not in ALLOWED_TISSUE_CATEGORY_VALUES:
            continue
        if cat == "other":
            if other_count >= MAX_OTHER_TAGS:
                continue
            other_count += 1
        cleaned.append({"category": cat, "value": val, "confidence": conf, "source": "llm_inferred"})
    return cleaned


def process_sra_project(client, project, retries=3):
    built = build_project_profile(project)
    if built is None:
        return None
    profile_text, n_samples = built

    for attempt in range(retries):
        try:
            tags = call_llm(client, profile_text)
            return {
                "project": project, "data_source": "sra", "n_samples": n_samples,
                "tags": tags, "raw_evidence": profile_text,
            }
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Failed {project} after {retries} attempts: {e}", file=sys.stderr)
                return None


def count_sra_samples(project: str):
    """Cheap row count (no full pandas parse) so we can rank/filter SRA
    projects by size before deciding which ones to send to the LLM."""
    path = sample_md_path("sra", project)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt") as f:
            return sum(1 for _ in f) - 1  # minus header row
    except Exception:
        return None


def load_sra_cache():
    if not SRA_CACHE_JSONL.exists():
        return {}
    cache = {}
    with open(SRA_CACHE_JSONL) as f:
        for line in f:
            try:
                row = json.loads(line)
                cache[row["project"]] = row
            except Exception:
                continue
    return cache


def append_sra_cache(row):
    with _write_lock:
        with open(SRA_CACHE_JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")


def run_sra_source(provider: str, model: str, workers: int, sra_limit: int | None,
                    min_samples: int = 0):
    sra_dir = METADATA_DIR / "sra"
    if not sra_dir.exists():
        print(f"Skipping sra: {sra_dir} not found", file=sys.stderr)
        return []

    all_projects = sorted(p.name for p in sra_dir.iterdir() if p.is_dir())
    cache = load_sra_cache()
    uncached = [p for p in all_projects if p not in cache]

    # Prioritize bigger, more substantial datasets: count samples for every
    # not-yet-tagged project, drop anything under min_samples, and sort the
    # rest largest-first -- so --sra-limit (if used) spends its LLM budget
    # on the most useful studies first instead of whatever came first
    # alphabetically.
    counted = []
    for p in uncached:
        n = count_sra_samples(p)
        if n is not None and n >= min_samples:
            counted.append((p, n))
    counted.sort(key=lambda pair: pair[1], reverse=True)
    todo = [p for p, _ in counted]
    if sra_limit is not None:
        todo = todo[:sra_limit]  # 0 means "tag no new SRA projects this run"

    client = LLMProvider(provider, model)
    print(f"{len(all_projects)} total SRA projects, {len(cache)} already tagged "
          f"(cached in {SRA_CACHE_JSONL}), {len(uncached) - len(counted)} skipped "
          f"(under {min_samples} samples or unreadable), {len(todo)} to process "
          f"this run (largest first), using {client}.")

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_sra_project, client, p): p for p in todo}
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    append_sra_cache(result)
                    cache[result["project"]] = result

    # tags.jsonl always reflects everything tagged so far (this run's new
    # projects plus anything cached from earlier runs) -- --sra-limit only
    # controls how many NEW projects get sent to the LLM this run, it does
    # not shrink what ends up in the merged output.
    return [cache[p] for p in all_projects if p in cache]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", choices=["gtex", "tcga"], default=None,
                         help="Print columns + example values for one downloaded project, then exit.")
    parser.add_argument("--sources", default="gtex,tcga,sra",
                         help="Comma-separated subset of sources to run, e.g. 'sra' for an SRA-only pass.")
    parser.add_argument("--sra-limit", type=int, default=None, dest="sra_limit",
                         help="Only send the first N not-yet-cached SRA projects to the LLM this run "
                              "(for a pilot). Does not affect GTEx/TCGA, and does not shrink the "
                              "final output -- tags.jsonl always includes every SRA project cached "
                              "so far, plus GTEx/TCGA in full.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--min-samples", type=int, default=100, dest="min_samples",
                         help="Never tag (or keep in the output) any project -- from any source "
                              "-- with fewer than this many samples. For SRA this also means "
                              "smaller studies are never sent to the LLM at all (saves cost); "
                              "remaining SRA projects are processed largest-first. Set to 0 to "
                              "disable this filter entirely.")
    parser.add_argument("--provider", choices=list(DEFAULT_MODELS), default="anthropic",
                         help="LLM provider for the SRA step.")
    parser.add_argument("--model", default=None,
                         help=f"Override the default model. Defaults: {DEFAULT_MODELS}")
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    rows = []

    if "gtex" in sources:
        rows += run_structured_source("gtex", summarize_gtex_project)
    if "tcga" in sources:
        rows += run_structured_source("tcga", summarize_tcga_project)
    if "sra" in sources:
        rows += run_sra_source(args.provider, args.model, args.workers, args.sra_limit,
                                args.min_samples)

    # Hard rule (any source): never keep a tag record for a dataset smaller
    # than --min-samples. SRA already skips tagging small studies before
    # spending on an LLM call; this also drops any small GTEx/TCGA project
    # (e.g. a niche tissue with few samples) and any small SRA project left
    # over from an earlier run with a looser/no filter.
    if args.min_samples:
        before = len(rows)
        rows = [r for r in rows if r["n_samples"] >= args.min_samples]
        dropped = before - len(rows)
        if dropped:
            print(f"Dropped {dropped} project(s) with fewer than {args.min_samples} samples.")

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} total project tag records to {args.out}")
    by_source = Counter(r["data_source"] for r in rows)
    for src in sorted(by_source):
        print(f"  {src}: {by_source[src]}")

    tissue_cat_tally = Counter(
        t["value"] for r in rows for t in r["tags"] if t["category"] == "tissue_category"
    )
    if tissue_cat_tally:
        print("\ntissue_category breakdown:")
        for value, count in tissue_cat_tally.most_common():
            print(f"  {value}: {count}")


if __name__ == "__main__":
    main()
