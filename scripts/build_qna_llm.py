"""
HISTORICAL / DO NOT RE-RUN: this generated the one-time LLM pilot that
produced qna_pilot.yaml (originally 5 each GTEx/TCGA/SRA, n_samples>=100,
later extended to all 75 datasets in a follow-up run).

Everything downstream (the app's build step) now runs off already-extracted
static data (enrichment/tags.jsonl + samples/samples.jsonl + the frozen
qna_pilot.yaml). This file is kept only as a record of how that pilot was
produced.

llm_providers.py now lives alongside this script in scripts/ (copied in,
read-only, from the original repo -- not touched there).

USAGE (historical reference only)
----------------------------------
    cd scripts
    OPENAI_API_KEY=... python3 build_qna_llm.py
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ROOT = HERE.parent
ENRICHED = ROOT / "enrichment"
SAMPLES_DIR = ROOT / "samples"
sys.path.insert(0, str(HERE))
from llm_providers import LLMProvider  # noqa: E402  -- now sits next to this script in scripts/

TAGS_JSONL = ENRICHED / "tags.jsonl"
SAMPLES_JSONL = SAMPLES_DIR / "samples.jsonl"
OUT_YAML = ENRICHED / "qna_pilot.yaml"

# Short labels -- shown to the user, kept as table row headers.
QUESTIONS = [
    "Sample/source type?",
    "Sequencing/assay type?",
    "Organism/species?",
    "Disease/condition/state?",
    "Experimental groups?",
    "Study population?",
    "Experimental design?",
    "Sample count & distribution?",
    "Sample-level characteristics?",
    "Concise summary?",
]

# Full instructions -- sent to the LLM for context, never shown in the UI.
QUESTION_DETAILS = [
    "What is the biological sample/source type? Identify whether the samples are Blood, Tissue, Saliva/Oral, Nasal/Respiratory, Stool/Feces, Urine, Bone Marrow, Cell Culture/Cell Line, etc. If applicable, identify the subtype (e.g., Blood -> PBMC, Whole Blood).",
    "What is the sequencing/assay type? Determine whether this is Bulk RNA-seq, single-cell RNA-seq, spatial RNA-seq, small RNA-seq, total RNA-seq, etc.",
    "What organism/species are the samples from? Identify Human, Mouse, Rat, etc.",
    "What disease, condition, or biological state is being studied? Identify the disease/condition and whether samples represent healthy, diseased, treated, infected, etc.",
    "What are the main experimental groups or sample groups? Identify groups such as case/control, treated/untreated, disease subtypes, stimulation conditions, etc.",
    "What is the study population? Identify relevant information such as adult/pediatric, sex, age range, patient/healthy volunteer, or other population characteristics when available.",
    "What is the experimental design? Determine whether the study is case-control, treatment-response, longitudinal, paired/unpaired, stimulated/unstimulated, time-course, etc.",
    "How many samples/subjects are represented, and how are they distributed across groups? Summarize the total number of samples and, where metadata allows, the approximate/group-wise distribution.",
    "What are the important sample-level characteristics or variations? Use the 5 sampled individual rows to identify recurring or meaningful characteristics such as tissue subtype, treatment, disease status, time point, replicate, collection method, or other sample attributes.",
    "What is the most useful concise description of this dataset for a biologist? Produce a short summary combining the sample source + organism + RNA/assay type + disease/condition + experimental design + population, while clearly stating any information that cannot be determined from the metadata.",
]

SYSTEM_PROMPT = f"""You are a genomics data curator. You will be given metadata for one \
RNA-seq dataset (project-level fields plus 5 individual sample rows). Answer \
the following 10 questions about it, in order, based ONLY on the metadata \
given -- do not invent facts. If something cannot be determined from the \
metadata, say so briefly rather than guessing.

Keep every answer to ONE short sentence, plain and easy to scan in a table \
-- a biologist should be able to read all 10 in a few seconds. Do not repeat \
the question back, do not hedge with multiple clauses, do not list every \
minor caveat -- state the single most important fact plainly. Aim for under \
20 words per answer.

Questions:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(QUESTION_DETAILS))}

Return ONLY a JSON array of exactly 10 short strings, the answers in order, \
no other text, no markdown fences."""


def load_tags():
    rows = []
    with TAGS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_samples():
    by_project = {}
    with SAMPLES_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_project[row["project"]] = row
    return by_project


def build_profile(tags_row, samples_row, k=5):
    study_info = (samples_row or {}).get("study_info", {}) or {}
    samples = (samples_row or {}).get("samples", []) or []
    subset = random.sample(samples, k) if len(samples) > k else samples

    lines = [
        f"Project: {tags_row['project']}",
        f"Data source: {tags_row['data_source']}",
        f"N samples (total): {tags_row['n_samples']}",
        f"Structured tags: {json.dumps(tags_row.get('tags', []))}",
        f"Raw evidence: {tags_row.get('raw_evidence','')}",
        "",
        "Study-level fields:",
    ]
    for k2, v in study_info.items():
        if v not in (None, ""):
            lines.append(f"- {k2}: {v}")

    lines.append("")
    lines.append(f"5 sampled rows (of {len(samples)} total):")
    for s in subset:
        lines.append(f"- run_id={s.get('run_id')}, tissue_hint={s.get('tissue_hint')}, "
                      f"disease_hint={s.get('disease_hint')}, raw={json.dumps(s.get('raw', {}))}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--per-source", type=int, default=5)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7, help="fixed by default so re-runs (e.g. after a prompt tweak) pick the same 15 datasets")
    ap.add_argument("--projects", nargs="+", default=None,
                     help="target specific project ids instead of the random per-source pilot pick, and MERGE into the existing qna_pilot.yaml rather than overwriting it")
    args = ap.parse_args()
    random.seed(args.seed)

    tags = load_tags()
    samples_by_project = load_samples()
    by_project = {r["project"]: r for r in tags}

    existing = {}
    if args.projects:
        if OUT_YAML.exists():
            existing = yaml.safe_load(OUT_YAML.read_text()) or {}
        picked = [by_project[p] for p in args.projects if p in by_project]
        missing = [p for p in args.projects if p not in by_project]
        if missing:
            print(f"Not found in tags.jsonl, skipping: {missing}")
    else:
        eligible = [r for r in tags if r["n_samples"] >= args.min_samples]
        by_source = {}
        for r in eligible:
            by_source.setdefault(r["data_source"], []).append(r)
        picked = []
        for source in ("gtex", "tcga", "sra"):
            pool = by_source.get(source, [])
            picked.extend(random.sample(pool, min(args.per_source, len(pool))))
        print(f"Eligible (n_samples >= {args.min_samples}): {len(eligible)} total "
              f"({', '.join(f'{s}:{len(v)}' for s, v in by_source.items())})")

    print(f"Picked {len(picked)} datasets: {[r['project'] for r in picked]}")

    provider = LLMProvider(args.provider, args.model)
    print(f"Using {provider}")

    out = dict(existing)  # merge mode: keep everything already in qna_pilot.yaml
    for row in picked:
        project = row["project"]
        profile = build_profile(row, samples_by_project.get(project))
        try:
            raw = provider.complete(SYSTEM_PROMPT, profile, max_tokens=1200)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            answers = json.loads(raw)
            if not isinstance(answers, list) or len(answers) != 10:
                raise ValueError(f"expected 10 answers, got {len(answers) if isinstance(answers, list) else type(answers)}")
            out[project] = {
                "data_source": row["data_source"],
                "n_samples": row["n_samples"],
                "qna": [[q, a] for q, a in zip(QUESTIONS, answers)],
            }
            print(f"  ok: {project} ({row['data_source']})")
        except Exception as e:
            print(f"  FAILED: {project} ({row['data_source']}) -- {e}")

    with OUT_YAML.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"\nWrote {OUT_YAML} with {len(out)}/{len(picked)} datasets.")


if __name__ == "__main__":
    main()
