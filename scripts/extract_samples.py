"""
Extracts per-sample metadata (Run ID + tissue/disease hints + full raw row)
for every project already present in tags.jsonl, and writes samples.jsonl —
one line per project, each holding its full list of samples.

Reuses the same metadata layout as extract_tags.py:

    metadata/<data_source>/<project>/<data_source>.<meta_type>.<project>.MD.gz

Hint extraction (_tissue_hint / _disease_hint) is ported from
anugene-ai/platform/recount3-labeler/metadata.py so hints come out the same
way they did in the old recount3-labeler tool: SRA's packed
"key;;value|key;;value" sample_attributes are parsed and matched against
tissue/disease keyword lists; GTEx/TCGA (no packed attributes column) fall
back to matching column names directly.

USAGE
-----
    python extract_samples.py                    # all projects in tags.jsonl
    python extract_samples.py --limit 5           # pilot on the first 5 (by tags.jsonl order)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

METADATA_DIR = Path("metadata")
TAGS_JSONL = Path("tags.jsonl")
OUT_JSONL = Path("samples.jsonl")

KEY_COLUMNS = ["rail_id", "external_id", "study"]
TISSUE_KEYWORDS = ["tissue", "source_name", "cell_type", "cell_line", "biosample"]
DISEASE_KEYWORDS = ["disease", "phenotype", "condition", "diagnosis"]


def _file_path(data_source: str, project: str, meta_type: str) -> Path:
    name = data_source if meta_type == "sample" else meta_type
    filename = f"{data_source}.{name}.{project}.MD.gz"
    return METADATA_DIR / data_source / project / filename


def _load_table(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _parse_encoded_attributes(raw) -> dict:
    """Parse recount3's "key;;value|key;;value" sample_attributes encoding."""
    pairs = {}
    for chunk in str(raw).split("|"):
        if ";;" in chunk:
            k, v = chunk.split(";;", 1)
            pairs[k.strip().lower()] = v.strip()
    return pairs


def _hint_series(df, keywords):
    attr_col = next((c for c in df.columns if c.endswith("sample_attributes")), None)
    if attr_col is not None:
        def pick(raw):
            pairs = _parse_encoded_attributes(raw)
            for kw in keywords:
                for k, v in pairs.items():
                    if kw in k:
                        return v
            return None
        return df[attr_col].fillna("").apply(pick)

    for kw in keywords:
        matches = [c for c in df.columns if kw in c]
        if matches:
            return df[matches[0]].astype(str).replace("nan", None)

    return pd.Series([None] * len(df), index=df.index)


def load_project_samples(data_source: str, project: str):
    """One row per sample, with _tissue_hint / _disease_hint columns added."""
    sample_df = _load_table(_file_path(data_source, project, "sample"))
    if sample_df is None:
        return None

    proj_df = _load_table(_file_path(data_source, project, "recount_project"))
    merged = sample_df
    if proj_df is not None:
        keys = [k for k in KEY_COLUMNS if k in sample_df.columns and k in proj_df.columns]
        if keys:
            merged = pd.merge(sample_df, proj_df, on=keys, how="left", suffixes=("", "__project"))

    id_col = next((c for c in ("external_id", "rail_id") if c in merged.columns), None)
    if id_col is None:
        return None
    merged = merged.rename(columns={id_col: "run_id"})
    merged["run_id"] = merged["run_id"].astype(str)

    merged["_tissue_hint"] = _hint_series(merged, TISSUE_KEYWORDS)
    merged["_disease_hint"] = _hint_series(merged, DISEASE_KEYWORDS)
    return merged


# Columns worth keeping in the per-sample "raw" panel -- everything else
# (GTEx's ~50 QC/library-stat columns like smrin, sme2mprt, smmppdpr, ...) is
# noise for a labeling decision and would bloat the built HTML for nothing.
RAW_FIELD_KEYWORDS = [
    "run_id", "run_acc", "sampid", "subjid", "sample_acc", "experiment_acc",
    "tcga_barcode", "study", "project", "title", "organism",
    "tissue", "disease", "phenotype", "condition", "diagnosis", "source_name",
    "cell_type", "cell_line", "biosample", "specimen", "sample_type",
    "sex", "age", "sample_attributes",
    "strategy", "platform", "assay", "library",
]


def _relevant_raw_columns(columns):
    return [c for c in columns if any(kw in c for kw in RAW_FIELD_KEYWORDS)]


def extract_project_samples(data_source: str, project: str):
    df = load_project_samples(data_source, project)
    if df is None or df.empty:
        return None

    raw_cols = _relevant_raw_columns(df.columns)

    # Study-level fields (study_title, study_abstract, organism, ...) come
    # out of the recount_project merge with the SAME value on every row --
    # storing them per-sample would repeat e.g. a 1300-char abstract on
    # every one of a project's 8000+ samples. Detect any column that's
    # constant across the whole project and hoist it to a project-level
    # dict once, instead of duplicating it per sample.
    study_info = {}
    per_sample_cols = []
    for c in raw_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique <= 1:
            vals = df[c].dropna()
            study_info[c] = None if vals.empty else vals.iloc[0]
        else:
            per_sample_cols.append(c)

    samples = []
    for _, row in df.iterrows():
        raw = {}
        for k in per_sample_cols:
            v = row[k]
            if pd.isna(v):
                v = None
            elif isinstance(v, str) and len(v) > 300:
                v = v[:300] + "…"
            raw[k] = v
        samples.append({
            "run_id": row["run_id"],
            "tissue_hint": None if pd.isna(row["_tissue_hint"]) else row["_tissue_hint"],
            "disease_hint": None if pd.isna(row["_disease_hint"]) else row["_disease_hint"],
            "raw": raw,
        })
    return {"data_source": data_source, "project": project, "study_info": study_info, "samples": samples}


def _json_default(o):
    # numpy scalar types (int64, float64, bool_, ...) aren't JSON-serializable
    # natively -- unwrap to the plain Python value.
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=Path, default=TAGS_JSONL)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N projects (pilot run).")
    args = parser.parse_args()

    if not args.tags.exists():
        raise SystemExit(f"{args.tags} not found -- run extract_tags.py first.")

    projects = []
    with open(args.tags) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            projects.append((row["data_source"], row["project"]))

    if args.limit:
        projects = projects[:args.limit]

    print(f"Extracting per-sample metadata for {len(projects)} projects...")
    written = 0
    with open(args.out, "w") as out:
        for data_source, project in projects:
            try:
                result = extract_project_samples(data_source, project)
            except Exception as e:
                print(f"  Failed on {data_source}/{project}: {e}", file=sys.stderr)
                continue
            if result is None:
                print(f"  Skipped {data_source}/{project}: no sample metadata found", file=sys.stderr)
                continue
            out.write(json.dumps(result, default=_json_default) + "\n")
            written += 1
            print(f"  {data_source}/{project}: {len(result['samples'])} samples")

    print(f"\nWrote {written} project sample records to {args.out}")


if __name__ == "__main__":
    main()
