"""
Pre-computes a synonym dictionary for every distinct tag value already in
tags.jsonl (e.g. "tuberculosis" -> ["tb", "mtb", "mycobacterium
tuberculosis", ...]), so the labeler UI can expand search terms locally
instead of calling an LLM live on every search.

Same intent as the in-browser expandQuery() in dataset_labeler.html, just
run once, offline, at build time instead of at search time.

USAGE
-----
    # pilot a small batch first
    python build_synonyms.py --limit 20 --provider openai

    # full run over every distinct tag value (resumable -- skips already-
    # cached terms on rerun)
    python build_synonyms.py --provider openai
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from llm_providers import LLMProvider, DEFAULT_MODELS

TAGS_JSONL = Path("tags.jsonl")
CACHE_JSONL = Path("synonyms.jsonl")
OUT_JSON = Path("synonyms.json")
MAX_WORKERS = 8

_write_lock = threading.Lock()

SYSTEM_PROMPT = (
    "Return ONLY a JSON array of 4-8 lowercase search terms (synonyms, "
    "abbreviations, related pathogen/disease names) for the given medical "
    "or biological term. No prose, no markdown fences. "
    'Example input: "tuberculosis" -> '
    '["tb","mtb","mycobacterium tuberculosis","pulmonary tb"]'
)


def load_distinct_terms(path: Path):
    terms = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for tag in row.get("tags", []):
                val = str(tag.get("value", "")).strip().lower()
                if val:
                    terms.add(val)
    return sorted(terms)


def load_cache():
    if not CACHE_JSONL.exists():
        return {}
    cache = {}
    with open(CACHE_JSONL) as f:
        for line in f:
            try:
                row = json.loads(line)
                cache[row["term"]] = row["synonyms"]
            except Exception:
                continue
    return cache


def append_cache(term, synonyms):
    with _write_lock:
        with open(CACHE_JSONL, "a") as f:
            f.write(json.dumps({"term": term, "synonyms": synonyms}) + "\n")


def fetch_synonyms(client: LLMProvider, term: str, retries=3):
    for attempt in range(retries):
        try:
            text = client.complete(SYSTEM_PROMPT, term, max_tokens=200)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            arr = json.loads(text.strip())
            if isinstance(arr, list) and arr:
                return [str(t).strip().lower() for t in arr if str(t).strip()]
            return []
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Failed '{term}' after {retries} attempts: {e}", file=sys.stderr)
                return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=Path, default=TAGS_JSONL)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only send the first N not-yet-cached terms this run (for a pilot).")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--provider", choices=list(DEFAULT_MODELS), default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    if not args.tags.exists():
        raise SystemExit(f"{args.tags} not found -- run extract_tags.py first.")

    all_terms = load_distinct_terms(args.tags)
    cache = load_cache()
    todo = [t for t in all_terms if t not in cache]
    if args.limit:
        todo = todo[:args.limit]

    client = LLMProvider(args.provider, args.model)
    print(f"{len(all_terms)} distinct tag values, {len(cache)} already cached, "
          f"{len(todo)} to process this run, using {client}.")

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_synonyms, client, t): t for t in todo}
            for future in tqdm(as_completed(futures), total=len(futures)):
                term = futures[future]
                result = future.result()
                if result is not None:
                    append_cache(term, result)
                    cache[term] = result

    final = {t: cache[t] for t in all_terms if t in cache}
    with open(args.out, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nWrote {len(final)} synonym entries to {args.out}")


if __name__ == "__main__":
    main()
