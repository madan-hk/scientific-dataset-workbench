# Scientific Dataset Curation and Labelling Workbench

## How to run on your machine

This is a plain `.html` file — no server, no build step. 75 datasets are
already pre-curated and committed (`generated/data.js` +
`enrichment/tags.jsonl` — tags, AI-Extracted Info), so a fresh clone has
something to work with immediately, no download or script run needed.

**1. Clone and install**
```
git clone <this-repo-url>
cd scientific-dataset-workbench
pip install -r requirements.txt
```

**2. Open the app**
```
open workflow/index.html
```
(or just double-click it — Chrome, Edge, or Brave; needs the [File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API), so not Safari/Firefox.)

**3. Connect a project folder**
On first load, click **"Connect Project Folder"** and, in the native folder
picker, select this same `scientific-dataset-workbench` folder (the repo
root you just cloned). That's the one folder-selection step the app ever
asks for — there's no server in between, the browser talks straight to
that folder.

The app creates a `tasks/` folder inside it and saves everything you do —
every Task, dataset decision, and sample label — there, as real JSON files
on your disk, one folder per Task:
```
scientific-dataset-workbench/
  tasks/
    <your-task-name>/
      task.json        Task definition (disease, label set, etc)
      <dataset-id>.json   Decision + sample labels for that dataset
```
Nothing here is committed to git (`tasks/` is in `.gitignore`) — it's your
own local work.

**4. Start working**
Create a Task, then work through **Dataset Discovery → Sample Curation →
Collection Review → Export** — see "How it works" below for what each
step does.

---

## What this is

A browser-based tool that lets a biologist or domain scientist review
public RNA-seq datasets (from SRA, GTEx, and TCGA) and label individual
samples, before that labeled data is used to train machine learning
models elsewhere.

It runs entirely client-side — no server and no login. Every change is
written straight to the local project folder as it happens (see step 3
above).

## Why this exists

Raw public RNA-seq data isn't usable for training a disease-classification
model on its own. A dataset can look relevant by title but turn out to be
the wrong tissue, the wrong assay type, or so skewed toward one label that
a model trained on it would just learn to recognize the dataset instead of
the actual disease.

This tool exists to catch that before training ever starts — turning an
unreviewed pile of public datasets into a labeled, audited Collection
ready to hand off to model training.

## How it works

The workflow follows one hierarchy: **Task → Collection → Datasets → Samples**.

1. **Task** — define the biological question once: the disease/condition being studied, the exact label set every sample will be judged against, and the expected sample type / sequence type.
2. **Dataset Discovery & Review** — search public datasets relevant to the Task. Each result shows AI-Extracted Information (a fixed set of questions answered ahead of time from the dataset's own raw metadata), a **Recount3 Info** panel fetched live from recount3's own public data (organism, sample count, study title/abstract — not pre-baked, pulled fresh from recount3's S3 bucket every time the card is opened), and the raw evidence itself, so nothing is taken on faith. Each dataset gets a decision: Include, Exclude, or Needs Review.
3. **Sample Curation & Labelling** — inside an included dataset, review each sample one at a time: confirm blood/tissue type, decide Keep / Reject / Skip, and assign a label from the Task's label set.
4. **Collection Review** — a rollup across every included dataset, with automated checks for label imbalance, dominant-label skew, high uncertainty, and sample-type coverage, before the Collection is considered ready.
5. **Export** — produces the labeled sample set (labels and metadata only, no expression values) for downstream model training.

## Tech stack

- Single-page vanilla HTML/CSS/JavaScript — no framework, no build step, no bundler.
- File System Access API for reading/writing the local project folder directly (Task/decision/label data).
- A direct, live `fetch()` against recount3's public S3 bucket for the Recount3 Info panel — no backend, no API key, CORS-open.
- Python scripts (offline, run ahead of time) for pulling and pre-processing public dataset metadata from [recount3](https://rna.recount.bio/).

## Project structure

```
workflow/index.html   The application itself (single page, no build step)
scripts/               Offline pipeline: download, extract, and pre-compute dataset metadata
enrichment/            Curated dataset-level tags/metadata (committed — small)
generated/data.js      Built dataset list the app reads (committed — small)
generated/samples_data.js   Built per-sample data (NOT committed — 54MB, regenerate via scripts/)
metadata/              Raw downloaded recount3 files per dataset (NOT committed — regenerate via scripts/)
samples/               Raw per-sample dump (NOT committed — regenerate via scripts/)
tasks/                 Created locally on first "Connect Project Folder" — your own work, never committed
```

## Status

Actively developed. Some flows (Collection "freeze"/version-lock,
ready-to-train export format) are placeholder wording for now — noted
inline where they appear in the app.
