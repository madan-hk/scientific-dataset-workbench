import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from pathlib import Path
from time import sleep


# Project metadata files for each data source
BASE = "https://recount-opendata.s3.amazonaws.com/recount3/release/human/data_sources"

DATA_SOURCES = {
    "sra": f"{BASE}/sra/metadata/sra.recount_project.MD.gz",
    "gtex": f"{BASE}/gtex/metadata/gtex.recount_project.MD.gz",
    "tcga": f"{BASE}/tcga/metadata/tcga.recount_project.MD.gz",
}

all_projects = []

for source, url in DATA_SOURCES.items():
    print(f"Loading {source}...")

    df = pd.read_csv(
        url,
        sep="\t",
        compression="gzip",
        low_memory=False
    )

    df["data_source"] = source
    all_projects.append(df)

# Combine all project tables
projects = pd.concat(all_projects, ignore_index=True)

print(f"Total projects: {len(projects):,}")

project_ids = (
    projects[["project", "data_source"]]
    .drop_duplicates()
    .sort_values(["data_source", "project"])
    .reset_index(drop=True)
)

print(f"Unique projects: {len(project_ids):,}")

BASE = "https://recount-opendata.s3.amazonaws.com/recount3/release/human/data_sources"

METADATA_TYPES = [
    "sample",           # sra / gtex / tcga metadata
    "recount_project",
    "recount_qc",
    "recount_seq_qc",
    "recount_pred",
]

def download_project_metadata(project, data_source, out_dir="metadata"):
    """
    Download all metadata files for one recount3 project.

    Parameters
    ----------
    project : str
        e.g. "SRP009615"
    data_source : str
        One of: "sra", "gtex", "tcga"
    out_dir : str

    Returns
    -------
    dict
        Mapping metadata type -> downloaded file path (or None if unavailable)
    """

    folder = project[-2:]

    # Put every project's metadata in its own folder
    project_dir = Path(out_dir) / data_source / project
    project_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}

    for meta_type in METADATA_TYPES:

        # The "sample metadata" file uses the datasource name
        name = data_source if meta_type == "sample" else meta_type

        filename = f"{data_source}.{name}.{project}.MD.gz"

        url = (
            f"{BASE}/{data_source}/metadata/"
            f"{folder}/{project}/{filename}"
        )

        outfile = project_dir / filename

        if outfile.exists():
            downloaded[meta_type] = outfile
            continue

        success = False

        for attempt in range(3):
            try:
                r = requests.get(url, stream=True, timeout=60)

                if r.status_code == 404:
                    break

                r.raise_for_status()

                with open(outfile, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

                downloaded[meta_type] = outfile
                success = True
                break

            except Exception as e:
                print(f"{project} [{meta_type}] attempt {attempt+1}: {e}")
                sleep(3)

        if not success:
            print(f"Failed: {project} {meta_type}")
            downloaded[meta_type] = None

    return downloaded


# Number of concurrent downloads
MAX_WORKERS = 16  # Try 8-32 depending on your internet connection

# Convert dataframe to list of tuples
tasks = list(project_ids[["project", "data_source"]].itertuples(index=False, name=None))

results = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(download_project_metadata, project, data_source): (project, data_source)
        for project, data_source in tasks
    }

    for future in tqdm(as_completed(futures), total=len(futures)):
        project, data_source = futures[future]

        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"{project} ({data_source}) failed: {e}")
