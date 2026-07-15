"""Download and validate a balanced, licensed Indian banknote dataset.

The source images are intentionally ignored by Git. This script writes a small
manifest with provenance, hashes, dimensions, and stable split groups so the
training run remains reproducible without redistributing the dataset.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from remotezip import RemoteZip
import requests

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError as exc:
    raise SystemExit("Install the training dependency with: pip install kaggle==1.7.4.5") from exc


DATASET_REF = "preetrank/indian-currency-real-vs-fake-notes-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_REF}"
DATASET_LICENSE = "CC BY-NC-SA 4.0"
DENOMINATIONS = ("10", "20", "50", "100", "500", "2000")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "training" / "currency"
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.json"
STAGING_DIR = OUTPUT_DIR / ".download_staging"
SOURCE_INDEX_PATH = OUTPUT_DIR / ".source_index.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def stable_order(name: str) -> str:
    return hashlib.sha256(f"fraud-shield-2026:{name}".encode("utf-8")).hexdigest()


def classify_source_path(name: str) -> tuple[str, str] | None:
    parts = [part.lower() for part in Path(name).parts]
    for label in ("real", "fake"):
        if label not in parts:
            continue
        index = parts.index(label)
        if index + 1 < len(parts) and parts[index + 1] in DENOMINATIONS:
            return ("genuine" if label == "real" else "counterfeit", parts[index + 1])
    return None


def enumerate_source_files(api: KaggleApi) -> list[dict]:
    if SOURCE_INDEX_PATH.exists():
        return json.loads(SOURCE_INDEX_PATH.read_text(encoding="utf-8"))
    records = []
    page_token = None
    while True:
        for attempt in range(5):
            try:
                response = api.dataset_list_files(DATASET_REF, page_token=page_token, page_size=200)
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        for dataset_file in response.dataset_files:
            source = classify_source_path(dataset_file.name)
            if source and Path(dataset_file.name).suffix.lower() in IMAGE_SUFFIXES:
                label, denomination = source
                records.append(
                    {
                        "source_file": dataset_file.name,
                        "source_bytes": int(dataset_file.total_bytes or 0),
                        "label": label,
                        "denomination": denomination,
                    }
                )
        page_token = response.next_page_token
        if not page_token:
            break
    SOURCE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_INDEX_PATH.write_text(json.dumps(records), encoding="utf-8")
    return records


def select_balanced(records: list[dict], per_stratum: int, min_source_bytes: int = 50_000) -> list[dict]:
    selected = []
    shortages = []
    for label in ("genuine", "counterfeit"):
        for denomination in DENOMINATIONS:
            stratum = [
                record
                for record in records
                if record["label"] == label and record["denomination"] == denomination
                and record["source_bytes"] >= min_source_bytes
            ]
            stratum.sort(key=lambda item: stable_order(item["source_file"]))
            if len(stratum) < per_stratum:
                shortages.append(f"{label}/{denomination}: {len(stratum)}")
            selected.extend(stratum[:per_stratum])
    if shortages:
        raise RuntimeError("Dataset does not satisfy the balanced target: " + ", ".join(shortages))
    return selected


def download_one(record: dict) -> dict:
    token = stable_order(record["source_file"])[:16]
    destination = STAGING_DIR / token
    destination.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(
        DATASET_REF,
        record["source_file"],
        path=str(destination),
        force=False,
        quiet=True,
    )
    candidates = [path for path in destination.iterdir() if path.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one downloaded file for {record['source_file']}")
    return {**record, "staged_path": str(candidates[0])}


def extract_selected(archive_path: Path, selected: list[dict]) -> list[dict]:
    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        missing = [record["source_file"] for record in selected if record["source_file"] not in members]
        if missing:
            raise RuntimeError(f"Archive is missing {len(missing)} selected files; first: {missing[0]}")
        for completed, record in enumerate(selected, start=1):
            token = stable_order(record["source_file"])[:16]
            destination = STAGING_DIR / token
            destination.mkdir(parents=True, exist_ok=True)
            suffix = Path(record["source_file"]).suffix.lower()
            staged_path = destination / f"source{suffix}"
            with archive.open(record["source_file"]) as source, staged_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            extracted.append({**record, "staged_path": str(staged_path)})
            if completed % 50 == 0 or completed == len(selected):
                print(f"Extracted {completed}/{len(selected)}")
    return extracted


def extract_selected_remote(selected: list[dict], workers: int) -> list[dict]:
    credentials_path = Path.home() / ".kaggle" / "kaggle.json"
    credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    endpoint = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
    redirect = requests.get(
        endpoint,
        auth=(credentials["username"], credentials["key"]),
        allow_redirects=False,
        timeout=30,
    )
    redirect.raise_for_status()
    signed_url = redirect.headers.get("Location")
    if not signed_url:
        raise RuntimeError("Kaggle did not provide a signed dataset archive URL")

    with RemoteZip(signed_url) as archive:
        members = set(archive.namelist())
    missing = [record["source_file"] for record in selected if record["source_file"] not in members]
    if missing:
        raise RuntimeError(f"Remote archive is missing {len(missing)} selected files; first: {missing[0]}")

    def extract_chunk(chunk: list[dict]) -> list[dict]:
        chunk_results = []
        with RemoteZip(signed_url) as archive:
            for record in chunk:
                token = stable_order(record["source_file"])[:16]
                destination = STAGING_DIR / token
                destination.mkdir(parents=True, exist_ok=True)
                suffix = Path(record["source_file"]).suffix.lower()
                staged_path = destination / f"source{suffix}"
                if not staged_path.exists() or staged_path.stat().st_size == 0:
                    with archive.open(record["source_file"]) as source, staged_path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                chunk_results.append({**record, "staged_path": str(staged_path)})
        return chunk_results

    worker_count = max(1, workers)
    chunks = [selected[index::worker_count] for index in range(worker_count)]
    extracted = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(extract_chunk, chunk) for chunk in chunks if chunk]
        for future in as_completed(futures):
            extracted.extend(future.result())
            print(f"Remote-extracted {len(extracted)}/{len(selected)}")
    return extracted


def difference_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 8))
    get_pixels = getattr(resized, "get_flattened_data", resized.getdata)
    pixels = list(get_pixels())
    bits = [pixels[row * 9 + col] > pixels[row * 9 + col + 1] for row in range(8) for col in range(8)]
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def validate_and_install(downloaded: list[dict]) -> list[dict]:
    manifest = []
    exact_hashes: dict[str, str] = {}
    for index, record in enumerate(sorted(downloaded, key=lambda item: stable_order(item["source_file"]))):
        source_path = Path(record["staged_path"])
        try:
            with Image.open(source_path) as image:
                image.verify()
            with Image.open(source_path) as image:
                width, height = image.size
                mode = image.mode
                perceptual_hash = difference_hash(image)
        except (OSError, UnidentifiedImageError) as exc:
            print(f"Skipping unreadable image {record['source_file']}: {exc}")
            continue
        if min(width, height) < 160:
            print(f"Skipping undersized image {record['source_file']}: {width}x{height}")
            continue

        sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if sha256 in exact_hashes:
            print(f"Skipping exact duplicate {record['source_file']}")
            continue
        exact_hashes[sha256] = record["source_file"]

        suffix = source_path.suffix.lower() if source_path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
        class_dir = OUTPUT_DIR / record["label"]
        class_dir.mkdir(parents=True, exist_ok=True)
        filename = f"kaggle_{record['label']}_{record['denomination']}_{index:04d}{suffix}"
        installed_path = class_dir / filename
        shutil.copy2(source_path, installed_path)
        manifest.append(
            {
                "path": installed_path.relative_to(OUTPUT_DIR).as_posix(),
                "label": record["label"],
                "denomination": record["denomination"],
                "source_dataset": DATASET_REF,
                "source_url": DATASET_URL,
                "source_file": record["source_file"],
                "source_label_verification": "publisher_label",
                "license": DATASET_LICENSE,
                "sha256": sha256,
                "difference_hash": perceptual_hash,
                "width": width,
                "height": height,
                "mode": mode,
                "split_group": hashlib.sha256(record["source_file"].encode("utf-8")).hexdigest()[:20],
            }
        )
    return manifest


def clear_generated_images() -> None:
    for label in ("genuine", "counterfeit"):
        class_dir = OUTPUT_DIR / label
        if not class_dir.exists():
            continue
        for path in class_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-stratum", type=int, default=43, help="Images per label and denomination")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--archive", type=Path, help="Previously downloaded Kaggle archive")
    parser.add_argument("--minimum-per-class", type=int, default=200)
    parser.add_argument("--min-source-bytes", type=int, default=50_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    print(f"Enumerating {DATASET_REF}...")
    records = enumerate_source_files(api)
    selected = select_balanced(records, args.per_stratum, args.min_source_bytes)
    print(f"Found {len(records)} labelled images; selected {len(selected)} balanced files")
    if args.dry_run:
        return

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    if args.archive:
        if not args.archive.is_file():
            raise FileNotFoundError(args.archive)
        downloaded = extract_selected(args.archive, selected)
    else:
        downloaded = extract_selected_remote(selected, args.workers)

    clear_generated_images()
    manifest = validate_and_install(downloaded)
    counts = {
        label: sum(record["label"] == label for record in manifest)
        for label in ("genuine", "counterfeit")
    }
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": DATASET_REF,
        "source_url": DATASET_URL,
        "license": DATASET_LICENSE,
        "label_assurance": "Publisher-labelled research data; not RBI forensic certification",
        "selection": {
            "seed": "fraud-shield-2026",
            "per_label_denomination": args.per_stratum,
            "requested_count": len(selected),
            "installed_count": len(manifest),
        },
        "counts": counts,
        "records": manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print(f"Installed {len(manifest)} validated images: {counts}")
    print(f"Manifest: {MANIFEST_PATH}")
    if min(counts.values()) < args.minimum_per_class:
        raise RuntimeError(
            f"Validation left fewer than {args.minimum_per_class} images in one class"
        )


if __name__ == "__main__":
    main()
