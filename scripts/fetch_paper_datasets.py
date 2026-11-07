"""Download the public datasets used to reproduce the EvoMDT evaluation.

Each dataset lands in its own folder under data/raw/benchmarks/, which is git-ignored. Downloaded
files are only ever parsed as data. Re-running skips files that are already complete.

MedQARo cannot be fetched by script: its authors publish it through a browser-only Proton Drive
link. Save it by hand to data/raw/benchmarks/medqaro/ and scripts/prepare_paper_data.py picks it up.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "raw" / "benchmarks"
HF = "https://huggingface.co/datasets"
ROND = (
    "https://raw.githubusercontent.com/Mayo-Clinic-RadOnc-Foundation-Models/"
    "Radiation-Oncology-NLP-Database/HEAD"
)
MAX_BYTES = 800 * 1024**2

FILES: dict[str, list[tuple[str, str]]] = {
    "medqa": [
        ("test.jsonl", f"{HF}/GBaker/MedQA-USMLE-4-options/resolve/main/phrases_no_exclude_test.jsonl"),
    ],
    "medmcqa": [
        ("validation.parquet", f"{HF}/openlifescienceai/medmcqa/resolve/main/data/validation-00000-of-00001.parquet"),
    ],
    "medquad": [
        ("train.parquet", f"{HF}/lavita/MedQuAD/resolve/main/data/train-00000-of-00001-e36383d177026d53.parquet"),
    ],
    "rond": [
        ("oncology_conversational.csv", f"{ROND}/6-Conversational/Oncology-Question.csv"),
        ("medical_physics_qa.csv", f"{ROND}/5-Question%20and%20answering%20(QA)/Medical-Physics-100questions-QA-format.csv"),
        ("LICENSE", f"{ROND}/LICENSE"),
    ],
    "pmc_patients": [
        ("PMC-Patients.csv", f"{HF}/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv"),
    ],
}


def fetch(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "adaptive-evomdt-data-fetch/1.0"})
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response:
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > MAX_BYTES:
            raise RuntimeError(f"{url} declares {declared} bytes, above the {MAX_BYTES} limit")
        written = 0
        with partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_BYTES:
                    raise RuntimeError(f"{url} exceeded the {MAX_BYTES} byte limit")
                handle.write(chunk)
    if declared and written != declared:
        raise RuntimeError(f"{url}: got {written} of {declared} bytes")
    partial.replace(target)


def main() -> int:
    failed = 0
    for name, files in FILES.items():
        folder = BENCH / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename, url in files:
            target = folder / filename
            if target.exists():
                print(f"have  {name}/{filename} ({target.stat().st_size:,} bytes)")
                continue
            try:
                fetch(url, target)
                print(f"got   {name}/{filename} ({target.stat().st_size:,} bytes)")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {name}/{filename}: {exc}")
    print("manual: MedQARo is browser-only; see https://github.com/ana-rogoz/MedQARo")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
