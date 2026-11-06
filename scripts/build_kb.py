"""Build the EvoMDT knowledge base: atomic units with provenance, version hashes and dense embeddings.

Output goes to data/kb/ (git-ignored): atoms.jsonl, embeddings.npy and kb_manifest.json.

Sources are limited to what the licences recorded in scripts/data_lock.json allow:

* NCCN PDFs are excluded. Their notice forbids use "with any artificial intelligence model or tool".
* recist_1_1.pdf and the ScienceDirect abstract print are excluded. They hold Elsevier copyright with
  a text-mining licence only, and retrieved text is sent to a hosted LLM.
* Open-access guidelines, PMC manuscripts released for text mining and openFDA labels (CC0) are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evomdt.retrieval import Embedder  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "kb"
LOCK = ROOT / "scripts" / "data_lock.json"

GRADE = re.compile(
    r"\b(?:category\s+[12][AB]?|level\s+(?:of\s+evidence\s+)?[IV]{1,3}[ABC]?|grade\s+[A-D]\b|"
    r"\[[IV]{1,3},\s*[A-E]\]|(?:strong|weak|moderate|conditional)\s+recommendation)",
    re.I,
)
LABEL_SECTIONS = {
    "boxed_warning": "Boxed warning",
    "indications_and_usage": "Indications and usage",
    "dosage_and_administration": "Dosage and administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and precautions",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse reactions",
    "drug_interactions": "Drug interactions",
    "use_in_specific_populations": "Use in specific populations",
}
TARGET_WORDS = 170
MAX_WORDS = 240
MIN_WORDS = 40


def cancer_tag(name: str) -> str:
    if name.startswith("breast"):
        return "breast"
    if name.startswith("lung"):
        return "lung"
    if name.startswith("lymphoma"):
        return "lymphoma"
    return "hcc"


def clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_words(text: str) -> list[str]:
    """Split on sentence boundaries into units of about TARGET_WORDS words."""
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z0-9\[(])", text)
    units: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > MAX_WORDS:
            words = words[:MAX_WORDS]
            sentence = " ".join(words)
        if current and count + len(words) > TARGET_WORDS:
            units.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)
    if current:
        units.append(" ".join(current))
    return [unit for unit in units if len(unit.split()) >= MIN_WORDS]


def atom(source_id: str, title: str, anchor: str, version: str, year: int | None, source_type: str, cancer: str, licence: str, text: str) -> dict:
    digest = hashlib.sha256(f"{source_id}|{anchor}|{text}".encode()).hexdigest()
    grade = GRADE.search(text)
    return {
        "evidence_id": f"E{digest[:8]}",
        "sha256": digest,
        "source_id": source_id,
        "title": title,
        "anchor": anchor,
        "version": version,
        "year": year,
        "source_type": source_type,
        "cancer": cancer,
        "licence": licence,
        "evidence_grade": grade.group(0) if grade else None,
        "text": text,
    }


def year_from(text: str) -> int | None:
    years = re.findall(r"(?:19|20)\d{2}", text)
    return int(years[-1]) if years else None


def from_pdf(path: Path, title: str, licence: str) -> list[dict]:
    import pymupdf

    atoms: list[dict] = []
    year = year_from(path.stem)
    with pymupdf.open(path) as document:
        for number, page in enumerate(document, start=1):
            raw = page.get_text("text")
            # Reference lists are bibliography, not guidance.
            match = re.search(r"^\s*(?:references|bibliography)\s*$", raw, re.I | re.M)
            if match and number > 2:
                raw = raw[: match.start()]
                done = True
            else:
                done = False
            for text in split_words(clean(raw)):
                atoms.append(atom(path.stem, title, f"p.{number}", str(year or "n/a"), year, "guideline", cancer_tag(path.stem), licence, text))
            if done:
                break
    return atoms


def from_xml(path: Path, title: str, licence: str, cancer: str, source_type: str) -> list[dict]:
    root = ET.parse(path).getroot()
    atoms: list[dict] = []
    year = year_from(path.stem)
    body = root.find(".//body")
    for section in body.iter("sec") if body is not None else []:
        heading = section.find("title")
        name = clean("".join(heading.itertext())) if heading is not None else "body"
        paragraphs = " ".join(clean("".join(p.itertext())) for p in section.findall("p"))
        for text in split_words(paragraphs):
            atoms.append(atom(path.stem, title, f"section: {name[:70]}", str(year or "n/a"), year, source_type, cancer, licence, text))
    return atoms


def from_label(path: Path, licence: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") or []
    main = next((l for l in labels if l.get("is_main_brand_label")), None)
    main = main or next((l for l in labels if l.get("has_openfda_block")), labels[0] if labels else None)
    if main is None:
        return []
    drug = payload["drug"]
    effective = str(main.get("effective_time") or "")
    year = int(effective[:4]) if effective[:4].isdigit() else None
    version = f"SPL v{main.get('version')} ({effective or 'undated'})"
    atoms: list[dict] = []
    for key, name in LABEL_SECTIONS.items():
        for block in main["label"].get(key) or []:
            for text in split_words(clean(block)):
                atoms.append(atom(f"label_{path.stem}", f"FDA label: {drug}", f"section: {name}", version, year, "drug_label", "pan", licence, text))
    return atoms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    licences = {Path(key).stem: value.get("licence", "") for key, value in lock.items()}
    titles = {Path(key).stem: value.get("source", "") for key, value in lock.items()}

    def title_of(stem: str) -> str:
        source = titles.get(stem, stem)
        return source.split(": ", 1)[1] if ": " in source else source

    atoms: list[dict] = []
    counts: dict[str, int] = {}
    for path in sorted((RAW / "open_guidelines").glob("*.pdf")):
        found = from_pdf(path, title_of(path.stem), licences.get(path.stem, ""))
        counts[path.stem] = len(found)
        atoms += found
    xml_sources = [
        (RAW / "open_guidelines" / "bclc_2022_reig_manuscript.xml", "hcc", "guideline"),
        (RAW / "references" / "mrecist_2010_lencioni_llovet_manuscript.xml", "hcc", "literature"),
        (RAW / "references" / "irecist_2017_seymour_manuscript.xml", "pan", "literature"),
    ]
    for path, cancer, source_type in xml_sources:
        found = from_xml(path, title_of(path.stem), licences.get(path.stem, ""), cancer, source_type)
        counts[path.stem] = len(found)
        atoms += found
    for path in sorted((RAW / "drug_labels").glob("*.json")):
        found = from_label(path, licences.get(path.stem, ""))
        counts[f"label_{path.stem}"] = len(found)
        atoms += found

    unique = {item["evidence_id"]: item for item in atoms}
    atoms = list(unique.values())
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        print(f"WARNING: no atoms from {empty}")

    OUT.mkdir(parents=True, exist_ok=True)
    vectors = Embedder(args.model).encode([item["text"] for item in atoms], batch_size=args.batch_size)
    np.save(OUT / "embeddings.npy", vectors.astype(np.float32))
    with (OUT / "atoms.jsonl").open("w", encoding="utf-8") as handle:
        for item in atoms:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    tags: dict[str, int] = {}
    for item in atoms:
        tags[item["cancer"]] = tags.get(item["cancer"], 0) + 1
    manifest = {
        "embedding_model": args.model,
        "atoms": len(atoms),
        "by_cancer": tags,
        "with_evidence_grade": sum(1 for item in atoms if item["evidence_grade"]),
        "by_source": counts,
        "excluded": {
            "NCCN guidelines": "EULA forbids use with any AI model or tool",
            "recist_1_1.pdf, irecist_sciencedirect_abstract_page.pdf": "Elsevier copyright, text-mining licence only",
        },
    }
    (OUT / "kb_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("atoms", "by_cancer", "with_evidence_grade")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
