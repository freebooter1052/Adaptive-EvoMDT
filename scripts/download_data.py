#!/usr/bin/env python3
"""Fetch and organise the external resources Adaptive EvoMDT needs. No processing.

Usage:
    python scripts/download_data.py              # Tiers 1 and 2
    python scripts/download_data.py --tier 1     # one tier only
    python scripts/download_data.py --refresh    # re-acquire even if hashes match

Idempotent: a file that is present and matches the SHA-256 in scripts/data_lock.json
is skipped. The lock also pins API snapshots (drug labels), so a re-run does not
silently change them; use --refresh to take a new snapshot.

Each run regenerates data/MANIFEST.md and docs/data_verification.md.

Access rules this script follows:
- PMC articles come only from the PMC Cloud Service. PMC's copyright notice forbids
  other automated downloads from its site, including for open-access articles.
- Publisher sites refuse scripted requests, so those items are listed as manual.
  Save a file at the path shown and the next run records it.
- Requests send a generic client name and never an email address.
- NCCN PDFs: only the page count and the version line are read. Their licence notice
  forbids use with any AI model or tool, so no other content is extracted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
RAW = "data/raw"
LOCK_PATH = REPO / "scripts" / "data_lock.json"
MANIFEST_PATH = REPO / "data" / "MANIFEST.md"
VERIFICATION_PATH = REPO / "docs" / "data_verification.md"
API_MODELS_PATH = REPO / "docs" / "api_models.md"

USER_AGENT = "adaptive-evomdt-data-fetch/1.0"
MAX_DOWNLOAD_BYTES = 500 * 1024**2  # Larger downloads need explicit approval.
MIN_FREE_BYTES = 1024**3
LAYOUT = ["hcc_tace", "guidelines", "references", "open_guidelines", "drug_labels", "benchmarks", "pubmed"]

PMC_CLOUD = "https://pmc-oa-opendata.s3.amazonaws.com"
OPENFDA = "https://api.fda.gov/drug/label.json"
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
TCIA_CLINICAL_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/HCC-TACE-Seg_clinical_data-V2.xlsx"

NCCN_LICENCE = "NCCN copyright and EULA: no redistribution, no use with any AI model or tool (notice in the PDF)"
ELSEVIER_COPYRIGHT = "Elsevier copyright; no open licence found (Crossref lists a text-mining licence only)"
MANUSCRIPT_LICENCE = "PMC author manuscript, publisher copyright: 'available for text mining' and fair use; no CC licence"
ELSEVIER_OA = "Elsevier open-access user licence: non-commercial download and text mining; no redistribution or display"
OPENFDA_LICENCE = "Public domain, CC0 1.0 (open.fda.gov/license)"
CC_BY_NC_YES = "yes (non-commercial, with attribution)"

log = logging.getLogger("download_data")


@dataclass(frozen=True)
class Item:
    tier: int
    dest: str
    kind: str  # copy | pmc | openfda | manual
    source: str
    licence: str
    redistributable: str
    local: Path | None = None
    fallback_url: str = ""
    pmcid: str = ""
    version: int = 0
    fmt: str = ""
    expect_licence: str = ""
    query: str = ""
    note: str = ""


def _nccn(original: str, snake: str) -> Item:
    return Item(1, f"{RAW}/guidelines/{snake}.pdf", "copy", f"local: ../guidelines/{original}.pdf",
                NCCN_LICENCE, "no", local=PROJECT / "guidelines" / f"{original}.pdf")


def _pmc(tier: int, dest: str, pmcid: str, version: int, fmt: str, licence_code: str, title: str, note: str = "") -> Item:
    open_access = licence_code != "TDM"
    return Item(tier, dest, "pmc", f"PMC Cloud Service {pmcid}.{version}: {title}",
                f"{licence_code} (per PMC metadata)" if open_access else MANUSCRIPT_LICENCE,
                CC_BY_NC_YES if open_access else "no",
                pmcid=pmcid, version=version, fmt=fmt, expect_licence=licence_code, note=note)


def _manual(dest: str, url: str, title: str, licence: str, note: str) -> Item:
    return Item(2, dest, "manual", f"{title}: {url}", licence, "no", note=note)


IRECIST_NAME = "iRECIST_ guidelines for response criteria for use in trials testing immunotherapeutics - ScienceDirect.pdf"

# Brand whose label is the main citation for each drug; None means generics only.
DRUGS = {
    "sorafenib": "NEXAVAR", "lenvatinib": "LENVIMA", "regorafenib": "STIVARGA", "cabozantinib": "CABOMETYX",
    "atezolizumab": "TECENTRIQ", "bevacizumab": "AVASTIN", "durvalumab": "IMFINZI", "tremelimumab": "IMJUDO",
    "nivolumab": "OPDIVO", "ipilimumab": "YERVOY", "doxorubicin": None, "cisplatin": None, "mitomycin": None,
}
DRUG_FILES = {name: ("mitomycin_c" if name == "mitomycin" else name) for name in DRUGS}

ITEMS: list[Item] = [
    Item(1, f"{RAW}/hcc_tace/HCC-TACE-Seg_clinical_data-V2.xlsx", "copy",
         f"local: ~/Downloads/HCC-TACE-Seg_clinical_data-V2.xlsx; official: {TCIA_CLINICAL_URL} (DOI 10.7937/TCIA.5FNA-0924)",
         "CC BY 4.0; TCIA citation required (Moawad et al. 2021)", "yes (with attribution)",
         local=Path.home() / "Downloads" / "HCC-TACE-Seg_clinical_data-V2.xlsx", fallback_url=TCIA_CLINICAL_URL),
    _nccn("hcc", "hcc"), _nccn("breast cancer", "breast_cancer"), _nccn("nscl", "nscl"),
    _nccn("b-cell", "b_cell"), _nccn("hodgkins", "hodgkins"), _nccn("t-cell", "t_cell"),
    Item(1, f"{RAW}/references/recist_1_1.pdf", "copy", "local: ../recist 1.1 (Eisenhauer et al., Eur J Cancer 2009)",
         ELSEVIER_COPYRIGHT, "no", local=PROJECT / "recist 1.1"),
    Item(1, f"{RAW}/references/irecist_sciencedirect_abstract_page.pdf", "copy", f"local: ../for staging/{IRECIST_NAME}",
         ELSEVIER_COPYRIGHT, "no", local=PROJECT / "for staging" / IRECIST_NAME,
         note="Browser print of the ScienceDirect abstract page, not the full article"),
    _pmc(1, f"{RAW}/references/mrecist_2010_lencioni_llovet_manuscript.xml", "PMC12268942", 1, "xml", "TDM",
         "Lencioni & Llovet, Semin Liver Dis 2010, mRECIST"),
    _pmc(1, f"{RAW}/references/irecist_2017_seymour_manuscript.xml", "PMC5648544", 1, "xml", "TDM",
         "Seymour et al., Lancet Oncol 2017, iRECIST", "Full text; the local PDF is only the abstract page"),
    _pmc(2, f"{RAW}/open_guidelines/bclc_2022_reig_manuscript.xml", "PMC8866082", 1, "xml", "TDM",
         "Reig et al., J Hepatol 2022, BCLC update"),
    _pmc(2, f"{RAW}/open_guidelines/bsg_hcc_2024.pdf", "PMC11287576", 1, "pdf", "CC BY-NC",
         "British Society of Gastroenterology HCC guideline, Gut 2024"),
    _pmc(2, f"{RAW}/open_guidelines/klca_ncc_hcc_2022.pdf", "PMC9597235", 1, "pdf", "CC BY-NC",
         "KLCA-NCC Korea HCC guideline, Clin Mol Hepatol 2022"),
    _pmc(2, f"{RAW}/open_guidelines/china_cnlc_hcc_2024.pdf", "PMC12503762", 2, "pdf", "CC BY-NC",
         "China Liver Cancer guideline 2024 edition, Liver Cancer 2025"),
    _pmc(2, f"{RAW}/open_guidelines/sitc_hcc_immunotherapy_2021.pdf", "PMC8438858", 1, "pdf", "CC BY-NC",
         "SITC HCC immunotherapy guideline, J Immunother Cancer 2021"),
    _pmc(2, f"{RAW}/open_guidelines/tace_consensus_2023.pdf", "PMC10366793", 1, "pdf", "CC BY-NC",
         "TACE expert-consensus recommendations, Clin Mol Hepatol 2023"),
    _manual(f"{RAW}/open_guidelines/easl_hcc_2018.pdf", "https://doi.org/10.1016/j.jhep.2018.03.019",
            "EASL HCC CPG 2018", "Elsevier copyright; free to read on the publisher site; no open licence found",
            "Superseded by EASL 2025"),
    _manual(f"{RAW}/open_guidelines/easl_hcc_2025.pdf", "https://doi.org/10.1016/j.jhep.2024.08.028",
            "EASL HCC CPG 2025", "Elsevier copyright, subscription; no open licence found",
            "Current EASL guideline; may need library access"),
    _manual(f"{RAW}/open_guidelines/esmo_hcc_2018.pdf", "https://doi.org/10.1093/annonc/mdy308",
            "ESMO HCC CPG 2018", ELSEVIER_OA, "Superseded by ESMO 2025; publisher refuses scripts (HTTP 403)"),
    _manual(f"{RAW}/open_guidelines/esmo_hcc_2025.pdf", "https://doi.org/10.1016/j.annonc.2025.02.006",
            "ESMO HCC CPG 2025", ELSEVIER_OA, "Current ESMO guideline; publisher refuses scripts (HTTP 403)"),
    _manual(f"{RAW}/open_guidelines/aasld_hcc_2023.pdf", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10663390/",
            "AASLD HCC guidance 2023", "PMC author manuscript, publisher copyright: fair use; no CC licence",
            "Not in the PMC Cloud Service"),
    _manual(f"{RAW}/open_guidelines/bclc_2022_reig.pdf", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8866082/",
            "BCLC 2022 (PDF form)", MANUSCRIPT_LICENCE, "Recommended: the XML has the staging flowchart's caption but not its image"),
    _manual(f"{RAW}/references/mrecist_2010_lencioni_llovet.pdf", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12268942/",
            "mRECIST 2010 (PDF form)", MANUSCRIPT_LICENCE, "Optional: XML already fetched; PDF needed only for page numbers"),
    _manual(f"{RAW}/references/irecist_2017_seymour.pdf", "https://pmc.ncbi.nlm.nih.gov/articles/PMC5648544/",
            "iRECIST 2017 (PDF form)", MANUSCRIPT_LICENCE, "Optional: XML already fetched; PDF needed only for page numbers"),
] + [
    Item(2, f"{RAW}/drug_labels/{DRUG_FILES[name]}.json", "openfda",
         f'openFDA drug label API, generic name or product elements: "{name}"', OPENFDA_LICENCE, "yes", query=name)
    for name in DRUGS
]


# --------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def human_size(size: int | float) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def http_open(url: str, headers: dict[str, str] | None = None, attempts: int = 3):
    """Open a URL, retrying on rate limits, server errors and network failures."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500 or attempt == attempts:
                raise
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict:
    with http_open(url, headers) as response:
        return json.load(response)


def download(url: str, target: Path) -> None:
    """Stream a URL to a file, refusing anything over the approval limit."""
    with http_open(url) as response:
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"{human_size(declared)} exceeds the 500 MB approval limit")
        written = 0
        with target.open("wb") as handle:
            while block := response.read(1024 * 1024):
                written += len(block)
                if written > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("download exceeded the 500 MB approval limit")
                handle.write(block)


# ------------------------------------------------------------------------ acquisition


def acquire_copy(item: Item, target: Path) -> None:
    if item.local and item.local.exists():
        shutil.copy2(item.local, target)
    elif item.fallback_url:
        log.info("local copy missing, using official URL: %s", item.fallback_url)
        download(item.fallback_url, target)
    else:
        raise FileNotFoundError(f"local source not found: {item.source}")


def acquire_pmc(item: Item, target: Path) -> None:
    key = f"{item.pmcid}.{item.version}"
    meta = fetch_json(f"{PMC_CLOUD}/{key}/{key}.json")
    if meta.get("license_code") != item.expect_licence:
        raise RuntimeError(f"licence changed: expected {item.expect_licence}, PMC says {meta.get('license_code')}")
    if meta.get("is_retracted"):
        raise RuntimeError(f"{key} is marked retracted")
    location = meta.get(f"{item.fmt}_url")
    if not location:
        raise RuntimeError(f"{key} has no {item.fmt} file in the PMC Cloud Service")
    parts = urllib.parse.urlsplit(location)
    download(f"{PMC_CLOUD}{parts.path}", target)
    expected_md5 = urllib.parse.parse_qs(parts.query).get("md5", [""])[0]
    if expected_md5 and md5_file(target) != expected_md5:
        raise RuntimeError(f"{key}: MD5 does not match PMC metadata")


def is_brand_label(label: dict, name: str, brand: str | None) -> bool:
    """True for the brand's own label, excluding variants such as OPDIVO QVANTIG."""
    if not brand:
        return False
    if brand in [value.upper() for value in label.get("openfda", {}).get("brand_name", [])]:
        return True
    elements = " ".join(label.get("spl_product_data_elements", [])).upper()
    return elements.startswith(f"{brand} {name.upper()}")


def dailymed_current(set_id: str) -> dict | None:
    """Latest published version of a label on DailyMed, which openFDA can lag behind."""
    url = f"{DAILYMED}/spls/{set_id}/history.json"
    history = fetch_json(url)["data"]["history"]
    if not history:
        return None
    latest = max(history, key=lambda entry: int(entry["spl_version"]))
    return {"set_id": set_id, "spl_version": latest["spl_version"], "published_date": latest["published_date"], "url": url}


def acquire_openfda(item: Item, target: Path) -> None:
    # Some current labels (LENVIMA, TECENTRIQ) have no openfda block, so a generic-name
    # search alone misses them. spl_product_data_elements is present on those records.
    search = urllib.parse.quote(f'openfda.generic_name:"{item.query}" spl_product_data_elements:"{item.query}"')
    labels: list[dict] = []
    disclaimer = ""
    while True:
        page = fetch_json(f"{OPENFDA}?search={search}&limit=100&skip={len(labels)}")
        disclaimer = page["meta"]["disclaimer"]
        labels.extend(page["results"])
        if len(labels) >= page["meta"]["results"]["total"]:
            break
        time.sleep(0.5)
    records = [
        {
            "set_id": label.get("set_id"),
            "version": label.get("version"),
            "effective_time": label.get("effective_time"),
            "spl_id": label.get("id"),
            "brand_name": label.get("openfda", {}).get("brand_name", []),
            "generic_name": label.get("openfda", {}).get("generic_name", []),
            "manufacturer_name": label.get("openfda", {}).get("manufacturer_name", []),
            "application_number": label.get("openfda", {}).get("application_number", []),
            "route": label.get("openfda", {}).get("route", []),
            "has_openfda_block": bool(label.get("openfda")),
            "is_main_brand_label": False,
            "label": label,
        }
        for label in sorted(labels, key=lambda entry: entry.get("set_id") or "")
    ]
    # Old relabelled copies can also match the brand, so take the newest.
    matches = [entry for entry in records if is_brand_label(entry["label"], item.query, DRUGS[item.query])]
    main = max(matches, key=lambda entry: entry["effective_time"] or "", default=None)
    if main:
        main["is_main_brand_label"] = True
    payload = {
        "drug": item.query,
        "source": f"{OPENFDA}?search={urllib.parse.unquote(search)}",
        "licence": OPENFDA_LICENCE,
        "openfda_disclaimer": disclaimer,
        "dailymed_current": dailymed_current(main["set_id"]) if main else None,
        "label_count": len(records),
        "labels": records,
    }
    target.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.5)


ACQUIRERS = {"copy": acquire_copy, "pmc": acquire_pmc, "openfda": acquire_openfda}


def record(item: Item, path: Path, lock: dict) -> None:
    sha = sha256_file(path)
    previous = lock.get(item.dest, {})
    lock[item.dest] = {
        "sha256": sha,
        "size": path.stat().st_size,
        "retrieved": previous["retrieved"] if previous.get("sha256") == sha else date.today().isoformat(),
        "source": item.source,
        "licence": item.licence,
        "redistributable": item.redistributable,
        "note": item.note,
    }


def ensure(item: Item, lock: dict, refresh: bool) -> str:
    """Make one item present. Returns skipped, fetched, recorded, manual or failed."""
    path = REPO / item.dest
    entry = lock.get(item.dest)
    if path.exists():
        sha = sha256_file(path)
        if entry and entry["sha256"] == sha and not refresh:
            if item.local and item.local.exists() and sha256_file(item.local) != sha:
                log.warning("SOURCE CHANGED %s: the local source differs; --refresh takes the new copy", item.dest)
            record(item, path, lock)
            log.info("SKIP    %s (SHA-256 matches)", item.dest)
            return "skipped"
        if entry and entry["sha256"] != sha:
            log.warning("HASH MISMATCH %s: differs from the lock", item.dest)
    if item.kind == "manual":
        if path.exists():
            record(item, path, lock)
            log.info("RECORD  %s (manually supplied)", item.dest)
            return "recorded"
        log.info("MANUAL  %s", item.source)
        return "manual"
    temp = path.with_name(path.name + ".part")
    try:
        ACQUIRERS[item.kind](item, temp)
        os.replace(temp, path)
    except Exception as exc:  # One failed source must not stop the others.
        temp.unlink(missing_ok=True)
        log.error("FAILED  %s: %s", item.dest, exc)
        return "failed"
    record(item, path, lock)
    log.info("FETCH   %s (%s)", item.dest, human_size(path.stat().st_size))
    return "fetched"


# ------------------------------------------------------------------ API model inventory


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip("\"'")
    return values


def classify_model(slug: str) -> list[str]:
    """Tag a model from its slug. These are name-based guesses for a person to review."""
    name = slug.lower()
    tags: list[str] = []
    is_deepseek = "deepseek" in name
    is_embedding = bool(re.search(r"embed|bge|nomic|e5-|rerank", name))
    if is_deepseek:
        tags.append("DeepSeek")
    if is_embedding:
        tags.append("embedding")
    size = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)b\b", name)
    is_small = bool(re.search(r"nano|mini|small|tiny", name)) or bool(size and float(size.group(1)) < 10)
    if is_small and not is_embedding:
        tags.append("small")
    non_chat = re.search(r"guard|safety|vision|vlm|paligemma|diffusion|flux|parse|ocr|asr|tts|whisper", name)
    if not (is_deepseek or is_embedding or is_small or non_chat):
        tags.append("judge candidate (non-DeepSeek)")
    return tags


def model_inventory() -> str:
    env = read_env(REPO / ".env")
    key, base = env.get("NIM_API_KEY"), env.get("NIM_BASE_URL")
    if not key or not base:
        return "skipped: NIM_API_KEY and NIM_BASE_URL are not both set in .env"
    if not (base.startswith("https://") or re.match(r"http://(127\.0\.0\.1|localhost)[:/]", base)):
        return "skipped: NIM_BASE_URL must use https, so the key is never sent in clear"
    try:
        payload = fetch_json(f"{base.rstrip('/')}/models", {"Authorization": f"Bearer {key}"})
    except urllib.error.HTTPError as exc:
        return f"failed: the model list endpoint returned HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return f"failed: could not reach the model list endpoint ({exc.reason})"
    slugs = sorted(model["id"] for model in payload.get("data", []))
    lines = [
        "# API model inventory",
        "",
        f"Listed {date.today().isoformat()} from the OpenAI-compatible `/models` endpoint at `{base}`. "
        f"{len(slugs)} models. Generated by `scripts/download_data.py`.",
        "",
        "Tags are guesses from the model name, not from provider specifications. Check parameter counts "
        "and judge suitability on the model card before relying on them.",
        "",
        "| Model slug | Tags |",
        "|---|---|",
    ]
    lines += [f"| `{slug}` | {', '.join(classify_model(slug)) or '-'} |" for slug in slugs]
    API_MODELS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"written: docs/api_models.md ({len(slugs)} models)"


# ------------------------------------------------------------------------ verification


def verify_xlsx(path: Path) -> list[tuple[str, str, str, bool]]:
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = {
        sheet.title: [row for row in sheet.iter_rows(values_only=True) if any(value is not None for value in row)]
        for sheet in book.worksheets
    }
    book.close()
    table = rows.get("data table", [[]])
    ids = {row[0] for row in table[1:]}
    columns = sum(value is not None for value in table[0])
    checks = [
        ("Sheets", '"data table", "glossary"', ", ".join(f'"{name}"' for name in rows), list(rows) == ["data table", "glossary"]),
        ("Patient rows", "105", str(len(table) - 1), len(table) - 1 == 105),
        ("Unique patient IDs (TCIA_ID)", "105", str(len(ids)), len(ids) == 105),
        ("Columns", "57", str(columns), columns == 57),
        ("Glossary entries (one per column)", "57", str(len(rows.get("glossary", []))), len(rows.get("glossary", [])) == 57),
    ]
    return checks


def verify_pdf(path: Path) -> dict:
    from pypdf import PdfReader

    result = {"file": path.name, "opens": False, "pages": "-", "encrypted": "-", "detail": ""}
    try:
        if path.read_bytes()[:5] != b"%PDF-":
            result["detail"] = "not a PDF file"
            return result
        reader = PdfReader(path)
        result.update(opens=True, pages=len(reader.pages), encrypted="yes" if reader.is_encrypted else "no")
        if path.parent.name == "guidelines":
            # NCCN: the version line only. See the licence note at the top of this file.
            match = re.search(r"Version\s+\d+\.\d{4},\s*\d\d/\d\d/\d\d", reader.pages[0].extract_text() or "")
            result["detail"] = match.group(0) if match else "version line not found"
        else:
            result["detail"] = str((reader.metadata or {}).get("/Title") or "")[:90]
    except Exception as exc:
        result["detail"] = f"error: {exc}"
    return result


def verify_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    title = root.find(".//article-title")
    return {
        "file": path.name,
        "title": " ".join("".join(title.itertext()).split())[:80] if title is not None else "",
        "words": len(" ".join(root.itertext()).split()),
        "sections": len(root.findall(".//sec")),
        "tables": len(root.findall(".//table-wrap")),
    }


def verify_drug(path: Path, name: str, brand: str | None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload["labels"]
    main = next((entry for entry in labels if entry["is_main_brand_label"]), None)
    complete = all(entry["set_id"] and entry["version"] and entry["effective_time"] for entry in labels)
    indications = " ".join((main or {}).get("label", {}).get("indications_and_usage", [])).lower()
    current = payload.get("dailymed_current") or {}
    return {
        "drug": DRUG_FILES[name],
        "labels": f"{len(labels)} ({sum(entry['has_openfda_block'] for entry in labels)})",
        "count": len(labels),
        "complete": complete,
        "brand": brand or "generics only",
        "found": main is not None if brand else None,
        "set_id": main["set_id"] if main else "-",
        "version": main["version"] if main else "-",
        "date": main["effective_time"] if main else "-",
        "dailymed": f"{current['spl_version']} ({current['published_date']})" if current else "-",
        "hcc": ("yes" if "hepatocellular" in indications else "no") if main else "-",
    }


def git_ignored(path: str) -> bool | None:
    try:
        done = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO, capture_output=True, check=False)
    except OSError:
        return None
    return done.returncode == 0


def write_verification(lock: dict, statuses: dict[str, str], inventory: str) -> bool:
    today = date.today().isoformat()
    free = shutil.disk_usage(REPO).free
    used = sum(file.stat().st_size for file in (REPO / RAW).rglob("*") if file.is_file())
    lines = [
        "# Data verification",
        "",
        f"Generated by `scripts/download_data.py` on {today}. Free disk: {human_size(free)}. `data/raw/` uses {human_size(used)}.",
        "",
    ]
    ok = True

    lines += ["## HCC-TACE-Seg clinical file", "", "| Check | Expected | Actual | Result |", "|---|---|---|---|"]
    xlsx = REPO / RAW / "hcc_tace" / "HCC-TACE-Seg_clinical_data-V2.xlsx"
    if xlsx.exists():
        for name, expected, actual, passed in verify_xlsx(xlsx):
            ok &= passed
            lines.append(f"| {name} | {cell(expected)} | {cell(actual)} | {'PASS' if passed else 'FAIL'} |")
    else:
        ok = False
        lines.append("| File present | yes | no | FAIL |")
    lines += ["", "The SHA-256 also matched the official TCIA download when checked on 2026-11-05.", ""]

    lines += ["## PDFs", "", "| Folder | File | Opens | Pages | Encrypted | Version or title |", "|---|---|---|---:|---|---|"]
    for folder in ("guidelines", "references", "open_guidelines"):
        for pdf in sorted((REPO / RAW / folder).glob("*.pdf")):
            result = verify_pdf(pdf)
            ok &= result["opens"]
            lines.append(
                f"| {folder} | {cell(result['file'])} | {'PASS' if result['opens'] else 'FAIL'} | {result['pages']} "
                f"| {result['encrypted']} | {cell(result['detail'])} |"
            )
    lines += [
        "",
        "The NCCN PDFs are encrypted but open without a password. "
        "`irecist_sciencedirect_abstract_page.pdf` holds only the abstract and section snippets, not the full article.",
        "",
    ]

    lines += ["## Author-manuscript XML", "", "| File | Article title | Words | Sections | Tables |", "|---|---|---:|---:|---:|"]
    for folder in ("references", "open_guidelines"):
        for xml_file in sorted((REPO / RAW / folder).glob("*.xml")):
            try:
                result = verify_xml(xml_file)
                lines.append(f"| {result['file']} | {cell(result['title'])} | {result['words']} | {result['sections']} | {result['tables']} |")
            except ET.ParseError as exc:
                ok = False
                lines.append(f"| {xml_file.name} | parse error: {cell(exc)} | - | - | - |")
    lines.append("")

    lines += [
        "## Drug labels (openFDA)",
        "",
        "Each file holds every label openFDA returned for the drug, with set ID, version and effective date.",
        "\"HCC in indications\" is a text search of the main brand's label. \"DailyMed version\" is the latest "
        "published version when the file was fetched; where it is higher, the openFDA text is behind. "
        "Labels without an openfda block include old or delisted copies, so check them before citing.",
        "",
        "| Drug | Labels (with openfda block) | All have IDs and dates | Main brand | Brand set ID | openFDA version | Effective | DailyMed version | HCC in indications |",
        "|---|---:|---|---|---|---:|---|---|---|",
    ]
    for name, brand in DRUGS.items():
        path = REPO / RAW / "drug_labels" / f"{DRUG_FILES[name]}.json"
        if not path.exists():
            ok = False
            lines.append(f"| {DRUG_FILES[name]} | missing | FAIL | - | - | - | - | - | - |")
            continue
        result = verify_drug(path, name, brand)
        brand_text = result["brand"] if result["found"] is not False else f"{result['brand']} NOT FOUND"
        ok &= result["complete"] and result["count"] > 0 and result["found"] is not False
        lines.append(
            f"| {result['drug']} | {result['labels']} | {'PASS' if result['complete'] else 'FAIL'} | {brand_text} "
            f"| {result['set_id']} | {result['version']} | {result['date']} | {result['dailymed']} | {result['hcc']} |"
        )
    lines.append("")

    lines += ["## Git rules", "", "| Path | Expected | Result |", "|---|---|---|"]
    for path, should_ignore in ((".env", True), ("data/raw/guidelines/hcc.pdf", True), ("data/MANIFEST.md", False), ("scripts/download_data.py", False)):
        ignored = git_ignored(path)
        passed = ignored is should_ignore
        ok &= passed
        lines.append(f"| `{path}` | {'ignored' if should_ignore else 'tracked'} | {'PASS' if passed else 'FAIL'} |")
    lines.append("")

    failed = [dest for dest, status in statuses.items() if status == "failed"]
    missing_manual = [item.dest for item in ITEMS if item.kind == "manual" and not (REPO / item.dest).exists()]
    lines += [
        "## Run status",
        "",
        f"- API model inventory: {inventory}",
        f"- Failed this run: {', '.join(failed) if failed else 'none'}",
        f"- Waiting for manual download: {len(missing_manual)} (see `data/MANIFEST.md`)",
        f"- Overall automated checks: {'PASS' if ok and not failed else 'FAIL'}",
        "",
    ]
    VERIFICATION_PATH.write_text("\n".join(lines), encoding="utf-8")
    return ok and not failed


def write_manifest(lock: dict) -> None:
    lines = [
        "# Data manifest",
        "",
        "Generated by `scripts/download_data.py` from `scripts/data_lock.json`. Do not edit by hand.",
        "",
        "Everything under `data/raw/` is git-ignored, whether or not its licence allows redistribution. "
        "Only this file and `data/samples/` are committed.",
        "",
        "## Files present",
        "",
        "| File | Source or local origin | Licence | Retrieved | Size | SHA-256 | Redistributable | Note |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for dest in sorted(lock):
        entry = lock[dest]
        lines.append(
            f"| `{dest.removeprefix('data/')}` | {cell(entry['source'])} | {cell(entry['licence'])} | {entry['retrieved']} "
            f"| {human_size(entry['size'])} | `{entry['sha256']}` | {entry['redistributable']} | {cell(entry.get('note', ''))} |"
        )
    pending = [item for item in ITEMS if item.kind == "manual" and item.dest not in lock]
    lines += [
        "",
        "## Manual downloads still needed",
        "",
        "These are not downloaded by script. Fetch them in a browser, save to the path shown, then re-run the script.",
        "",
        "| Save as | Document and URL | Licence | Note |",
        "|---|---|---|---|",
    ]
    lines += [f"| `{item.dest.removeprefix('data/')}` | {cell(item.source)} | {cell(item.licence)} | {cell(item.note)} |" for item in pending]
    lines += [
        "",
        "## Terms to keep in mind",
        "",
        "- **NCCN:** the PDFs state: \"you MAY NOT distribute this Content or use it with any artificial intelligence model or tool\".",
        "- **HCC-TACE-Seg citation:** Moawad AW, Fuentes D, Morshid A, et al. (2021). Multimodality annotated HCC cases with and "
        "without advanced imaging segmentation [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.5FNA-0924",
        "- **openFDA:** its own disclaimer says not to rely on it for medical-care decisions and to assume results are unvalidated. "
        "Each label file repeats the disclaimer. Label files are dated snapshots.",
        "",
    ]
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=["1", "2", "3", "all"], default="all", help="tier to fetch; 'all' means Tiers 1 and 2")
    parser.add_argument("--refresh", action="store_true", help="re-acquire files even if their hashes match")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout)
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)

    if args.tier == "3":
        log.info("Tier 3 (benchmarks, PubMed, local embedder) waits for approval and is not implemented. Nothing fetched.")
        return 0

    free = shutil.disk_usage(REPO).free
    log.info("Free disk: %s", human_size(free))
    if free < MIN_FREE_BYTES:
        log.error("Less than 1 GB free; stopping.")
        return 1

    for folder in LAYOUT:
        (REPO / RAW / folder).mkdir(parents=True, exist_ok=True)
    (REPO / "docs").mkdir(exist_ok=True)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8")) if LOCK_PATH.exists() else {}
    tiers = {1, 2} if args.tier == "all" else {int(args.tier)}
    statuses = {item.dest: ensure(item, lock, args.refresh) for item in ITEMS if item.tier in tiers}
    LOCK_PATH.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n", encoding="utf-8")

    inventory = model_inventory() if 2 in tiers else "not run (Tier 2 not selected)"
    log.info("API model inventory: %s", inventory)

    write_manifest(lock)
    passed = write_verification(lock, statuses, inventory)

    counts = {status: list(statuses.values()).count(status) for status in sorted(set(statuses.values()))}
    log.info("Done: %s", counts)
    log.info("Automated checks: %s (docs/data_verification.md)", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
