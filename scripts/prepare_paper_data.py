"""Turn the raw datasets into the case files used by scripts/run_paper_eval.py.

  benchmarks   Layer 1. MedQA, MedQA-Cancer, MedMCQA-Cancer, MedQuAD-CancerGov and ROND as case JSONL.
  realworld    Layer 2. MDT dossiers and reference plans from published case reports (PMC-Patients).

The paper's Layer 2 cohort (four hospitals, n=405) is private. PMC-Patients is the open substitute:
each report records what the treating team did, which plays the part of the expert plan.

A case is kept only if a verbatim quote confirms the treatment, and no reference-plan therapy appears
in the dossier. Every exclusion is counted in data/paper_eval/realworld_flow.json. The 'audit' step has a
different model from the extractor check a random sample of the kept cases against the source reports.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "benchmarks"
OUT = ROOT / "data" / "paper_eval"
SEED = 7
CANCER_TERMS = re.compile(
    r"\b(cancer|carcinoma|adenocarcinoma|sarcoma|lymphoma|leukemia|leukaemia|melanoma|myeloma|malignan\w+|tumou?rs?|"
    r"neoplas\w+|metasta\w+|chemotherap\w+|radiotherap\w+|glioblastoma|glioma|mesothelioma|blastoma|oncolog\w+)\b",
    re.I,
)
LETTERS = "ABCDEFGH"


def mcq_case(case_id: str, dataset: str, question: str, options: dict[str, str], answer: str) -> dict:
    text = question.strip() + "\n" + "\n".join(f"({label}) {value}" for label, value in options.items())
    labels = list(options)
    return {
        "case_id": case_id,
        "domain": "evomdt_benchmark",
        "task_type": "mcq",
        "question": text,
        "options": labels,
        "reference_answer": answer,
        "metadata": {
            "dataset": dataset,
            "output_contract": {"type": "enum_choice", "allowed_values": labels, "final_answer_only": True},
        },
    }


def generation_case(case_id: str, dataset: str, question: str, answer: str) -> dict:
    return {
        "case_id": case_id,
        "domain": "evomdt_benchmark",
        "task_type": "generation",
        "question": question.strip(),
        "reference_answer": answer.strip(),
        "metadata": {"dataset": dataset},
    }


def write_jsonl(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")


def prepare_benchmarks(args: argparse.Namespace) -> None:
    import pandas as pd

    rng = random.Random(SEED)

    with (RAW / "medqa" / "test.jsonl").open(encoding="utf-8") as handle:
        medqa = [json.loads(line) for line in handle]
    cases = [mcq_case(f"medqa-{i:04d}", "medqa", row["question"], row["options"], row["answer_idx"]) for i, row in enumerate(medqa)]
    cancer = [case for case, row in zip(cases, medqa) if CANCER_TERMS.search(row["question"])]
    order = cases[:]
    rng.shuffle(order)
    dev_ids = {case["case_id"] for case in order[-args.dev:]}
    sample = order[: args.medqa]
    write_jsonl("medqa", sample)
    write_jsonl("medqa_cancer", [case for case in cancer if case["case_id"] not in dev_ids])
    write_jsonl("dev_medqa", order[-args.dev :])

    frame = pd.read_parquet(RAW / "medmcqa" / "validation.parquet")
    rows = []
    for _, row in frame.iterrows():
        if not CANCER_TERMS.search(str(row["question"])):
            continue
        options = {"A": row["opa"], "B": row["opb"], "C": row["opc"], "D": row["opd"]}
        rows.append(mcq_case(f"medmcqa-{row['id'][:8]}", "medmcqa_cancer", str(row["question"]), {k: str(v) for k, v in options.items()}, LETTERS[int(row["cop"])]))
    rng.shuffle(rows)
    print(f"MedMCQA validation questions matching cancer terms: {len(rows)}")
    write_jsonl("medmcqa_cancer", rows[: args.medmcqa])

    frame = pd.read_parquet(RAW / "medquad" / "train.parquet")
    frame = frame[(frame.document_source == "CancerGov") & frame.answer.notna()]
    rows = [generation_case(f"medquad-{row['question_id']}", "medquad", str(row["question"]), str(row["answer"])) for _, row in frame.iterrows()]
    rng.shuffle(rows)
    write_jsonl("medquad", rows[: args.medquad])

    frame = pd.read_csv(RAW / "rond" / "oncology_conversational.csv", encoding="latin-1")
    write_jsonl("rond", [generation_case(f"rond-{i:03d}", "rond", str(row["Question"]), str(row["Answer"])) for i, row in frame.iterrows()])

    medqaro = RAW / "medqaro"
    if not any(medqaro.glob("*")) if medqaro.exists() else True:
        print("MedQARo: not present (browser-only download), so that benchmark is skipped.")


# ---------------------------------------------------------------------------- Layer 2

COHORTS = {
    "breast_cancer": (re.compile(r"breast (?:cancer|carcinoma)|carcinoma of the breast|invasive (?:ductal|lobular) carcinoma", re.I), 100),
    "hepatocellular_carcinoma": (re.compile(r"hepatocellular carcinoma|\bHCC\b"), 105),
    "lung_adenocarcinoma": (re.compile(r"(?:lung|pulmonary) adenocarcinoma|adenocarcinoma of the (?:lung|right lung|left lung)", re.I), 100),
    "lymphoma": (re.compile(r"\blymphoma\b", re.I), 100),
}
NOT_INITIAL = re.compile(r"recurren|relaps|refractory|metachronous|post-?transplant|after (?:treatment|chemotherapy|mastectomy|resection)", re.I)
EXTRACT_PROMPT = """You prepare a standardized tumour-board dossier from a published case report, for a study that compares a decision-support system against what the treating team did.

Target cancer: {cancer}

Decide first whether the case is eligible. It is eligible only if ALL of these hold:
- the patient's main cancer is the target cancer, in an adult;
- the report describes the anti-cancer treatment decided at FIRST diagnosis or initial presentation of this cancer;
- the pre-treatment findings are detailed enough for a board to discuss.

Then return JSON only, with these fields:
- eligible: true or false
- exclusion_reason: short text, or null
- dossier: an object using ONLY information known BEFORE the treatment decision:
    age (int), sex, performance_status, stage, stage_system, pathology_summary,
    lesions (list of objects: site, size_cm (number), description, stage_context),
    biomarkers (list of objects: name, value),
    organ_function (object of name to value),
    comorbidities (list of strings), prior_treatments (list of strings: therapy for OTHER, earlier illnesses only),
    presentation (2-4 sentences on the complaint and work-up)
  Use null or an empty list for anything the report does not state. Never infer, compute or guess a value.
  The dossier MUST NOT mention the treatment given for this cancer, the response, the outcome, surgery findings that came from the treatment, or follow-up results.
- reference_plan_text: the plan the treating team actually carried out, in the form "Primary Treatment: ...; Follow up Management: ...". Use only what the report states, in about 15-50 words. Leave out outcomes.
- reference_plan_items: list of objects with action and priority (1 for the primary treatment, 2 for adjuncts, 3 for follow-up), one per distinct action.
- treatment_quote: a verbatim excerpt of 5-30 words copied from the report that states the primary treatment.

CASE REPORT
{note}
"""
THERAPY_WORDS = {
    "resection", "resected", "lobectomy", "mastectomy", "lumpectomy", "hepatectomy", "transplant", "transplantation", "ablation",
    "chemoembolization", "tace", "tare", "radioembolization", "radiotherapy", "chemotherapy", "immunotherapy", "cisplatin", "carboplatin",
    "pemetrexed", "paclitaxel", "docetaxel", "doxorubicin", "cyclophosphamide", "rituximab", "chop", "abvd", "osimertinib", "gefitinib",
    "erlotinib", "afatinib", "alectinib", "crizotinib", "pembrolizumab", "nivolumab", "atezolizumab", "bevacizumab", "sorafenib", "lenvatinib",
    "tamoxifen", "letrozole", "anastrozole", "exemestane", "trastuzumab", "pertuzumab", "palbociclib", "ribociclib", "abemaciclib", "capecitabine",
    "entecavir", "tenofovir", "methotrexate", "ibrutinib", "brentuximab", "bendamustine", "gemcitabine", "vincristine", "etoposide", "ifosfamide",
}


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def therapy_terms(text: str) -> set[str]:
    return set(squash(text).split()) & THERAPY_WORDS


def dossier_text(dossier: dict) -> str:
    """Dossier text for the leak check. Prior therapy for other illnesses is legitimately pre-treatment."""
    visible = {key: value for key, value in dossier.items() if key != "prior_treatments"}
    return json.dumps(visible, ensure_ascii=False)


def text_or_none(value: object) -> str | None:
    if value is None or value == "":
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def labelled(value: object, label: str) -> str | None:
    if value is None or value == "":
        return None
    return value if isinstance(value, str) else f"{label} {value}"


def to_case(uid: str, cancer: str, record: dict, row: dict) -> dict:
    dossier = record["dossier"]
    missing = [
        label
        for label, value in [
            ("performance status", dossier.get("performance_status")),
            ("stage", dossier.get("stage")),
            ("organ function", dossier.get("organ_function")),
            ("biomarkers", dossier.get("biomarkers")),
            ("comorbidities", dossier.get("comorbidities")),
        ]
        if not value
    ]
    lesions = []
    for lesion in dossier.get("lesions") or []:
        if not isinstance(lesion, dict):
            continue
        size = lesion.get("size_cm")
        lesions.append({
            "site": str(lesion.get("site") or "not available"),
            "size_cm": size if isinstance(size, (int, float)) else None,
            "description": text_or_none(lesion.get("description")),
            "stage_context": text_or_none(lesion.get("stage_context")),
        })
    biomarkers = [
        {"name": str(item.get("name")), "value": str(item.get("value"))}
        for item in dossier.get("biomarkers") or []
        if isinstance(item, dict) and item.get("name") and item.get("value") is not None
    ]
    age = dossier.get("age")
    # Models occasionally return numbers here. A bare number for sex has no reliable meaning, so it counts
    # as missing; stage and performance status keep their value under an explicit label.
    sex = dossier.get("sex") if isinstance(dossier.get("sex"), str) else None
    performance = labelled(dossier.get("performance_status"), "performance status")
    stage = labelled(dossier.get("stage"), "stage")
    return {
        "case_id": f"pmc-{uid}",
        "cancer_type": cancer,
        "domain": "evomdt_oncology",
        "task_type": "plan_eval",
        "patient_context": {
            "age": age if isinstance(age, int) else None,
            "sex": sex,
            "performance_status": performance,
            "summary": text_or_none(dossier.get("presentation")),
        },
        "stage": stage,
        "stage_system": text_or_none(dossier.get("stage_system")),
        "pathology_summary": text_or_none(dossier.get("pathology_summary")),
        "lesions": lesions,
        "biomarkers": biomarkers,
        "organ_function": {str(k): str(v) for k, v in (dossier.get("organ_function") or {}).items() if v is not None},
        "comorbidities": [str(item) for item in dossier.get("comorbidities") or []],
        "prior_treatments": [str(item) for item in dossier.get("prior_treatments") or []],
        "missing_data": missing,
        "question": "Provide an MDT-style recommendation for initial management of this patient, including treatment, safety checks and surveillance.",
        "metadata": {
            "dataset": "pmc_patients",
            "reference_plan": record["reference_plan_items"],
            "reference_plan_text": record["reference_plan_text"],
            "source_pmid": str(row["PMID"]),
            "source_title": str(row["title"])[:200],
        },
    }


async def prepare_realworld(args: argparse.Namespace) -> None:
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from evomdt.config import load_env_file
    from evomdt.llm import parse_json_content
    from evomdt.models import CaseDossier

    from openai import AsyncOpenAI

    load_env_file(ROOT / ".env")
    client = AsyncOpenAI(api_key=os.environ["NIM_API_KEY"], base_url=os.environ["NIM_BASE_URL"], timeout=300, max_retries=1)
    cache = OUT / "extraction_cache"
    cache.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RAW / "pmc_patients" / "PMC-Patients.csv", usecols=["patient_uid", "PMID", "title", "patient", "age"])
    frame = frame[frame.patient.str.len().between(args.min_chars, args.max_chars)]
    flow: dict[str, dict] = {}
    gate = asyncio.Semaphore(args.concurrency)

    def is_adult(raw: object) -> bool:
        try:
            pairs = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return False
        return any(unit == "year" and value >= 18 for value, unit in pairs)

    for cancer, (pattern, target) in COHORTS.items():
        # Screen on the title: reports that only mention the cancer in passing are rarely eligible.
        match = frame[frame.title.str.contains(pattern, na=False) & ~frame.title.str.contains(NOT_INITIAL, na=False)]
        match = match[match.age.map(is_adult)].drop_duplicates("PMID")
        rows = match.sample(frac=1, random_state=SEED).to_dict("records")
        wanted = target + args.dev_per_cancer
        counts: Counter[str] = Counter(candidates=len(rows))
        kept: list[dict] = []

        async def one(row: dict) -> dict | None:
            uid = str(row["patient_uid"])
            path = cache / f"{uid}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            prompt = EXTRACT_PROMPT.format(cancer=cancer.replace("_", " "), note=row["patient"])
            async with gate:
                for attempt in range(3):
                    try:
                        reply = await client.chat.completions.create(
                            model=args.model,
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"},
                            temperature=0,
                            seed=SEED,
                        )
                        record = parse_json_content(reply.choices[0].message.content or "")
                        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
                        return record
                    except Exception as exc:
                        print(f"  extraction retry {uid}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                        await asyncio.sleep(10 * (attempt + 1))
            return None

        position = 0
        while len(kept) < wanted and position < len(rows):
            batch = rows[position : position + args.batch]
            position += len(batch)
            counts["screened"] += len(batch)
            records = await asyncio.gather(*(one(row) for row in batch))
            for row, record in zip(batch, records):
                if record is None:
                    counts["extraction_failed"] += 1
                elif not record.get("eligible"):
                    counts["not_eligible"] += 1
                elif not record.get("reference_plan_items") or not record.get("reference_plan_text") or not isinstance(record.get("dossier"), dict):
                    counts["incomplete_extraction"] += 1
                elif squash(str(record.get("treatment_quote") or "")) not in squash(row["patient"]) or not record.get("treatment_quote"):
                    counts["quote_not_in_report"] += 1
                elif therapy_terms(record["reference_plan_text"]) & therapy_terms(dossier_text(record["dossier"])):
                    counts["treatment_leaked_into_dossier"] += 1
                else:
                    case = to_case(str(row["patient_uid"]), cancer, record, row)
                    CaseDossier.model_validate(case)
                    kept.append(case)
            print(f"{cancer}: screened {counts['screened']}, kept {len(kept)}/{wanted}, {dict(counts)}", flush=True)

        counts["kept"] = len(kept)
        flow[cancer] = dict(counts)
        write_jsonl(f"real_{cancer}", kept[:target])
        write_jsonl(f"dev_real_{cancer}", kept[target:wanted])
    (OUT / "realworld_flow.json").write_text(json.dumps(flow, indent=1), encoding="utf-8")
    print(json.dumps(flow, indent=1))


AUDIT_PROMPT = """You audit data prepared from a published case report. Compare the prepared items with the report.

Return JSON only:
- plan_faithful: true if the reference plan states only treatment that the report describes, and names its primary treatment correctly
- dossier_faithful: true if every dossier value is supported by the report
- no_leak: true if the dossier does not reveal the treatment given for this cancer, the response or the outcome
- problem: short text naming any problem, or null

CASE REPORT
{note}

PREPARED DOSSIER
{dossier}

PREPARED REFERENCE PLAN
{plan}
"""


async def audit_realworld(args: argparse.Namespace) -> None:
    """Independent check of a random sample of kept extractions, by a different model from the extractor."""
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from evomdt.config import load_env_file
    from evomdt.llm import parse_json_content

    from openai import AsyncOpenAI

    load_env_file(ROOT / ".env")
    client = AsyncOpenAI(api_key=os.environ["NIM_API_KEY"], base_url=os.environ["NIM_BASE_URL"], timeout=300, max_retries=1)
    notes = pd.read_csv(RAW / "pmc_patients" / "PMC-Patients.csv", usecols=["patient_uid", "patient"]).set_index("patient_uid").patient.to_dict()
    rng = random.Random(SEED)
    sample: list[tuple[str, dict]] = []
    for cancer in COHORTS:
        with (OUT / f"real_{cancer}.jsonl").open(encoding="utf-8") as handle:
            cases = [json.loads(line) for line in handle]
        rng.shuffle(cases)
        sample += [(cancer, case) for case in cases[: args.audit_per_cancer]]
    gate = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    cache = OUT / "audit_cache"
    cache.mkdir(exist_ok=True)

    async def one(cancer: str, case: dict) -> None:
        path = cache / f"{case['case_id']}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
            return
        keys = ["patient_context", "stage", "pathology_summary", "lesions", "biomarkers", "organ_function", "comorbidities", "prior_treatments"]
        dossier = json.dumps({key: case.get(key) for key in keys}, ensure_ascii=False)
        prompt = AUDIT_PROMPT.format(note=notes[case["case_id"].removeprefix("pmc-")], dossier=dossier, plan=case["metadata"]["reference_plan_text"])
        async with gate:
            for attempt in range(8):
                try:
                    reply = await client.chat.completions.create(
                        model=args.audit_model, messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}, temperature=0, seed=SEED,
                    )
                    record = parse_json_content(reply.choices[0].message.content or "")
                    audited = {"case_id": case["case_id"], "cancer": cancer, **record}
                    path.write_text(json.dumps(audited, ensure_ascii=False, indent=1), encoding="utf-8")
                    results.append(audited)
                    print(f"audited {len(results)}/{len(sample)}", flush=True)
                    return
                except Exception as exc:
                    await asyncio.sleep(min(180, 20 * (attempt + 1)))  # Hosted limits can last minutes.
                    error = str(exc)[:120]
            print(f"audit failed for {case['case_id']}: {error}", flush=True)

    await asyncio.gather(*(one(cancer, case) for cancer, case in sample))
    summary = {
        "auditor": args.audit_model,
        "audited": len(results),
        "plan_faithful": sum(r.get("plan_faithful") is True for r in results),
        "dossier_faithful": sum(r.get("dossier_faithful") is True for r in results),
        "no_leak": sum(r.get("no_leak") is True for r in results),
        "cases": results,
    }
    (OUT / "realworld_audit.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=1))
    for result in results:
        if result.get("problem"):
            print(result["case_id"], str(result["problem"])[:200])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["benchmarks", "realworld", "audit"])
    parser.add_argument("--audit-model", default="mistralai/mistral-large-4-1t-a52b")
    parser.add_argument("--audit-per-cancer", type=int, default=15)
    parser.add_argument("--medqa", type=int, default=200)
    parser.add_argument("--medmcqa", type=int, default=250)
    parser.add_argument("--medquad", type=int, default=100)
    parser.add_argument("--dev", type=int, default=20)
    parser.add_argument("--model", default=None, help="Extraction model. Default: NIM_MODEL_JUDGE from .env.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--dev-per-cancer", type=int, default=5)
    parser.add_argument("--min-chars", type=int, default=700)
    parser.add_argument("--max-chars", type=int, default=7000)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.step == "benchmarks":
        prepare_benchmarks(args)
    elif args.step == "audit":
        asyncio.run(audit_realworld(args))
    else:
        if args.model is None:
            sys.path.insert(0, str(ROOT))
            from evomdt.config import load_env_file

            load_env_file(ROOT / ".env")
            args.model = os.environ["NIM_MODEL_JUDGE"]
        asyncio.run(prepare_realworld(args))


if __name__ == "__main__":
    main()
