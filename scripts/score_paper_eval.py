"""Score the EvoMDT evaluation and write the paper's tables and figures to results/paper_reproduction/.

  judge    run the blinded LLM judges (cached, so safe to resume)
  report   compute every metric and write tables, figures and summary.json

Layer 3 of the paper is a physician rating that cannot be reproduced without physicians. This script
writes a blinded rating sheet, and runs the same analysis on it once it is filled in. Until then the
LLM raters stand in, and everything derived from them is labelled as a proxy.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evomdt.config import load_env_file  # noqa: E402
from evomdt.llm import parse_json_content  # noqa: E402

DATA = ROOT / "data" / "paper_eval"
RUNS = ROOT / "artifacts" / "paper_eval" / "runs"
JUDGE_DIR = ROOT / "artifacts" / "paper_eval" / "judge"
RESULTS = ROOT / "results" / "paper_reproduction"
SEED = 7
SIMPLE_BASELINE = "Take cold medicine."  # The paper's example of an obviously invalid plan.
CANCERS = ["breast_cancer", "hepatocellular_carcinoma", "lung_adenocarcinoma", "lymphoma"]
CANCER_LABEL = {"breast_cancer": "Breast", "hepatocellular_carcinoma": "HCC", "lung_adenocarcinoma": "Lung", "lymphoma": "Lymphoma"}
MCQ_SETS = {"medqa": "MedQA", "medqa_cancer": "MedQA-Cancer", "medmcqa_cancer": "MedMCQA-Cancer"}
GEN_SETS = {"medquad": "MedQuAD", "rond": "ROND"}
PRIORITY_WEIGHT = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
DIMENSIONS = ["Clinical Appropriateness", "Response Completeness", "Response Efficiency", "Evidence Utilization", "Patient-Centeredness", "Safety & Ethics"]

# Paper values, used only as comparison points in the report and figures.
PAPER = {
    "medqa": 0.86, "medqa_cancer": 0.89, "medmcqa_cancer": 0.75, "medquad": 0.63, "rond": 0.68, "medqaro": 0.66,
    "rouge1": {"breast_cancer": 0.05, "hepatocellular_carcinoma": 0.04, "lung_adenocarcinoma": 0.06, "lymphoma": 0.05},
    "similarity": {"breast_cancer": 0.68, "hepatocellular_carcinoma": 0.62, "lung_adenocarcinoma": 0.66, "lymphoma": 0.68},
    "ai": {"breast_cancer": 7.0, "hepatocellular_carcinoma": 7.0, "lung_adenocarcinoma": 9.0, "lymphoma": 9.0},
    "composite": 3.44, "time_median_min": 10.6,
}

JUDGE_PROMPT = """You are an oncologist scoring how well a candidate treatment plan matches the plan an expert team chose for the same patient.

Score from 0 to 10:
0 = clinically invalid or unrelated to this disease
2-3 = right disease area, but the wrong approach for this patient
5 = same general strategy, with important components missing or different
7-8 = same primary treatment, with minor differences
10 = clinically equivalent to the expert plan

Extra detail that is safe and reasonable does not lower the score. Judge clinical meaning, not wording. A candidate that offers several options without committing to one scores lower than one that commits to the expert's option.

Return JSON only: {{"score": <integer 0-10>, "items_covered": [<true or false for each numbered expert item, in order>], "reason": "<one sentence>"}}

PATIENT
{dossier}

EXPERT PLAN
{reference}

Numbered expert items:
{items}

CANDIDATE PLAN
{candidate}
"""

RUBRIC = {
    "Clinical Appropriateness": "1 off-guideline; 2 major deviations; 3 broadly aligned but thin; 4 highly consistent (minor issues); 5 strictly guideline-concordant with sound judgment",
    "Response Completeness": "1 key steps missing; 2 fragmented; 3 basically complete; 4 clear loop; 5 fully closed loop with roles and timing",
    "Response Efficiency": "1 verbose or chaotic; 2 loose or missing; 3 clear but unquantified; 4 concise with thresholds; 5 conclusion-first, fully actionable thresholds",
    "Evidence Utilization": "1 none or outdated; 2 sparse; 3 partial grading; 4 mostly high-level; 5 up-to-date per-item mapping with grade, year and system",
    "Patient-Centeredness": "1 one-size-fits-all; 2 vague; 3 key factors; 4 thorough plus shared decisions; 5 full plan including fertility, genetics, psychology and cost",
    "Safety & Ethics": "1 ignored; 2 adverse events listed only; 3 thin measures; 4 clear monitoring and prevention; 5 prevent, monitor, manage and refer, compliant",
}
RUBRIC_FOCUS = {
    "lung_adenocarcinoma": "Stage fit; NGS and PD-L1; perioperative IO or TKI; brain-metastasis local therapy; CT or PET-CT and brain MRI; irAEs, TKI toxicity, RT pneumonitis.",
    "breast_cancer": "Subtype and stage fit; ER/PR/HER2 and Ki-67; clear (neo)adjuvant pathway; SLNB and axilla; bone health; HER2 cardiotoxicity, endocrine and VTE risk, fertility.",
    "hepatocellular_carcinoma": "BCLC with Child-Pugh and ECOG; resection, ablation, TACE, TARE, transplant or systemic options; PVTT plan; HBV/HCV antivirals; AFP; varices screen; liver and bleeding risk.",
    "lymphoma": "Histology with Ann Arbor or Lugano; IPI or CNS-IPI; first-line regimen; PET-CT with Deauville; marrow and CNS work-up; HBV screen; TLS, anthracycline cardiotoxicity, neuropathy.",
}
RATER_PROMPT = """You are a board-certified oncologist rating two tumour-board plans for the same patient. You do not know who or what wrote either plan. Rate each plan on its own merits against the rubric.

Rubric (score each dimension 1 to 5, integers only):
{rubric}

Disease-specific points for this case: {focus}

Return JSON only: {{"A": {{<dimension name>: <score>, ...}}, "B": {{<dimension name>: <score>, ...}}}}, using these exact dimension names: {names}.

PATIENT DOSSIER
{dossier}

PLAN A
{plan_a}

PLAN B
{plan_b}
"""
BCLC_LABEL_PROMPT = """You label an oncology tumour-board plan. Report what the plan says. Do not judge whether it is correct.

Return JSON only:
- primary_modality: the anti-cancer treatment the plan makes its main recommendation. One of: {modalities}. Work-up before treatment does not count. Use no_treatment_named only if the plan commits to no modality.
- quote: a verbatim excerpt of 5 to 30 words copied from the plan that supports primary_modality.

PLAN
{plan}
"""
MODALITIES = ["resection", "ablation", "transplant", "tace", "tare", "radiotherapy", "systemic", "best_supportive_care", "no_treatment_named", "other"]
# Reig et al., J Hepatol 2022, Fig. 1.
BCLC_2022 = {"A": {"resection", "ablation", "transplant"}, "B": {"transplant", "tace", "systemic"}, "C": {"systemic"}, "D": {"best_supportive_care"}}


# ------------------------------------------------------------------------------ loading


def read_cases(name: str) -> list[dict]:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_runs(system: str) -> dict[str, dict]:
    folder = RUNS / re.sub(r"[^a-z0-9._-]+", "_", system.lower())
    if not folder.exists():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json") if p.name != "failures.json"}


def answer_of(result: dict) -> str:
    return result["coordinator_decision"]["final_answer"] if "coordinator_decision" in result else result["final_answer"]


def plan_text(result: dict) -> str:
    """A system's full care plan: the narrative plus, for EvoMDT, its itemized plan and safety and monitoring text."""
    if "coordinator_decision" not in result:
        return result["final_answer"]
    decision = result["coordinator_decision"]
    parts = [decision["final_answer"]]
    if decision.get("final_plan"):
        parts.append("Plan items:\n" + "\n".join(f"- {item['action']}" for item in decision["final_plan"]))
    if decision.get("safety_assessment"):
        parts.append(f"Safety: {decision['safety_assessment']}")
    if decision.get("monitoring_protocol"):
        parts.append(f"Monitoring: {decision['monitoring_protocol']}")
    return "\n".join(parts)


def dossier_for_prompt(case: dict) -> str:
    keys = ["cancer_type", "patient_context", "stage", "stage_system", "pathology_summary", "lesions", "biomarkers", "organ_function", "comorbidities", "prior_treatments", "missing_data"]
    return json.dumps({key: case.get(key) for key in keys if case.get(key)}, ensure_ascii=False)


def shuffled_references(cases: list[dict]) -> dict[str, str]:
    """Paper's random baseline: another patient's plan, same cancer. A rotation guarantees no self-match."""
    rng = random.Random(SEED)
    order = cases[:]
    rng.shuffle(order)
    return {
        case["case_id"]: order[(index + 1) % len(order)]["metadata"]["reference_plan_text"]
        for index, case in enumerate(order)
    }


PER_CANCER: int | None = None  # Set from --per-cancer. Every Layer 2 step uses this same cohort.


def layer2_cases() -> dict[str, list[dict]]:
    return {cancer: read_cases(f"real_{cancer}")[:PER_CANCER] for cancer in CANCERS}


def layer2_runs() -> dict[str, dict]:
    """EvoMDT results for the Layer 2 cohort only."""
    results = read_runs("evomdt")
    return {case["case_id"]: results[case["case_id"]] for cases in layer2_cases().values() for case in cases if case["case_id"] in results}


# -------------------------------------------------------------------------------- judges


async def call_judge(client, model: str, prompt: str, path: Path, gate: asyncio.Semaphore, validate, stats: dict) -> None:
    if path.exists():
        return
    async with gate:
        for attempt in range(8):
            try:
                reply = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}, temperature=0, seed=SEED,
                )
                record = parse_json_content(reply.choices[0].message.content or "")
                validate(record)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
                stats["ok"] += 1
                if stats["ok"] % 25 == 0:
                    print(f"  {model}: {stats['ok']} done, {stats['failed']} failed", flush=True)
                return
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:140]}"
                await asyncio.sleep(min(180, 20 * (attempt + 1)))  # Hosted limits can last minutes.
        stats["failed"] += 1
        print(f"  judge gave up on {path.name}: {error}", flush=True)


def validate_score(record: dict) -> None:
    if not 0 <= float(record["score"]) <= 10:
        raise ValueError("score out of range")


def validate_rubric(record: dict) -> None:
    for plan in ("A", "B"):
        for dimension in DIMENSIONS:
            if not 1 <= float(record[plan][dimension]) <= 5:
                raise ValueError("rubric score out of range")


def ab_order(case_id: str, rater: str) -> bool:
    """Deterministic but unpredictable A/B assignment per case and rater. True puts EvoMDT in slot A."""
    return int(hashlib.sha256(f"{case_id}|{rater}".encode()).hexdigest(), 16) % 2 == 0


async def run_judges(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    load_env_file(ROOT / ".env")
    client = AsyncOpenAI(api_key=os.environ["NIM_API_KEY"], base_url=os.environ["NIM_BASE_URL"], timeout=300, max_retries=2)
    evomdt = read_runs("evomdt")
    single = read_runs(args.single) if args.single else {}
    tasks = []
    for model in args.models:
        gate = asyncio.Semaphore(args.concurrency)
        stats = {"ok": 0, "failed": 0}
        tag = re.sub(r"[^a-z0-9]+", "_", model.lower())
        for cancer, cases in layer2_cases().items():
            random_refs = shuffled_references(cases) if cases else {}
            for index, case in enumerate(cases):
                result = evomdt.get(case["case_id"])
                if result is None or (model != args.models[0] and index >= args.second_judge_per_cancer):
                    continue
                meta = case["metadata"]
                items = "\n".join(f"{i}. {item['action']}" for i, item in enumerate(meta["reference_plan"], 1))
                conditions = {"evomdt": plan_text(result)}
                # The first judge scores every arm; further judges score a sample of EvoMDT, for agreement.
                # The random and invalid-text baselines need only a sample: they exist to show the scale's floor.
                if model == args.models[0]:
                    if index < args.random_per_cancer:
                        conditions["random"] = random_refs[case["case_id"]]
                    if case["case_id"] in single:
                        conditions["single"] = single[case["case_id"]]["final_answer"]
                    if index < args.simple_per_cancer:
                        conditions["simple"] = SIMPLE_BASELINE
                for condition, candidate in conditions.items():
                    prompt = JUDGE_PROMPT.format(dossier=dossier_for_prompt(case), reference=meta["reference_plan_text"], items=items, candidate=candidate)
                    tasks.append(call_judge(client, model, prompt, JUDGE_DIR / tag / condition / f"{case['case_id']}.json", gate, validate_score, stats))
                slot_a_is_evomdt = ab_order(case["case_id"], model)
                plans = (plan_text(result), meta["reference_plan_text"])
                plan_a, plan_b = plans if slot_a_is_evomdt else plans[::-1]
                prompt = RATER_PROMPT.format(
                    rubric="\n".join(f"- {name}: {text}" for name, text in RUBRIC.items()),
                    focus=RUBRIC_FOCUS[cancer], names=", ".join(DIMENSIONS),
                    dossier=dossier_for_prompt(case), plan_a=plan_a, plan_b=plan_b,
                )
                tasks.append(call_judge(client, model, prompt, JUDGE_DIR / tag / "rubric" / f"{case['case_id']}.json", gate, validate_rubric, stats))
        if model == args.models[0]:
            label_gate = asyncio.Semaphore(args.concurrency)
            for case_id, result in read_runs("evomdt_hcc_tace").items():
                prompt = BCLC_LABEL_PROMPT.format(modalities=", ".join(MODALITIES), plan=plan_text(result))
                tasks.append(call_judge(client, model, prompt, JUDGE_DIR / tag / "bclc_label" / f"{case_id}.json", label_gate, lambda r: r["primary_modality"], stats))
    print(f"{len(tasks)} judge tasks (cached ones finish immediately)", flush=True)
    await asyncio.gather(*tasks)
    print("judging complete", flush=True)


# ------------------------------------------------------------------------------ metrics


def wilson(hits: int, total: int) -> tuple[float, float]:
    if not total:
        return (float("nan"), float("nan"))
    z = 1.96
    p = hits / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    denom = 1 + z * z / total
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def bootstrap_ci(values: list[float], seed: int = SEED, rounds: int = 2000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = rng.choice(array, size=(rounds, len(array))).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def five_seed_mean(values: list[float]) -> tuple[float, float]:
    """The paper's 'resampling over 5 random seeds': mean and SD of five bootstrap means."""
    array = np.asarray(values, dtype=float)
    means = [np.random.default_rng(seed).choice(array, size=len(array)).mean() for seed in range(5)]
    return (float(np.mean(means)), float(np.std(means)))


class TokenEncoder:
    """Token-level contextual embeddings for BERTScore (Zhang et al. 2020), usable with any encoder."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, texts: list[str]):
        torch = self.torch
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt", return_special_tokens_mask=True)
        special = batch.pop("special_tokens_mask").bool()
        batch = batch.to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        hidden = torch.nn.functional.normalize(hidden, dim=-1)
        keep = batch["attention_mask"].bool() & ~special.to(self.device)
        return [hidden[i][keep[i]] for i in range(len(texts))]

    def bertscore(self, candidates: list[str], references: list[str], batch_size: int = 8) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(candidates), batch_size):
            cand = self.encode(candidates[start : start + batch_size])
            refs = self.encode(references[start : start + batch_size])
            for c, r in zip(cand, refs):
                if not len(c) or not len(r):
                    scores.append(0.0)
                    continue
                sim = c @ r.T
                precision = sim.max(dim=1).values.mean().item()
                recall = sim.max(dim=0).values.mean().item()
                scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
        return scores

    def close(self) -> None:
        del self.model
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def icc_2_1(matrix: np.ndarray) -> float:
    """ICC(2,1), Shrout and Fleiss: two-way random effects, absolute agreement, single rater."""
    n, k = matrix.shape
    grand = matrix.mean()
    ms_rows = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((matrix.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + grand
    ms_error = (residual**2).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def icc_with_ci(matrix: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(500):
        sample = matrix[rng.integers(0, len(matrix), len(matrix))]
        estimate = icc_2_1(sample)
        if not math.isnan(estimate):
            values.append(estimate)
    return (icc_2_1(matrix), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def write_table(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = RESULTS / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# -------------------------------------------------------------------------------- layers


def score_layer1(systems: list[str], encoders: dict[str, TokenEncoder]) -> dict:
    from scipy.stats import binomtest, wilcoxon

    runs = {system: read_runs(system) for system in systems}
    rows, gen_rows, out = [], [], {}
    for key, label in MCQ_SETS.items():
        cases = read_cases(key)
        out[key] = {}
        correct_by_system: dict[str, dict[str, bool]] = {}
        for system in systems:
            hits = {}
            for case in cases:
                result = runs[system].get(case["case_id"])
                if result is not None:
                    hits[case["case_id"]] = answer_of(result).strip().upper().strip("().") == case["reference_answer"].upper()
            correct_by_system[system] = hits
            if not hits:
                continue
            n, k = len(hits), sum(hits.values())
            low, high = wilson(k, n)
            mean, sd = five_seed_mean([float(v) for v in hits.values()])
            out[key][system] = {"accuracy": round(k / n, 4), "n": n, "of": len(cases), "ci95": [round(low, 3), round(high, 3)]}
            rows.append({"dataset": label, "system": system, "n": n, "planned": len(cases), "accuracy": round(k / n, 4), "ci95_low": round(low, 3), "ci95_high": round(high, 3), "five_seed_mean": round(mean, 4), "five_seed_sd": round(sd, 4), "paper_evomdt": PAPER[key]})
        base = systems[0]
        for system in systems[1:]:
            shared = set(correct_by_system[base]) & set(correct_by_system[system])
            only_base = sum(correct_by_system[base][c] and not correct_by_system[system][c] for c in shared)
            only_other = sum(correct_by_system[system][c] and not correct_by_system[base][c] for c in shared)
            if only_base + only_other:
                p_value = binomtest(only_base, only_base + only_other, 0.5).pvalue
                out[key].setdefault("mcnemar", {})[system] = {"evomdt_only_correct": only_base, "other_only_correct": only_other, "p": round(float(p_value), 4)}

    for key, label in GEN_SETS.items():
        cases = read_cases(key)
        out[key] = {}
        per_case: dict[str, dict[str, float]] = {}
        for system in systems:
            present = [case for case in cases if case["case_id"] in runs[system]]
            if not present:
                continue
            candidates = [answer_of(runs[system][case["case_id"]]) for case in present]
            references = [case["reference_answer"] for case in present]
            by_encoder = {name: encoder.bertscore(candidates, references) for name, encoder in encoders.items()}
            combined = np.mean(list(by_encoder.values()), axis=0).tolist()
            per_case[system] = {case["case_id"]: value for case, value in zip(present, combined)}
            mean, sd = five_seed_mean(combined)
            low, high = bootstrap_ci(combined)
            out[key][system] = {"bertscore_f1": round(float(np.mean(combined)), 4), "n": len(present), "of": len(cases), "ci95": [round(low, 3), round(high, 3)], "by_encoder": {name: round(float(np.mean(values)), 4) for name, values in by_encoder.items()}}
            gen_rows.append({"dataset": label, "system": system, "n": len(present), "planned": len(cases), "bertscore_f1_mean_of_encoders": round(float(np.mean(combined)), 4), **{f"f1_{name}": round(float(np.mean(values)), 4) for name, values in by_encoder.items()}, "ci95_low": round(low, 3), "ci95_high": round(high, 3), "five_seed_mean": round(mean, 4), "five_seed_sd": round(sd, 4), "paper_evomdt": PAPER[key]})
        base = systems[0]
        for system in systems[1:]:
            if base in per_case and system in per_case:
                shared = sorted(set(per_case[base]) & set(per_case[system]))
                difference = [per_case[base][c] - per_case[system][c] for c in shared]
                if any(difference):
                    out[key].setdefault("wilcoxon", {})[system] = {"mean_difference": round(float(np.mean(difference)), 4), "p": round(float(wilcoxon(difference).pvalue), 4)}
    write_table("layer1_multiple_choice", rows)
    write_table("layer1_generative_bertscore", gen_rows)
    return out


def judge_scores(models: list[str], condition: str, case_id: str) -> list[dict]:
    found = []
    for model in models:
        path = JUDGE_DIR / re.sub(r"[^a-z0-9]+", "_", model.lower()) / condition / f"{case_id}.json"
        if path.exists():
            found.append(json.loads(path.read_text(encoding="utf-8")))
    return found


def judge_agreement(models: list[str]) -> dict:
    """How closely the second judge's 0-10 scores track the primary judge's, on the cases both scored."""
    from scipy.stats import spearmanr

    if len(models) < 2:
        return {}
    pairs = []
    for cases in layer2_cases().values():
        for case in cases:
            judged = [judge_scores([model], "evomdt", case["case_id"]) for model in models[:2]]
            if all(judged):
                pairs.append([float(judged[0][0]["score"]), float(judged[1][0]["score"])])
    if len(pairs) < 3:
        return {"n": len(pairs)}
    matrix = np.array(pairs)
    icc, low, high = icc_with_ci(matrix)
    rho = spearmanr(matrix[:, 0], matrix[:, 1]).statistic
    return {
        "n": len(pairs), "primary": models[0], "second": models[1],
        "primary_mean": round(float(matrix[:, 0].mean()), 3), "second_mean": round(float(matrix[:, 1].mean()), 3),
        "icc_2_1": round(icc, 3), "icc_ci95": [round(low, 3), round(high, 3)],
        "spearman": None if math.isnan(rho) else round(float(rho), 3),
    }


def covered_weight(case: dict, judged: list[dict]) -> float | None:
    """Weighted plan concordance: judged coverage of the reference items, weighted by priority."""
    items = case["metadata"]["reference_plan"]
    values = []
    for record in judged:
        flags = record.get("items_covered") or []
        if len(flags) != len(items):
            continue
        weights = [PRIORITY_WEIGHT.get(int(item.get("priority", 3)), 0.2) for item in items]
        values.append(sum(w for w, flag in zip(weights, flags) if flag is True) / sum(weights))
    return float(np.mean(values)) if values else None


def score_layer2(models: list[str], single_system: str | None, encoders: dict[str, TokenEncoder]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    evomdt = read_runs("evomdt")
    single = read_runs(single_system) if single_system else {}
    from evomdt.retrieval import Embedder

    embedder = Embedder("BAAI/bge-m3")
    rows, per_case, out = [], [], {}
    for cancer, cases in layer2_cases().items():
        present = [case for case in cases if case["case_id"] in evomdt]
        out[cancer] = {"planned": len(cases), "completed": len(present)}
        if not present:
            continue
        references = [case["metadata"]["reference_plan_text"] for case in present]
        shuffled = shuffled_references(cases)
        candidates = {
            "evomdt": [answer_of(evomdt[case["case_id"]]) for case in present],
            "random": [shuffled[case["case_id"]] for case in present],
            "simple": [SIMPLE_BASELINE] * len(present),
        }
        if single:
            candidates["single"] = [single.get(case["case_id"], {}).get("final_answer", "") for case in present]
        reference_vectors = embedder.encode(references)
        for condition, texts in candidates.items():
            mask = [bool(text) for text in texts]
            kept_refs = [ref for ref, keep in zip(references, mask) if keep]
            kept_cases = [case for case, keep in zip(present, mask) if keep]
            kept_texts = [text for text in texts if text]
            if not kept_texts:
                continue
            similarity = (embedder.encode(kept_texts) * reference_vectors[np.array(mask)]).sum(axis=1)
            bert = np.mean([encoder.bertscore(kept_texts, kept_refs) for encoder in encoders.values()], axis=0)
            # Standard recall: overlap / reference length. The second direction is overlap / candidate
            # length, the only reading under which the paper's 0.04-0.06 is reachable for long plans.
            forward = [scorer.score(ref, text) for ref, text in zip(kept_refs, kept_texts)]
            reverse = [scorer.score(text, ref) for ref, text in zip(kept_refs, kept_texts)]
            ai_scores, coverage = [], []
            for case in kept_cases:
                judged = judge_scores(models[:1], condition, case["case_id"])
                ai_scores.append(float(np.mean([float(r["score"]) for r in judged])) if judged else None)
                coverage.append(covered_weight(case, judged))
            for index, case in enumerate(kept_cases):
                per_case.append({
                    "case_id": case["case_id"], "cancer": cancer, "condition": condition,
                    "rouge1_recall": round(forward[index]["rouge1"].recall, 4), "rougeL_recall": round(forward[index]["rougeL"].recall, 4),
                    "rouge1_over_candidate": round(reverse[index]["rouge1"].recall, 4), "rougeL_over_candidate": round(reverse[index]["rougeL"].recall, 4),
                    "cosine_similarity": round(float(similarity[index]), 4), "bertscore_f1": round(float(bert[index]), 4),
                    "ai_score": None if ai_scores[index] is None else round(ai_scores[index], 2),
                    "wpc": None if coverage[index] is None else round(coverage[index], 4),
                })
            judged_scores = [value for value in ai_scores if value is not None]
            judged_cover = [value for value in coverage if value is not None]
            summary = {
                "n": len(kept_cases),
                "rouge1_recall": round(float(np.mean([f["rouge1"].recall for f in forward])), 4),
                "rougeL_recall": round(float(np.mean([f["rougeL"].recall for f in forward])), 4),
                "rouge1_over_candidate": round(float(np.mean([r["rouge1"].recall for r in reverse])), 4),
                "rougeL_over_candidate": round(float(np.mean([r["rougeL"].recall for r in reverse])), 4),
                "cosine_similarity": round(float(similarity.mean()), 4),
                "bertscore_f1": round(float(bert.mean()), 4),
                "ai_score_0_10": round(float(np.mean(judged_scores)), 3) if judged_scores else None,
                "ai_score_median": round(float(np.median(judged_scores)), 2) if judged_scores else None,
                "ai_score_n": len(judged_scores),
                "weighted_plan_concordance": round(float(np.mean(judged_cover)), 4) if judged_cover else None,
            }
            out[cancer][condition] = summary
            rows.append({"cancer": CANCER_LABEL[cancer], "condition": condition, **summary})

        results = [evomdt[case["case_id"]] for case in present]
        metrics = [result["evaluation"]["metrics"] for result in results]
        durations = [result["duration_seconds"] for result in results]
        out[cancer]["pipeline"] = {
            "evidence_traceability": round(float(np.mean([m.get("evidence_traceability_coverage") or 0 for m in metrics])), 4),
            "safety_violation_rate": round(float(np.mean([m.get("safety_violation_rate") or 0 for m in metrics])), 4),
            "guardrail_removals": sum(1 for r in results for entry in r["coordinator_decision"]["audit_trace"] if entry.startswith("Guardrail removed")),
            "coordinator_fallbacks": sum(1 for r in results if not any(t["role"] == "coordinator" for t in r["agent_traces"])),
            "dropped_citations": sum(t.get("dropped_citations", 0) for r in results for t in r["agent_traces"]),
            "median_seconds": round(float(np.median(durations)), 1),
            "median_tokens": int(np.median([sum(t["usage"].get("total_tokens") or 0 for t in r["agent_traces"]) for r in results])),
        }
    write_table("layer2_by_cancer_and_condition", rows)
    write_table("layer2_per_case", per_case)
    return out


def audit_sensitivity() -> dict:
    """Layer 2 scores with and without the audited cases whose reference plan the auditor rated unfaithful."""
    audit_path, table_path = DATA / "realworld_audit.json", RESULTS / "tables" / "layer2_per_case.csv"
    if not audit_path.exists() or not table_path.exists():
        return {}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cohort = {case["case_id"] for cases in layer2_cases().values() for case in cases}
    audited = [case for case in audit["cases"] if case["case_id"] in cohort]
    flagged = {case["case_id"] for case in audited if case["plan_faithful"] is False}
    with table_path.open(encoding="utf-8") as handle:
        per_case = list(csv.DictReader(handle))

    def mean(rows: list[dict], column: str) -> float | None:
        values = [float(row[column]) for row in rows if row[column] not in ("", None)]
        return round(float(np.mean(values)), 4) if values else None

    rows = []
    for condition in ("evomdt", "single"):
        part = [row for row in per_case if row["condition"] == condition]
        for label, subset in (("all cases", part), ("without flagged plans", [row for row in part if row["case_id"] not in flagged])):
            if subset:
                rows.append({"condition": condition, "cases": label, "n": len(subset), **{column: mean(subset, column) for column in ("ai_score", "wpc", "cosine_similarity", "bertscore_f1", "rouge1_recall")}})
    write_table("layer2_audit_sensitivity", rows)
    return {
        "auditor": audit["auditor"],
        "audited_whole_pool": audit["audited"],
        "audited_in_cohort": len(audited),
        "plan_faithful_in_cohort": sum(case["plan_faithful"] is True for case in audited),
        "dossier_faithful_in_cohort": sum(case["dossier_faithful"] is True for case in audited),
        "no_leak_in_cohort": sum(case["no_leak"] is True for case in audited),
        "flagged_plans_excluded": sorted(flagged),
        "scores": rows,
    }


def score_hcc_tace(models: list[str]) -> dict:
    """Guideline concordance against BCLC 2022, on the real HCC-TACE-Seg cohort (rule-based, no LLM score)."""
    results = read_runs("evomdt_hcc_tace")
    if not results or not models:
        return {}
    tag = re.sub(r"[^a-z0-9]+", "_", models[0].lower())
    cases = {case["case_id"]: case for case in read_cases("hcc_tace")}
    per_stage: dict[str, list[bool]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    verified = 0
    for case_id, result in results.items():
        path = JUDGE_DIR / tag / "bclc_label" / f"{case_id}.json"
        if not path.exists():
            continue
        label = json.loads(path.read_text(encoding="utf-8"))
        stage = cases[case_id]["stage"].replace("BCLC ", "")
        primary = label["primary_modality"]
        counts[primary] += 1
        per_stage[stage].append(primary in BCLC_2022[stage])
        plan = re.sub(r"[^a-z0-9]+", " ", plan_text(result).lower())
        verified += re.sub(r"[^a-z0-9]+", " ", str(label.get("quote", "")).lower()).strip() in plan
    pooled = [value for values in per_stage.values() for value in values]
    if not pooled:
        return {}
    low, high = wilson(sum(pooled), len(pooled))
    metrics = [result["evaluation"]["metrics"] for result in results.values()]
    summary = {
        "n_completed": len(results),
        "n_labelled": len(pooled),
        "bclc_2022_guideline_concordance": round(sum(pooled) / len(pooled), 4),
        "ci95": [round(low, 3), round(high, 3)],
        "by_stage": {stage: {"n": len(values), "concordant": sum(values)} for stage, values in sorted(per_stage.items())},
        "primary_modality_counts": dict(counts),
        "label_quote_verified": verified,
        "evidence_traceability": round(float(np.mean([m.get("evidence_traceability_coverage") or 0 for m in metrics])), 4),
        "safety_violation_rate": round(float(np.mean([m.get("safety_violation_rate") or 0 for m in metrics])), 4),
    }
    write_table("hcc_tace_bclc_concordance", [{"stage": stage, **values} for stage, values in summary["by_stage"].items()])
    return summary


def collect_ratings(models: list[str]) -> list[dict]:
    rows = []
    case_cancer = {case["case_id"]: cancer for cancer, cases in layer2_cases().items() for case in cases}
    for model in models:
        tag = re.sub(r"[^a-z0-9]+", "_", model.lower())
        for path in (JUDGE_DIR / tag / "rubric").glob("*.json"):
            case_id = path.stem
            if case_id not in case_cancer:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            slot_a_is_evomdt = ab_order(case_id, model)
            for slot, plan in (("A", "evomdt" if slot_a_is_evomdt else "documented"), ("B", "documented" if slot_a_is_evomdt else "evomdt")):
                for dimension in DIMENSIONS:
                    rows.append({"case_id": case_id, "cancer": case_cancer[case_id], "rater": model, "plan": plan, "dimension": dimension, "score": float(record[slot][dimension])})
    return rows


def analyse_ratings(rows: list[dict], *, prefix: str, primary: str | None = None) -> dict:
    """The paper's Layer 3 analysis (Table 1; Fig. 4a, c-f), for physician ratings or the LLM proxy."""
    from scipy.stats import kruskal, mannwhitneyu, spearmanr

    if not rows:
        return {}
    raters = sorted({row["rater"] for row in rows})
    every_rater = [row for row in rows if row["plan"] == "evomdt"]
    # With a primary rater, a second rater who scored only a sample cannot skew the means or tests.
    rows = [row for row in rows if primary is None or row["rater"] == primary]
    evomdt = [row for row in rows if row["plan"] == "evomdt"]
    out: dict = {
        "raters": raters, "primary_rater": primary, "cases": len({row["case_id"] for row in evomdt}),
        "cases_per_rater": {rater: len({row["case_id"] for row in every_rater if row["rater"] == rater}) for rater in raters},
    }

    table = []
    for cancer in CANCERS:
        part = [row for row in every_rater if row["cancer"] == cancer]
        if not part:
            continue
        overall: dict[str, object] = {"Cancer": CANCER_LABEL[cancer], "Dimension": "Overall (/5)"}
        for rater in raters:
            overall[rater] = round(float(np.mean([row["score"] for row in part if row["rater"] == rater] or [float("nan")])), 2)
        table.append(overall)
        for dimension in DIMENSIONS:
            line: dict[str, object] = {"Cancer": CANCER_LABEL[cancer], "Dimension": dimension}
            for rater in raters:
                line[rater] = round(float(np.mean([row["score"] for row in part if row["rater"] == rater and row["dimension"] == dimension] or [float("nan")])), 2)
            table.append(line)
    write_table(f"{prefix}_table1", table)

    def case_means(subset: list[dict]) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in subset:
            grouped[row["case_id"]].append(row["score"])
        return {key: float(np.mean(values)) for key, values in grouped.items()}

    composite = case_means(evomdt)
    low, high = bootstrap_ci(list(composite.values()))
    out["composite_mean"] = round(float(np.mean(list(composite.values()))), 3)
    out["composite_ci95"] = [round(low, 3), round(high, 3)]
    comparator = case_means([row for row in rows if row["plan"] == "documented"])
    if comparator:
        out["documented_plan_composite_mean"] = round(float(np.mean(list(comparator.values()))), 3)
    out["dimensions"] = {}
    for dimension in DIMENSIONS:
        values = [row["score"] for row in evomdt if row["dimension"] == dimension]
        lo, hi = bootstrap_ci(list(case_means([row for row in evomdt if row["dimension"] == dimension]).values()))
        distribution = {str(score): round(100 * sum(1 for v in values if round(v) == score) / len(values), 1) for score in range(1, 6)}
        out["dimensions"][dimension] = {"mean": round(float(np.mean(values)), 3), "ci95": [round(lo, 3), round(hi, 3)], "high_score_rate_pct": round(100 * float(np.mean(np.array(values) >= 4)), 1), "distribution_pct": distribution}

    by_cancer = [list(case_means([row for row in evomdt if row["cancer"] == cancer]).values()) for cancer in CANCERS]
    by_cancer = [values for values in by_cancer if values]
    if len(by_cancer) > 1:
        out["kruskal_wallis_between_cancers_p"] = round(float(kruskal(*by_cancer).pvalue), 4)
    if comparator:
        shared = sorted(set(composite) & set(comparator))
        test = mannwhitneyu([composite[c] for c in shared], [comparator[c] for c in shared])
        out["mann_whitney_vs_documented"] = {"p": float(test.pvalue), "rank_biserial": round(float(2 * test.statistic / len(shared) ** 2 - 1), 3)}

    if len(raters) > 1:
        out["icc_2_1"] = {}
        for dimension in DIMENSIONS:
            lookup = {(row["case_id"], row["rater"]): row["score"] for row in every_rater if row["dimension"] == dimension}
            complete = [case for case in composite if all((case, rater) in lookup for rater in raters)]
            if len(complete) > 2:
                matrix = np.array([[lookup[(case, rater)] for rater in raters] for case in complete])
                icc, lo, hi = icc_with_ci(matrix)
                out["icc_2_1"][dimension] = {"icc": round(icc, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": len(complete)}

    per_dimension = {dimension: case_means([row for row in evomdt if row["dimension"] == dimension]) for dimension in DIMENSIONS}
    cases = sorted(composite)
    out["spearman"] = {}
    for left in DIMENSIONS:
        out["spearman"][left] = {}
        for right in DIMENSIONS:
            rho = spearmanr([per_dimension[left].get(c, np.nan) for c in cases], [per_dimension[right].get(c, np.nan) for c in cases], nan_policy="omit").statistic
            out["spearman"][left][right] = None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else round(float(rho), 3)
    return out


def write_rating_sheet(path: Path) -> None:
    """A blinded sheet for real physicians: columns match analyse_ratings after unblinding."""
    evomdt = read_runs("evomdt")
    rows = []
    for cancer, cases in layer2_cases().items():
        for case in cases:
            result = evomdt.get(case["case_id"])
            if result is None:
                continue
            slot_a_is_evomdt = ab_order(case["case_id"], "physician")
            plans = (plan_text(result), case["metadata"]["reference_plan_text"])
            plan_a, plan_b = plans if slot_a_is_evomdt else plans[::-1]
            line = {"case_id": case["case_id"], "cancer": cancer, "dossier": dossier_for_prompt(case), "plan_A": plan_a, "plan_B": plan_b}
            line.update({f"{slot}: {dimension} (1-5)": "" for slot in "AB" for dimension in DIMENSIONS})
            rows.append(line)
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def timing_summary() -> dict:
    durations = [result["duration_seconds"] / 60 for result in layer2_runs().values()]
    if not durations:
        return {}
    q1, median, q3 = np.percentile(durations, [25, 50, 75])
    return {"n": len(durations), "median_min": round(float(median), 2), "iqr_min": [round(float(q1), 2), round(float(q3), 2)], "mean_min": round(float(np.mean(durations)), 2), "sd_min": round(float(np.std(durations)), 2)}


def representative_cases(models: list[str]) -> str:
    """One case per cancer, closest to that cohort's median AI score, so the examples are not cherry-picked."""
    evomdt = read_runs("evomdt")
    blocks = ["# Representative cases (paper, Fig. 5)", "", "Each is the case nearest the cohort's median AI score. Source case reports: PMC-Patients (CC BY-NC-SA 4.0).", ""]
    for cancer, cases in layer2_cases().items():
        scored = []
        for case in cases:
            judged = judge_scores(models[:1], "evomdt", case["case_id"])
            if case["case_id"] in evomdt and judged:
                scored.append((float(np.mean([float(r["score"]) for r in judged])), case))
        if not scored:
            continue
        median = float(np.median([score for score, _ in scored]))
        score, case = min(scored, key=lambda pair: (abs(pair[0] - median), pair[1]["case_id"]))
        result = evomdt[case["case_id"]]
        decision = result["coordinator_decision"]
        citations = [
            f"{c['source_id']} ({c['anchor']}; {c['version']})"
            for item in decision["final_plan"] for c in item["citations"]
        ]
        blocks += [
            f"## {CANCER_LABEL[cancer]}: {case['case_id']} (AI score {score:.1f}/10, cohort median {median:.1f})", "",
            f"**Patient:** {case['patient_context'].get('summary') or ''}", "",
            f"**Stage:** {case.get('stage') or 'not available'} | **Pathology:** {case.get('pathology_summary') or 'not available'}", "",
            f"**EvoMDT recommendation:** {decision['final_answer']}", "",
            f"**Documented plan (reference):** {case['metadata']['reference_plan_text']}", "",
            f"**Evidence cited:** {'; '.join(dict.fromkeys(citations)) or 'none'}", "",
            f"**Reasoning trace:** {len(decision['conflicts'])} conflicts logged; {len(decision['rejected_actions'])} options rejected; confidence {decision['final_confidence']:.2f}.", "",
        ]
    return "\n".join(blocks)


# -------------------------------------------------------------------------------- figures

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style(plt) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.bottom": False, "ytick.left": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def grouped_bars(ax, labels, series: dict[str, list[float]], colors: dict[str, str], *, paper=None, ylim=(0, 1), ylabel="") -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width * 0.9, color=colors[name], label=name, zorder=2)
        for xpos, value in zip(x + offset, values):
            if not math.isnan(value) and index == 0:
                ax.text(xpos, value + (ylim[1] - ylim[0]) * 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    if paper:
        ax.scatter(x, paper, marker="D", s=46, facecolor=SURFACE, edgecolor=INK, linewidth=1.3, zorder=3, label="Paper, EvoMDT")
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def make_figures(layer1: dict, layer2: dict, ratings: dict, systems: list[str], names: dict[str, str], timing: dict, proxy_prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    folder = RESULTS / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    colors = {names[system]: SERIES[index % len(SERIES)] for index, system in enumerate(systems)}

    # Fig. 2, only when Layer 1 was run.
    if any(layer1[key].get(system) for key in [*MCQ_SETS, *GEN_SETS] for system in systems):
        # Fig. 2b
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        labels = list(MCQ_SETS.values())
        series = {names[s]: [layer1[k].get(s, {}).get("accuracy", float("nan")) for k in MCQ_SETS] for s in systems if any(s in layer1[k] for k in MCQ_SETS)}
        grouped_bars(axes[0], labels, series, colors, paper=[PAPER[k] for k in MCQ_SETS], ylim=(0, 1.05), ylabel="Accuracy")
        axes[0].set_title("Multiple-choice benchmarks")
        labels = list(GEN_SETS.values())
        series = {names[s]: [layer1[k].get(s, {}).get("bertscore_f1", float("nan")) for k in GEN_SETS] for s in systems if any(s in layer1[k] for k in GEN_SETS)}
        grouped_bars(axes[1], labels, series, colors, paper=[PAPER[k] for k in GEN_SETS], ylim=(0, 1.05), ylabel="BERTScore F1")
        axes[1].set_title("Free-response benchmarks (MedQARo not run)")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Layer 1: benchmark performance (paper Fig. 2b)", x=0.01, ha="left", fontweight="semibold", color=INK)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(folder / "fig2b_benchmarks.png", dpi=170)
        plt.close(fig)

        # Fig. 2a
        keys = ["medqa", "medqa_cancer", "medmcqa_cancer", "rond", "medquad"]
        titles = [MCQ_SETS.get(k) or GEN_SETS[k] for k in keys]
        angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
        fig, ax = plt.subplots(figsize=(6.4, 6), subplot_kw={"polar": True})
        for system in systems:
            values = [layer1[k].get(system, {}).get("accuracy", layer1[k].get(system, {}).get("bertscore_f1", float("nan"))) for k in keys]
            if all(math.isnan(v) for v in values):
                continue
            ax.plot(angles + angles[:1], values + values[:1], color=colors[names[system]], linewidth=2, marker="o", markersize=5, label=names[system])
        paper_values = [PAPER[k] for k in keys]
        ax.plot(angles + angles[:1], paper_values + paper_values[:1], color=INK, linewidth=1.2, marker="D", markersize=5, markerfacecolor=SURFACE, label="Paper, EvoMDT")
        ax.set_xticks(angles, titles)
        ax.set_ylim(0, 1)
        ax.set_rlabel_position(20)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
        ax.spines["polar"].set_color(AXIS)
        ax.set_title("Accuracy or BERTScore, 0 to 1 (paper Fig. 2a)", pad=22)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
        fig.tight_layout()
        fig.savefig(folder / "fig2a_radar.png", dpi=170)
        plt.close(fig)

    # Fig. 3
    per_case = []
    path = RESULTS / "tables" / "layer2_per_case.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            per_case = list(csv.DictReader(handle))
    cancers = [c for c in CANCERS if "evomdt" in layer2.get(c, {})]
    if cancers:
        label_list = [CANCER_LABEL[c] for c in cancers]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        metric_rows = {
            "ROUGE-1 recall": [layer2[c]["evomdt"]["rouge1_recall"] for c in cancers],
            "ROUGE-L recall": [layer2[c]["evomdt"]["rougeL_recall"] for c in cancers],
            "Cosine similarity": [layer2[c]["evomdt"]["cosine_similarity"] for c in cancers],
            "AI score / 10": [(layer2[c]["evomdt"]["ai_score_0_10"] or float("nan")) / 10 for c in cancers],
        }
        cancer_colors = dict(zip(label_list, SERIES))
        by_cancer = {label: [metric_rows[m][i] for m in metric_rows] for i, label in enumerate(label_list)}
        grouped_bars(axes[0, 0], list(metric_rows), by_cancer, cancer_colors, ylim=(0, 1.05), ylabel="Score")
        for container in axes[0, 0].texts[:]:
            container.remove()
        axes[0, 0].set_title("a  EvoMDT by cancer type")
        axes[0, 0].legend(ncol=4, loc="upper left")

        def boxes(ax, column: str, title: str, ylabel: str, ylim) -> None:
            data = [[float(r[column]) for r in per_case if r["cancer"] == c and r["condition"] == "evomdt" and r[column] not in ("", None)] for c in cancers]
            plot = ax.boxplot(data, widths=0.42, patch_artist=True, showfliers=False, medianprops={"color": INK, "linewidth": 1.4}, whiskerprops={"color": MUTED}, capprops={"color": MUTED}, boxprops={"linewidth": 0})
            for patch in plot["boxes"]:
                patch.set_facecolor(SERIES[0])
                patch.set_alpha(0.75)
            for offset, (condition, color, label, marker) in enumerate([("random", SERIES[1], "Random baseline (mean)", "s"), ("simple", SERIES[3], "Simple baseline (mean)", "^")]):
                means = []
                for c in cancers:
                    values = [float(r[column]) for r in per_case if r["cancer"] == c and r["condition"] == condition and r[column] not in ("", None)]
                    means.append(np.mean(values) if values else float("nan"))
                ax.scatter(np.arange(1, len(cancers) + 1) + 0.3, means, color=color, marker=marker, s=46, zorder=3, label=label)
            ax.set_xticks(range(1, len(cancers) + 1), label_list)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
            ax.legend(loc="lower right")

        boxes(axes[0, 1], "ai_score", "b  AI evaluation score (LLM judges)", "Score, 0-10", (-0.3, 10.3))
        boxes(axes[1, 0], "rouge1_recall", "c  ROUGE-1 recall", "Recall", (0, 1.02))
        boxes(axes[1, 1], "cosine_similarity", "d  Semantic similarity to the reference plan", "Cosine similarity (BGE-M3)", (0, 1))
        fig.suptitle("Layer 2: treatment plans against documented plans, n per cancer as in the report (paper Fig. 3)", x=0.01, ha="left", fontweight="semibold", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(folder / "fig3_real_world.png", dpi=170)
        plt.close(fig)

    # Fig. 4, always titled as the LLM proxy unless physician ratings were supplied.
    if ratings:
        proxy = proxy_prefix != "physician"
        heading = "LLM-rater proxy, NOT physician ratings" if proxy else "Physician ratings"
        fig = plt.figure(figsize=(13, 8.6))
        grid = fig.add_gridspec(2, 3)
        ax = fig.add_subplot(grid[0, 0:2])
        steps = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
        y = np.arange(len(DIMENSIONS))[::-1]
        left = np.zeros(len(DIMENSIONS))
        for score, color in zip("12345", steps):
            widths = np.array([ratings["dimensions"][d]["distribution_pct"][score] for d in DIMENSIONS])
            ax.barh(y, widths, left=left, color=color, height=0.62, label=f"{score}", edgecolor=SURFACE, linewidth=1.5)
            left += widths
        ax.set_yticks(y, DIMENSIONS)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of ratings (%)")
        ax.grid(False)
        ax.legend(title="Score", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        ax.set_title("a  Score distribution by dimension")

        ax = fig.add_subplot(grid[0, 2])
        evomdt = layer2_runs()
        minutes = {c: [r["duration_seconds"] / 60 for r in evomdt.values() if r["case"].get("cancer_type") == c] for c in cancers}
        ax.boxplot([minutes[c] for c in cancers], widths=0.42, patch_artist=True, showfliers=False, medianprops={"color": INK}, whiskerprops={"color": MUTED}, capprops={"color": MUTED}, boxprops={"facecolor": SERIES[0], "alpha": 0.75, "linewidth": 0})
        ax.set_xticks(range(1, len(cancers) + 1), [CANCER_LABEL[c] for c in cancers])
        ax.set_ylabel("Minutes per case")
        ax.grid(axis="x", visible=False)
        ax.set_title("b  EvoMDT wall-clock time (no physician arm)")

        ax = fig.add_subplot(grid[1, 0])
        icc = ratings.get("icc_2_1", {})
        names_icc = [d for d in DIMENSIONS if d in icc]
        if names_icc:
            positions = np.arange(len(names_icc))[::-1]
            values = [icc[d]["icc"] for d in names_icc]
            errors = [[icc[d]["icc"] - icc[d]["ci95"][0] for d in names_icc], [icc[d]["ci95"][1] - icc[d]["icc"] for d in names_icc]]
            ax.errorbar(values, positions, xerr=errors, fmt="o", color=SERIES[0], ecolor=MUTED, markersize=7, capsize=3)
            ax.set_yticks(positions, names_icc)
            ax.set_xlim(min(-0.1, min(values) - 0.2), 1)
            ax.set_xlabel("ICC(2,1) with 95% CI")
            ax.grid(axis="y", visible=False)
        ax.set_title("e  Agreement between raters")

        ax = fig.add_subplot(grid[1, 1:])
        matrix = np.array([[ratings["spearman"][a][b] if ratings["spearman"][a][b] is not None else np.nan for b in DIMENSIONS] for a in DIMENSIONS], dtype=float)
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
        image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
        short = [d.split(" ")[0].replace("-Centeredness", "") for d in DIMENSIONS]
        ax.set_xticks(range(6), short, rotation=30, ha="right")
        ax.set_yticks(range(6), short)
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if abs(matrix[i, j]) > 0.6 else INK)
        fig.colorbar(image, ax=ax, fraction=0.03, label="Spearman ρ")
        ax.set_title("f  Correlation between dimensions")
        fig.suptitle(f"Layer 3: {heading} (paper Fig. 4 layout)", x=0.01, ha="left", fontweight="semibold", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(folder / f"fig4_{proxy_prefix}.png", dpi=170)
        plt.close(fig)


def report(args: argparse.Namespace) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    systems = ["evomdt", *args.baselines]
    names = {"evomdt": "EvoMDT (this run)"}
    names.update({system: f"Single-agent {system.split('/')[-1]}" for system in args.baselines})
    encoders = {}
    for name, model in (("bge_m3", "BAAI/bge-m3"), ("qwen3", args.qwen)):
        encoders[name] = TokenEncoder(model)
    layer1 = score_layer1(systems, encoders)
    layer2 = score_layer2(args.models, args.baselines[0] if args.baselines else None, encoders)
    for encoder in encoders.values():
        encoder.close()
    hcc = score_hcc_tace(args.models)
    timing = timing_summary()
    proxy = analyse_ratings(collect_ratings(args.models), prefix="layer3_llm_proxy", primary=args.models[0])
    physician = {}
    if args.physician_ratings:
        with open(args.physician_ratings, encoding="utf-8-sig") as handle:
            rows = [{**row, "score": float(row["score"])} for row in csv.DictReader(handle)]
        physician = analyse_ratings(rows, prefix="layer3_physician")
    write_rating_sheet(ROOT / "artifacts" / "paper_eval" / "physician_rating_sheet.csv")
    (RESULTS / "representative_cases.md").write_text(representative_cases(args.models), encoding="utf-8")
    flow_path = DATA / "realworld_flow.json"
    kb_path = ROOT / "data" / "kb" / "kb_manifest.json"
    state_path = ROOT / "artifacts" / "paper_eval" / "evolution_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    summary = {
        "layer1": layer1,
        "layer2": layer2,
        "layer2_per_cancer_cap": PER_CANCER,
        "extraction_audit": audit_sensitivity(),
        "hcc_tace_guideline_concordance": hcc,
        "layer3_llm_proxy": proxy,
        "layer3_physician": physician or "not run: needs physician raters",
        "timing_layer2_evomdt": timing,
        "cohort_flow": json.loads(flow_path.read_text(encoding="utf-8")) if flow_path.exists() else {},
        "knowledge_base": {k: v for k, v in json.loads(kb_path.read_text(encoding="utf-8")).items() if k != "by_source"} if kb_path.exists() else {},
        "evolution": {key: state.get(key) for key in ("role_weights", "prompt_refinements", "retrieval_top_k")} | {"events": len(state.get("history", []))},
        "failures": {p.parent.name: len(json.loads(p.read_text(encoding="utf-8"))) for p in RUNS.glob("*/failures.json") if not p.parent.name.startswith("pilot")},
        "judges": {"primary_all_arms": args.models[0], "second_sample_only": args.models[1:]},
        "judge_agreement_ai_score": judge_agreement(args.models),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    make_figures(layer1, layer2, physician or proxy, systems, names, timing, "physician" if physician else "llm_proxy")
    print(json.dumps({k: v for k, v in summary.items() if k in ("layer1", "hcc_tace_guideline_concordance", "timing_layer2_evomdt", "failures")}, indent=1)[:6000])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["judge", "report"])
    parser.add_argument("--models", nargs="+", default=["minimaxai/minimax-m3.1-flash", "mistralai/mistral-large-4-1t-a52b"], help="The first judge scores every arm, and the others a sample.")
    parser.add_argument("--simple-per-cancer", type=int, default=10)
    parser.add_argument("--random-per-cancer", type=int, default=25)
    parser.add_argument("--second-judge-per-cancer", type=int, default=10)
    parser.add_argument("--baselines", nargs="*", default=["single:nvidia/nemotron-3-ultra-550b-a55b", "single:mistralai/mistral-large-4-1t-a52b"])
    parser.add_argument("--single", default="single:nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--qwen", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--physician-ratings", help="CSV with case_id, cancer, rater, plan, dimension, score.")
    parser.add_argument("--per-cancer", type=int, help="Score only the first N Layer 2 cases of each cancer.")
    args = parser.parse_args()
    global PER_CANCER
    PER_CANCER = args.per_cancer
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.step == "judge":
        asyncio.run(run_judges(args))
    else:
        report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
