"""Offline end-to-end check of the vendored OpenEvoMDT baseline.

Drives every documented command (run-case, run-benchmark, evolve, inspect-run)
through the real EvoMDTSystem pipeline, swapping only the LLM provider for the
deterministic fake providers shipped in tests/conftest.py. Also stress-tests
the safety guardrail and the coordinator-failure fallback.

Run from the repo root:
    .venv/Scripts/python results/openevomdt_check/run_offline_check.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # sample paths and config.toml are relative to the repo root

from evomdt.config import AppConfig  # noqa: E402
from evomdt.llm import LLMTrace, OpenAIChatProvider  # noqa: E402
from evomdt.pipeline import EvoMDTSystem  # noqa: E402
from tests.conftest import BiomedicalFakeProvider, FakeProvider, MinimalTaskFakeProvider  # noqa: E402

ARTIFACTS = HERE / "artifacts"
HCC_CASE = "data/samples/cases/sample_hcc_case.json"
BIO_CASE = "data/samples/cases/sample_biomedical_hyperpyrexia_case.json"
HCC_BENCH = "data/samples/benchmarks/sample_benchmark.jsonl"
BIO_BENCH = "data/samples/benchmarks/sample_biomedical_benchmark.jsonl"


class UnsafeCoordinatorProvider(FakeProvider):
    """Coordinator ignores the Safety veto and accepts 'Major hepatectomy'."""

    async def generate_structured(self, messages, parser):
        trace = await super().generate_structured(messages, parser)
        if "coordinator agent" not in messages[0].content.lower():
            return trace
        payload = json.loads(trace.raw_content)
        payload["final_plan"].insert(0, {
            "action": "Major hepatectomy", "owner_role": "treatment",
            "rationale": "Injected unsafe action.", "priority": 1, "score": 0.95, "citations": [],
        })
        payload["accepted_actions"].insert(0, "Major hepatectomy")
        payload["rejected_actions"] = []
        payload["conflicts"] = []
        raw = json.dumps(payload)
        return LLMTrace(raw_content=raw, parsed=parser(raw), model_name="unsafe-fake", usage={"total_tokens": 1})


class CrashingCoordinatorProvider(FakeProvider):
    """Coordinator returns non-JSON, forcing the deterministic fallback."""

    async def generate_structured(self, messages, parser):
        if "coordinator agent" in messages[0].content.lower():
            parser("this is not json")  # raises like a malformed model reply would
        return await super().generate_structured(messages, parser)


def make_config(name: str) -> AppConfig:
    config = AppConfig.load("config.toml").model_copy(deep=True)
    config.runtime.artifacts_dir = str(ARTIFACTS / name)
    config.runtime.evolution_state_path = str(ARTIFACTS / name / "evolution_state.json")
    return config


def system(name: str, provider_cls) -> EvoMDTSystem:
    return EvoMDTSystem(make_config(name), provider_factory=lambda _key: provider_cls())


def case_summary(result) -> dict:
    decision = result.coordinator_decision.model_dump()
    return {
        "run_id": result.run_id,
        "duration_seconds": result.duration_seconds,
        "agents_run": [t.role for t in result.agent_traces],
        "final_answer": decision.get("final_answer"),
        "accepted_actions": decision.get("accepted_actions"),
        "rejected_actions": decision.get("rejected_actions"),
        "conflicts": [
            {k: c.get(k) for k in ("action", "objectors", "outcome")} for c in decision.get("conflicts", [])
        ],
        "metrics": result.evaluation.metrics if result.evaluation else None,
    }


async def main() -> dict:
    shutil.rmtree(ARTIFACTS, ignore_errors=True)
    report: dict = {}

    report["run_case_hcc"] = case_summary(await system("run_case_hcc", FakeProvider).run_case_file(HCC_CASE))
    report["run_case_biomedical"] = case_summary(
        await system("run_case_biomedical", BiomedicalFakeProvider).run_case_file(BIO_CASE)
    )
    report["run_benchmark_hcc"] = await system("run_benchmark_hcc", FakeProvider).run_benchmark(HCC_BENCH)
    report["run_benchmark_biomedical"] = await system("run_benchmark_biomedical", BiomedicalFakeProvider).run_benchmark(BIO_BENCH)

    evo = system("evolve_biomedical", BiomedicalFakeProvider)
    report["evolve_baseline_weights"] = make_config("x").agents.roles["safety"].weight
    report["evolve_biomedical"] = await evo.evolve(BIO_BENCH)

    hcc_sys = system("run_case_hcc", FakeProvider)
    inspected = hcc_sys.inspect_run(report["run_case_hcc"]["run_id"])
    report["inspect_run"] = {"keys": sorted(inspected.keys()), "run_id_matches": inspected["run_id"] == report["run_case_hcc"]["run_id"]}

    report["guardrail_unsafe_coordinator"] = case_summary(
        await system("guardrail_unsafe", UnsafeCoordinatorProvider).run_case_file(HCC_CASE)
    )
    report["fallback_crashing_coordinator"] = case_summary(
        await system("fallback_crash", CrashingCoordinatorProvider).run_case_file(HCC_CASE)
    )

    try:
        OpenAIChatProvider(make_config("x").llm_for("deepseek/deepseek-chat"))
        report["real_provider_without_key"] = "constructed (a key is set)"
    except ValueError as exc:
        report["real_provider_without_key"] = f"ValueError: {exc}"

    _ = MinimalTaskFakeProvider  # minimal-format path is covered by tests/test_minimal_benchmark_format.py
    return report


if __name__ == "__main__":
    result = asyncio.run(main())
    (HERE / "offline_check_report.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
