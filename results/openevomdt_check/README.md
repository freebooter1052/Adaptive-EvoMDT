# OpenEvoMDT baseline: functional check and result analysis

**Date:** 2026-11-03 · **Baseline:** the repository root (`evomdt/`, `main.py`, `tests/`, `data/` and the rest), an unmodified copy of [yhzhu99/OpenEvoMDT](https://github.com/yhzhu99/OpenEvoMDT) at commit `a3fa7c8` (MIT). Its upstream README is at [`docs/OpenEvoMDT_README.md`](../../docs/OpenEvoMDT_README.md).

> **Scope.** This check shows that the baseline **pipeline works**: orchestration, guardrail, evaluation, evolution and artifacts. It does **not** measure clinical decision quality. No LLM backend was available (no `DEEPSEEK_API_KEY`, and no local Ollama or LM Studio model), so every agent reply came from the deterministic fake providers that ship with the repo in `tests/conftest.py`. All metric values below come from scripted answers. Latency and token counts are not meaningful.

## 1. Environment

| Item | Value |
|---|---|
| OS / GPU | Windows 11, RTX 4070 Laptop (8 GB) |
| Python | 3.12.6, venv at `.venv` in the repo root (uses the system torch 2.6.0+cu124) |
| Installed | `pip install -e .`, adding bert-score 0.3.13, rouge-score 0.1.2 and pytest 9.2.0 |
| Config | stock `config.toml`: all five roles on `deepseek/deepseek-chat`, parallel specialists, BERTScore off, LLM judge off |

## 2. What was run

1. **Unit tests:** `pytest -v`, 18 tests.
2. **Real CLI, no key:** `python main.py run-case --input data/samples/cases/sample_hcc_case.json`.
3. **Offline end-to-end harness:** [`run_offline_check.py`](run_offline_check.py) drives the real `EvoMDTSystem` and replaces only the provider. It covers every documented command and two fault scenarios:

| Scenario | Command path | Provider |
|---|---|---|
| `run_case_hcc` | `run_case_file` on the HCC case | `FakeProvider` |
| `run_case_biomedical` | `run_case_file` on the hyperpyrexia A/B case | `BiomedicalFakeProvider` |
| `run_benchmark_hcc` | `run_benchmark` on `sample_benchmark.jsonl` (HCC + breast) | `FakeProvider` |
| `run_benchmark_biomedical` | `run_benchmark` on `sample_biomedical_benchmark.jsonl` | `BiomedicalFakeProvider` |
| `evolve_biomedical` | `evolve` on the biomedical benchmark | `BiomedicalFakeProvider` |
| `inspect_run` | `inspect_run` on the saved HCC run | n/a |
| `guardrail_unsafe` | HCC case; the coordinator **accepts** the Safety-vetoed "Major hepatectomy" | injected unsafe coordinator |
| `fallback_crash` | HCC case; the coordinator returns non-JSON | injected crashing coordinator |

The raw output is in [`offline_check_report.json`](offline_check_report.json). Per-run artifacts (one JSON per agent, plus coordinator, evaluation and summary files) are in [`artifacts/`](artifacts).

## 3. Results

### 3.1 Tests and CLI

- **18/18 tests passed** in 0.47 s. They cover config, consensus, evaluation, evolution, normalization, the pipeline, benchmarks, the minimal JSONL format and biomedical generation.
- **The real CLI without a key** fails immediately with `ValueError: Missing API key for model 'deepseek-chat'. Set DEEPSEEK_API_KEY.` and writes no artifacts. This is the correct behaviour: it fails before any partial run starts.

### 3.2 Single cases

| Metric | HCC (oncology) | Hyperpyrexia (biomedical A/B) |
|---|---|---|
| Agents run | Diagnostic, Treatment, Safety, Monitoring, Coordinator | same |
| Final answer | limited resection, entecavir and surveillance | `A` (correct) |
| Rejected actions | Major hepatectomy (Safety veto) | Provide reassurance only |
| `weighted_plan_concordance` | 1.0 | n/a |
| `guideline_concordance` | 1.0 | n/a |
| `safety_violation_rate` | 0.0 | n/a |
| `evidence_traceability_coverage` | **0.0** | n/a |
| `accuracy` / `format_compliance` | n/a | 1.0 / 1.0 |
| `supporting_fact_coverage` | n/a | 0.5 |
| `traceability_score` | n/a | 0.75 |

### 3.3 Benchmarks

| Benchmark | Cases | Aggregate metrics |
|---|---|---|
| `sample_benchmark.jsonl` | 2 | plan concordance 1.0, guideline concordance 1.0, safety violations 0.0, evidence traceability 0.0, **computed from 1 case only** (see F1) |
| `sample_biomedical_benchmark.jsonl` | 2 | accuracy 0.5, format compliance 0.5, supporting-fact coverage 0.375, audit trace 1.0, safety escalation 1.0, traceability 0.458 |

On the biomedical benchmark, the second case (`bio-low-fever-002`) is scripted to answer `"Maybe B"`. The output contract rejects it as non-compliant, and it scores as wrong. This confirms that the A/B contract is enforced and not loosely matched.

### 3.4 Evolution

| Role | Weight before | Weight after 2 cases | Prompt refinements added |
|---|---|---|---|
| Diagnostic | 1.0 | 1.1 | `traceability_gap`, `accuracy_gap` |
| Treatment | 1.0 | 1.1 | `traceability_gap`, `accuracy_gap` |
| Safety | 1.0 | 1.0 | none |
| Monitoring | 1.0 | 1.0 | none |
| Coordinator | none (no vote) | none | `format_gap` |

Each case moves a weight by `weight_step` = 0.05, and `max_weight_delta` = 0.15 caps the total change. The state was written to `evolution_state.json` with 2 history entries. Plain `run-case` and `run-benchmark` runs do not load this state ("frozen by default", confirmed by `test_run_case_uses_baseline_state_by_default`).

### 3.5 Safety paths

- **Guardrail (`guardrail_unsafe`):** the coordinator put "Major hepatectomy" first in its plan and accepted it. The deterministic guardrail **removed** it from the accepted actions, moved it to the rejected list and recorded a conflict (objector: Safety, outcome: rejected). The final `safety_violation_rate` is 0.0. **The guardrail works as described.**
- **Coordinator failure (`fallback_crash`):** the invalid coordinator reply was logged with a traceback, and the pipeline fell back to `deterministic_coordinator`. That fallback still excluded the vetoed action, but it also added "Pre-operative surgical fitness assessment" to the plan. With it, `guideline_concordance` fell from 1.0 to 0.8, and the rationale became a generic "Fallback coordinator summary...". **The run degrades gracefully and stays safe, but answer quality drops.**

## 4. Findings

| # | Finding | Severity | Evidence |
|---|---|---|---|
| F1 | **Benchmark aggregates silently drop cases that have no computable metric.** The breast case has only a free-text `reference_answer`, so with BERTScore off its evaluation is `{"bertscore_f1": null}`. The summary still reports `"cases": 2`, but every aggregate is the HCC case's score alone. | High for evaluation validity | `artifacts/run_benchmark_hcc/runs/*breast-002/evaluation.json`; `aggregate_benchmark_results` skips `None` values |
| F2 | **Run IDs only resolve to the second.** The HCC, guardrail and fallback runs all got `20261103T131212+0000_sample-hcc-001`. In a shared artifacts directory, a repeated case or a fast rerun **overwrites** the earlier run. The `+00:00` to `Z` replacement never fires, because the colons are stripped first, so directory names contain a literal `+`. | Medium | `run_id` values in `offline_check_report.json`; `pipeline.py` lines 142-143 |
| F3 | **Evidence traceability is structurally 0.** RAG is off by design, so no plan item carries citations. The metric can't separate systems until retrieval or citations are added. | Expected / by design | `evidence_traceability_coverage = 0.0` in every oncology run |
| F4 | **The fallback coordinator records a duplicate conflict** for the same action (one entry `flagged`, one `rejected`). Conflict counts are inflated when the fallback runs. | Low | `fallback_crash` conflicts |
| F5 | **Every case runs the full team.** All five agents ran on every case, including the trivial A/B tasks. This is the cost that Adaptive EvoMDT targets. | Baseline behaviour | `agents_run` in every scenario |
| F6 | The fake providers return `total_tokens: 1` and the runs take 2-4 ms. The pipeline does record per-agent `usage` and `duration_seconds`, so with a real backend these become the cost and latency baselines. | Informational | per-agent `*.json` artifacts |

## 5. What this means for Adaptive EvoMDT

- **The baseline is usable as the "full-team" comparison arm.** The orchestration contract, artifacts and metrics all work, and the provider is injected through `provider_factory`. An Ollama (Tier 1) or NVIDIA NIM DeepSeek (Tier 2) provider plugs in there; both expose OpenAI-compatible APIs, so mostly only `config.toml` changes.
- **Fix or work around F1 before reporting numbers.** Report the number of cases each metric was computed on, or enable BERTScore (`enable_bertscore = true`) so that free-text cases get a score.
- **Fix F2 before large benchmark runs.** Add sub-second time or a short UUID to the run ID.
- **The baseline's safety guardrail is a deterministic post-check.** It removes vetoed actions after synthesis. The confidence and safety gate in Adaptive EvoMDT goes further (re-consult, escalate, human review), and the guardrail and fallback scenarios in this harness can be reused as unit tests for the safety invariants.
- **To show a cost reduction** (F5 to F6), the next step is a real-model run that collects tokens and latency per agent from these same artifacts.

## 6. Reproduce

From the repository root (Git Bash):

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest -v
.venv/Scripts/python results/openevomdt_check/run_offline_check.py
```

The harness clears and rewrites `results/openevomdt_check/artifacts/` on each run.

**Real-model run** (not done here): set `DEEPSEEK_API_KEY`, or point a `[llm.*]` entry in `config.toml` at a local OpenAI-compatible server (for example Ollama at `http://localhost:11434/v1`). Then run:

```bash
.venv/Scripts/python main.py run-benchmark --input data/samples/benchmarks/sample_benchmark.jsonl
```

*Research prototype. Not a medical device. Not for clinical use.*
