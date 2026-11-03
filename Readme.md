# Adaptive EvoMDT

**Adaptive, Resource-Aware and Safety-Gated Multi-Agent Orchestration for Oncology Clinical Decision Support**

Christin Jose Biju (26MAI0055) · M.Tech CSE (AI & ML), VIT Vellore · Guide: Dr. R. Pushpalakshmi

> **Status: planning / early implementation.** This README describes the intended system. Sections marked TODO are not built or verified yet.
> **Research prototype. Not a medical device. Not for clinical use.** Outputs are decision support for research evaluation only.

---

## What this is

EvoMDT (Liu et al., *npj Digital Medicine* 9, 124, 2026) models a tumour board with five LLM agents: Diagnostic, Treatment, Safety, Monitoring and Coordinator. It runs the full team on every case, which wastes tokens and time on simple cases and gives no explicit guarantee of deeper review on hard ones.

Adaptive EvoMDT is an orchestration layer on top of an EvoMDT-style system. For each case it decides:

1. **Which agents run**, chosen from complexity, risk and uncertainty scores (Safety and Coordinator always run).
2. **In what order**, using a task graph (DAG) with critical-path priority, so independent work runs in parallel.
3. **On which model**, using an SLM-first cascade: a small local model (Tier 1) is the default, and a larger hosted DeepSeek model (Tier 2) is used only on escalation or for high-risk cases.
4. **When to stop**, only when confidence, agent agreement, evidence coverage and the safety check all pass. Otherwise it re-consults, escalates the model, adds agents, or flags human review.

**Design principle:** resource limits may delay a mandatory task but never cause a mandatory safety check to be skipped.

**Research question:** can safety-constrained adaptive orchestration reduce computational cost and latency of multi-agent clinical decision support while keeping decision quality, safety and evidence coverage comparable?

## Architecture

```
Case + budget (tokens, RPM, VRAM/RAM, deadline)
  -> 1 Case profiler            (Tier 1 only; raw text never leaves the machine)
  -> 2 Complexity/risk scorer   C, R, U + red-flag rules
  -> 3 Agent selector + DAG     smallest safe agent set
  -> 4 Resource-aware scheduler dispatch feasible tasks in parallel
  -> 5 Specialist agents + RAG  Diagnostic, Treatment, Safety, Monitoring
  -> 6 Confidence + safety gate PASS -> Coordinator -> recommendation + evidence trace
                                FAIL -> escalate (retry, re-consult, model escalation, add agent, human review)
  Audit log: every decision, tier, token count, timing and evidence ID
```

### Model tiers

| Tier | Model | Runs on | Used for |
|---|---|---|---|
| 1 | Small local model, 4-bit, up to ~8B parameters (chosen by pilot) | Ollama on the local GPU | Default for all agents; the only tier that sees raw case text |
| 2 | DeepSeek V4.1 Flash, hosted on NVIDIA NIM (for now) | OpenAI-compatible API | Escalations, high-risk cases, optionally the Coordinator |

Which agent starts on which tier is set in `configs/tier_table.yaml`, filled from the pilot. The Safety agent always runs on Tier 2 for red-flag and high-risk cases.

### Privacy-aware routing

Raw case text goes to Tier 1 only. Tier 2 receives only the structured, de-identified profile plus retrieved guideline snippets. Every Tier 2 call is written to an egress log. The study uses public, de-identified data. Privacy routing is a design property and has not been shown to give real-world privacy guarantees.

### Safety invariants (each has a unit test)

- Safety and Coordinator run in every pathway.
- Red-flag cases always take the full pathway.
- Model choice never skips or downgrades Safety for high-risk cases.
- An escalated answer must still pass all four gate conditions.
- In privacy mode, zero raw-text bytes reach Tier 2.
- Resource or budget limits delay mandatory tasks; they never drop them.

## Data

| Source | Use | Notes |
|---|---|---|
| HCC-TACE-Seg clinical spreadsheet (105 patients, TCIA, DOI 10.7937/TCIA.5FNA-0924, CC BY 4.0) | Real case dossiers | Imaging (28 GB) is not used. TODO: confirm the field list after download |
| Synthetic vignettes (~60–100) | Covers the high-risk stratum | Always labelled `synthetic=true` and reported separately |
| NCCN guidelines (HCC primary), RECIST 1.1, iRECIST | RAG knowledge base | Copyrighted: local research use only. Do not redistribute |

`data/` and `.env` are git-ignored. Do not commit guideline PDFs, chunks or API keys.

## Requirements

- Python 3.11 or 3.12
- Ollama (for Tier 1) and an NVIDIA GPU with 8 GB or more VRAM (developed on an RTX 4070 Laptop, 8 GB)
- An NVIDIA NIM API key from build.nvidia.com (Tier 2). DeepSeek's own API works as a drop-in alternative
- Node 20 or later (UI only, later phase)

## Setup

```powershell
git clone <your-repo-url> adaptive_evomdt
cd adaptive_evomdt
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Pull the Tier 1 candidates (final choice comes from the pilot; model names are placeholders, confirm them in the Ollama library):

```powershell
ollama pull <qwen-7b-instruct-q4>
ollama pull <llama-8b-instruct-q4>
ollama pull <small-4b-instruct-q4>
```

Create `.env` (never commit it):

```
NIM_API_KEY=your-key-here
NIM_BASE_URL=https://integrate.api.nvidia.com/v1   # TODO: verify against NVIDIA docs
NIM_MODEL=<deepseek slug from your catalog>        # TODO: confirm exact slug
OLLAMA_URL=http://localhost:11434
# Optional fallback
DEEPSEEK_API_KEY=
```

### Smoke test (TODO)

```powershell
python -m experiments.pilot_models --smoke
```

This should print tokens/s and VRAM for each local model, confirm the NIM slug, and measure your requests-per-minute limit with a burst test.

## Configuration

| File | Purpose |
|---|---|
| `configs/models.yaml` | Tier 1 and Tier 2 backends, base URLs, model names, temperature, seeds |
| `configs/tier_table.yaml` | Starting tier per agent and risk level (from the pilot) |
| `configs/thresholds.yaml` | Gate thresholds (θ, δ, ε), risk-adaptive values |
| `configs/weights.yaml` | Complexity and risk feature weights, agent reliability weights |
| `configs/budgets.yaml` | Token budget, requests-per-minute cap, deadline, `max_parallel`, VRAM/RAM floors |

## Usage (TODO: endpoints not built yet)

Run the API:

```powershell
uvicorn app.main:app --reload
```

Solve one case:

```powershell
curl -X POST http://localhost:8000/solve -H "Content-Type: application/json" -d @data/cases/example.json
```

Run experiments:

```powershell
python -m experiments.run_baselines --split dev
python -m experiments.run_adaptive --split dev
python -m experiments.ablations --split test
python -m experiments.analyse
```

## Testing

```powershell
pytest
```

Core tests run on a mock backend with no LLM and no network. The safety-invariant tests are the ones to read first: `tests/test_safety_invariants.py`, `tests/test_gate.py` and `tests/test_router.py`.

## Repository layout

```
configs/          YAML configuration
app/
  main.py         FastAPI: POST /solve, GET /runs/{id}
  orchestrator.py solve_case loop
  profiling.py    case profiler (HCC schema)
  scoring.py      complexity, risk, red flags, pathway selection
  dag.py          task graph, topological sort, upward-rank priority
  scheduler.py    resource-aware scheduler, budgets, safety-reserved budgeting
  llm/            base interface, Ollama backend, OpenAI-compatible backend, mock backend
  router.py       tier selection, model escalation, privacy filter, egress log
  agents.py       role prompts and structured outputs
  rag.py          chunking, hybrid BM25 + dense retrieval, provenance
  gate.py         confidence, disagreement, evidence coverage, safety gate
  escalation.py   escalation ladder, targeted re-consultation
  coordinator.py  weighted consensus, final recommendation
  audit.py        JSON audit log, calibration (ECE)
data/             git-ignored: raw data, corpus, cases
experiments/      pilot, baselines, adaptive runs, ablations, replay, analysis
tests/            unit and safety-invariant tests
frontend/         React UI (last phase)
results/          run logs, tables, figures
```

## Evaluation design

**Systems compared** (backbone × pipeline):

| | B1 single model | B2 fixed sequential | B3 fixed parallel | Adaptive |
|---|---|---|---|---|
| **S** (small model only) | B1-S | B2-S | B3-S | Adaptive-S |
| **L** (DeepSeek only) | B1-L | B2-L (EvoMDT-style) | B3-L | Adaptive-L |
| **C** (cascade) | – | – | – | **Adaptive-C (proposed)** |

Ablations on Adaptive-C: no agent selection, no DAG scheduling, no early exit, no resource awareness, no SRB, no RAT, no TRC, no model escalation, no privacy routing.

**Decision quality** (the HCC dataset has no ground-truth treatment plan, since all patients had TACE): guideline concordance against a rule table derived from NCCN HCC, non-inferiority against the fixed pipeline on the same backbone, and a rubric LLM-judge with a manual check of 20–30 cases.

**Safety:** deterministic rule-based oracle; safety-violation rate, missed-escalation rate (target 0), cases where Safety could not run under tight budgets (target 0 with SRB).

**Efficiency:** tokens, agent calls, cost per case at a stated reference list price, latency (rate-limit wait reported separately), peak RAM/VRAM, share of calls on Tier 1, model-escalation rate and precision.

**Reliability:** expected calibration error per tier, agent disagreement, evidence coverage, provenance completeness.

**Statistics:** paired Wilcoxon, McNemar, bootstrap 95% CIs, Holm correction, per-stratum results. Real and synthetic cases are reported separately. Splits: dev ≈ 35 / val ≈ 20 / test ≈ 50 real patients; the test split is run once.

## Reproducibility

- Temperature 0 and fixed seeds for agents; three seeds for the headline comparison only.
- Every model response is cached by (model, prompt, seed). Re-running tables from the cache needs no API calls.
- Each result records date, endpoint, model name, and git commit.
- API models can change or retire; the pinned model name and cache are the record of what was used.

## Limitations

- Small real dataset (105 patients), all treated with TACE; synthetic cases fill the high-risk stratum.
- No clinician-validated gold standard; quality is judged by guideline rules, the fixed pipeline and a model judge.
- The judge and the Tier 2 model are both DeepSeek-family, so the judge may favour similar answers. A manual check mitigates but does not remove this.
- NIM trial access has unpublished rate limits and may change. TODO: check the trial terms for research use.
- One 8 GB local GPU limits local parallelism.
- Not clinically validated. Do not use on real patients.

## Roadmap

| Weeks | Phase |
|---|---|
| 1–2 | Setup, data download, model smoke tests, small-model vs DeepSeek pilot, go/no-go |
| 2–3 | Scoring, DAG, scheduler, router, gate, escalation, audit on a mock backend, with safety tests |
| 3–5 | RAG corpus, agents, full pilot, baselines on both backbones |
| 6–8 | Adaptive orchestrator end to end (Review 2) |
| 9–10 | Gate calibration, own algorithms (SRB, RAT, PAC, TRC), privacy routing, API and UI |
| 11–13 | Final experiments, ablations, statistics |
| 14–15 | Report, paper draft, demo |

## References

- Q. Liu et al., "EvoMDT: A Self-Evolving Multi-Agent System for Structured Clinical Decision-Making in Multi-Cancer," *npj Digital Medicine* 9, 124, 2026. doi:10.1038/s41746-025-02304-8
- A. W. Moawad et al., "Multimodality Annotated HCC Cases with and without Advanced Imaging Segmentation (HCC-TACE-Seg)," The Cancer Imaging Archive, 2021. doi:10.7937/TCIA.5FNA-0924 (CC BY 4.0, citation required)
- H. Topcuoglu, S. Hariri, M.-Y. Wu, "Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing," *IEEE TPDS* 13(3), 2002 (HEFT upward-rank priority)
- Full literature mapping: see `docs/literature.md` (TODO)

## License

TODO: choose a license for the code. Third-party data and guidelines keep their own terms (HCC-TACE-Seg: CC BY 4.0; NCCN guidelines: copyrighted, not redistributed).