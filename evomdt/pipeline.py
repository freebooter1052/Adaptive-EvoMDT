from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from .artifacts import inspect_run as read_run_summary
from .artifacts import save_case_run
from .config import AppConfig
from .consensus import apply_safety_guardrail, deterministic_coordinator, weighted_consensus
from .contracts import resolve_output_contract
from .evaluation import aggregate_benchmark_results, evaluate_case
from .evolution import apply_feedback, initial_state, load_state, save_state
from .io import iter_dataset_tasks, load_case, write_json
from .llm import ChatMessage, OpenAIChatProvider, StructuredLLM, parse_json_content
from .models import (
    AgentExecutionTrace,
    AgentOutput,
    CaseDossier,
    CaseRunResult,
    CitationProvenance,
    CoordinatorOutput,
    EvolutionState,
    SPECIALIST_ROLES,
    SpecialistRoleName,
    utc_timestamp,
)
from .normalization import normalize_agent_output_payload, normalize_coordinator_output_payload
from .prompts import build_coordinator_system_prompt, build_coordinator_user_prompt, build_system_prompt, build_user_prompt
from .retrieval import EvidenceSnippet, KnowledgeBase, render_evidence, retrieve_for_role

LOGGER = logging.getLogger("evomdt.pipeline")


def _bind_citations(citations: list[CitationProvenance], evidence: dict[str, EvidenceSnippet]) -> tuple[list[CitationProvenance], int]:
    """Keep citations that name retrieved evidence, with full provenance. Everything else is dropped."""
    bound: dict[str, CitationProvenance] = {}
    dropped = 0
    for citation in citations:
        # Models often put the supplied ID in source_id, since the citation schema names that field.
        snippet = evidence.get(citation.evidence_id or "") or evidence.get(citation.source_id or "")
        if snippet is None:
            dropped += 1
            continue
        bound[snippet.evidence_id] = snippet.citation()
    return list(bound.values()), dropped


def _bind_output_citations(output: AgentOutput | CoordinatorOutput, evidence: dict[str, EvidenceSnippet]) -> int:
    holders: list = list(output.supporting_facts)
    holders += output.candidate_actions if isinstance(output, AgentOutput) else output.final_plan
    dropped = 0
    for holder in holders:
        holder.citations, count = _bind_citations(holder.citations, evidence)
        dropped += count
    return dropped


class EvoMDTSystem:
    def __init__(
        self,
        config: AppConfig,
        provider_factory: Callable[[str], StructuredLLM] | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ):
        self.config = config
        self.provider_factory = provider_factory or self._default_provider_factory
        self.knowledge_base = knowledge_base
        if self.knowledge_base is None and config.retrieval.enabled:
            self.knowledge_base = KnowledgeBase(config.retrieval)

    def _default_provider_factory(self, model_key: str) -> StructuredLLM:
        return OpenAIChatProvider(self.config.llm_for(model_key))

    async def _retrieve(self, case: CaseDossier, role: SpecialistRoleName, state: EvolutionState) -> list[EvidenceSnippet] | None:
        if self.knowledge_base is None:
            return None
        top_k = state.retrieval_top_k.get(role, self.config.retrieval.top_k)
        return await asyncio.to_thread(retrieve_for_role, self.knowledge_base, case, role, top_k)

    async def _run_specialist(
        self,
        case: CaseDossier,
        role: SpecialistRoleName,
        refinement_ids: list[str],
        state: EvolutionState,
    ) -> tuple[AgentExecutionTrace, list[EvidenceSnippet]]:
        role_config = self.config.agents.roles[role]
        provider = self.provider_factory(role_config.llm)
        evidence = await self._retrieve(case, role, state)
        messages = [
            ChatMessage(role="system", content=build_system_prompt(case, role, refinement_ids, with_evidence=evidence is not None)),
            ChatMessage(
                role="user",
                content=build_user_prompt(case, role, render_evidence(evidence) if evidence is not None else None),
            ),
        ]
        trace = await provider.generate_structured(
            messages,
            parser=lambda content: AgentOutput.model_validate(normalize_agent_output_payload(parse_json_content(content), role)),
        )
        parsed_output = AgentOutput.model_validate(trace.parsed)
        if parsed_output.role != role:
            parsed_output.role = role
        dropped = 0
        if evidence is not None:
            dropped = _bind_output_citations(parsed_output, {item.evidence_id: item for item in evidence})
        return (
            AgentExecutionTrace(
                role=role,
                model_name=trace.model_name,
                raw_content=trace.raw_content,
                usage=trace.usage,
                parsed_output=parsed_output,
                retrieved_evidence_ids=[item.evidence_id for item in evidence or []],
                dropped_citations=dropped,
            ),
            evidence or [],
        )

    async def _run_coordinator(
        self,
        case: CaseDossier,
        outputs: list[AgentOutput],
        refinement_ids: list[str],
        evidence: dict[str, EvidenceSnippet] | None,
        role_weights: dict[SpecialistRoleName, float],
    ) -> AgentExecutionTrace:
        role_config = self.config.agents.roles["coordinator"]
        provider = self.provider_factory(role_config.llm)
        evidence_text = render_evidence(list(evidence.values()), max_chars=600) if evidence is not None else None
        consensus = weighted_consensus(case, outputs, role_weights) if resolve_output_contract(case) is not None else None
        messages = [
            ChatMessage(
                role="system",
                content=build_coordinator_system_prompt(case, refinement_ids, with_evidence=evidence is not None),
            ),
            ChatMessage(
                role="user",
                content=build_coordinator_user_prompt(case, outputs, evidence_text=evidence_text, consensus=consensus),
            ),
        ]
        trace = await provider.generate_structured(
            messages,
            parser=lambda content: CoordinatorOutput.model_validate(
                normalize_coordinator_output_payload(parse_json_content(content), case)
            ),
        )
        parsed_output = CoordinatorOutput.model_validate(trace.parsed)
        dropped = _bind_output_citations(parsed_output, evidence) if evidence is not None else 0
        return AgentExecutionTrace(
            role="coordinator",
            model_name=trace.model_name,
            raw_content=trace.raw_content,
            usage=trace.usage,
            parsed_output=parsed_output,
            retrieved_evidence_ids=list(evidence or {}),
            dropped_citations=dropped,
        )

    async def run_case(
        self,
        case: CaseDossier,
        *,
        persist: bool = True,
        state: EvolutionState | None = None,
    ) -> CaseRunResult:
        state = state or initial_state(self.config)
        started_at = perf_counter()
        refinement_ids_by_role = state.prompt_refinements

        if self.config.agents.parallel:
            specialist_runs = await asyncio.gather(
                *[
                    self._run_specialist(case, role, refinement_ids_by_role.get(role, []), state)
                    for role in SPECIALIST_ROLES
                ]
            )
        else:
            specialist_runs = []
            for role in SPECIALIST_ROLES:
                specialist_runs.append(await self._run_specialist(case, role, refinement_ids_by_role.get(role, []), state))

        traces = [run[0] for run in specialist_runs]
        outputs = [trace.parsed_output for trace in traces]
        evidence: dict[str, EvidenceSnippet] | None = None
        if self.knowledge_base is not None:
            evidence = {item.evidence_id: item for _, snippets in specialist_runs for item in snippets}
        coordinator_trace: AgentExecutionTrace | None = None
        try:
            coordinator_trace = await self._run_coordinator(
                case,
                outputs,
                refinement_ids_by_role.get("coordinator", []),
                evidence,
                state.role_weights,
            )
            traces.append(coordinator_trace)
            coordinator_decision = apply_safety_guardrail(
                case,
                outputs,
                CoordinatorOutput.model_validate(coordinator_trace.parsed_output),
                state.role_weights,
            )
        except Exception:
            LOGGER.exception("Coordinator agent failed; falling back to deterministic synthesis.")
            coordinator_decision = deterministic_coordinator(case, outputs, state.role_weights)
        evaluation = evaluate_case(case, outputs, coordinator_decision, self.config)
        timestamp = utc_timestamp().replace(":", "").replace("-", "").replace("+00:00", "Z")
        run_id = f"{timestamp}_{case.case_id.replace(' ', '-').lower()}"
        duration_seconds = round(perf_counter() - started_at, 4)
        result = CaseRunResult(
            run_id=run_id,
            case=case,
            agent_traces=traces,
            coordinator_decision=coordinator_decision,
            evaluation=evaluation,
            duration_seconds=duration_seconds,
            artifacts_dir=str(self.config.artifacts_dir / "runs" / run_id),
        )
        if persist:
            save_case_run(self.config, result)
        return result

    async def run_case_file(self, path: str | Path, *, persist: bool = True) -> CaseRunResult:
        return await self.run_case(load_case(path), persist=persist)

    async def run_benchmark(self, path: str | Path, *, output_path: str | Path | None = None) -> dict:
        results: list[CaseRunResult] = []
        evaluations = []
        for case in iter_dataset_tasks(path):
            case_result = await self.run_case(case)
            results.append(case_result)
            if case_result.evaluation is not None:
                evaluations.append(case_result.evaluation)
        summary = {
            "cases": len(results),
            "aggregate_metrics": aggregate_benchmark_results(evaluations),
            "run_ids": [result.run_id for result in results],
        }
        if output_path is not None:
            write_json(output_path, summary)
        else:
            write_json(self.config.artifacts_dir / "benchmark_summary.json", summary)
        return summary

    async def evolve(self, path: str | Path) -> dict:
        state = load_state(self.config)
        evolved_cases = 0
        if Path(path).suffix == ".jsonl":
            cases = iter_dataset_tasks(path)
        else:
            cases = [load_case(path)]
        for case in cases:
            result = await self.run_case(case, state=state)
            if result.evaluation is not None:
                state = apply_feedback(self.config, state, result.evaluation)
                evolved_cases += 1
        save_state(self.config, state)
        return {
            "updated_cases": evolved_cases,
            "role_weights": state.role_weights,
            "prompt_refinements": state.prompt_refinements,
            "history_entries": len(state.history),
        }

    def inspect_run(self, run_id: str) -> dict:
        return read_run_summary(self.config, run_id)
