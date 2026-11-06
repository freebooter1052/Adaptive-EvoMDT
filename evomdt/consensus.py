from __future__ import annotations

import re

from .contracts import normalize_answer, resolve_output_contract
from .models import (
    AgentOutput,
    CaseDossier,
    ConflictRecord,
    CoordinatorDecision,
    CoordinatorOutput,
    PlanItem,
    RiskAlert,
    RoleName,
    SpecialistRoleName,
)


def normalize_action(action: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", action.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def case_role_modifiers(case: CaseDossier) -> dict[SpecialistRoleName, float]:
    modifiers: dict[SpecialistRoleName, float] = {
        "diagnostic": 1.0,
        "treatment": 1.0,
        "safety": 1.0,
        "monitoring": 1.0,
    }
    if len(case.lesions) > 1 or len(case.biomarkers) > 2:
        modifiers["diagnostic"] += 0.15
    if case.prior_treatments:
        modifiers["treatment"] += 0.15
    if case.patient_context.performance_status and any(
        marker in case.patient_context.performance_status.lower() for marker in ["2", "3", "4", "poor"]
    ):
        modifiers["safety"] += 0.2
    if any("child" in value.lower() or "bilirubin" in key.lower() for key, value in case.organ_function.items()):
        modifiers["safety"] += 0.15
    if "met" in case.question.lower() or "follow-up" in case.question.lower():
        modifiers["monitoring"] += 0.1
    return modifiers


SEVERITY_RISK: dict[str, float] = {"absolute": 1.0, "major": 0.8, "moderate": 0.5, "minor": 0.2, "info": 0.0}
# The paper defines S_conflict = alpha*D_semantic + beta*R_risk + gamma*I_implementation but publishes no
# coefficients. These put risk first, in line with its safety-first resolution.
CONFLICT_ALPHA, CONFLICT_BETA, CONFLICT_GAMMA = 0.3, 0.5, 0.2
# n = 4 voting specialists: a two-thirds quorum means one faulty specialist cannot decide the outcome.
CONSENSUS_QUORUM = 2 / 3
_STOP_WORDS = {"a", "an", "the", "of", "for", "with", "and", "or", "as", "to", "in", "if", "on", "at", "by", "plus"}


def conflict_severity_score(conflict: ConflictRecord) -> float:
    voters = len(conflict.recommenders) + len(conflict.objectors)
    semantic = len(conflict.objectors) / voters if voters else 0.5
    implementation = 1.0 if conflict.conflict_type == "implementation" else 0.0
    score = (
        CONFLICT_ALPHA * semantic
        + CONFLICT_BETA * SEVERITY_RISK.get(conflict.severity, 0.0)
        + CONFLICT_GAMMA * implementation
    )
    return round(score, 4)


def action_tokens(text: str) -> set[str]:
    return {word for word in normalize_action(text).split() if word not in _STOP_WORDS}


def actions_match(left: str, right: str, threshold: float = 0.6) -> bool:
    """True when the smaller action's content words mostly appear in the other action."""
    left_tokens, right_tokens = action_tokens(left), action_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens)) >= threshold


def weighted_consensus(
    case: CaseDossier,
    specialist_outputs: list[AgentOutput],
    role_weights: dict[SpecialistRoleName, float],
) -> dict[str, object]:
    """Compute D_final = sum(w_i * C_i) per enum answer, with the case-adjusted role weights."""
    effective = _role_weights_with_modifiers(case, role_weights)
    allowed = {normalize_answer(option) for option in (case.options or [])}
    tally: dict[str, float] = {}
    votes: dict[str, str] = {}
    for output in specialist_outputs:
        answer = normalize_answer(output.answer_candidate)
        if not answer or (allowed and answer not in allowed):
            continue
        votes[output.role] = answer
        tally[answer] = tally.get(answer, 0.0) + effective[output.role] * output.confidence
    total = sum(tally.values())
    if not total:
        return {"votes": votes, "tally": {}, "leader": None, "share": 0.0, "quorum_reached": False}
    leader = max(tally, key=lambda key: tally[key])
    share = tally[leader] / total
    return {
        "votes": votes,
        "tally": {key: round(value, 4) for key, value in tally.items()},
        "leader": leader,
        "share": round(share, 4),
        # At least 3 votes are required, or one or two agents could reach the quorum alone.
        "quorum_reached": share >= CONSENSUS_QUORUM and len(votes) >= 3,
    }


def score_plan_items(
    case: CaseDossier,
    specialist_outputs: list[AgentOutput],
    plan: list[PlanItem],
    role_weights: dict[SpecialistRoleName, float],
) -> None:
    """Replace self-assigned scores with sum(w_i * C_i) over supporters, normalized by total weight."""
    effective = _role_weights_with_modifiers(case, role_weights)
    total = sum(effective.values()) or 1.0
    for item in plan:
        support = 0.0
        for output in specialist_outputs:
            supported = output.role == item.owner_role or any(
                action.stance != "avoid" and actions_match(item.action, action.action)
                for action in output.candidate_actions
            )
            if supported:
                support += effective[output.role] * output.confidence
        item.score = round(min(1.0, support / total), 4)


def _safety_vetoes(safety_output: AgentOutput) -> dict[str, RiskAlert]:
    vetoes: dict[str, RiskAlert] = {}
    for alert in safety_output.risks_or_alerts:
        if alert.severity not in {"absolute", "major"}:
            continue
        for related_action in alert.related_actions:
            vetoes[normalize_action(related_action)] = alert
    for action in safety_output.candidate_actions:
        if action.stance == "avoid":
            vetoes.setdefault(
                normalize_action(action.action),
                RiskAlert(
                    severity="major",
                    concern=action.rationale,
                    mitigation="Avoid this action unless new data changes the risk profile.",
                    related_actions=[action.action],
                ),
            )
    return vetoes


def _role_weights_with_modifiers(
    case: CaseDossier,
    role_weights: dict[SpecialistRoleName, float],
) -> dict[SpecialistRoleName, float]:
    modifiers = case_role_modifiers(case)
    return {role: round(role_weights[role] * modifiers[role], 4) for role in role_weights}


def _case_label(case: CaseDossier) -> str:
    return case.cancer_type or case.domain


def _normalize_if_short(answer: str) -> str:
    stripped = answer.strip()
    if len(stripped) <= 8 and re.fullmatch(r"[A-Za-z0-9]+", stripped):
        return normalize_answer(stripped)
    return stripped


def _resolved_final_answer(case: CaseDossier, coordinator_output: CoordinatorOutput, top_action: str) -> str:
    explicit_answer = coordinator_output.final_answer.strip()
    if explicit_answer:
        return _normalize_if_short(explicit_answer) if case.task_type in {"generation", "mcq"} else explicit_answer
    if case.task_type in {"generation", "mcq"}:
        if coordinator_output.summary.strip():
            return _normalize_if_short(coordinator_output.summary)
        return _normalize_if_short(top_action)
    return coordinator_output.summary if coordinator_output.final_plan else top_action


def apply_safety_guardrail(
    case: CaseDossier,
    specialist_outputs: list[AgentOutput],
    coordinator_output: CoordinatorOutput,
    role_weights: dict[SpecialistRoleName, float],
) -> CoordinatorDecision:
    outputs_by_role = {output.role: output for output in specialist_outputs}
    safety_output = outputs_by_role["safety"]
    vetoes = _safety_vetoes(safety_output)

    final_plan: list[PlanItem] = []
    accepted_actions: list[str] = []
    rejected_actions = list(dict.fromkeys(coordinator_output.rejected_actions))
    conflicts = list(coordinator_output.conflicts)
    audit_trace = list(coordinator_output.audit_trace)

    for item in coordinator_output.final_plan:
        key = normalize_action(item.action)
        veto = vetoes.get(key)
        if veto is None:
            final_plan.append(item)
            accepted_actions.append(item.action)
            continue

        rejected_actions.append(item.action)
        conflicts.append(
            ConflictRecord(
                action=item.action,
                recommenders=[item.owner_role],
                objectors=["safety"],
                severity=veto.severity,
                conflict_type="risk",
                outcome="rejected",
                rationale=f"Safety guardrail vetoed coordinator plan item: {veto.concern}",
            )
        )
        audit_trace.append(f"Guardrail removed '{item.action}' because Safety flagged {veto.severity} risk.")

    if not accepted_actions:
        for action in coordinator_output.accepted_actions:
            if normalize_action(action) not in vetoes:
                accepted_actions.append(action)
            else:
                rejected_actions.append(action)
                audit_trace.append(f"Guardrail removed accepted action '{action}' because Safety vetoed it.")

    score_plan_items(case, specialist_outputs, final_plan, role_weights)
    for conflict in conflicts:
        conflict.severity_score = conflict_severity_score(conflict)
    final_plan.sort(key=lambda item: (item.priority, item.owner_role != "treatment", -item.score, item.action))
    accepted_actions = list(dict.fromkeys(accepted_actions))
    rejected_actions = list(dict.fromkeys(rejected_actions))
    effective_weights = _role_weights_with_modifiers(case, role_weights)

    top_action = accepted_actions[0] if accepted_actions else safety_output.summary
    final_answer = _resolved_final_answer(case, coordinator_output, top_action)

    consensus = weighted_consensus(case, specialist_outputs, role_weights)
    if (
        resolve_output_contract(case) is not None
        and consensus["quorum_reached"]
        and normalize_answer(final_answer) != consensus["leader"]
    ):
        audit_trace.append(
            f"Weighted consensus ({consensus['share']:.0%} of vote weight for {consensus['leader']}) "
            f"overrode the coordinator's answer '{final_answer}'."
        )
        final_answer = str(consensus["leader"])

    response_lines = [
        f"Case: {case.case_id} ({_case_label(case)})",
        f"Question: {case.question}",
    ]
    if case.task_type == "plan_eval":
        response_lines.extend(["", f"Integrated MDT Recommendation: {final_answer}"])
        if coordinator_output.diagnosis_and_staging:
            response_lines.extend(["", f"Primary Diagnosis and Staging: {coordinator_output.diagnosis_and_staging}"])
        response_lines.extend(["", "Treatment Plan:"])
        if final_plan:
            for item in final_plan:
                cited = [
                    f"{citation.source_id}, {citation.anchor}, {citation.version}"
                    for citation in item.citations
                    if citation.source_id
                ]
                evidence = f" | Evidence: {'; '.join(cited)}" if cited else " | Evidence: none retrieved"
                response_lines.append(
                    f"- {item.action} [{item.owner_role}, priority {item.priority}, consensus {item.score:.2f}]{evidence}"
                )
        else:
            response_lines.append("- No safe consensus action was accepted.")
        if coordinator_output.safety_assessment:
            response_lines.extend(["", f"Safety Assessment: {coordinator_output.safety_assessment}"])
        if coordinator_output.monitoring_protocol:
            response_lines.extend(["", f"Monitoring Protocol: {coordinator_output.monitoring_protocol}"])
        response_lines.extend(["", f"Overall Confidence: {coordinator_output.final_confidence:.2f}"])
    else:
        response_lines.extend(
            [
                "",
                f"Final Answer: {final_answer}",
            ]
        )
        if coordinator_output.summary:
            response_lines.extend(["", f"Coordinator Summary: {coordinator_output.summary}"])
    if safety_output.risks_or_alerts:
        response_lines.append("")
        response_lines.append("Safety Signals:")
        for alert in safety_output.risks_or_alerts:
            response_lines.append(f"- {alert.severity.upper()}: {alert.concern} | Mitigation: {alert.mitigation}")
    response_lines.append("")
    response_lines.append(f"Decision Rationale: {coordinator_output.decision_rationale}")
    response_lines.append("")
    response_lines.append("Audit Trace:")
    response_lines.extend(f"- {entry}" for entry in audit_trace)

    return CoordinatorDecision(
        final_plan=final_plan,
        final_answer=final_answer,
        diagnosis_and_staging=coordinator_output.diagnosis_and_staging,
        safety_assessment=coordinator_output.safety_assessment,
        monitoring_protocol=coordinator_output.monitoring_protocol,
        consensus=consensus,
        accepted_actions=accepted_actions,
        rejected_actions=rejected_actions,
        conflicts=conflicts,
        role_weights=effective_weights,
        decision_rationale=coordinator_output.decision_rationale,
        audit_trace=audit_trace,
        response_text="\n".join(response_lines),
        final_confidence=coordinator_output.final_confidence,
    )


def deterministic_coordinator(
    case: CaseDossier,
    specialist_outputs: list[AgentOutput],
    role_weights: dict[SpecialistRoleName, float],
) -> CoordinatorDecision:
    outputs_by_role = {output.role: output for output in specialist_outputs}
    safety_output = outputs_by_role["safety"]
    effective_weights = _role_weights_with_modifiers(case, role_weights)

    plan_items: list[PlanItem] = []
    conflicts: list[ConflictRecord] = []
    audit_trace: list[str] = []

    for output in specialist_outputs:
        for action in output.candidate_actions:
            if action.stance == "avoid":
                conflicts.append(
                    ConflictRecord(
                        action=action.action,
                        recommenders=[],
                        objectors=[output.role],
                        severity="major",
                        conflict_type="risk",
                        outcome="flagged",
                        rationale=action.rationale,
                    )
                )
                continue
            score = round(effective_weights[output.role] * output.confidence, 4)
            plan_items.append(
                PlanItem(
                    action=action.action,
                    owner_role=output.role,
                    rationale=action.rationale,
                    priority=action.priority,
                    score=score,
                    citations=action.citations,
                )
            )
            audit_trace.append(f"Fallback coordinator kept '{action.action}' from {output.role}.")

    fallback = CoordinatorOutput(
        summary="Fallback coordinator summary generated from specialist actions.",
        final_answer="",
        final_plan=plan_items,
        accepted_actions=[item.action for item in plan_items],
        rejected_actions=[],
        conflicts=conflicts,
        decision_rationale="Fallback deterministic coordinator used because LLM coordinator output was unavailable.",
        audit_trace=audit_trace,
        final_confidence=max((output.confidence for output in specialist_outputs), default=0.0),
        supporting_facts=[],
    )
    return apply_safety_guardrail(case, specialist_outputs, fallback, role_weights)
