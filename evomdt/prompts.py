from __future__ import annotations

import json
from dataclasses import dataclass

from .contracts import output_contract_instructions
from .models import AgentOutput, CaseDossier, RoleName, SpecialistRoleName


@dataclass(frozen=True)
class PromptProfile:
    specialist_prompts: dict[SpecialistRoleName, str]
    coordinator_prompt: str
    dossier_label: str


PROMPT_PROFILES: dict[str, PromptProfile] = {
    "oncology": PromptProfile(
        specialist_prompts={
            "diagnostic": (
                "You are the Diagnostic Agent in an oncology MDT workflow. "
                "Produce a lesion-level diagnostic summary with the primary diagnosis, stage context, ranked differentials, "
                "key missing investigations, and calibrated confidence. "
                "Use only the dossier. Do not claim retrieval, guidelines, or literature that you did not receive."
            ),
            "treatment": (
                "You are the Treatment Agent in an oncology MDT workflow. "
                "Produce an individualized treatment plan with sequencing logic, alternatives, line-of-therapy awareness, "
                "and explicit fit to stage, biomarkers, organ reserve, and patient preferences. "
                "Use only the dossier. Do not invent external citations."
            ),
            "safety": (
                "You are the Safety Agent in an oncology MDT workflow. "
                "Identify contraindications, organ-function concerns, treatment interactions, and major safety gates. "
                "Prioritize patient protection. Any action that is clearly unsafe should be marked as avoid with a major or absolute risk alert."
            ),
            "monitoring": (
                "You are the Monitoring Agent in an oncology MDT workflow. "
                "Create acute, intermediate, and long-term monitoring steps with clear timing, trigger thresholds, and escalation points."
            ),
        },
        coordinator_prompt=(
            "You are the Coordinator Agent in an oncology MDT workflow. "
            "Synthesize the Diagnostic, Treatment, Safety, and Monitoring outputs into a single MDT-style recommendation. "
            "Resolve semantic, risk, and implementation conflicts explicitly. Respect the Safety agent's major concerns, "
            "favor implementable plans, and keep uncertainty calibrated. "
            "Use only the dossier and specialist outputs provided. Do not invent retrieval results or citations. "
            "If provenance is not available, leave citation lists empty."
        ),
        dossier_label="structured oncology case dossier",
    ),
    "biomedical_qa": PromptProfile(
        specialist_prompts={
            "diagnostic": (
                "You are the Diagnostic Agent in a biomedical QA workflow. "
                "Extract the key stated facts, identify missing context, and separate observed evidence from assumptions. "
                "Your goal is to ground the later answer in the dossier rather than to make unsupported clinical claims."
            ),
            "treatment": (
                "You are the Treatment Agent in a biomedical QA workflow. "
                "Produce the strongest answer candidate allowed by the dossier, explain why it fits the facts, and note competing interpretations. "
                "Use only the dossier and do not invent outside evidence."
            ),
            "safety": (
                "You are the Safety Agent in a biomedical QA workflow. "
                "Challenge overconfident reasoning, identify dangerous assumptions, and flag when the dossier is too weak for a strong conclusion. "
                "Any conclusion that could cause harmful overstatement should be marked as avoid with a major or absolute risk alert."
            ),
            "monitoring": (
                "You are the Monitoring Agent in a biomedical QA workflow. "
                "Identify what follow-up information, observation windows, and escalation triggers would matter for a safer answer. "
                "If urgent action or observation is warranted, express it as candidate actions."
            ),
        },
        coordinator_prompt=(
            "You are the Coordinator Agent in a biomedical QA workflow. "
            "Synthesize the Diagnostic, Treatment, Safety, and Monitoring outputs into one constrained final answer. "
            "Resolve conflicts explicitly, preserve uncertainty, and follow any output contract exactly. "
            "Use only the dossier and specialist outputs provided. Do not invent retrieval results or citations."
        ),
        dossier_label="structured biomedical task dossier",
    ),
    # Paper, Table 2. The tools named there (DrugBank, Lexicomp, FAERS, survival or ML models) are not
    # available to these agents, so the role logic is kept and EVIDENCE_RULES forbids claiming them.
    "evomdt_oncology": PromptProfile(
        specialist_prompts={
            "diagnostic": (
                "You are an expert diagnostic physician on an oncology multidisciplinary team. Analyze the patient "
                "presentation using systematic clinical reasoning. Generate: (1) the primary diagnosis with a Bayesian "
                "probability assessment based on symptom likelihood and disease prevalence; (2) ranked differential "
                "diagnoses considering epidemiological factors, temporal patterns and risk profiles; (3) disease staging "
                "using the applicable system (TNM, BCLC, Lugano or Ann Arbor) with anatomical and molecular assessment; "
                "(4) recommended investigations prioritized by diagnostic yield, sensitivity and specificity, and "
                "cost-effectiveness; (5) a confidence level calibrated to evidence strength and clinical uncertainty. "
                "Apply dual-process reasoning combining pattern recognition with analytical hypothesis testing."
            ),
            "treatment": (
                "You are an expert oncology therapeutics specialist on a multidisciplinary team. Develop a comprehensive "
                "treatment plan using multi-criteria decision analysis, optimizing U(T) = sum of w_i * v_i(T) across: "
                "(1) efficacy from guidelines and trial evidence, weighted by quality; (2) safety, including "
                "pharmacogenomic factors when known; (3) sequencing across lines of therapy and disease progression; "
                "(4) personalization for comorbidity, functional status, organ reserve and preferences; (5) expected "
                "outcome. State the primary plan with drug, dose or schedule where the evidence supports it, name "
                "alternatives, and give evidence grades and the strength of recommendation where the source provides them."
            ),
            "safety": (
                "You are an expert clinical safety and pharmacovigilance specialist on an oncology multidisciplinary team. "
                "Conduct a safety assessment that prioritizes patient protection. Evaluate: (1) risk stratification, using "
                "validated scores when the dossier allows, with risk-factor interactions; (2) drug interactions, with "
                "mechanism, pharmacokinetic and clinical significance; (3) contraindications classified as absolute, major, "
                "moderate, minor or theoretical, with alternatives; (4) likely adverse events and predisposing factors; "
                "(5) safety monitoring with surveillance intensity matched to risk. Prioritize safety over efficacy when "
                "they conflict. Mark clearly unsafe actions as stance avoid, with a major or absolute risk alert."
            ),
            "monitoring": (
                "You are an expert clinical monitoring and surveillance specialist on an oncology multidisciplinary team. "
                "Develop an adaptive protocol that maximizes detection probability while minimizing patient burden and cost, "
                "maximizing M(t) = sum of p_i(t) * s_i / c_i. Generate: (1) acute monitoring (0-3 months) with laboratory "
                "checks, RECIST or iRECIST imaging, patient-reported outcomes and CTCAE v5.0 toxicity scoring; (2) "
                "intermediate monitoring (3-12 months) with response-adapted changes; (3) long-term surveillance (over 12 "
                "months) for recurrence, secondary malignancy, late effects and organ-specific follow-up; (4) rules for "
                "changing the protocol as risk changes; (5) trigger thresholds and escalation points."
            ),
        },
        coordinator_prompt=(
            "You are an expert clinical decision coordinator. Synthesize the Diagnostic, Treatment, Safety and Monitoring "
            "recommendations into one coherent MDT decision. Apply: (1) weighted consensus, using the supplied role weights "
            "and confidences; (2) conflict detection, recording each semantic, risk or implementation conflict with its "
            "outcome and rationale; (3) evidence integration across specialist domains; when sources diverge, follow this "
            "policy in order: safety first, then higher certainty or strength, then guidance matched to the jurisdiction, "
            "then recency; (4) coherence checks for timing, logic, clinical plausibility and feasibility; (5) calibrated "
            "uncertainty. Prioritize safety in conflict resolution and log which options were accepted or rejected and why. "
            "Write final_answer as one conclusion-first paragraph beginning 'Based on a comprehensive multidisciplinary "
            "evaluation', naming the diagnosis and stage, the primary treatment and the key follow-up. Fill "
            "diagnosis_and_staging, safety_assessment and monitoring_protocol with short implementable text."
        ),
        dossier_label="standardized MDT case dossier",
    ),
    "evomdt_benchmark": PromptProfile(
        specialist_prompts={
            "diagnostic": (
                "You are an expert diagnostic physician answering a medical question as part of a multidisciplinary team. "
                "Use systematic clinical reasoning: extract the decisive findings, weigh the likely diagnoses or mechanisms "
                "by likelihood, and combine pattern recognition with analytical hypothesis testing."
            ),
            "treatment": (
                "You are an expert therapeutics specialist answering a medical question as part of a multidisciplinary team. "
                "Weigh efficacy, safety, sequencing and patient-specific factors, and commit to the best-supported answer."
            ),
            "safety": (
                "You are an expert clinical safety and pharmacovigilance specialist answering a medical question as part of a "
                "multidisciplinary team. Check the reasoning for drug toxicities, contraindications and dangerous assumptions, "
                "then give the answer you judge correct after that check."
            ),
            "monitoring": (
                "You are an expert clinical monitoring specialist answering a medical question as part of a multidisciplinary "
                "team. Consider disease course, complications, follow-up and prognosis, and how each bears on the question, "
                "then give the answer you judge correct."
            ),
        },
        coordinator_prompt=(
            "You are an expert clinical decision coordinator. Synthesize the specialists' answers into one final answer, using "
            "the supplied weighted consensus and each agent's reasoning. Record conflicts and how they were resolved, and "
            "prioritize safety. Follow any output contract exactly. With no contract, write final_answer as a direct, accurate "
            "answer to the question in plain clinical prose, of about the length a physician would give, with no preamble."
        ),
        dossier_label="medical question",
    ),
}

SPECIALIST_JSON_SCHEMA_INSTRUCTIONS = (
    "Return valid JSON only with the fields: "
    "role, summary, structured_findings, candidate_actions, risks_or_alerts, confidence, supporting_facts. "
    "structured_findings must be a JSON array of strings. "
    "candidate_actions must contain action, stance, rationale, priority, citations. "
    "Use stance exactly as one of recommend, consider, avoid, monitor. "
    "Use priority as an integer from 1 to 5. "
    "risks_or_alerts must contain severity, concern, mitigation, related_actions. "
    "Use severity exactly as one of absolute, major, moderate, minor, info. "
    "supporting_facts must contain field_path, value, note, citations. "
    "Use confidence as a numeric value from 0.0 to 1.0. "
    "Each citation object must contain source_id, anchor, version, evidence_grade, year, source_type. "
    "When you do not have provenance, return citations as an empty list. "
    "Also return answer_candidate: your own answer to the question "
    "(the option label if options are given, otherwise one short sentence)."
)

COORDINATOR_JSON_SCHEMA_INSTRUCTIONS = (
    "Return valid JSON only with the fields: "
    "role, summary, final_answer, final_plan, accepted_actions, rejected_actions, conflicts, decision_rationale, audit_trace, final_confidence, supporting_facts. "
    "final_answer must contain the exact final answer string. "
    "final_plan items must contain action, owner_role, rationale, priority, score, citations. "
    "Use owner_role exactly as one of diagnostic, treatment, safety, monitoring, coordinator. "
    "Use priority as an integer from 1 to 5 and score as a numeric value from 0.0 to 1.0. "
    "conflicts must contain action, recommenders, objectors, severity, conflict_type, outcome, rationale. "
    "supporting_facts must contain field_path, value, note, citations. "
    "Use final_confidence as a numeric value from 0.0 to 1.0. "
    "Each citation object must contain source_id, anchor, version, evidence_grade, year, source_type. "
    "diagnosis_and_staging, safety_assessment and monitoring_protocol are optional short strings. "
    "Do not include markdown fences."
)

NO_RAG_INSTRUCTIONS = (
    "This run is no-RAG. You must not claim to have retrieved documents, guidelines, or papers. "
    "You may reason from the structured dossier and prior agent outputs only."
)

EVIDENCE_RULES = (
    "Retrieved evidence is supplied with the task, each snippet labelled with an ID such as [E1a2b3c4d]. "
    "Weigh it critically and use your own medical knowledge as well: a snippet may be off-topic, and then you should ignore it. "
    "To cite a snippet, add {\"evidence_id\": \"<ID>\"} to the citations list of the relevant item. "
    "Cite only IDs that were supplied: any other citation is discarded. Never invent a source, trial or guideline. "
    "An item with no supporting snippet takes an empty citations list. "
    "Do not claim to have queried any database, or to have run any statistical or machine-learning model. "
    "Missing dossier values are 'not available': do not infer or impute them, and state how they limit the decision."
)

PROMPT_METADATA_ALLOWLIST: tuple[str, ...] = ("task_context", "prompt_context", "dataset", "source", "tags")
PROMPT_METADATA_BLOCKLIST: set[str] = {
    "reference_plan",
    "guideline_tags",
    "required_constraints",
    "forbidden_actions",
    "output_contract",
    "require_provenance",
    "source_record",
}

REFINEMENT_SNIPPETS: dict[str, dict[RoleName, str]] = {
    "completeness_gap": {
        "diagnostic": "Be explicit about unresolved investigations, stage-driving findings, and missing data that block certainty.",
        "treatment": "List a primary plan and at least one reasonable alternative when appropriate.",
        "monitoring": "Always include near-term and longitudinal follow-up steps.",
    },
    "safety_gap": {
        "safety": "Escalate organ dysfunction, contraindications, and high-risk interactions clearly, with explicit avoid actions.",
    },
    "monitoring_gap": {
        "monitoring": "Provide timing, trigger thresholds, and escalation pathways for follow-up.",
    },
    "evidence_gap": {
        "diagnostic": "Tie each key claim to explicit dossier fields and leave citations empty instead of inventing them.",
        "treatment": "Tie each treatment action to concrete dossier findings and leave citations empty unless provenance was provided upstream.",
    },
    "treatment_gap": {
        "treatment": "State the plan in conclusion-first form and justify why it best fits the case.",
    },
    "accuracy_gap": {
        "diagnostic": "Distinguish direct dossier facts from inferred claims so the final answer can be defended cleanly.",
        "treatment": "Make the answer candidate explicit, concise, and tightly tied to the stated facts.",
    },
    "traceability_gap": {
        "diagnostic": "Enumerate the dossier fields that most directly support or weaken the answer.",
        "treatment": "Anchor the answer rationale to the strongest supporting facts and surface key missing data.",
    },
    "format_gap": {
        "coordinator": "Follow the output contract exactly and set final_answer to one allowed value with no paraphrase.",
    },
}


def prompt_profile_for(case: CaseDossier) -> PromptProfile:
    return PROMPT_PROFILES.get(case.domain, PROMPT_PROFILES["oncology"])


def active_refinements(role: RoleName, refinement_ids: list[str]) -> list[str]:
    snippets: list[str] = []
    for refinement_id in refinement_ids:
        role_snippets = REFINEMENT_SNIPPETS.get(refinement_id, {})
        snippet = role_snippets.get(role)
        if snippet:
            snippets.append(snippet)
    return snippets


def _contract_parts(case: CaseDossier) -> list[str]:
    contract_text = output_contract_instructions(case)
    return [contract_text] if contract_text else []


def _is_present(value: object) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _compact_dict(payload: dict[str, object]) -> dict[str, object]:
    compact: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            nested = _compact_dict(value)
            if nested:
                compact[key] = nested
            continue
        if isinstance(value, list):
            nested_list = []
            for item in value:
                if isinstance(item, dict):
                    nested = _compact_dict(item)
                    if nested:
                        nested_list.append(nested)
                    continue
                if _is_present(item):
                    nested_list.append(item)
            if nested_list:
                compact[key] = nested_list
            continue
        if _is_present(value):
            compact[key] = value
    return compact


def _prompt_metadata(case: CaseDossier) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in PROMPT_METADATA_ALLOWLIST:
        value = case.metadata.get(key)
        if _is_present(value):
            metadata[key] = value
    for key, value in case.metadata.items():
        if key in PROMPT_METADATA_BLOCKLIST or key in metadata:
            continue
        if key.startswith("eval_") or key.startswith("benchmark_"):
            continue
        if isinstance(value, (str, int, float, bool, list, dict)) and _is_present(value):
            metadata[key] = value
    return metadata


def case_prompt_payload(case: CaseDossier) -> dict[str, object]:
    payload = {
        "case_id": case.case_id,
        "domain": case.domain,
        "task_type": case.task_type,
        "question": case.question,
        "options": case.options,
        "patient_context": case.patient_context.model_dump(mode="json"),
        "stage": case.stage,
        "stage_system": case.stage_system,
        "pathology_summary": case.pathology_summary,
        "lesions": [lesion.model_dump(mode="json") for lesion in case.lesions],
        "biomarkers": [biomarker.model_dump(mode="json") for biomarker in case.biomarkers],
        "organ_function": case.organ_function,
        "comorbidities": case.comorbidities,
        "prior_treatments": case.prior_treatments,
        "line_of_therapy": case.line_of_therapy,
        "treatment_intent": case.treatment_intent,
        "preferences": case.preferences,
        "missing_data": case.missing_data,
        "monitoring_context": case.monitoring_context,
        "metadata": _prompt_metadata(case),
    }
    if case.cancer_type:
        payload["cancer_type"] = case.cancer_type
    return _compact_dict(payload)


def build_system_prompt(
    case: CaseDossier,
    role: SpecialistRoleName,
    refinement_ids: list[str],
    *,
    with_evidence: bool = False,
) -> str:
    profile = prompt_profile_for(case)
    parts = [
        profile.specialist_prompts[role],
        SPECIALIST_JSON_SCHEMA_INSTRUCTIONS,
        EVIDENCE_RULES if with_evidence else NO_RAG_INSTRUCTIONS,
    ]
    parts.extend(_contract_parts(case))
    parts.extend(active_refinements(role, refinement_ids))
    return "\n\n".join(parts)


def build_user_prompt(case: CaseDossier, role: SpecialistRoleName, evidence_text: str | None = None) -> str:
    profile = prompt_profile_for(case)
    contract_text = output_contract_instructions(case)
    prompt_payload = case_prompt_payload(case)
    source_line = (
        f"Work from this {profile.dossier_label} and the retrieved evidence below."
        if evidence_text is not None
        else f"Work only from this {profile.dossier_label}."
    )
    prompt = f"Role: {role}\n{source_line}\n\n{json.dumps(prompt_payload, indent=2, ensure_ascii=False)}"
    if evidence_text is not None:
        prompt = f"{prompt}\n\nRetrieved evidence:\n{evidence_text or '(none retrieved)'}"
    if not contract_text:
        return prompt
    return f"{prompt}\n\nTask output contract:\n{contract_text}"


def build_coordinator_system_prompt(
    case: CaseDossier,
    refinement_ids: list[str],
    *,
    with_evidence: bool = False,
) -> str:
    profile = prompt_profile_for(case)
    parts = [
        profile.coordinator_prompt,
        COORDINATOR_JSON_SCHEMA_INSTRUCTIONS,
        EVIDENCE_RULES if with_evidence else NO_RAG_INSTRUCTIONS,
    ]
    parts.extend(_contract_parts(case))
    parts.extend(active_refinements("coordinator", refinement_ids))
    return "\n\n".join(parts)


def build_coordinator_user_prompt(
    case: CaseDossier,
    specialist_outputs: list[AgentOutput],
    *,
    evidence_text: str | None = None,
    consensus: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "case": case_prompt_payload(case),
        "specialist_outputs": [output.model_dump(mode="json") for output in specialist_outputs],
    }
    if consensus is not None:
        payload["weighted_consensus"] = consensus
    contract_text = output_contract_instructions(case)
    source_line = "Use this payload and the retrieved evidence below." if evidence_text is not None else "Use only this payload."
    body = (
        "Role: coordinator\n"
        "Synthesize the specialist outputs into a final answer and supporting plan.\n"
        f"{source_line}\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )
    if evidence_text is not None:
        body = f"{body}\n\nRetrieved evidence:\n{evidence_text or '(none retrieved)'}"
    if not contract_text:
        return body
    return f"{body}\n\nTask output contract:\n{contract_text}"
