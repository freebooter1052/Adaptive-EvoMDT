"""Behaviour added to match the paper's Methods: retrieval, citation binding, consensus, evolution scope."""
import asyncio
import json

import numpy as np

from evomdt.config import RetrievalConfig
from evomdt.consensus import actions_match, conflict_severity_score, weighted_consensus
from evomdt.evaluation import weighted_plan_concordance
from evomdt.evolution import apply_feedback, initial_state
from evomdt.llm import LLMTrace
from evomdt.models import (
    AgentOutput,
    CaseDossier,
    ConflictRecord,
    CoordinatorDecision,
    EvaluationResult,
    PlanItem,
)
from evomdt.pipeline import EvoMDTSystem
from evomdt.retrieval import BM25, KnowledgeBase, tokenize


class FixedEmbedder:
    def __init__(self, vector):
        self.vector = np.array(vector, dtype=np.float32)

    def encode(self, texts, batch_size=32):
        return np.tile(self.vector, (len(texts), 1))


def make_kb(tmp_path, dense_query=(1.0, 0.0)) -> KnowledgeBase:
    atoms = [
        {"evidence_id": "Eaaaa1111", "source_id": "guide_a", "title": "Guide A", "anchor": "p.3", "version": "2024", "year": 2024,
         "source_type": "guideline", "cancer": "hcc", "licence": "CC BY", "evidence_grade": None, "text": "resection is recommended for early hepatocellular carcinoma"},
        {"evidence_id": "Ebbbb2222", "source_id": "guide_b", "title": "Guide B", "anchor": "p.9", "version": "2021", "year": 2021,
         "source_type": "guideline", "cancer": "breast", "licence": "CC BY", "evidence_grade": None, "text": "endocrine therapy for luminal breast cancer"},
        {"evidence_id": "Ecccc3333", "source_id": "label_x", "title": "FDA label: x", "anchor": "section: Contraindications", "version": "SPL v2", "year": 2025,
         "source_type": "drug_label", "cancer": "pan", "licence": "CC0", "evidence_grade": None, "text": "contraindicated in severe hepatic impairment"},
    ]
    (tmp_path / "atoms.jsonl").write_text("\n".join(json.dumps(atom) for atom in atoms), encoding="utf-8")
    np.save(tmp_path / "embeddings.npy", np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32))
    config = RetrievalConfig(enabled=True, kb_dir=str(tmp_path))
    return KnowledgeBase(config, embedder=FixedEmbedder(dense_query))


def hcc_case(**overrides) -> CaseDossier:
    values = {"case_id": "c1", "cancer_type": "hepatocellular_carcinoma", "question": "Plan?", "stage": "BCLC A"}
    return CaseDossier(**{**values, **overrides})


def specialist(role: str, answer: str = "", confidence: float = 0.8) -> AgentOutput:
    return AgentOutput(role=role, summary="s", confidence=confidence, answer_candidate=answer)


def test_bm25_prefers_documents_with_query_terms():
    bm25 = BM25([tokenize("resection for liver cancer"), tokenize("endocrine therapy")])
    scores = bm25.scores(tokenize("liver resection"))
    assert scores[0] > scores[1] == 0


def test_search_filters_to_the_case_cancer_and_pan_cancer_sources(tmp_path):
    hits = make_kb(tmp_path).search("resection", top_k=3, cancer="hcc")
    assert {hit.evidence_id for hit in hits} == {"Eaaaa1111", "Ecccc3333"}
    assert hits[0].evidence_id == "Eaaaa1111"
    citation = hits[0].citation()
    assert (citation.source_id, citation.anchor, citation.version) == ("guide_a", "p.3", "2024")


class CitingProvider:
    """Cites one real and one invented ID in every reply, to exercise citation binding."""

    async def generate_structured(self, messages, parser):
        is_coordinator = "Role: coordinator" in messages[1].content
        if is_coordinator:
            payload = {
                "summary": "s", "final_answer": "Based on a comprehensive multidisciplinary evaluation, resect.",
                "final_plan": [{"action": "Resection", "owner_role": "treatment", "rationale": "r", "priority": 1, "score": 0.1,
                                "citations": [{"source_id": "Eaaaa1111", "evidence_grade": "made up"}, "Efabricated"]}],
                "decision_rationale": "r", "final_confidence": 0.8,
            }
        else:
            payload = {
                "summary": "s", "confidence": 0.8, "answer_candidate": "Resection",
                "candidate_actions": [{"action": "Resection", "stance": "recommend", "rationale": "r", "priority": 1,
                                       "citations": [{"evidence_id": "Eaaaa1111"}, {"source_id": "NCCN made up"}]}],
            }
        content = json.dumps(payload)
        return LLMTrace(raw_content=content, parsed=parser(content), model_name="fake", usage={"total_tokens": 1})


def test_run_binds_citations_to_retrieved_evidence_and_drops_the_rest(tmp_path, test_config):
    system = EvoMDTSystem(test_config, provider_factory=lambda _: CitingProvider(), knowledge_base=make_kb(tmp_path))
    result = asyncio.run(system.run_case(hcc_case(), persist=False))

    item = result.coordinator_decision.final_plan[0]
    assert [citation.evidence_id for citation in item.citations] == ["Eaaaa1111"]
    assert item.citations[0].source_id == "guide_a"
    assert item.citations[0].evidence_grade is None  # KB provenance replaces whatever the model reported.
    assert result.evaluation.metrics["evidence_traceability_coverage"] == 1.0
    assert all(trace.dropped_citations == 1 for trace in result.agent_traces)
    assert all(trace.retrieved_evidence_ids for trace in result.agent_traces)
    assert "guide_a, p.3, 2024" in result.coordinator_decision.response_text


def test_consensus_quorum_needs_two_thirds_of_weight_and_three_votes():
    mcq = hcc_case(options=["A", "B", "C", "D"], task_type="mcq")
    weights = {role: 1.0 for role in ("diagnostic", "treatment", "safety", "monitoring")}

    agreed = weighted_consensus(mcq, [specialist(r, "b") for r in ("diagnostic", "treatment", "safety")] + [specialist("monitoring", "A")], weights)
    assert (agreed["leader"], agreed["quorum_reached"]) == ("B", True)

    split = weighted_consensus(mcq, [specialist("diagnostic", "A"), specialist("treatment", "A"), specialist("safety", "B"), specialist("monitoring", "B")], weights)
    assert split["quorum_reached"] is False

    invalid = weighted_consensus(mcq, [specialist("diagnostic", "Z"), specialist("treatment", "A")], weights)
    assert invalid["votes"] == {"treatment": "A"} and invalid["quorum_reached"] is False


def test_conflict_severity_puts_risk_above_semantic_disagreement():
    risky = ConflictRecord(action="x", objectors=["safety"], recommenders=["treatment"], severity="absolute", conflict_type="risk", outcome="rejected", rationale="r")
    mild = ConflictRecord(action="x", objectors=["safety"], recommenders=["treatment"], severity="minor", conflict_type="semantic", outcome="flagged", rationale="r")
    assert conflict_severity_score(risky) > conflict_severity_score(mild)


def test_plan_concordance_matches_reworded_actions_but_not_unrelated_ones():
    assert actions_match("Laparoscopic liver resection", "Perform a laparoscopic resection of the liver segment")
    assert not actions_match("Laparoscopic liver resection", "Atezolizumab plus bevacizumab")

    case = hcc_case(metadata={"reference_plan": [{"action": "Laparoscopic liver resection", "priority": 1}, {"action": "Entecavir antiviral therapy", "priority": 1}]})
    decision = CoordinatorDecision(
        final_plan=[PlanItem(action="Perform laparoscopic resection of the liver", owner_role="treatment", rationale="r", priority=1, score=0.9)],
        final_answer="x", role_weights={}, decision_rationale="r", response_text="x", final_confidence=0.8,
    )
    assert weighted_plan_concordance(case, decision) == 0.5


def test_evolution_widens_retrieval_scope_up_to_the_configured_cap(test_config):
    test_config.retrieval.top_k = 5
    test_config.retrieval.max_top_k = 6
    state = initial_state(test_config)
    feedback = EvaluationResult(metrics={}, feedback_tags=["evidence_gap"], summary="s")

    state = apply_feedback(test_config, state, feedback)
    assert state.retrieval_top_k["treatment"] == 6
    assert state.retrieval_top_k["safety"] == 5
    assert state.history[-1].retrieval_updates == {"diagnostic": 6, "treatment": 6}

    state = apply_feedback(test_config, state, feedback)
    assert state.retrieval_top_k["treatment"] == 6
