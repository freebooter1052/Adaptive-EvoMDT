"""Hybrid evidence retrieval over the local knowledge base (paper, Methods: 'Knowledge base construction').

Semantic and keyword scores are min-max normalized per query, then combined with the paper's weights
(0.8 dense, 0.2 BM25). Every hit carries the provenance triplet (source ID, anchor, version), so an
agent's citation can be bound to a real KB entry rather than to text the model made up.
"""
from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RetrievalConfig
from .models import CaseDossier, CitationProvenance, RoleName

TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]+")

ROLE_QUERY_HINTS: dict[RoleName, str] = {
    "diagnostic": "diagnosis staging classification work-up investigations biomarker testing",
    "treatment": "recommended treatment first-line therapy surgery systemic regimen sequencing indication",
    "safety": "contraindications warnings adverse reactions toxicity drug interactions organ function dose modification",
    "monitoring": "follow-up surveillance response assessment monitoring schedule imaging interval",
    "coordinator": "recommended treatment guideline consensus",
}


@dataclass(frozen=True)
class EvidenceSnippet:
    evidence_id: str
    source_id: str
    title: str
    anchor: str
    version: str
    year: int | None
    source_type: str
    evidence_grade: str | None
    text: str
    score: float

    def citation(self) -> CitationProvenance:
        return CitationProvenance(
            evidence_id=self.evidence_id,
            source_id=self.source_id,
            anchor=self.anchor,
            version=self.version,
            evidence_grade=self.evidence_grade,
            year=self.year,
            source_type=self.source_type,
        )


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.lengths = np.array([len(doc) for doc in documents], dtype=np.float32)
        self.average = float(self.lengths.mean()) if len(documents) else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, doc in enumerate(documents):
            for term, count in Counter(doc).items():
                self.postings.setdefault(term, []).append((index, count))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - len(hits) + 0.5) / (len(hits) + 0.5))
            for term, hits in self.postings.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        result = np.zeros(len(self.lengths), dtype=np.float32)
        for term in set(query):
            for index, count in self.postings.get(term, []):
                norm = 1 - self.b + self.b * self.lengths[index] / self.average
                result[index] += self.idf[term] * count * (self.k1 + 1) / (count + self.k1 * norm)
        return result


class Embedder:
    """BGE-style dense encoder: CLS pooling, L2-normalized (the model is loaded on first use)."""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch

        with self._lock:
            if self._model is None:
                self._load()
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(texts), batch_size):
                    batch = self._tokenizer(
                        texts[start : start + batch_size],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    hidden = self._model(**batch).last_hidden_state[:, 0]
                    chunks.append(torch.nn.functional.normalize(hidden, dim=-1).float().cpu().numpy())
            return np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)


def _minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class KnowledgeBase:
    def __init__(self, config: RetrievalConfig, embedder: Embedder | None = None):
        kb_dir = Path(config.kb_dir)
        atoms_path = kb_dir / "atoms.jsonl"
        if not atoms_path.exists():
            raise FileNotFoundError(f"{atoms_path} not found. Build it with scripts/build_kb.py.")
        with atoms_path.open(encoding="utf-8") as handle:
            self.atoms = [json.loads(line) for line in handle]
        self.embeddings = np.load(kb_dir / "embeddings.npy")
        if len(self.embeddings) != len(self.atoms):
            raise ValueError("Knowledge base embeddings and atoms have different lengths; rebuild the KB.")
        self.config = config
        self.embedder = embedder or Embedder(config.embedding_model)
        self.bm25 = BM25([tokenize(atom["text"]) for atom in self.atoms])
        self.cancers = np.array([atom["cancer"] for atom in self.atoms])
        self.by_id = {atom["evidence_id"]: atom for atom in self.atoms}

    def search(
        self,
        query: str,
        *,
        top_k: int,
        cancer: str | None = None,
        source_types: set[str] | None = None,
    ) -> list[EvidenceSnippet]:
        dense = self.embeddings @ self.embedder.encode([query])[0]
        sparse = self.bm25.scores(tokenize(query))
        mask = np.ones(len(self.atoms), dtype=bool)
        if cancer:
            mask &= np.isin(self.cancers, [cancer, "pan"])
        if source_types:
            mask &= np.array([atom["source_type"] in source_types for atom in self.atoms])
        if not mask.any():
            return []
        combined = np.full(len(self.atoms), -np.inf, dtype=np.float32)
        combined[mask] = (
            self.config.dense_weight * _minmax(dense[mask]) + self.config.sparse_weight * _minmax(sparse[mask])
        )

        hits: list[EvidenceSnippet] = []
        seen: set[str] = set()
        for index in np.argsort(-combined):
            if not np.isfinite(combined[index]) or len(hits) >= top_k:
                break
            atom = self.atoms[int(index)]
            fingerprint = atom["text"][:160].lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            hits.append(
                EvidenceSnippet(
                    evidence_id=atom["evidence_id"],
                    source_id=atom["source_id"],
                    title=atom["title"],
                    anchor=atom["anchor"],
                    version=atom["version"],
                    year=atom.get("year"),
                    source_type=atom["source_type"],
                    evidence_grade=atom.get("evidence_grade"),
                    text=atom["text"],
                    score=round(float(combined[index]), 4),
                )
            )
        return hits


# Dossier cancer labels mapped to KB cancer tags.
CANCER_TAGS = {
    "hepatocellular_carcinoma": "hcc",
    "hcc": "hcc",
    "breast_cancer": "breast",
    "breast": "breast",
    "lung_adenocarcinoma": "lung",
    "lung": "lung",
    "lymphoma": "lymphoma",
}


def case_query(case: CaseDossier, role: RoleName) -> str:
    parts: list[str] = [ROLE_QUERY_HINTS[role]]
    if case.cancer_type:
        parts.append(case.cancer_type.replace("_", " "))
    parts.extend(filter(None, [case.stage, case.pathology_summary, case.line_of_therapy]))
    parts.extend(f"{biomarker.name} {biomarker.value}" for biomarker in case.biomarkers[:6])
    parts.extend(f"{lesion.site} {lesion.description or ''}" for lesion in case.lesions[:2])
    if role == "safety":
        parts.extend(f"{key} {value}" for key, value in list(case.organ_function.items())[:4])
        parts.extend(case.comorbidities[:5])
    if case.task_type != "plan_eval":
        parts.append(case.question)
    return " ".join(parts)[:1200]


def retrieve_for_role(kb: KnowledgeBase, case: CaseDossier, role: RoleName, top_k: int) -> list[EvidenceSnippet]:
    return kb.search(
        case_query(case, role),
        top_k=top_k,
        cancer=CANCER_TAGS.get((case.cancer_type or "").lower()),
    )


def render_evidence(evidence: list[EvidenceSnippet], max_chars: int = 900) -> str:
    blocks = []
    for item in evidence:
        grade = f"; grade: {item.evidence_grade}" if item.evidence_grade else ""
        blocks.append(
            f"[{item.evidence_id}] {item.title} | {item.anchor} | version {item.version} | "
            f"{item.source_type}{grade}\n{item.text[:max_chars]}"
        )
    return "\n\n".join(blocks)
