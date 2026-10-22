"""Hybrid retrieval (Sections 3.4 and 4).

* GraphRetriever: exact traversal over WaterOnt. Every hit carries the explicit
  path of (subject, RELATIONSHIP, object) triples that produced it, so a
  recommendation can be audited edge by edge.
* VectorIndex: similarity search over EngineeringGuideline text. TF-IDF by
  default (no downloads, deterministic); pass any `embed(list[str]) -> ndarray`
  callable to use dense embeddings instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .decision import ACTION_RANK, pipe_facts
from .kg import KnowledgeGraph

Triple = tuple[str, str, str]
BAND_REL = {"pof": "HAS_FAILURE_PROBABILITY", "hydraulic": "HAS_HYDRAULIC_CRITICALITY",
            "service": "HAS_SERVICE_CRITICALITY", "overall": "HAS_OVERALL_RISK",
            "monitoring": "HAS_MONITORING_STATUS"}
BAND_PREFIX = {"pof": "FP", "hydraulic": "HC", "service": "SC", "overall": "OR", "monitoring": "MS"}


class VectorIndex:
    def __init__(self, docs: list[dict], text_key: str = "text",
                 embed: Callable[[list[str]], np.ndarray] | None = None):
        self.docs = docs
        texts = [f"{d.get('title', '')}. {d[text_key]}" for d in docs]
        self.embed = embed
        if embed is None:
            self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.M = self.vec.fit_transform(texts)
        else:
            E = np.asarray(embed(texts), dtype=float)
            self.M = E / np.linalg.norm(E, axis=1, keepdims=True)

    def search(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        if self.embed is None:
            sims = (self.M @ self.vec.transform([query]).T).toarray().ravel()
        else:
            q = np.asarray(self.embed([query]), dtype=float).ravel()
            sims = self.M @ (q / np.linalg.norm(q))
        order = np.argsort(-sims)[:k]
        return [(self.docs[i], float(sims[i])) for i in order if sims[i] > 0]


@dataclass
class Evidence:
    pipe_uid: str
    facts: dict
    action: str
    rule: str
    guidelines: list[str]
    path: list[Triple] = field(default_factory=list)


class GraphRetriever:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    # -------------------------------------------------------------- candidates
    def pipes_by_bands(self, **bands: list[str]) -> tuple[set[str], dict[str, list[Triple]]]:
        """Intersection of RiskProfiles reached *from* the band nodes (multi-hop pattern)."""
        rp_sets, paths = [], {}
        for dim, wanted in bands.items():
            hits = set()
            for b in wanted:
                band_node = f"{BAND_PREFIX[dim]}:{b}"
                if not self.kg.has(band_node):
                    continue
                for rp in self.kg.inc(band_node, BAND_REL[dim]):
                    hits.add(rp)
                    paths.setdefault(rp, []).append((rp, BAND_REL[dim], band_node))
            rp_sets.append(hits)
        rps = set.intersection(*rp_sets) if rp_sets else set(self.kg.nodes("RiskProfile"))
        pipes, out_paths = set(), {}
        for rp in rps:
            for p in self.kg.inc(rp, "HAS_RISK_PROFILE"):
                pipes.add(p)
                out_paths[p] = [(p, "HAS_RISK_PROFILE", rp), *paths.get(rp, [])]
        return pipes, out_paths

    def pipes_by_action(self, actions: list[str]) -> set[str]:
        out = set()
        for a in actions:
            for dr in self.kg.inc(f"MA:{a}", "RECOMMENDS"):
                out |= set(self.kg.out(dr, "CONCERNS"))
        return out

    def pipes_by_asset(self, material: list[str] | None = None, installed_before: int | None = None,
                       min_failures: int | None = None) -> set[str]:
        out = set()
        for p in self.kg.nodes("Pipe"):
            ar = self.kg.node(self.kg.one(p, "HAS_ASSET_RECORD"))
            if material and ar["material"] not in material:
                continue
            if installed_before and ar["installationYear"] >= installed_before:
                continue
            if min_failures is not None and ar["failureCount"] < min_failures:
                continue
            out.add(p)
        return out

    def pipes_beyond_design_life(self) -> set[str]:
        out = set()
        for p in self.kg.nodes("Pipe"):
            ar = self.kg.node(self.kg.one(p, "HAS_ASSET_RECORD"))
            if ar["age"] > ar["designLife"]:
                out.add(p)
        return out

    def pipes_serving_facilities(self) -> set[str]:
        out = set()
        for fac in self.kg.nodes("CriticalFacility"):
            for j in self.kg.inc(fac, "SERVES_FACILITY"):
                out |= set(self.kg.inc(j, "STARTS_AT")) | set(self.kg.inc(j, "ENDS_AT"))
        return {p for p in out if "Pipe" in self.kg.node(p)["labels"]}

    def pipes_low_pressure(self, threshold: float) -> set[str]:
        return {p for p in self.kg.nodes("Pipe")
                if self.kg.node(self.kg.one(p, "HAS_RISK_PROFILE"))["minEndPressure"] < threshold}

    def zones(self) -> dict[str, dict]:
        return {z: self.kg.node(z) for z in self.kg.nodes("PressureZone") if self.kg.node(z).get("isSupplyZone")}

    def resolve_zone(self, ref: str) -> str | None:
        """Match a zone by a device/source that feeds it (PRV-3, PUMP_1, T1) or 'tank'/'storage'."""
        ref = ref.lower()
        for z, d in self.zones().items():
            members = d["name"].lower().split("+")
            if ref in members or ref == d["name"].lower():
                return z
            if ref in ("tank", "storage") and d.get("containsStorage"):
                return z
        return None

    def pipes_in_zone(self, zone_uid: str) -> set[str]:
        return {p for p in self.kg.inc(zone_uid, "IN_ZONE") if "Pipe" in self.kg.node(p)["labels"]}

    def pipes_incident_on(self, junction_name: str) -> set[str]:
        j = f"J:{junction_name}"
        if not self.kg.has(j):
            return set()
        return {p for p in self.kg.inc(j, "STARTS_AT") + self.kg.inc(j, "ENDS_AT")
                if "Pipe" in self.kg.node(p)["labels"]}

    def pipe_by_name(self, name: str) -> str | None:
        uid = f"P:{name}"
        return uid if self.kg.has(uid) else None

    # ---------------------------------------------------------------- evidence
    def evidence(self, pipe_uid: str, prefix_path: list[Triple] | None = None) -> Evidence:
        kg = self.kg
        f = pipe_facts(kg, pipe_uid)
        dr = kg.inc(pipe_uid, "CONCERNS")[0]
        action_node = kg.one(dr, "RECOMMENDS")
        gls = kg.out(dr, "SUPPORTED_BY")
        ar, rp = kg.one(pipe_uid, "HAS_ASSET_RECORD"), kg.one(pipe_uid, "HAS_RISK_PROFILE")
        path = list(prefix_path or [])
        path += [(pipe_uid, "HAS_ASSET_RECORD", ar), (pipe_uid, "HAS_RISK_PROFILE", rp),
                 (rp, "HAS_OVERALL_RISK", kg.one(rp, "HAS_OVERALL_RISK")),
                 (dr, "CONCERNS", pipe_uid), (dr, "RECOMMENDS", action_node),
                 (dr, "BASED_ON", ar), (dr, "BASED_ON", rp),
                 *[(dr, "SUPPORTED_BY", g) for g in gls]]
        seen, uniq = set(), []
        for t in path:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return Evidence(pipe_uid, f, kg.node(dr)["action"], kg.node(dr)["rule"], gls, uniq)

    def verify_path(self, path: list[Triple]) -> bool:
        return bool(path) and all(self.kg.has_edge(s, r, o) for s, r, o in path)

    @staticmethod
    def rank(evs: list[Evidence]) -> list[Evidence]:
        return sorted(evs, key=lambda e: (ACTION_RANK[e.action], -e.facts["priority"], e.pipe_uid))
