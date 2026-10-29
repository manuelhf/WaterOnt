"""Labelled property graph used as the single source of truth for WaterOnt.

The in-memory store (a NetworkX MultiDiGraph) keeps the whole pipeline runnable
without a database. `to_neo4j` pushes the same graph into Neo4j, applying the
schema constraints listed in Appendix A.7 of the paper.

Conventions
-----------
* Every node has a unique `id` and a list of `labels` (most specific first).
* Relationship types are UPPER_SNAKE_CASE, mirroring the ontology's camelCase
  object properties (wno:startsAt -> STARTS_AT).
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Iterable, Iterator

import networkx as nx

REL_TO_PROPERTY = {
    "STARTS_AT": "startsAt", "ENDS_AT": "endsAt",
    "CONTAINS_NODE": "containsNode", "CONTAINS_LINK": "containsLink",
    "FOLLOWS_PATTERN": "followsPattern", "HAS_CURVE": "hasCurve",
    "HAS_STATUS": "hasStatus", "GOVERNED_BY": "governedBy",
    "HAS_ASSET_RECORD": "hasAssetRecord", "EXPERIENCED_FAILURE": "experiencedFailure",
    "HAS_INSPECTION": "hasInspection", "HAS_RISK_PROFILE": "hasRiskProfile",
    "HAS_FAILURE_PROBABILITY": "hasFailureProbability",
    "HAS_HYDRAULIC_CRITICALITY": "hasHydraulicCriticality",
    "HAS_SERVICE_CRITICALITY": "hasServiceCriticality",
    "HAS_OVERALL_RISK": "hasOverallRisk", "SERVES_FACILITY": "servesFacility",
    "CONCERNS": "concerns", "RECOMMENDS": "recommends", "SUPPORTED_BY": "supportedBy",
    "BASED_ON": "basedOn", "OBSERVED_BY": "observedBy", "IN_ZONE": "inZone",
    "FEEDS_ZONE": "feedsZone", "HAS_ZONE": "hasZone",
    "HAS_MONITORING_STATUS": "hasMonitoringStatus",
}


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self._by_label: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------ write
    def add_node(self, node_id: str, *labels: str, **props) -> str:
        props.pop("id", None)
        if node_id in self.g:
            self.g.nodes[node_id].update(props)
            for lab in labels:
                if lab not in self.g.nodes[node_id]["labels"]:
                    self.g.nodes[node_id]["labels"].append(lab)
                    self._by_label[lab].add(node_id)
            return node_id
        self.g.add_node(node_id, id=node_id, labels=list(labels), **props)
        for lab in labels:
            self._by_label[lab].add(node_id)
        return node_id

    def add_edge(self, src: str, rel: str, dst: str, **props) -> None:
        if src not in self.g or dst not in self.g:
            raise KeyError(f"Edge {src}-[{rel}]->{dst}: endpoint missing")
        if not any(d["rel"] == rel for d in self.g.get_edge_data(src, dst, default={}).values()):
            self.g.add_edge(src, dst, key=rel, rel=rel, **props)

    def remove_edge(self, src: str, rel: str, dst: str) -> None:
        if self.g.has_edge(src, dst, key=rel):
            self.g.remove_edge(src, dst, key=rel)

    # ------------------------------------------------------------------- read
    def node(self, node_id: str) -> dict:
        return self.g.nodes[node_id]

    def has(self, node_id: str) -> bool:
        return node_id in self.g

    def nodes(self, label: str) -> list[str]:
        return sorted(self._by_label.get(label, ()))

    def out(self, node_id: str, rel: str | None = None) -> list[str]:
        return [v for _, v, d in self.g.out_edges(node_id, data=True) if rel is None or d["rel"] == rel]

    def inc(self, node_id: str, rel: str | None = None) -> list[str]:
        return [u for u, _, d in self.g.in_edges(node_id, data=True) if rel is None or d["rel"] == rel]

    def one(self, node_id: str, rel: str) -> str | None:
        targets = self.out(node_id, rel)
        return targets[0] if targets else None

    def has_edge(self, src: str, rel: str, dst: str) -> bool:
        return any(d["rel"] == rel for d in self.g.get_edge_data(src, dst, default={}).values())

    def edges(self) -> Iterator[tuple[str, str, str]]:
        for u, v, d in self.g.edges(data=True):
            yield u, d["rel"], v

    def stats(self) -> dict:
        counts = {lab: len(ids) for lab, ids in sorted(self._by_label.items())}
        rels = defaultdict(int)
        for _, r, _ in self.edges():
            rels[r] += 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "labels": counts, "relationships": dict(sorted(rels.items()))}

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "KnowledgeGraph":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------ neo4j
    NEO4J_CONSTRAINTS = [
        # Shared super-label so relationship import can use an index-backed MATCH
        "CREATE CONSTRAINT wno_id IF NOT EXISTS FOR (n:WNO) REQUIRE n.id IS UNIQUE",
        *(f"CREATE CONSTRAINT {lab.lower()}_id IF NOT EXISTS FOR (n:{lab}) REQUIRE n.id IS UNIQUE"
          for lab in ["WaterNetwork", "Junction", "Reservoir", "Tank", "Pipe", "Pump", "Valve",
                      "DemandPattern", "PumpCurve", "AssetRecord", "FailureEvent", "RiskProfile",
                      "EngineeringGuideline", "DecisionRecord", "MaintenanceAction"]),
        # Property-existence constraints require Neo4j Enterprise; skipped gracefully otherwise.
        "CREATE CONSTRAINT pipe_len IF NOT EXISTS FOR (p:Pipe) REQUIRE p.length IS NOT NULL",
        "CREATE CONSTRAINT pipe_dia IF NOT EXISTS FOR (p:Pipe) REQUIRE p.diameter IS NOT NULL",
    ]

    def to_neo4j(self, uri: str, user: str, password: str, database: str | None = None,
                 wipe: bool = False, batch: int = 500, keep_existing: bool = False) -> int:
        """Push the graph to Neo4j. By default the previous WaterOnt graph (nodes labelled WNO)
        is removed first, so that a changed graph (another seed, network, asset register or
        policy) replaces the old one instead of merging with it; MERGE alone would leave stale
        relationships, e.g. a decision record with two recommended actions. Other data in the
        database is untouched. wipe=True deletes everything; keep_existing=True merges.
        Returns the number of previous WaterOnt nodes removed."""
        from neo4j import GraphDatabase  # optional dependency

        def chunks(seq: list, n: int) -> Iterable[list]:
            for i in range(0, len(seq), n):
                yield seq[i:i + n]

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as s:
            removed = 0
            if wipe:
                s.run("MATCH (n) DETACH DELETE n")
            elif not keep_existing:
                while True:     # in batches, so large graphs do not exhaust transaction memory
                    rec = s.run("MATCH (n:WNO) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS c").single()
                    c = rec["c"] if rec is not None else 0
                    removed += c
                    if c == 0:
                        break
            for c in self.NEO4J_CONSTRAINTS:
                try:
                    s.run(c)
                except Exception as exc:  # Community edition rejects existence constraints
                    print(f"[neo4j] skipped constraint ({exc.__class__.__name__}): {c[:60]}...")
            by_labels: dict[tuple, list] = defaultdict(list)
            for nid, data in self.g.nodes(data=True):
                props = {k: v for k, v in data.items() if k != "labels" and _neo4j_ok(v)}
                by_labels[tuple(data["labels"])].append(props)
            for labels, rows in by_labels.items():
                lab = ":".join(["WNO", *(f"`{l}`" for l in labels)])
                for part in chunks(rows, batch):
                    s.run(f"UNWIND $rows AS r MERGE (n:{lab} {{id: r.id}}) SET n += r", rows=part)
            by_rel: dict[str, list] = defaultdict(list)
            for u, rel, v in self.edges():
                by_rel[rel].append({"u": u, "v": v})
            for rel, rows in by_rel.items():
                for part in chunks(rows, batch):
                    s.run(f"UNWIND $rows AS r MATCH (a:WNO {{id: r.u}}), (b:WNO {{id: r.v}}) "
                          f"MERGE (a)-[:{rel}]->(b)", rows=part)
        driver.close()
        return removed


def _neo4j_ok(v) -> bool:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return True
    return isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)
