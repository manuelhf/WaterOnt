"""RDF/OWL export so the Appendix A.6 competency questions run as SPARQL.

Instance IRIs follow the paper's sketches: an EPANET element named `P` becomes
wno:P. Because EPANET lets a junction and a pipe share a name, the type prefix is
kept when needed (wno:J_10 vs wno:P_10); `iri_mode="paper"` drops prefixes and
is only safe for networks without such collisions.
"""
from __future__ import annotations

import re

from rdflib import RDF, XSD, Graph, Literal, Namespace

from .kg import REL_TO_PROPERTY, KnowledgeGraph

WNO = Namespace("http://www.ncl.ac.uk/waternetwork-ontology#")
SKIP = {"id", "labels", "multipliers", "xValues", "yValues", "controlledBy"}
STATUS_PROPS = {"value": "value", "reason": "reason"}


def _local(node_id: str, mode: str) -> str:
    kind, _, name = node_id.partition(":")
    base = name if mode == "paper" and name else node_id
    return re.sub(r"[^A-Za-z0-9_\-]", "_", base)


def to_rdf(kg: KnowledgeGraph, iri_mode: str = "prefixed") -> Graph:
    g = Graph()
    g.bind("wno", WNO)
    for nid, data in kg.g.nodes(data=True):
        s = WNO[_local(nid, iri_mode)]
        for lab in data["labels"]:
            if re.fullmatch(r"[A-Za-z]+", lab):
                g.add((s, RDF.type, WNO[lab]))
        for k, v in data.items():
            if k in SKIP or v is None or isinstance(v, (list, dict)):
                continue
            dt = XSD.boolean if isinstance(v, bool) else XSD.integer if isinstance(v, int) \
                else XSD.float if isinstance(v, float) else None
            g.add((s, WNO[k], Literal(v, datatype=dt) if dt else Literal(str(v))))
    for u, rel, v in kg.edges():
        g.add((WNO[_local(u, iri_mode)], WNO[REL_TO_PROPERTY.get(rel, rel.lower())],
               WNO[_local(v, iri_mode)]))
    return g


COMPETENCY_QUESTIONS = {
    "CQ1": "SELECT ?n WHERE {{ wno:{P} wno:startsAt|wno:endsAt ?n }}",
    "CQ2": "SELECT ?p WHERE {{ ?p wno:startsAt|wno:endsAt wno:{J} ; a wno:Pipe }}",
    "CQ3": "SELECT ?d ?l WHERE {{ wno:{P} wno:diameter ?d ; wno:length ?l }}",
    "CQ4": "SELECT ?v WHERE {{ ?v a wno:PressureReducingValve }}",
    "CQ5": "SELECT ?c WHERE {{ wno:{K} wno:hasCurve ?c }}",
    "CQ6": "SELECT ?j WHERE {{ ?j wno:followsPattern wno:{T} }}",
    "CQ7": 'SELECT ?l ?r WHERE {{ ?l wno:hasStatus ?s . ?s wno:value "Closed" ; wno:reason ?r }}',
    "CQ8": "SELECT ?f WHERE {{ ?n a wno:WaterNetwork ; wno:headlossFormula ?f }}",
    # Upper-layer competency questions from Table 1
    "CQ-Asset": """SELECT ?p ?y ?n WHERE {{ ?p wno:hasAssetRecord ?a . ?a wno:material "CI" ;
                   wno:installationYear ?y ; wno:failureCount ?n . FILTER(?y < 1970 && ?n > 2) }}""",
    "CQ-Risk": """SELECT ?p WHERE {{ ?p wno:hasRiskProfile ?rp . ?rp wno:hasHydraulicCriticality wno:HC_Critical ;
                  wno:hasServiceCriticality wno:SC_Critical ; wno:hasOverallRisk wno:OR_Major }}""",
    "CQ-Decision": """SELECT ?p ?g WHERE {{ ?d wno:concerns ?p ; wno:recommends wno:MA_Replacement ;
                      wno:supportedBy ?g }}""",
}


def run_cq(g: Graph, cq: str, **params) -> list[tuple]:
    q = "PREFIX wno: <http://www.ncl.ac.uk/waternetwork-ontology#>\n" + COMPETENCY_QUESTIONS[cq].format(**params)
    return [tuple(str(x).split("#")[-1] for x in row) for row in g.query(q)]
