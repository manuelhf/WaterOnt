import json
import os
import types

import pytest
from rdflib import Graph

from wateront.config import Config
from wateront.graphrag import GraphRAG, plan_rules
from wateront.rdf_export import run_cq, to_rdf

ROOT = os.path.dirname(os.path.dirname(__file__))


# ---------------------------------------------------------------- Physical layer
def test_every_link_has_exactly_one_start_and_end(built):
    kg = built.kg
    for link in kg.nodes("Link"):
        assert len(kg.out(link, "STARTS_AT")) == 1
        assert len(kg.out(link, "ENDS_AT")) == 1


def test_head_pumps_have_exactly_one_pump_curve(built):
    kg = built.kg
    for pump in kg.nodes("Pump"):
        if kg.node(pump)["pumpType"] == "HeadPump":
            curves = [c for c in kg.out(pump, "HAS_CURVE") if "PumpCurve" in kg.node(c)["labels"]]
            assert len(curves) == 1


def test_counts_match_epanet(built):
    wn, kg = built.wn, built.kg
    assert len(kg.nodes("Pipe")) == wn.num_pipes
    assert len(kg.nodes("Node")) == wn.num_nodes
    assert len(kg.out("NET:Net3", "CONTAINS_LINK")) == wn.num_links


def test_controlled_link_is_governed(built):
    # Net3: pipe 330 is closed/opened by tank-level controls
    assert built.kg.out("P:330", "GOVERNED_BY")


# ------------------------------------------------------------------ upper layers
def test_every_pipe_has_full_stack(built):
    kg = built.kg
    for p in kg.nodes("Pipe"):
        rp = kg.one(p, "HAS_RISK_PROFILE")
        assert kg.one(p, "HAS_ASSET_RECORD") and rp
        for rel in ("HAS_FAILURE_PROBABILITY", "HAS_HYDRAULIC_CRITICALITY",
                    "HAS_SERVICE_CRITICALITY", "HAS_OVERALL_RISK"):
            assert len(kg.out(rp, rel)) == 1
        dr = kg.inc(p, "CONCERNS")
        assert len(dr) == 1 and kg.out(dr[0], "SUPPORTED_BY")


def test_replacement_implies_policy_conditions(built):
    for d in built.decisions:
        if d.action == "Replacement":
            assert d.facts["overall"] == "Intolerable" and d.facts["age"] > d.facts["design_life"]
            assert "G-REPL-01" in d.guidelines


def test_reproducible(built):
    from wateront.pipeline import build_knowledge_graph
    again = build_knowledge_graph("Net3", Config(), verbose=False)
    assert [d.action for d in again.decisions] == [d.action for d in built.decisions]


# ------------------------------------------------------------------ RDF / SPARQL
def test_ontology_parses():
    g = Graph().parse(os.path.join(ROOT, "ontology", "wateront.ttl"))
    assert len(g) > 200


def test_competency_questions(built):
    g = to_rdf(built.kg)
    assert {r[0] for r in run_cq(g, "CQ1", P="P_101")} == {"J_10", "J_101"}
    assert run_cq(g, "CQ2", J="J_10")
    assert run_cq(g, "CQ3", P="P_101")
    assert run_cq(g, "CQ5", K="PU_10")
    assert run_cq(g, "CQ8")[0][0] == "H-W"
    repl = run_cq(g, "CQ-Decision")
    assert repl and all(gid.startswith("G-") for _, gid in repl)


# ---------------------------------------------------------------------- GraphRAG
def test_planner():
    p = plan_rules("Which hydraulically critical pipes also have a high failure probability? top 5")
    assert p.bands == {"hydraulic": ["High", "Critical"], "pof": ["High", "VeryHigh"]} and p.top_k == 5
    assert plan_rules("Which pipes should be replaced?").actions == ["Replacement"]


def test_answers_are_traceable_and_grounded(built):
    rag = GraphRAG(built.kg, use_llm=False)
    ans = rag.ask("Which pipes should be replaced and which standard authorises it?")
    assert ans.recommendations and not ans.unsupported_citations
    for e in ans.recommendations:
        assert e.action == "Replacement"
        assert rag.retriever.verify_path(e.path)


def test_verifier_catches_hallucinated_citation(built):
    rag = GraphRAG(built.kg, use_llm=False)
    ans = rag.ask("What is recommended for pipe 101?")
    ans.text += " Also see [G-FAKE-99] and [P:999999]."
    assert set(rag.verify(ans)) == {"G-FAKE-99", "P:999999"}


class _FakeClient:
    """Stands in for anthropic.Anthropic() so LLM code paths run offline."""
    def __init__(self, reply):
        self.messages = types.SimpleNamespace(create=lambda **kw: types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=reply)]))


def test_llm_paths_with_mock(built, monkeypatch):
    import wateront.graphrag as gr
    plan_json = json.dumps({"actions": ["Replacement"], "top_k": 3})
    monkeypatch.setattr(gr, "_anthropic_client", lambda: _FakeClient(plan_json))
    plan = gr.plan_llm("renewal candidates?", Config())
    assert plan.actions == ["Replacement"] and plan.top_k == 3

    rag = GraphRAG(built.kg, use_llm=True)
    first = rag.retrieve(plan)[0].pipe_uid
    monkeypatch.setattr(gr, "_anthropic_client",
                        lambda: _FakeClient(f"Replace [{first}] per [G-REPL-01]; see [G-NOPE]."))
    ans = rag.ask("Which pipes should be replaced?", plan=plan)
    assert ans.generator == "llm" and ans.unsupported_citations == ["G-NOPE"]


# -------------------------------------------------------------------- evaluation
def test_evaluation_runs(built):
    import pandas as pd
    from wateront.asset import simulate_future_failures
    from wateront.baselines import FlatRAGBaseline, MLBaseline
    from wateront.evaluation import run_evaluation
    cfg = Config()
    labels = simulate_future_failures(built.assets, built.wn, cfg)
    prev = pd.Series([d.action for d in built.decisions]).value_counts().to_dict()
    systems = {"graphrag": GraphRAG(built.kg, cfg, use_llm=False),
               "flat": FlatRAGBaseline(built.kg, cfg), "ml": MLBaseline(built.kg, labels, prev, cfg)}
    per_query, summary = run_evaluation(built.kg, systems, cfg)
    assert summary.loc["graphrag", "traceability"] == 1.0
    assert summary.loc["graphrag", "policy_recall"] >= summary.loc["flat", "policy_recall"]
    assert summary.loc["ml", "policy_recall"] == 0.0


# ------------------------------------------------------------------------ L-TOWN
LTOWN = os.path.join(ROOT, "data", "networks", "L-TOWN.inp")
needs_ltown = pytest.mark.skipif(not os.path.exists(LTOWN), reason="L-TOWN.inp not present")


@pytest.fixture(scope="session")
def ltown():
    from wateront.pipeline import build_knowledge_graph
    return build_knowledge_graph(LTOWN, verbose=False)


@needs_ltown
def test_ltown_inventory(ltown):
    kg = ltown.kg
    assert len(kg.nodes("Pipe")) == 905 and len(kg.nodes("Junction")) == 782
    assert len(kg.nodes("PressureReducingValve")) == 3
    assert len(kg.out("PU:PUMP_1", "GOVERNED_BY")) == 2          # tank-level controls


@needs_ltown
def test_ltown_multi_category_demand_preserved(ltown):
    n1 = ltown.kg.node("J:n1")                                    # industrial-only junction
    assert n1["demand_industrial"] > 0 and n1["baseDemand"] == pytest.approx(n1["demand_industrial"])
    assert "FAC:n1" in ltown.kg.out("J:n1", "SERVES_FACILITY")


@needs_ltown
def test_ltown_sensors_from_inp_comments(ltown):
    kg = ltown.kg
    assert (len(kg.nodes("PressureSensor")), len(kg.nodes("AMRMeter")),
            len(kg.nodes("FlowSensor")), len(kg.nodes("LevelSensor"))) == (33, 82, 3, 1)
    assert kg.out("P:p227", "OBSERVED_BY") and kg.out("T:T1", "OBSERVED_BY")


@needs_ltown
def test_ltown_supply_zones(ltown):
    kg = ltown.kg
    zones = {kg.node(z)["name"]: kg.node(z) for z in kg.nodes("PressureZone") if kg.node(z)["isSupplyZone"]}
    assert set(zones) == {"PRV-1+PRV-2", "PRV-3", "PUMP_1+T1"}
    assert zones["PUMP_1+T1"]["containsStorage"]
    assert "Zone:PRV-3" in kg.out("V:PRV-3", "FEEDS_ZONE")


@needs_ltown
def test_ltown_per_km_banding_gives_nontrivial_risk(ltown):
    bands = ltown.risk.pof_band.value_counts()
    assert {"High", "VeryHigh"} <= set(bands.index)              # per-segment banding gave none
    assert (ltown.risk.overall_risk == "Intolerable").any()


@needs_ltown
def test_ltown_zone_and_monitoring_queries(ltown):
    rag = GraphRAG(ltown.kg, use_llm=False)
    a = rag.ask("Which pipes downstream of PRV-3 need intervention?")
    assert a.recommendations and all(e.facts["zone"] == "PRV-3" for e in a.recommendations)
    b = rag.ask("Which unmonitored hydraulically critical mains have intolerable risk?")
    assert b.recommendations and all(e.facts["monitored"] is False for e in b.recommendations)
    assert all(rag.retriever.verify_path(e.path) for e in a.recommendations + b.recommendations)


def test_failure_loader_and_holdout_capture(tmp_path):
    from wateront.asset import load_failure_events
    from wateront.evaluation import holdout_capture
    f = tmp_path / "leaks.csv"
    f.write_text("pipe,start,mode\np1,2019-03-01,Leak\np1,2019-08-01,\np7,2018-01-01,Burst\n")
    ev = load_failure_events(str(f))
    assert [y for y, _ in ev["p1"]] == [2019, 2019] and ev["p7"] == [(2018, "Burst")]
    cap = holdout_capture({"s": ["a", "b", "c"]}, {"b"}, {"a": 10, "b": 10, "c": 80}, budgets=(0.2,))
    assert cap.captured[0] == 1 and cap.lift[0] == 5.0


def test_external_likelihood_identity_and_provenance(built):
    """Feeding the Weibull probabilities back as 'external' scores must reproduce every
    decision; feeding other scores keeps band sizes and records the source."""
    from wateront.pipeline import build_knowledge_graph
    w = built.risk.pof_km.to_dict()
    same = build_knowledge_graph("Net3", likelihood_scores=w, likelihood_source="weibull copy", verbose=False)
    assert [d.action for d in same.decisions] == [d.action for d in built.decisions]
    rev = {p: -v for p, v in w.items()}                       # reversed ranking
    other = build_knowledge_graph("Net3", likelihood_scores=rev, likelihood_source="reversed", verbose=False)
    assert other.risk.pof_band.value_counts().to_dict() == built.risk.pof_band.value_counts().to_dict()
    assert other.kg.node("RP:101")["likelihoodSource"] == "reversed"
    assert [d.action for d in other.decisions] != [d.action for d in built.decisions]


def test_flat_rag_llm_path_with_mock(built, monkeypatch):
    """The 'no KG, with LLM' cell: the model sees asset records and the full policy,
    returns JSON decisions, and citations outside its context are flagged."""
    import wateront.graphrag as gr
    from wateront.baselines import FlatRAGBaseline
    flat = FlatRAGBaseline(built.kg, use_llm=True)
    seen = {}
    def fake_client():
        def create(**kw):
            seen["prompt"] = kw["messages"][0]["content"]
            reply = json.dumps([{"pipe": "P:101", "action": "Replacement", "guidelines": ["G-REPL-01", "G-FAKE-1"]}])
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=reply)])
        return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    monkeypatch.setattr(gr, "_anthropic_client", fake_client)
    ans = flat.ask("Which pipes should be replaced?")
    assert all(f"[{g['id']}]" in seen["prompt"] for g in flat.guidelines)        # full policy in context
    assert [(e.pipe_uid, e.action) for e in ans.recommendations] == [("P:101", "Replacement")]
    assert "G-FAKE-1" in ans.unsupported_citations


def test_table_rules_baseline_matches_graph_answers(built):
    """Same attributes + same rules + same planner in a table gives the same answers:
    the policy, not the graph, determines fidelity. The table has no evidence paths."""
    from wateront.baselines import TableRulesBaseline
    from wateront.evaluation import INTENTS
    rag, tab = GraphRAG(built.kg, use_llm=False), TableRulesBaseline(built.kg)
    for it in INTENTS:
        for q in it.paraphrases:
            a, b = rag.ask(q), tab.ask(q)
            assert [(e.pipe_uid, e.action, e.guidelines) for e in a.recommendations] == \
                   [(e.pipe_uid, e.action, e.guidelines) for e in b.recommendations], q
            assert all(not e.path for e in b.recommendations)


def _failing_client():
    def create(**kw):
        raise RuntimeError("401 invalid x-api-key")
    return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))


def test_strict_mode_never_hides_llm_failures(built, monkeypatch):
    """Experiments use strict=True: an LLM failure must raise, never fall back silently."""
    import wateront.graphrag as gr
    monkeypatch.setattr(gr, "_anthropic_client", _failing_client)
    with pytest.raises(RuntimeError, match="401"):
        GraphRAG(built.kg, use_llm=True, strict=True).ask("Which pipes should be replaced?")
    relaxed = GraphRAG(built.kg, use_llm=True).ask("Which pipes should be replaced?")   # app keeps working
    assert relaxed.generator == "template"


def test_experiments_stop_on_invalid_key(monkeypatch):
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "paper"))
    import experiments
    import wateront.graphrag as gr
    monkeypatch.setattr(gr, "_anthropic_client", _failing_client)
    with pytest.raises(SystemExit, match="LLM check failed"):
        experiments.check_api(Config())


def test_flat_llm_reads_json_inside_prose(built, monkeypatch):
    import wateront.graphrag as gr
    from wateront.baselines import FlatRAGBaseline
    reply = 'Here are my decisions:\n[{"pipe": "P:101", "action": "Replacement", "guidelines": ["G-REPL-01"]}]\nThanks.'
    monkeypatch.setattr(gr, "_anthropic_client", lambda: _FakeClient(reply))
    ans = FlatRAGBaseline(built.kg, use_llm=True).ask("Which pipes should be replaced?")
    assert [(e.pipe_uid, e.action) for e in ans.recommendations] == [("P:101", "Replacement")]


def test_flat_llm_recovers_entries_from_truncated_answer(built, monkeypatch):
    """An answer cut off by the output limit keeps every complete entry (seen on Colab)."""
    import wateront.graphrag as gr
    from wateront.baselines import FlatRAGBaseline
    reply = ('```json\n[\n{"pipe":"P:101","action":"Inspection","guidelines":["G-MON-02"]},\n'
             '{"pipe":"P:105","action":"Replacement","guidelines":["G-REPL-01"]},\n{"pipe":"P:1')
    monkeypatch.setattr(gr, "_anthropic_client", lambda: _FakeClient(reply))
    ans = FlatRAGBaseline(built.kg, use_llm=True).ask("Which pipes need an inspection?")
    assert [(e.pipe_uid, e.action) for e in ans.recommendations] == [("P:101", "Inspection"), ("P:105", "Replacement")]


class _SequenceClient:
    """Returns the given replies in turn (one per API call)."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.messages = types.SimpleNamespace(create=self._create)
    def _create(self, **kw):
        self.calls += 1
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=self.replies.pop(0))])


def test_unreadable_plan_is_retried_then_scored_as_failure(built, monkeypatch):
    import wateront.graphrag as gr
    good = json.dumps({"actions": ["Replacement"]})
    # first answer truncated, retry succeeds, then the generator answers
    seq = _SequenceClient(['{"actions": ["Repla', good, "Replace [P:101] per [G-REPL-01]."])
    monkeypatch.setattr(gr, "_anthropic_client", lambda: seq)
    ans = GraphRAG(built.kg, use_llm=True, strict=True).ask("Which pipes should be replaced?")
    assert ans.recommendations and not getattr(ans, "planning_failed", False) and seq.calls == 3
    # both planner answers unreadable: scored empty answer, no exception, flagged
    seq2 = _SequenceClient(['{"actions": ["Repla', 'not json at all'])
    monkeypatch.setattr(gr, "_anthropic_client", lambda: seq2)
    ans2 = GraphRAG(built.kg, use_llm=True, strict=True).ask("Which pipes should be replaced?")
    assert ans2.recommendations == [] and ans2.planning_failed and seq2.calls == 2


def test_api_errors_still_abort_in_strict_mode(built, monkeypatch):
    import wateront.graphrag as gr
    monkeypatch.setattr(gr, "_anthropic_client", _failing_client)
    with pytest.raises(RuntimeError, match="401"):
        GraphRAG(built.kg, use_llm=True, strict=True).ask("Which pipes should be replaced?")
