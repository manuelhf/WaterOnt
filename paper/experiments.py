"""Experiments behind Sections 5 and 6 of the paper.

    python paper/experiments.py      # writes paper/results/*.csv and meta.json

Comparison design (2 x 2): with or without the knowledge graph, and with or without an LLM.

                      without LLM                        with LLM (given context)
    without KG        table: same attributes and rules   flat-text RAG (records and full policy)
    with KG           WaterOnt, deterministic            WaterOnt GraphRAG (LLM plans and explains)

The two LLM cells run only when ANTHROPIC_API_KEY is set; everything else is deterministic.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wateront.baselines import FlatRAGBaseline, TableRulesBaseline  # noqa: E402  (flat RAG: LLM cell)
from wateront.config import Config  # noqa: E402
from wateront.decision import apply_policy  # noqa: E402
from wateront.evaluation import default_intents, reference, run_evaluation, score  # noqa: E402
from wateront.graphrag import GraphRAG, QueryPlan  # noqa: E402
from wateront.pipeline import build_knowledge_graph  # noqa: E402
from wateront.risk import compute_risk_table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LTOWN = os.path.join(os.path.dirname(HERE), "data", "networks", "L-TOWN.inp")
OUT = os.path.join(HERE, "results")
KG_NO_LLM, TABLE_NO_LLM = "KG, no LLM", "No KG, no LLM (table and rules)"
KG_LLM, FLAT_LLM = "KG, LLM", "No KG, LLM (flat-text RAG)"


def systems(res, cfg, llm: bool):
    out = {TABLE_NO_LLM: TableRulesBaseline(res.kg, cfg), KG_NO_LLM: GraphRAG(res.kg, cfg, use_llm=False)}
    if llm:
        out[KG_LLM] = GraphRAG(res.kg, cfg, use_llm=True, strict=True)
        out[FLAT_LLM] = FlatRAGBaseline(res.kg, cfg, use_llm=True)
    return out


# Correct query plan for each intent: isolates planning errors from retrieval and reasoning.
GOLD = {
    "replacement": QueryPlan(actions=["Replacement"]),
    "rehabilitation": QueryPlan(actions=["Rehabilitation"]),
    "inspection": QueryPlan(actions=["Inspection"]),
    "critical_high_pof": QueryPlan(bands={"hydraulic": ["High", "Critical"], "pof": ["High", "VeryHigh"]}),
    "critical_customers": QueryPlan(critical_facilities=True, exclude_monitor=True),
    "multimodal_pressure_age_policy": QueryPlan(material=["CI"], low_pressure=True, exclude_monitor=True),
    "monitoring_gap": QueryPlan(unmonitored=True, bands={"hydraulic": ["High", "Critical"], "overall": ["Intolerable"]}),
    "zone_prv3": QueryPlan(zones=["PRV-3"], exclude_monitor=True),
}


def gold_plan(res, cfg):
    rag = GraphRAG(res.kg, cfg, use_llm=False)
    rows = []
    for it in default_intents(res.kg):
        ans = rag.ask(it.paraphrases[0], plan=GOLD[it.key])
        sc = score(res.kg, ans, reference(res.kg, it, cfg))
        rows.append(dict(intent=it.key, decision_f1=sc["decision_f1"], policy_recall=sc["policy_recall"],
                         traceability=sc["traceability"]))
    return pd.DataFrame(rows)


def decisions_from_tables(assets, risk, cfg):
    a = assets.set_index("pipe")
    out = {}
    for pipe, r in risk.iterrows():
        x = a.loc[pipe]
        f = dict(pipe=pipe, overall=r.overall_risk, age=cfg.reference_year - int(x.install_year),
                 design_life=cfg.design_life[x.material], condition_grade=int(x.condition_grade),
                 last_inspection_year=int(x.last_inspection_year), facilities=r.facilities,
                 monitored=bool(r.monitored), hydraulic=r.hydraulic_band, priority=float(r.priority))
        out[pipe] = apply_policy(f, cfg)
    return out


def sensitivity(res, base_cfg):
    """One-at-a-time perturbations; asset histories fixed, Risk and Decision Layers recomputed."""
    from wateront.physical import pipes_near_sensor
    from wateront.risk import choose_critical_facilities
    wn, assets = res.wn, res.assets
    fac = choose_critical_facilities(wn, base_cfg)
    mon = pipes_near_sensor(wn, res.kg, "PressureSensor", hops=2)

    def run(cfg):
        risk, _ = compute_risk_table(wn, assets, cfg, fac, mon)
        return decisions_from_tables(assets, risk, cfg)

    base = run(base_cfg)
    kg_actions = {d.facts["pipe"]: d.action for d in res.decisions}
    assert all(base[p].action == kg_actions[p] for p in base), "fast path must match the graph"
    rows = []

    def record(group, factor, cfg):
        d = run(cfg)
        rows.append(dict(group=group, factor=factor,
                         share_changed=sum(d[p].action != base[p].action for p in base) / len(base),
                         replacements=sum(v.action == "Replacement" for v in d.values())))
    for f in (0.8, 0.9, 1.1, 1.2):
        record("Design life", f, replace(base_cfg, design_life={k: v * f for k, v in base_cfg.design_life.items()}))
        record("Inspection intervals", f,
               replace(base_cfg, inspection_interval={k: v * f for k, v in base_cfg.inspection_interval.items()}))
        record("Weibull scale", f, replace(base_cfg, weibull={k: (b, e * f) for k, (b, e) in base_cfg.weibull.items()}))
        record("Hazard scale", f, replace(base_cfg, hazard_scale=base_cfg.hazard_scale * f))
        record("PoF band limits", f, replace(base_cfg, pof_bands=tuple((t * f if t < 1 else t, n) for t, n in base_cfg.pof_bands)))
    return pd.DataFrame(rows), sum(v.action == "Replacement" for v in base.values())


def banding_data(net, cfg):
    res = build_knowledge_graph(net, cfg, verbose=False)
    r = res.risk
    return pd.DataFrame({"network": res.network_id, "pipe": r.index, "pof_segment": r.pof.values,
                         "pof_km": r.pof_km.values, "length_m": [res.wn.get_link(p).length for p in r.index]})


def check_api(cfg):
    """Fail fast with a clear message if the key or model does not work, before any results
    are computed, so an invalid key can never produce mislabelled LLM results."""
    from wateront.graphrag import _anthropic_client
    try:
        _anthropic_client().messages.create(model=cfg.llm_model, max_tokens=5,
                                            messages=[{"role": "user", "content": "Reply with OK"}])
    except Exception as exc:
        raise SystemExit(f"LLM check failed, nothing was run: {exc}")
    print(f"API check passed (model {cfg.llm_model})")


def main():
    os.makedirs(OUT, exist_ok=True)
    llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    cfg = Config(seed=42)
    if llm:
        check_api(cfg)
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "seed": 42, "llm_cells_run": llm}

    print(f"[1] 2x2 comparison on L-TOWN, seed 42 (LLM cells: {'run' if llm else 'skipped, no API key'})")
    res = build_knowledge_graph(LTOWN, cfg, verbose=False)
    per_query, summary = run_evaluation(res.kg, systems(res, cfg, llm), cfg)
    per_query.to_csv(os.path.join(OUT, "comparison_per_query.csv"), index=False)
    if llm:
        fails = per_query[per_query.planning_failed]
        meta["llm_planning_failures"] = int(len(fails))
        print(f"LLM planning failures (scored as empty answers): {len(fails)} of "
              f"{int((per_query.system == KG_LLM).sum())} questions")
    summary.to_csv(os.path.join(OUT, "comparison_summary.csv"))
    meta["ref_sizes"] = per_query.groupby("intent").n_ref.first().to_dict()

    print("[2] correct-plan analysis")
    gold_plan(res, cfg).to_csv(os.path.join(OUT, "gold_plan.csv"), index=False)

    print("[3] ten synthetic utilities (no-LLM cells)")
    rows = []
    for seed in range(42, 52):
        c = Config(seed=seed)
        r = build_knowledge_graph(LTOWN, c, verbose=False)
        _, s = run_evaluation(r.kg, systems(r, c, llm=False), c)
        rows += [dict(seed=seed, system=n, **row.to_dict()) for n, row in s.iterrows()]
        acts = pd.Series([d.action for d in r.decisions]).value_counts()
        rows.append(dict(seed=seed, system="_actions", **{k: int(v) for k, v in acts.items()}))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "seeds.csv"), index=False)

    print("[4] Net3 and Net6 (no-LLM cells)")
    for net in ("Net3", "Net6"):
        r = build_knowledge_graph(net, cfg, verbose=False)
        _, s = run_evaluation(r.kg, systems(r, cfg, llm=False), cfg)
        s.to_csv(os.path.join(OUT, f"{net.lower()}_summary.csv"))

    print("[5] banding basis")
    pd.concat([banding_data("Net3", cfg), banding_data(LTOWN, cfg)]).to_csv(os.path.join(OUT, "banding.csv"), index=False)

    print("[6] sensitivity")
    sens, base_repl = sensitivity(res, cfg)
    sens.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
    meta["base_replacements"] = int(base_repl)

    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=float)
    print("done:", OUT)


if __name__ == "__main__":
    main()
