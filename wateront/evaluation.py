"""Evaluation (Section 5).

Metrics (per query, then averaged)
* Decision F1        F1 over (pipe, action) pairs against the reference top-k.
* Policy Recall      mean over reference pipes of |cited clauses ∩ authorising clauses|
                     / |authorising clauses| (0 when the pipe is missed).
* Traceability       share of returned recommendations whose every cited clause is
                     reachable in the KG via DecisionRecord-[:SUPPORTED_BY]->Guideline
                     for that pipe (no citation = not traceable).
* Evidence consistency  mean pairwise Jaccard of evidence-id sets (pipes + clauses)
                     across paraphrases of the same intent.
* Unsupported citations  ids cited in generated text but absent from retrieved
                     evidence (hallucination check; meaningful when an LLM generates).

CAVEAT (state this in the paper): reference decisions come from applying the
same encoded policy to the recorded asset data. Decision F1 therefore measures
*faithful, query-correct policy application*, not real-world decision quality.
For the latter, replace `reference()` with expert-labelled decisions via
`load_expert_reference` (CSV: intent,pipe,action,guidelines).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .config import Config
from .decision import ACTION_RANK, apply_policy, pipe_facts
from .graphrag import Answer
from .kg import KnowledgeGraph


@dataclass
class Intent:
    key: str
    spec: Callable[[dict, Config], bool]
    paraphrases: list[str]
    top_k: int = 25


INTENTS = [
    Intent("replacement", lambda f, d, c: d.action == "Replacement", [
        "Which pipes should be replaced, and which standard authorises each replacement?",
        "List the mains recommended for renewal together with the supporting policy.",
        "Which assets have reached the end of their life and need to be swapped out?"]),
    Intent("rehabilitation", lambda f, d, c: d.action == "Rehabilitation", [
        "Which mains are candidates for rehabilitation?",
        "Which deteriorated pipes should be relined rather than replaced?",
        "Where should we carry out structural rehabilitation this period?"]),
    Intent("assessment", lambda f, d, c: d.action == "Assessment", [
        "Which pipes need a condition assessment?",
        "Which mains are overdue for a condition survey?",
        "Where should condition assessment crews be sent next?"]),
    Intent("critical_high_pof", lambda f, d, c: f["hydraulic"] in ("High", "Critical")
           and f["pof_band"] in ("High", "VeryHigh"), [
        "Which hydraulically critical pipes also have a high failure probability?",
        "Show backbone mains that are likely to fail.",
        "Find bottleneck pipes with a high or very high probability of failure."]),
    Intent("critical_customers", lambda f, d, c: bool(f["facilities"]) and d.action != "Monitor", [
        "Which mains serving hospitals, schools or care homes need intervention?",
        "Which pipes supplying critical customers should be prioritised?",
        "Do any pipes feeding schools or hospitals require work?"]),
    Intent("multimodal_pressure_age_policy", lambda f, d, c: f["material"] == "CI"
           and f["min_end_pressure"] < c.min_service_pressure_m + c.pressure_margin_m
           and d.action != "Monitor", [
        "Which cast iron mains at low service pressure need intervention?",
        "Prioritise cast iron pipes with a low pressure margin.",
        "Which old cast iron pipes with a pressure deficit require attention?"]),
]


# Intents that need sensor metadata / multiple supply zones (L-TOWN style models)
INSTRUMENTED_INTENTS = [
    Intent("monitoring_gap", lambda f, d, c: f.get("monitored") is False
           and f["hydraulic"] in ("High", "Critical") and f["overall"] == "Major", [
        "Which unmonitored hydraulically critical mains have major risk?",
        "Show critical pipes with major risk and no pressure sensor nearby.",
        "Where are the monitoring blind spots on high-risk backbone mains?"]),
    Intent("zone_prv3", lambda f, d, c: f.get("zone") == "PRV-3" and d.action != "Monitor", [
        "Which pipes downstream of PRV-3 need intervention?",
        "Prioritise work in the zone supplied through PRV-3.",
        "What maintenance is needed in the PRV-3 pressure zone?"]),
]


def default_intents(kg: KnowledgeGraph) -> list[Intent]:
    out = list(INTENTS)
    if kg.nodes("PressureSensor"):
        out.append(INSTRUMENTED_INTENTS[0])
    if any(kg.node(z)["name"] == "PRV-3" for z in kg.nodes("PressureZone")):
        out.append(INSTRUMENTED_INTENTS[1])
    return out


def reference(kg: KnowledgeGraph, intent: Intent, cfg: Config,
              capped: bool = True) -> list[tuple[str, str, list[str]]]:
    """Pipes satisfying the intent, with their policy action and clauses, ranked by action
    urgency and priority. capped=True keeps the top_k (the ranked task scored by decision F1);
    capped=False keeps every qualifying pipe (used by policy precision)."""
    rows = []
    for p in kg.nodes("Pipe"):
        f = pipe_facts(kg, p)
        d = apply_policy(f, cfg)
        if intent.spec(f, d, cfg):
            rows.append((p, d.action, d.guidelines, f["priority"]))
    rows.sort(key=lambda r: (ACTION_RANK[r[1]], -r[3], r[0]))
    return [(p, a, g) for p, a, g, _ in (rows[: intent.top_k] if capped else rows)]


def load_expert_reference(path: str) -> dict[str, list[tuple[str, str, list[str]]]]:
    df = pd.read_csv(path, dtype=str)
    out: dict[str, list] = {}
    for r in df.itertuples(index=False):
        out.setdefault(r.intent, []).append((f"P:{r.pipe}", r.action, str(r.guidelines).split(";")))
    return out


def _traceable(kg: KnowledgeGraph, pipe_uid: str, cited: list[str]) -> bool:
    if not cited:
        return False
    drs = kg.inc(pipe_uid, "CONCERNS")
    return bool(drs) and all(kg.has_edge(drs[0], "SUPPORTED_BY", g) for g in cited)


def score(kg: KnowledgeGraph, ans: Answer, ref: list[tuple[str, str, list[str]]],
          full_ref: list[tuple[str, str, list[str]]] | None = None) -> dict:
    """Decision F1 scores the ranked task against the capped reference. Policy precision is the
    share of returned (pipe, action) pairs that are correct under the policy for the question,
    with no cap, so correct answers outside the top k are not penalised."""
    pred = {(e.pipe_uid, e.action) for e in ans.recommendations}
    gold = {(p, a) for p, a, _ in ref}
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    rec = tp / len(gold) if gold else (1.0 if not pred else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    cited = {e.pipe_uid: set(e.guidelines) for e in ans.recommendations}
    pr = [len(cited.get(p, set()) & set(g)) / len(g) for p, _, g in ref] if ref else []
    tr = [_traceable(kg, e.pipe_uid, e.guidelines) for e in ans.recommendations]
    return dict(n_ref=len(gold), n_pred=len(pred), decision_precision=prec, decision_recall=rec,
                decision_f1=f1, policy_recall=sum(pr) / len(pr) if pr else float("nan"),
                traceability=sum(tr) / len(tr) if tr else float("nan"),
                unsupported_citations=len(ans.unsupported_citations),
                policy_precision=(len(pred & {(p, a) for p, a, _ in full_ref}) / len(pred)
                                  if full_ref is not None and pred else float("nan")))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def run_evaluation(kg: KnowledgeGraph, systems: dict, cfg: Config | None = None,
                   intents: list[Intent] | None = None,
                   expert: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or Config()
    rows, answers = [], {}
    skipped = []
    for it in intents or default_intents(kg):
        ref = expert[it.key] if expert and it.key in expert else reference(kg, it, cfg)
        full_ref = None if expert and it.key in expert else reference(kg, it, cfg, capped=False)
        if not ref:   # an empty reference makes F1 meaningless; report instead of scoring
            skipped.append(it.key)
            continue
        for name, sysm in systems.items():
            for i, q in enumerate(it.paraphrases):
                ans = sysm.ask(q)
                answers[(name, it.key, i)] = ans
                rows.append(dict(system=name, intent=it.key, paraphrase=i, query=q, **score(kg, ans, ref, full_ref),
                                 planning_failed=bool(getattr(ans, "planning_failed", False)),
                                 generator=ans.generator,
                                 predictions=json.dumps([[e.pipe_uid, e.action, e.guidelines]
                                                         for e in ans.recommendations])))
    if skipped:
        print(f"[eval] skipped intents with empty reference on this network: {skipped}")
    per_query = pd.DataFrame(rows)
    cons = []
    for (name, key), grp in per_query.groupby(["system", "intent"]):
        sets = [answers[(name, key, i)].evidence_ids() for i in grp.paraphrase]
        pairs = list(itertools.combinations(sets, 2))
        cons.append(dict(system=name, intent=key,
                         evidence_consistency=sum(jaccard(a, b) for a, b in pairs) / len(pairs)))
    cons = pd.DataFrame(cons)
    metrics = ["decision_f1", "policy_recall", "traceability", "unsupported_citations", "policy_precision"]
    summary = per_query.groupby("system")[metrics].mean().join(
        cons.groupby("system")["evidence_consistency"].mean())
    order = [s for s in systems if s in summary.index]
    return per_query.merge(cons, on=["system", "intent"]), summary.loc[order]


def holdout_capture(rankings: dict[str, list[str]], failed: set[str], length_m: dict[str, float],
                    budgets=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Non-circular check against OBSERVED failures withheld from the graph (e.g. build
    with BattLeDIM 2018 leaks, test on 2019 leaks).

    Budgets are fractions of total network LENGTH, because renewal and inspection are
    planned in km: a pipe-count top-k unfairly favours rankings that pick long segments
    (failures scale with length) over per-km risk rankings. `lift` = capture rate /
    budget, i.e. 1.0 is no better than choosing mains at random."""
    total = sum(length_m.values())
    rows = []
    for name, ranked in rankings.items():
        for b in budgets:
            chosen, used = set(), 0.0
            for p in ranked:
                if used >= b * total:
                    break
                chosen.add(p)
                used += length_m[p]
            cap = len(chosen & failed) / len(failed) if failed else float("nan")
            rows.append(dict(system=name, length_budget=f"{b:.0%}", km=round(used / 1000, 2),
                             captured=len(chosen & failed), n_failed=len(failed),
                             capture_rate=round(cap, 3), lift=round(cap / b, 2) if failed else float("nan")))
    return pd.DataFrame(rows)


def synthetic_utilities(inp: str, seeds, verbose: bool = True) -> pd.DataFrame:
    """Evaluate on several synthetic utilities sharing one real network: each seed draws a
    different but equally plausible asset history. Reporting mean and sd across seeds shows
    that conclusions do not hinge on one random draw (the main weakness of synthetic data)."""
    from .asset import simulate_future_failures
    from .baselines import FlatRAGBaseline, MLBaseline
    from .graphrag import GraphRAG
    from .pipeline import build_knowledge_graph
    rows = []
    for seed in seeds:
        cfg = Config(seed=seed)
        res = build_knowledge_graph(inp, cfg, verbose=False)
        labels = simulate_future_failures(res.assets, res.wn, cfg)
        prev = pd.Series([d.action for d in res.decisions]).value_counts().to_dict()
        systems = {"WaterOnt GraphRAG": GraphRAG(res.kg, cfg, use_llm=False),
                   "Flat-document RAG": FlatRAGBaseline(res.kg, cfg),
                   "ML (GBM rank)": MLBaseline(res.kg, labels, prev, cfg)}
        _, summary = run_evaluation(res.kg, systems, cfg)
        for system, r in summary.iterrows():
            rows.append(dict(seed=seed, system=system, **r.to_dict()))
        if verbose:
            print(f"[seeds] seed {seed} done")
    df = pd.DataFrame(rows)
    metrics = ["decision_f1", "policy_recall", "traceability", "evidence_consistency"]
    agg = df.groupby("system")[metrics].agg(["mean", "std"])
    order = ["WaterOnt GraphRAG", "Flat-document RAG", "ML (GBM rank)"]
    return agg.loc[[o for o in order if o in agg.index]]
