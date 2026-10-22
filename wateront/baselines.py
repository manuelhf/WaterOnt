"""Baselines for Section 5.

MLBaseline       "predict and rank": gradient boosting on asset features trained on
                 synthetic horizon-failure labels (out-of-fold predictions). Ranked
                 pipes are mapped to actions with cut-offs calibrated to the
                 reference action prevalence (its best case). No policy citations.
FlatRAGBaseline  conventional document RAG: asset-register rows rendered as text
                 cards plus guideline clauses in one flat TF-IDF index. It sees the
                 same attributes and risk bands as WaterOnt, but no graph structure.
                 The action for each retrieved card comes from the action-bearing
                 guideline most similar to that card, which is also what it cites.

Both expose `ask(query) -> Answer` so they are scored by the same evaluator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import Config
from .decision import ACTION_RANK, ACTIONS, pipe_facts
from .graphrag import Answer, plan_rules
from .kg import KnowledgeGraph
from .retrieval import Evidence, VectorIndex

TOPIC_ACTION = {"replacement": "Replacement", "rehabilitation": "Rehabilitation",
                "inspection": "Inspection", "monitoring": "Monitor"}


def _features(kg: KnowledgeGraph) -> pd.DataFrame:
    rows = []
    for p in kg.nodes("Pipe"):
        ar = kg.node(kg.one(p, "HAS_ASSET_RECORD"))
        n = kg.node(p)
        rows.append(dict(uid=p, name=n["name"], age=ar["age"], diameter=n["diameter"],
                         length=n["length"], failures=ar["failureCount"],
                         grade=ar["conditionGrade"], material=ar["material"]))
    df = pd.DataFrame(rows).set_index("uid")
    return pd.get_dummies(df, columns=["material"], dtype=float)


class MLBaseline:
    name = "ML (GBM rank)"

    def __init__(self, kg: KnowledgeGraph, labels: pd.Series, action_prevalence: dict[str, int],
                 cfg: Config | None = None):
        self.kg, self.cfg = kg, cfg or Config()
        X = _features(kg)
        y = labels.reindex(X["name"]).fillna(0).astype(int).values
        Xn = X.drop(columns=["name"])
        clf = GradientBoostingClassifier(random_state=self.cfg.seed)
        n_splits = max(2, min(5, int(min(np.bincount(y, minlength=2)))))
        if len(set(y)) > 1:
            proba = cross_val_predict(clf, Xn, y, method="predict_proba",
                                      cv=StratifiedKFold(n_splits, shuffle=True, random_state=self.cfg.seed))[:, 1]
        else:
            proba = np.full(len(y), float(y.mean()))
        self.proba = pd.Series(proba, index=X.index).sort_values(ascending=False)
        try:
            from sklearn.metrics import roc_auc_score
            self.auc = float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan")
        except Exception:
            self.auc = float("nan")
        self.action = {}
        order, i = list(self.proba.index), 0
        for a in ACTIONS[:-1]:
            for uid in order[i:i + action_prevalence.get(a, 0)]:
                self.action[uid] = a
            i += action_prevalence.get(a, 0)
        for uid in order[i:]:
            self.action[uid] = "Monitor"

    def ask(self, query: str) -> Answer:
        plan = plan_rules(query)
        cand = list(self.proba.index)
        if plan.actions:
            cand = [u for u in cand if self.action[u] in plan.actions]
        elif plan.exclude_monitor:
            cand = [u for u in cand if self.action[u] != "Monitor"]
        if plan.pipes:
            cand = [u for u in cand if self.kg.node(u)["name"] in plan.pipes]
        recs = []
        for u in cand[: plan.top_k]:
            f = pipe_facts(self.kg, u)
            recs.append(Evidence(u, f, self.action[u], "ML", [], []))
        recs.sort(key=lambda e: (ACTION_RANK[e.action], -self.proba[e.pipe_uid]))
        text = "\n".join(f"[{e.pipe_uid}] {e.action} (p_fail={self.proba[e.pipe_uid]:.2f})" for e in recs)
        return Answer(query, plan, recs, [], text, generator="ml")


def asset_card(f: dict) -> str:
    fac = f"Serves {', '.join(f['facilities'])}." if f["facilities"] else ""
    return (f"Pipe {f['pipe']}: {f['material']} main, {f['diameter']:.0f} mm, {f['length']:.0f} m long, "
            f"age {f['age']} years, design life {f['design_life']} years. Condition grade "
            f"{f['condition_grade']}. {f['failure_count']} recorded failures. Failure probability "
            f"{f['pof_band']}. Hydraulic criticality {f['hydraulic']}. Service criticality "
            f"{f['service']}. Overall risk {f['overall']}. Minimum pressure {f['min_end_pressure']} m. "
            f"Last inspected {f['last_inspection_year']}. {fac}")


class FlatRAGBaseline:
    name = "Flat-document RAG"

    def __init__(self, kg: KnowledgeGraph, cfg: Config | None = None, use_llm: bool = False,
                 k_cards: int = 40):
        self.kg, self.cfg, self.use_llm, self.k_cards = kg, cfg or Config(), use_llm, k_cards
        if use_llm:
            self.name = "Flat-document RAG + LLM"
        self.facts = {p: pipe_facts(kg, p) for p in kg.nodes("Pipe")}
        self.guidelines = [kg.node(g) for g in kg.nodes("EngineeringGuideline")]
        docs = [dict(id=p, kind="asset", title="", text=asset_card(f)) for p, f in self.facts.items()]
        docs += [dict(id=g["id"], kind="policy", title=g["title"], text=g["text"]) for g in self.guidelines]
        self.index = VectorIndex(docs)
        self.policy_index = VectorIndex([g for g in self.guidelines if g["topic"] in TOPIC_ACTION])

    def ask(self, query: str) -> Answer:
        if self.use_llm:
            return self._ask_llm(query)
        plan = plan_rules(query)
        hits = self.index.search(query, k=len(self.index.docs))
        policy = [d for d, _ in hits if d["kind"] == "policy"][:3]
        recs = []
        for d, _ in hits:
            if d["kind"] != "asset":
                continue
            f = self.facts[d["id"]]
            match = self.policy_index.search(d["text"], k=2)   # action from the asset's own card
            if not match:
                continue
            action = TOPIC_ACTION[match[0][0]["topic"]]
            if plan.actions and action not in plan.actions:
                continue
            if plan.exclude_monitor and action == "Monitor":
                continue
            recs.append(Evidence(d["id"], f, action, "flat", [m["id"] for m, _ in match], []))
            if len(recs) >= plan.top_k:
                break
        text = "\n".join(f"[{e.pipe_uid}] {e.action} per " + ", ".join(f"[{g}]" for g in e.guidelines)
                         for e in recs)
        return Answer(query, plan, recs, policy, text, generator="flat")

    LLM_SYSTEM = ("You are a water-utility asset engineer. Using ONLY the asset records and policy "
                  "clauses provided, answer the question by deciding an action (Replacement, "
                  "Rehabilitation, Inspection or Monitor) for each relevant pipe and citing the clause "
                  "ids that authorise it. Reply with JSON only: a list of objects with keys "
                  "pipe (the id in brackets, e.g. P:101), action, guidelines (list of clause ids).")

    def _ask_llm(self, query: str) -> Answer:
        """Conventional RAG: top-k flat chunks by similarity, LLM reasons over the text."""
        import json
        import re

        from .graphrag import _anthropic_client
        plan = plan_rules(query)
        hits = self.index.search(query, k=self.k_cards)
        cards = [d for d, _ in hits if d["kind"] == "asset"]
        policy = [d for d, _ in self.index.search(query, k=len(self.index.docs)) if d["kind"] == "policy"][:5]
        ctx = "\n".join(f"[{d['id']}] {d['text']}" for d in cards) + "\n\nPolicy:\n" + \
            "\n".join(f"[{g['id']}] {g['title']}: {g['text']}" for g in policy)
        msg = _anthropic_client().messages.create(
            model=self.cfg.llm_model, max_tokens=self.cfg.llm_max_tokens, system=self.LLM_SYSTEM,
            messages=[{"role": "user", "content": f"{ctx}\n\nQuestion: {query}"}])
        raw = "".join(b.text for b in msg.content if b.type == "text")
        try:
            items = json.loads(re.sub(r"```(json)?", "", raw).strip())
        except json.JSONDecodeError:
            items = []
        recs = []
        for it in items[: plan.top_k]:
            uid = str(it.get("pipe", "")).strip("[]")
            if uid in self.facts and it.get("action") in ACTIONS:
                recs.append(Evidence(uid, self.facts[uid], it["action"], "flat-llm",
                                     [str(g) for g in it.get("guidelines", [])], []))
        ans = Answer(query, plan, recs, policy, raw, generator="flat-llm")
        allowed = {d["id"] for d in cards} | {g["id"] for g in policy}
        ans.unsupported_citations = sorted({g for e in recs for g in e.guidelines if g not in allowed}
                                           | {e.pipe_uid for e in recs if e.pipe_uid not in allowed})
        return ans
