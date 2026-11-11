"""Full-context LLM baseline: can an LLM given every pipe description at once (no retrieval)
apply the maintenance policy without the knowledge graph?

    python paper/experiments_full_context.py        # needs ANTHROPIC_API_KEY

The LLM receives all pipe descriptions of L-TOWN (seed 42) and the full policy, with the same
instruction as flat-text RAG plus one sentence: list at most 25 pipes, most urgent first (the
size of the reference answers and of the GraphRAG answers). Nothing else is run, and the
results reported earlier are not touched; everything is written to new files in paper/results/:

    full_context_per_query.csv    results per question, with every returned (pipe, action, clauses)
    full_context_summary.csv      Table 5 metrics plus policy precision
    full_context_meta.json        model, date, context size
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wateront.baselines import FlatRAGBaseline  # noqa: E402
from wateront.config import Config  # noqa: E402
from wateront.evaluation import run_evaluation  # noqa: E402
from wateront.pipeline import build_knowledge_graph  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LTOWN = os.path.join(os.path.dirname(HERE), "data", "networks", "L-TOWN.inp")
OUT = os.path.join(HERE, "results")
SYSTEM = "No KG, LLM (full context)"


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set: nothing was run")
    os.makedirs(OUT, exist_ok=True)
    cfg = Config(seed=42)
    from experiments import check_api
    check_api(cfg)
    res = build_knowledge_graph(LTOWN, cfg, verbose=False)
    full = FlatRAGBaseline(res.kg, cfg, use_llm=True, retriever="all")
    cards = [d for d in full.index.docs if d["kind"] == "asset"]
    chars = sum(len(d["text"]) + len(d["id"]) + 4 for d in cards)
    print(f"full context: {len(cards)} pipe descriptions, about {chars // 4:,} tokens per question")
    t0 = time.time()
    per_query, summary = run_evaluation(res.kg, {SYSTEM: full}, cfg)
    per_query.to_csv(os.path.join(OUT, "full_context_per_query.csv"), index=False)
    summary.to_csv(os.path.join(OUT, "full_context_summary.csv"))
    print(summary.round(3).to_string())
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "seed": 42, "llm_model": cfg.llm_model,
            "pipe_descriptions": len(cards), "approx_context_tokens": chars // 4,
            "answer_cap": full.full_context_top_k, "output_limit_tokens": cfg.llm_max_tokens_flat,
            "questions_without_recommendation": int((per_query.n_pred == 0).sum()),
            "minutes": round((time.time() - t0) / 60, 1)}
    try:
        import anthropic
        meta["anthropic"] = anthropic.__version__
    except Exception:
        pass
    with open(os.path.join(OUT, "full_context_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("done:", OUT)


if __name__ == "__main__":
    main()
