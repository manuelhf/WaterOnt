"""Stronger retrieval for the flat-text RAG baseline (reviewer question: would better
retrieval narrow the gap to WaterOnt GraphRAG?).

    python paper/experiments_retrieval.py                  # recommended: all LLM systems in one run
    python paper/experiments_retrieval.py --no-graphrag    # flat-text RAG variants only

Only the retrieval method of flat-text RAG changes: the same combined index of 905 pipe
descriptions and 12 policy clauses, the same 40 most similar entries, the full policy and
the same prompt. By default WaterOnt GraphRAG is run in the same session, so that every LLM
row comes from one run with every answer saved (LLM output is not repeatable, and earlier runs
did not save answers). The earlier result files are not touched; everything is written to
new files in paper/results/:

    retrieval_analysis.csv          per question and retriever: pipe records supplied, and
                                    reference pipes among them (no LLM needed)
    retrieval_rag_per_query.csv     LLM results per question, including every returned
                                    (pipe, action, clauses) in the 'predictions' column
    retrieval_rag_summary.csv       Table 5 metrics plus policy precision (uncapped)
    retrieval_meta.json             models, versions, date
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wateront.baselines import FlatRAGBaseline  # noqa: E402
from wateront.config import Config  # noqa: E402
from wateront.evaluation import default_intents, reference, run_evaluation  # noqa: E402
from wateront.pipeline import build_knowledge_graph  # noqa: E402
from wateront.retrieval import DENSE_MODEL, DENSE_QUERY_PREFIX, sentence_transformer_embedder  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LTOWN = os.path.join(os.path.dirname(HERE), "data", "networks", "L-TOWN.inp")
OUT = os.path.join(HERE, "results")
LABEL = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "dense"}


def flat_rag(res, cfg, retriever, embed=None, use_llm=True):
    if retriever == "dense":
        return FlatRAGBaseline(res.kg, cfg, use_llm=use_llm, retriever="dense",
                               embed=embed, query_prefix=DENSE_QUERY_PREFIX)
    return FlatRAGBaseline(res.kg, cfg, use_llm=use_llm, retriever=retriever)


def retrieval_analysis(res, cfg, retrievers, embed=None) -> pd.DataFrame:
    """What each retriever puts in the LLM's context, measured without calling the LLM."""
    rows = []
    for r in retrievers:
        flat = flat_rag(res, cfg, r, embed, use_llm=False)
        for it in default_intents(res.kg):
            ref = {p for p, _, _ in reference(res.kg, it, cfg)}
            for i, q in enumerate(it.paraphrases):
                cards = {d["id"] for d, _ in flat.index.search(q, k=flat.k_cards) if d["kind"] == "asset"}
                rows.append(dict(retriever=LABEL[r], intent=it.key, paraphrase=i, query=q,
                                 pipe_records=len(cards), reference_pipes=len(ref),
                                 reference_in_context=len(ref & cards)))
    return pd.DataFrame(rows)


def summarise_retrieval(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("retriever", sort=False)
    out = pd.DataFrame({
        "questions_without_pipe_records": g.pipe_records.apply(lambda x: int((x == 0).sum())),
        "pipe_records_min": g.pipe_records.min(), "pipe_records_max": g.pipe_records.max(),
        "reference_in_context": g.reference_in_context.sum(), "reference_pipes": g.reference_pipes.sum()})
    out["share_in_context"] = (out.reference_in_context / out.reference_pipes).round(3)
    return out


def main(argv=None, embed=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", nargs="+", default=["tfidf", "dense"], choices=["dense", "bm25", "tfidf"],
                    help="flat-text RAG retrievers to run WITH the LLM (default: tfidf dense)")
    ap.add_argument("--no-graphrag", action="store_true", help="do not run WaterOnt GraphRAG in this session")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    cfg = Config(seed=42)
    res = build_knowledge_graph(LTOWN, cfg, verbose=False)
    if embed is None:
        print(f"loading dense embedding model {DENSE_MODEL}")
        embed = sentence_transformer_embedder(DENSE_MODEL)

    print("[1] retrieval analysis (no LLM): TF-IDF, BM25, dense")
    ra = retrieval_analysis(res, cfg, ["tfidf", "bm25", "dense"], embed)
    ra.to_csv(os.path.join(OUT, "retrieval_analysis.csv"), index=False)
    summary = summarise_retrieval(ra)
    print(summary.to_string())

    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "seed": 42, "dense_model": DENSE_MODEL,
            "llm_model": cfg.llm_model, "llm_retrievers": args.retrievers,
            "graphrag_in_run": not args.no_graphrag, "llm_run": False}
    if os.environ.get("ANTHROPIC_API_KEY"):
        from experiments import check_api
        check_api(cfg)
        print(f"[2] flat-text RAG with the LLM, retrievers: {', '.join(args.retrievers)}")
        systems = {f"No KG, LLM ({LABEL[r]} RAG)": flat_rag(res, cfg, r, embed) for r in args.retrievers}
        if not args.no_graphrag:
            from wateront.graphrag import GraphRAG
            systems["KG, LLM (WaterOnt GraphRAG)"] = GraphRAG(res.kg, cfg, use_llm=True, strict=True)
        per_query, summ = run_evaluation(res.kg, systems, cfg)
        per_query.to_csv(os.path.join(OUT, "retrieval_rag_per_query.csv"), index=False)
        summ.to_csv(os.path.join(OUT, "retrieval_rag_summary.csv"))
        print(summ.round(3).to_string())
        meta["llm_run"] = True
        meta["no_recommendation_questions"] = {
            s: int((per_query[per_query.system == s].n_pred == 0).sum()) for s in systems}
        meta["llm_planning_failures"] = int(per_query.planning_failed.sum())
    else:
        print("[2] skipped: no ANTHROPIC_API_KEY set")
    try:
        import anthropic
        import sentence_transformers
        meta["versions"] = {"anthropic": anthropic.__version__, "sentence_transformers": sentence_transformers.__version__}
    except Exception:
        pass
    with open(os.path.join(OUT, "retrieval_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("done:", OUT)


if __name__ == "__main__":
    main()
