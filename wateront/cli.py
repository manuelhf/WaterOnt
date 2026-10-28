"""Command line interface.

  python -m wateront build    --inp Net3 --out kg.pkl [--ttl kg.ttl] [--neo4j bolt://localhost:7687]
  python -m wateront ask      --kg kg.pkl "Which pipes should be replaced?" [--json] [--llm]
  python -m wateront evaluate --inp Net3 [--out results/]
"""
from __future__ import annotations

import argparse
import json
import os

from .config import Config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="wateront")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build the four-layer knowledge graph")
    b.add_argument("--inp", default="Net3", help=".inp path or WNTR library name (Net3, Net6, ...)")
    b.add_argument("--assets", help="optional asset-register CSV (see asset.load_asset_table)")
    b.add_argument("--guidelines", help="optional guideline JSON (see decision.load_guidelines)")
    b.add_argument("--failures", help="observed failure CSV (pipe, year|date, [mode]); replaces synthetic history")
    b.add_argument("--facilities", help="critical customers CSV (junction, type); replaces synthetic ones")
    b.add_argument("--sensors", help="sensor layout CSV (element, type); replaces sensors marked in the .inp")
    b.add_argument("--sim-hours", type=float, help="simulation length (default: model duration, max 168 h)")
    b.add_argument("--out", default="kg.pkl")
    b.add_argument("--ttl", help="also export instance RDF (Turtle)")
    b.add_argument("--neo4j", help="bolt URI; credentials from NEO4J_USER / NEO4J_PASSWORD")
    b.add_argument("--wipe", action="store_true", help="delete ALL existing Neo4j data first")
    b.add_argument("--keep-existing", action="store_true",
                   help="merge into the existing WaterOnt graph instead of replacing it (not recommended)")
    b.add_argument("--seed", type=int, default=42)

    a = sub.add_parser("ask", help="Ask the GraphRAG engine a question")
    a.add_argument("question")
    a.add_argument("--kg", default="kg.pkl")
    a.add_argument("--llm", action="store_true", help="use Claude for planning/generation (needs ANTHROPIC_API_KEY)")
    a.add_argument("--json", action="store_true", help="print the full trace as JSON")

    e = sub.add_parser("evaluate", help="Compare WaterOnt GraphRAG with the baselines")
    e.add_argument("--inp", default="Net3")
    e.add_argument("--out", default="results")
    e.add_argument("--llm", action="store_true")
    e.add_argument("--expert", help="optional expert reference CSV (intent,pipe,action,guidelines)")
    e.add_argument("--failures", help="observed failures used as HISTORY in the graph (e.g. BattLeDIM 2018)")
    e.add_argument("--holdout-failures", help="observed failures WITHHELD from the graph (e.g. BattLeDIM 2019); "
                                              "used as ML labels and for capture@k validation")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--seeds", type=int, default=1,
                   help="evaluate N synthetic utilities (seeds seed..seed+N-1) and report mean and sd")

    w = sub.add_parser("web", help="Write the self-contained web demonstrator (one HTML file)")
    w.add_argument("--inp", default="Net3")
    w.add_argument("--out", default="demo/index.html")
    w.add_argument("--assets"); w.add_argument("--failures"); w.add_argument("--facilities"); w.add_argument("--sensors")
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--json", action="store_true", help="write a network file for the page's “Open network” button instead")

    c = sub.add_parser("convert", help="EPANET model + optional CSV files -> network file for the explorer")
    c.add_argument("files", nargs="+", help="one .inp and any asset register / inspection log, failure list, "
                                            "sensor layout or critical customers CSV (recognised by their columns)")
    c.add_argument("--out-dir", default=".")
    c.add_argument("--seed", type=int, default=42)

    args = ap.parse_args(argv)

    if args.cmd == "build":
        from .pipeline import build_knowledge_graph
        res = build_knowledge_graph(args.inp, Config(seed=args.seed, sim_duration_h=args.sim_hours),
                                    args.assets, args.guidelines, args.failures, args.facilities, args.sensors)
        res.kg.save(args.out)
        print(f"saved {args.out}: {res.kg.stats()['nodes']} nodes, {res.kg.stats()['edges']} edges")
        if args.ttl:
            from .rdf_export import to_rdf
            to_rdf(res.kg).serialize(args.ttl, format="turtle")
            print(f"saved {args.ttl}")
        if args.neo4j:
            removed = res.kg.to_neo4j(args.neo4j, os.environ.get("NEO4J_USER", "neo4j"),
                                      os.environ.get("NEO4J_PASSWORD", "password"), wipe=args.wipe,
                                      keep_existing=args.keep_existing)
            note = ("merged into the existing graph" if args.keep_existing else
                    "database wiped first" if args.wipe else f"replaced the previous WaterOnt graph ({removed} nodes)")
            print(f"pushed to {args.neo4j}: {note}")

    elif args.cmd == "web":
        from .pipeline import build_knowledge_graph
        from .web_export import write_web
        cfg = Config(seed=args.seed)
        res = build_knowledge_graph(args.inp, cfg, args.assets, failures_csv=args.failures,
                                    facilities_csv=args.facilities, sensors_csv=args.sensors)
        if args.json:
            from .web_export import write_json
            out = args.out if args.out.endswith(".json") else f"{res.network_id}_seed{args.seed}.json"
            print(f"wrote {write_json(res, out, cfg)} (open it with the explorer's “Open network” button)")
        else:
            print(f"wrote {write_web(res, args.out, cfg)} (open it in any browser; no server needed)")

    elif args.cmd == "convert":
        import sys
        from .convert import convert
        try:
            _, summary = convert(args.files, out_dir=args.out_dir, seed=args.seed)
        except ValueError as err:
            print(f"Nothing was converted: {err}")
            sys.exit(1)
        print("\n".join(summary))

    elif args.cmd == "ask":
        from .graphrag import GraphRAG
        from .kg import KnowledgeGraph
        rag = GraphRAG(KnowledgeGraph.load(args.kg), use_llm=args.llm)
        ans = rag.ask(args.question)
        print(json.dumps(ans.to_dict(), indent=2, default=str) if args.json else ans.text)
        if ans.unsupported_citations:
            print(f"\nWARNING unsupported citations: {ans.unsupported_citations}")

    elif args.cmd == "evaluate" and args.seeds > 1:
        import pandas as pd
        from .evaluation import synthetic_utilities
        table = synthetic_utilities(args.inp, range(args.seed, args.seed + args.seeds))
        os.makedirs(args.out, exist_ok=True)
        table.to_csv(os.path.join(args.out, "multi_seed_summary.csv"))
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
            print(table)

    elif args.cmd == "evaluate":
        import pandas as pd
        from .asset import simulate_future_failures
        from .baselines import FlatRAGBaseline, MLBaseline
        from .evaluation import holdout_capture, load_expert_reference, run_evaluation
        from .graphrag import GraphRAG
        from .pipeline import build_knowledge_graph

        cfg = Config(seed=args.seed)
        res = build_knowledge_graph(args.inp, cfg, failures_csv=args.failures)
        holdout = None
        if args.holdout_failures:
            from .asset import load_failure_events
            holdout = set(load_failure_events(args.holdout_failures)) & set(res.assets["pipe"])
            labels = pd.Series({p: int(p in holdout) for p in res.assets["pipe"]})
            print(f"[eval] {len(holdout)} held-out failed pipes used as ML labels and for capture@k")
        else:
            labels = simulate_future_failures(res.assets, res.wn, cfg)
        prevalence = pd.Series([d.action for d in res.decisions]).value_counts().to_dict()
        ml = MLBaseline(res.kg, labels, prevalence, cfg)
        print(f"[ml] out-of-fold AUC on synthetic horizon failures: {ml.auc:.3f}")
        systems = {"WaterOnt GraphRAG": GraphRAG(res.kg, cfg, use_llm=args.llm),
                   FlatRAGBaseline.name: FlatRAGBaseline(res.kg, cfg), MLBaseline.name: ml}
        if args.llm:
            systems["Flat-document RAG + LLM"] = FlatRAGBaseline(res.kg, cfg, use_llm=True)
        expert = load_expert_reference(args.expert) if args.expert else None
        per_query, summary = run_evaluation(res.kg, systems, cfg, expert=expert)
        os.makedirs(args.out, exist_ok=True)
        per_query.to_csv(os.path.join(args.out, f"{res.network_id}_per_query.csv"), index=False)
        summary.to_csv(os.path.join(args.out, f"{res.network_id}_summary.csv"))
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
            print(per_query.groupby(["intent", "system"])[["n_ref", "decision_f1", "policy_recall",
                                                             "evidence_consistency"]].mean())
            print("\nSUMMARY\n", summary)
        if holdout:
            from .decision import ACTION_RANK, pipe_facts
            facts = [pipe_facts(res.kg, p) for p in res.kg.nodes("Pipe")]
            act = {d.facts["uid"]: d.action for d in res.decisions}
            graph_rank = [f["pipe"] for f in sorted(facts, key=lambda f: (ACTION_RANK[act[f["uid"]]], -f["priority"]))]
            ml_rank = [res.kg.node(u)["name"] for u in ml.proba.index]
            lengths = {f["pipe"]: f["length"] for f in facts}
            # risk ranking (likelihood x consequence) vs likelihood-only ranking: failure capture
            # measures likelihood, so report both to keep the comparison fair
            pof_rank = [f["pipe"] for f in sorted(facts, key=lambda f: -f["pof_km"])]
            cap = holdout_capture({"WaterOnt GraphRAG (risk)": graph_rank, "WaterOnt PoF/km": pof_rank,
                                   MLBaseline.name: ml_rank}, holdout, lengths)
            cap.to_csv(os.path.join(args.out, f"{res.network_id}_holdout_capture.csv"), index=False)
            print("\nHOLDOUT CAPTURE (observed failures withheld from the graph)\n", cap.to_string(index=False))


if __name__ == "__main__":
    main()
