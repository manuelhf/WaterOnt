"""EPANET model (+ optional CSV files) -> network file for the web explorer.

Used by demo/WaterOnt_convert.ipynb, so that colleagues without Python can open their own
network in the explorer. Each CSV is recognised from its columns, not its name:

    asset register / inspection log   a pipe column plus register columns
                                      (install year, material, condition, last inspection, ...)
    failure list                      a pipe column plus a year or date column, nothing else
    sensor layout                     an element column and a type column with sensor types
    critical customers                an element/junction column and a type column (Hospital, ...)
"""
from __future__ import annotations

import os

from .tables import read_table

SENSOR_WORDS = {"pressure", "pressuresensor", "flow", "flowsensor", "level", "levelsensor", "amr", "amrmeter", "meter"}
REGISTER_COLS = {"install_year", "condition_grade", "last_inspection_year", "material", "soil_corrosivity",
                 "failure_years", "lining", "joint"}


def classify(path: str) -> str:
    """'inp', 'assets', 'failures', 'sensors', 'facilities' or 'unknown'."""
    if path.lower().endswith(".inp"):
        return "inp"
    try:
        head, rows, _ = read_table(path)
    except Exception:
        return "unknown"
    if "pipe" in head:
        if REGISTER_COLS & set(head):
            return "assets"
        if {"year", "date", "start", "start_time"} & set(head):
            return "failures"
        return "unknown"
    if "element" in head and "type" in head:
        ct = head.index("type")
        types = {r[ct].strip().lower().replace(" ", "").replace("_", "") for r in rows if ct < len(r) and r[ct].strip()}
        return "sensors" if types and types <= SENSOR_WORDS else "facilities"
    return "unknown"


def convert(paths: list[str], out_dir: str = ".", seed: int = 42) -> tuple[str, list[str]]:
    """Build the knowledge graph from the uploaded files and write the explorer's network
    file. Returns (path of the .json, plain-language summary lines)."""
    from .config import Config
    from .pipeline import build_knowledge_graph
    from .web_export import write_json

    kinds: dict[str, list[str]] = {}
    for p in paths:
        kinds.setdefault(classify(p), []).append(p)
    inps = kinds.get("inp", [])
    if len(inps) != 1:
        raise ValueError(f"upload exactly one EPANET model (.inp); found {len(inps)}")
    for k in ("assets", "failures", "sensors", "facilities"):
        if len(kinds.get(k, [])) > 1:
            raise ValueError(f"more than one {k} file: {', '.join(os.path.basename(x) for x in kinds[k])}; "
                             f"combine them or upload one at a time")
    one = lambda k: kinds.get(k, [None])[0]
    inp = inps[0]
    summary = [f"Model: {os.path.basename(inp)}"]
    for k, label in (("assets", "Asset register / inspection log"), ("failures", "Failure list"),
                     ("sensors", "Sensor layout"), ("facilities", "Critical customers")):
        if one(k):
            summary.append(f"{label}: {os.path.basename(one(k))}")
    for p in kinds.get("unknown", []):
        summary.append(f"Not used (columns not recognised): {os.path.basename(p)}")

    res = build_knowledge_graph(inp, Config(seed=seed), asset_csv=one("assets"), failures_csv=one("failures"),
                                facilities_csv=one("facilities"), sensors_csv=one("sensors"), verbose=False)
    wn = res.wn
    acts: dict[str, int] = {}
    for d in res.decisions:
        acts[d.action] = acts.get(d.action, 0) + 1
    summary.append(f"Built: {wn.num_pipes} pipes, {wn.num_junctions} junctions; recommended "
                   + ", ".join(f"{acts.get(a, 0)} {a}" for a in ("Replacement", "Rehabilitation", "Inspection", "Monitor")))
    notes = []
    if not one("assets"):
        notes.append("No asset register: pipe ages, materials, condition and inspections are SYNTHETIC "
                     "(illustrative only). Upload a register or inspection log for real results.")
    else:
        from .asset import load_asset_overrides
        rep: dict = {}
        recs = load_asset_overrides(one("assets"), Config(seed=seed), report=rep)
        names = set(res.assets["pipe"])
        matched = sum(1 for p in recs if p in names)
        notes.append(f"Your file covers {matched} of {wn.num_pipes} pipes"
                     + (f"; {len(recs) - matched} rows name pipes not in the model and were skipped" if len(recs) > matched else "")
                     + (f"; {rep['combined']} extra rows for repeated pipes were combined (latest inspection wins)" if rep["combined"] else "")
                     + ".")
        given = set(rep["columns_used"])
        missing = [label for col, label in (("install_year", "installation years"), ("material", "materials"),
                                             ("condition_grade", "condition grades"),
                                             ("last_inspection_year", "inspection dates"))
                   if col not in given]
        if missing:
            notes.append("Not in your file, so still SYNTHETIC for every pipe: " + ", ".join(missing) + ".")
        if matched < wn.num_pipes:
            notes.append(f"The other {wn.num_pipes - matched} pipes keep synthetic values for everything.")
    if not one("failures") and not one("assets"):
        notes.append("No failure records: failure histories are synthetic.")
    if not res.kg.nodes("PressureSensor") and not one("sensors"):
        notes.append("No pressure sensors in the model or uploaded: monitoring status is not used in the decisions.")
    coords = [n.coordinates for _, n in wn.nodes()]
    if len({tuple(c) for c in coords}) <= 1:
        notes.append("The model has no node coordinates: the explorer cannot draw the map (questions still work).")
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(inp))[0] + "_wateront.json")
    write_json(res, out)
    summary += [f"Note: {n}" for n in notes]
    summary.append(f"Written: {out}")
    return out, summary
