"""Export a built WaterOnt graph to a self-contained web demonstrator.

    python -m wateront web --inp data/networks/L-TOWN.inp --out demo/index.html
    python -m wateront web --inp Net3 --seed 7 --json --out net3_seed7.json   # open via the page

The page can also open a network file exported with --json (button or drag and
drop); the file is read in the browser and never uploaded.

The page needs no server: the graph facts are embedded as JSON, and the policy
rules (risk banding, ISO-style matrix, rules R0-R5) are re-implemented in
JavaScript so visitors can change the policy and see decisions recompute.
`tests/test_web_parity.py` checks that the JavaScript rules reproduce the
Python decisions exactly with default settings.
"""
from __future__ import annotations

import json
import os
from importlib import resources

from .config import Config
from .decision import pipe_facts
from .pipeline import BuildResult


def export_payload(res: BuildResult, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    kg, wn = res.kg, res.wn
    node_names = list(wn.node_name_list)
    idx = {n: i for i, n in enumerate(node_names)}
    sensors_at: dict[str, list[str]] = {}
    for cls in ("PressureSensor", "FlowSensor", "LevelSensor", "AMRMeter"):
        for s in kg.nodes(cls):
            elem = kg.node(s)["name"].split(" @ ")[1]
            sensors_at.setdefault(elem, []).append(cls)
    nodes = []
    for n in node_names:
        node = wn.get_node(n)
        x, y = node.coordinates if node.coordinates else (0.0, 0.0)
        kind = type(node).__name__[0]  # J, R, T
        nodes.append([n, round(float(x), 2), round(float(y), 2), kind, sensors_at.get(n, [])])

    risk = res.risk
    actions = {d.facts["uid"]: d.action for d in res.decisions}
    assets = res.assets.set_index("pipe")
    pipes = []
    for p in kg.nodes("Pipe"):
        f = pipe_facts(kg, p)
        name = f["pipe"]
        link = wn.get_link(name)
        r = risk.loc[name]
        pipes.append(dict(
            id=name, s=idx[link.start_node_name], e=idx[link.end_node_name],
            len=round(f["length"], 1), dia=round(f["diameter"]), mat=f["material"],
            msrc=f["material_source"], reg=f["provided"], age=f["age"], grade=f["condition_grade"],
            insp=f["last_inspection_year"], fails=f["failure_count"],
            pofKm=round(float(r.pof_km), 6), pof=round(float(r.pof), 4),
            pofBand=f["pof_band"], hyd=f["hydraulic"], svc=f["service"],
            hScore=round(float(r.hydraulic_score), 6), sScore=round(float(r.service_score), 6),
            mon=f["monitored"], zone=f["zone"], fac=f["facilities"],
            minP=round(min(float(f["min_end_pressure"]), 9999.0), 2),
            sensors=sensors_at.get(name, []),
            soil=str(assets.loc[name, "soil_corrosivity"]),
            failYears=[int(y) for y in assets.loc[name, "failure_years"]],
            failModes=[str(m) for m in assets.loc[name, "failure_modes"]],
            # Python's decision, used by the parity test and shown as the baseline
            pyAction=actions[p],
        ))
    others = []
    for name, link in wn.links():
        if type(link).__name__ != "Pipe":
            others.append(dict(id=name, kind=type(link).__name__, s=idx[link.start_node_name],
                               e=idx[link.end_node_name], sensors=sensors_at.get(name, []),
                               setting=getattr(link, "initial_setting", None)))
    guidelines = [{k: kg.node(g)[k] for k in ("id", "framework", "title", "text")}
                  for g in kg.nodes("EngineeringGuideline")]
    zones = [{"id": kg.node(z)["name"], "nJunctions": kg.node(z)["nJunctions"],
              "storage": kg.node(z)["containsStorage"]}
             for z in kg.nodes("PressureZone") if kg.node(z)["isSupplyZone"]]
    stats = kg.stats()
    ltown = "TOWN" in res.network_id.upper()
    return dict(
        format="wateront-web/1",
        notice=(("Network data derived from L-TOWN, © KIOS Research and Innovation Center of Excellence, "
                 "EUPL-1.2; converted to JSON and combined with synthetic asset records by Herrera et al.")
                if ltown else "Asset records are synthetic."),
        network=res.network_id, nodes=nodes, pipes=pipes, devices=others,
        guidelines=guidelines, zones=zones,
        graph={"nodes": stats["nodes"], "edges": stats["edges"]},
        config=dict(referenceYear=cfg.reference_year, pofBands=[t for t, _ in cfg.pof_bands[:3]],
                    designLife=cfg.design_life, inspectionInterval=cfg.inspection_interval,
                    minServicePressure=cfg.min_service_pressure_m,
                    pressureMargin=cfg.pressure_margin_m, seed=cfg.seed,
                    # failure model, so the browser can recompute PoF after an edit
                    weibull={m: list(v) for m, v in cfg.weibull.items()},
                    hazardScale=cfg.hazard_scale, failureCoef=cfg.failure_count_hazard_coef,
                    horizon=cfg.planning_horizon_years, soilFactor=cfg.soil_corrosivity_factor),
        edited=[],
    )


def write_json(res: BuildResult, out: str, cfg: Config | None = None) -> str:
    """Network file for the explorer's “Open network” button (read locally in the browser)."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(export_payload(res, cfg), fh, separators=(",", ":"))
    return out


def write_web(res: BuildResult, out: str, cfg: Config | None = None) -> str:
    payload = json.dumps(export_payload(res, cfg), separators=(",", ":"))
    template = resources.files("wateront").joinpath("web_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__WATERONT_DATA__*/null", payload.replace("</", "<\\/"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out
