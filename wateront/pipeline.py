"""Builds the four WaterOnt layers from an EPANET .inp file (Section 3.5, Figure 1)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import wntr

from .asset import (apply_failure_history, build_asset_layer, load_asset_table,
                    load_failure_events, synthesise_asset_table)
from .config import Config
from .decision import Decision, build_decision_layer, load_guidelines
from .kg import KnowledgeGraph
from .physical import (build_instrumentation, build_physical_layer, build_pressure_zones,
                       pipes_near_sensor)
from .risk import build_risk_layer, choose_critical_facilities, compute_risk_table


@dataclass
class BuildResult:
    kg: KnowledgeGraph
    wn: wntr.network.WaterNetworkModel
    assets: pd.DataFrame
    risk: pd.DataFrame
    decisions: list[Decision]
    network_id: str


def resolve_inp(name_or_path: str) -> str:
    """Accept a path, or the name of a network shipped with WNTR (Net1, Net3, Net6, ...)."""
    if os.path.exists(name_or_path):
        return name_or_path
    import wntr.library
    lib = os.path.join(os.path.dirname(wntr.library.__file__), "networks")
    cand = os.path.join(lib, name_or_path if name_or_path.endswith(".inp") else name_or_path + ".inp")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"{name_or_path}: not a file and not a WNTR library network")


def build_knowledge_graph(inp: str, cfg: Config | None = None, asset_csv: str | None = None,
                          guidelines_json: str | None = None, failures_csv: str | None = None,
                          verbose: bool = True) -> BuildResult:
    cfg = cfg or Config()
    path = resolve_inp(inp)
    network_id = os.path.splitext(os.path.basename(path))[0]
    wn = wntr.network.WaterNetworkModel(path)
    kg = KnowledgeGraph()
    log = print if verbose else (lambda *a, **k: None)

    build_physical_layer(wn, kg, network_id)
    n_sensors = build_instrumentation(wn, kg, path)
    build_pressure_zones(wn, kg, network_id)
    zones = [kg.node(z)["name"] for z in kg.nodes("PressureZone") if kg.node(z)["isSupplyZone"]]
    log(f"[physical] {len(kg.nodes('Pipe'))} pipes, {len(kg.nodes('Node'))} nodes, "
        f"{n_sensors} sensors, supply zones: {zones}")

    assets = synthesise_asset_table(wn, cfg)
    if asset_csv:  # real register overrides priors wherever it has a row
        real = load_asset_table(asset_csv).set_index("pipe")
        known = set(assets["pipe"])
        unknown = set(real.index) - known
        if unknown:
            log(f"[asset]    ignoring {len(unknown)} register rows with no matching pipe")
        rows = assets.to_dict("records")
        hit = []
        for row in rows:
            if row["pipe"] in real.index:
                row.update(real.loc[row["pipe"]].to_dict())
                hit.append(row["pipe"])
        assets = pd.DataFrame(rows, columns=assets.columns)
        log(f"[asset]    {len(hit)}/{len(assets)} pipes from register, rest from priors")
    if failures_csv:
        events = load_failure_events(failures_csv)
        assets = apply_failure_history(assets, events)
        log(f"[asset]    observed failure history from {failures_csv}: "
            f"{sum(map(len, events.values()))} events on {len(events)} pipes")
    build_asset_layer(wn, kg, assets, cfg)
    log(f"[asset]    material provenance: "
        f"{assets['material_source'].str.split(':').str[:2].str.join(':').value_counts().to_dict()}")
    log(f"[asset]    {len(kg.nodes('AssetRecord'))} asset records, "
        f"{len(kg.nodes('FailureEvent'))} failure events")

    facilities = choose_critical_facilities(wn, cfg)
    monitored = pipes_near_sensor(wn, kg, "PressureSensor", hops=2) if kg.nodes("PressureSensor") else None
    risk, node_state = compute_risk_table(wn, assets, cfg, facilities, monitored)
    if monitored is None:   # no sensor metadata: treat monitoring status as unknown, not absent
        risk["monitored"] = True
    build_risk_layer(wn, kg, risk, node_state, facilities)
    log(f"[risk]     overall risk: {risk.overall_risk.value_counts().to_dict()}")

    gl = load_guidelines(guidelines_json) if guidelines_json else None
    decisions = build_decision_layer(kg, cfg, gl)
    log(f"[decision] {pd.Series([d.action for d in decisions]).value_counts().to_dict()}")
    return BuildResult(kg, wn, assets, risk, decisions, network_id)
