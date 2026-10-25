"""Risk Layer (Section 3.3).

Each pipe gets a RiskProfile linked to *shared* band nodes, e.g.
    (:Pipe)-[:HAS_RISK_PROFILE]->(:RiskProfile)-[:HAS_OVERALL_RISK]->(:OverallRisk {band:'Intolerable'})
so "hydraulically critical AND high PoF" is a two-hop pattern intersection rather
than a numeric filter evaluated by the language model.

Indicators
* FailureProbability: conditional Weibull probability of >=1 failure over the
  planning horizon given survival to current age, with proportional-hazards uplift
  for past failures and soil corrosivity, scaled by length.
* HydraulicCriticality: mean percentile rank of edge betweenness (topological, after
  Giudicianni et al.) and mean absolute simulated flow (max-flow-contribution proxy).
* ServiceCriticality: demand served within two hops, critical facilities at the pipe
  ends (ISO 24510-style consumer consequence), raised one band where the simulated
  minimum pressure at an end node is below the service threshold.
* OverallRisk: likelihood x consequence matrix (ISO 55000-style), four categories.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import wntr

from .asset import weibull_cum_hazard
from .config import Config
from .kg import KnowledgeGraph
from .physical import link_uid, node_uid, total_demand

LEVEL = {"Low": 1, "Medium": 2, "High": 3, "VeryHigh": 4, "Critical": 4}
CRIT_BANDS = ["Low", "Medium", "High", "Critical"]
OVERALL = ["Negligible", "Tolerable", "Significant", "Intolerable"]


def overall_category(pof_level: int, consequence_level: int) -> str:
    score = pof_level * consequence_level
    if score <= 2:
        return "Negligible"
    if score <= 4:
        return "Tolerable"
    if score <= 8:
        return "Significant"
    return "Intolerable"


def _band(value: float, bands) -> str:
    return next(name for thr, name in bands if value < thr)


def run_hydraulics(wn: wntr.network.WaterNetworkModel, cfg: Config):
    hours = cfg.sim_duration_h or min(wn.options.time.duration / 3600.0, 168.0) or 24.0
    wn.options.time.duration = int(hours * 3600)   # weekly models (L-TOWN) keep weekday/weekend cycle
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:  # keep EPANET scratch files out of the cwd
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "wateront"))
    flow = res.link["flowrate"].abs().mean() * 1000.0     # L/s
    p = res.node["pressure"]
    return flow, p.min(), p.mean()


def compute_risk_table(wn: wntr.network.WaterNetworkModel, assets: pd.DataFrame,
                       cfg: Config, facilities: dict[str, str],
                       monitored: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow, pmin, pmean = run_hydraulics(wn, cfg)
    G = wn.to_graph().to_undirected()
    ebc_raw = nx.edge_betweenness_centrality(G, normalized=True)
    ebc = {k: v for (u, w, k), v in ebc_raw.items()}

    a = assets.set_index("pipe")
    junctions = set(wn.junction_name_list)   # pressure margin only meaningful at demand nodes
    rows = []
    for name, pipe in wn.pipes():
        r = a.loc[name]
        beta, eta = cfg.weibull[r.material]
        age = cfg.reference_year - int(r.install_year)
        km = max(pipe.length / 1000.0, 0.05)
        dH = weibull_cum_hazard(age + cfg.planning_horizon_years, beta, eta) - weibull_cum_hazard(age, beta, eta)
        mult = np.exp(cfg.failure_count_hazard_coef * len(r.failure_years)) \
            * cfg.soil_corrosivity_factor[r.soil_corrosivity]
        pof = float(1 - np.exp(-dH * km * cfg.hazard_scale * mult))        # this segment
        pof_km = float(1 - np.exp(-dH * 1.0 * cfg.hazard_scale * mult))    # per km of main
        ends = [pipe.start_node_name, pipe.end_node_name]
        near = set(ends)
        for n in ends:
            near |= set(nx.single_source_shortest_path_length(G, n, cutoff=2))
        demand = sum(total_demand(wn.get_node(n)) for n in near if n in wn.junction_name_list)
        rows.append(dict(pipe=name, pof=pof, pof_km=pof_km, monitored=name in (monitored or set()),
                         betweenness=ebc.get(name, 0.0),
                         mean_flow=float(flow.get(name, 0.0)), demand_served=float(demand),
                         min_end_pressure=float(min([pmin[n] for n in ends if n in junctions] or [np.inf])),
                         facilities=[facilities[n] for n in ends if n in facilities]))
    df = pd.DataFrame(rows).set_index("pipe")
    df["hydraulic_score"] = 0.5 * df.betweenness.rank(pct=True) + 0.5 * df.mean_flow.rank(pct=True)
    df["hydraulic_score"] = df.hydraulic_score.rank(pct=True)
    df["service_score"] = df.demand_served.rank(pct=True)
    basis = df.pof_km if cfg.pof_basis == "per_km" else df.pof
    df["pof_band"] = basis.map(lambda v: _band(v, cfg.pof_bands))
    df["hydraulic_band"] = df.hydraulic_score.map(lambda v: _band(v, cfg.criticality_percentiles))

    def service_band(row) -> str:
        lvl = LEVEL[_band(row.service_score, cfg.criticality_percentiles)]
        if "Hospital" in row.facilities:
            lvl = 4
        elif row.facilities:   # schools, care homes, industrial customers
            lvl = max(lvl, 3)
        if row.min_end_pressure < cfg.min_service_pressure_m:
            lvl = min(lvl + 1, 4)
        return CRIT_BANDS[lvl - 1]

    df["service_band"] = df.apply(service_band, axis=1)
    df["consequence_level"] = df[["hydraulic_band", "service_band"]].apply(
        lambda r: max(LEVEL[r.hydraulic_band], LEVEL[r.service_band]), axis=1)
    df["pof_level"] = df.pof_band.map(LEVEL)
    df["risk_score"] = df.pof_level * df.consequence_level
    df["overall_risk"] = [overall_category(p, c) for p, c in zip(df.pof_level, df.consequence_level)]
    # Continuous tie-breaker for ranking within the same matrix cell
    df["priority"] = df.risk_score + basis * np.maximum(df.hydraulic_score, df.service_score)

    nodes = pd.DataFrame({"min_pressure": pmin, "mean_pressure": pmean})
    return df, nodes


def choose_critical_facilities(wn: wntr.network.WaterNetworkModel, cfg: Config,
                               known: dict[str, str] | None = None) -> dict[str, str]:
    """Critical consumers. Junctions with an industrial demand category in the model are
    real evidence and become IndustrialCustomer. If `known` (from a facilities CSV) is
    given it is used as-is and nothing is invented; otherwise hospitals, schools and care
    homes are synthetic and seeded."""
    from .physical import demand_by_category
    out = {n: "IndustrialCustomer" for n, j in wn.junctions()
           if demand_by_category(j).get("industrial", 0.0) > 0}
    if known is not None:
        unknown = sorted(set(known) - set(wn.junction_name_list))
        if unknown:
            raise ValueError(f"facilities CSV names junctions not in the model: {unknown[:5]}")
        out.update(known)
        return out
    rng = np.random.default_rng(cfg.seed + 7)
    cands = sorted(n for n, j in wn.junctions() if total_demand(j) > 0 and n not in out)
    k = min(cfg.n_critical_facilities, len(cands))
    picks = rng.choice(cands, size=k, replace=False) if k else []
    kinds = ["Hospital", "Hospital", "School", "School", "School", "CareHome", "CareHome", "CareHome"]
    out.update({str(n): kinds[i % len(kinds)] for i, n in enumerate(picks)})
    return out


def apply_likelihood_scores(df: pd.DataFrame, scores: dict[str, float], source: str) -> pd.DataFrame:
    """Replace the Weibull likelihood with an external model's scores (e.g. a utility's
    failure-prediction model), keeping consequence, policy and traceability unchanged.

    Scores are mapped onto the likelihood bands by rank, so that each band keeps exactly
    the number of pipes the Weibull model assigns to it. Decisions then differ only in
    *which* pipes a model ranks as likely to fail, not in how its scores are calibrated.
    """
    out = df.copy()
    missing = set(out.index) - set(scores)
    if missing:
        raise ValueError(f"likelihood scores missing for {len(missing)} pipes, e.g. {sorted(missing)[:3]}")
    s = pd.Series({p: float(scores[p]) for p in out.index})
    counts = out.pof_band.value_counts()
    order = s.sort_values(kind="mergesort").index           # lowest likelihood first
    bands, k = {}, 0
    for b in ["Low", "Medium", "High", "VeryHigh"]:
        for p in order[k:k + int(counts.get(b, 0))]:
            bands[p] = b
        k += int(counts.get(b, 0))
    out["pof_band"] = pd.Series(bands)
    out["pof_level"] = out.pof_band.map(LEVEL)
    out["risk_score"] = out.pof_level * out.consequence_level
    out["overall_risk"] = [overall_category(p, c) for p, c in zip(out.pof_level, out.consequence_level)]
    pct = s.rank(pct=True)
    out["priority"] = out.risk_score + pct * np.maximum(out.hydraulic_score, out.service_score)
    out["likelihood_score"] = s
    out["likelihood_source"] = source
    return out


def build_risk_layer(wn: wntr.network.WaterNetworkModel, kg: KnowledgeGraph,
                     risk: pd.DataFrame, nodes: pd.DataFrame, facilities: dict[str, str]) -> None:
    for b in ["Low", "Medium", "High", "VeryHigh"]:
        kg.add_node(f"FP:{b}", "FailureProbability", band=b, level=LEVEL[b])
    for b in CRIT_BANDS:
        kg.add_node(f"HC:{b}", "HydraulicCriticality", band=b, level=LEVEL[b])
        kg.add_node(f"SC:{b}", "ServiceCriticality", band=b, level=LEVEL[b])
    for i, b in enumerate(OVERALL, start=1):
        kg.add_node(f"OR:{b}", "OverallRisk", band=b, level=i)
    for b in ("Monitored", "Unmonitored"):
        kg.add_node(f"MS:{b}", "MonitoringStatus", band=b)

    for n, row in nodes.iterrows():
        if n in wn.junction_name_list:
            kg.node(node_uid(wn, n)).update(minPressure=float(row.min_pressure),
                                            meanPressure=float(row.mean_pressure))
    for n, kind in facilities.items():
        fid = kg.add_node(f"FAC:{n}", "CriticalFacility", kind, facilityType=kind, name=f"{kind} @ {n}")
        kg.add_edge(node_uid(wn, n), "SERVES_FACILITY", fid)

    for name, r in risk.iterrows():
        pid = link_uid(wn, name)
        rp = kg.add_node(f"RP:{name}", "RiskProfile", failureProbability=round(r.pof, 4),
                         failureProbabilityPerKm=round(r.pof_km, 4),
                         betweenness=round(r.betweenness, 5), meanFlow=round(r.mean_flow, 3),
                         demandServed=round(r.demand_served, 3),
                         minEndPressure=round(min(r.min_end_pressure, 9999.0), 2),
                         hydraulicScore=round(r.hydraulic_score, 3),
                         serviceScore=round(r.service_score, 3),
                         riskScore=int(r.risk_score), priority=round(float(r.priority), 4),
                         likelihoodSource=str(r.get("likelihood_source", "weibull")),
                         likelihoodScore=round(float(r.get("likelihood_score", r.pof_km)), 6))
        kg.add_edge(pid, "HAS_RISK_PROFILE", rp)
        kg.add_edge(rp, "HAS_FAILURE_PROBABILITY", f"FP:{r.pof_band}")
        kg.add_edge(rp, "HAS_HYDRAULIC_CRITICALITY", f"HC:{r.hydraulic_band}")
        kg.add_edge(rp, "HAS_SERVICE_CRITICALITY", f"SC:{r.service_band}")
        kg.add_edge(rp, "HAS_OVERALL_RISK", f"OR:{r.overall_risk}")
        kg.add_edge(rp, "HAS_MONITORING_STATUS", "MS:Monitored" if r.monitored else "MS:Unmonitored")
