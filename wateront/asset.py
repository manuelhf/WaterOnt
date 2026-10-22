"""Asset Layer (Section 3.2).

Benchmark networks carry no utility asset data, so attributes are generated from
literature-shaped priors (see `Config`). Everything is seeded for reproducibility.
Replace `synthesise_asset_table` with a loader for real asset registers via
`load_asset_table` (CSV with the same columns) to run on utility data.

Generation logic
* Install year follows network development: pipes topologically closer to the
  sources are older (an "urban core first" growth assumption), plus noise.
* Material is drawn from the install-era chronology, unless the hydraulic model
  itself carries evidence (`Config.use_model_hints`): a diameter on the metric
  plastic OD series (63, 75, 110, 160, 225 mm ...) implies PE/PVC laid after ~1985,
  and a low Hazen-Williams C implies an older, rougher main. Each record stores
  `materialSource` so the provenance of every attribute is queryable.
* Past failures are Poisson draws from the Weibull hazard over the history window.
* Condition grade (UKWIR-style 1 = very good ... 5 = very poor) rises with
  age / design life and past failures.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import wntr

from .config import Config
from .kg import KnowledgeGraph
from .physical import link_uid

ASSET_COLUMNS = ["pipe", "install_year", "material", "lining", "joint", "soil_corrosivity",
                 "condition_grade", "last_inspection_year", "failure_years", "failure_modes",
                 "material_source"]


def weibull_cum_hazard(t: np.ndarray | float, beta: float, eta: float):
    return (np.asarray(t, dtype=float) / eta) ** beta


def _draw(rng: np.random.Generator, dist: dict) -> str:
    keys = list(dist)
    p = np.array([dist[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


def synthesise_asset_table(wn: wntr.network.WaterNetworkModel, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    G = wn.to_graph().to_undirected()
    sources = list(wn.reservoir_name_list) + list(wn.tank_name_list)
    dist = nx.multi_source_dijkstra_path_length(G, sources) if sources else {}
    maxd = max(dist.values()) if dist else 1.0
    # Percentile rank of distance-from-source -> uniform spread of install eras
    ranked = pd.Series(dist, dtype=float).rank(pct=True).to_dict() if dist else {}

    rows = []
    span = cfg.newest_install_year - cfg.oldest_install_year
    for name, pipe in wn.pipes():
        d = 0.5 * (ranked.get(pipe.start_node_name, 1.0) + ranked.get(pipe.end_node_name, 1.0))
        year = int(np.clip(cfg.oldest_install_year + span * d + rng.normal(0, 8),
                           cfg.oldest_install_year, cfg.newest_install_year))
        source = "prior"
        dmm = round(pipe.diameter * 1000)
        hw = str(wn.options.hydraulic.headloss).upper().startswith("H")
        if cfg.use_model_hints and dmm in cfg.plastic_od_series_mm:
            year = int(np.clip(max(year, 1985 + rng.integers(0, 30)), 1985, cfg.newest_install_year))
            material, source = _draw(rng, {"HDPE": 0.55, "PVC": 0.45}), f"model-hint:plastic-OD-{dmm}mm"
        else:
            if cfg.use_model_hints and hw and pipe.roughness <= cfg.old_main_max_hw_c:
                year = int(np.clip(year - 25, cfg.oldest_install_year, cfg.newest_install_year))
                source = f"model-hint:HW-C{pipe.roughness:g}"
            material = next(_draw(rng, mix) for lo, hi, mix in cfg.material_chronology if lo <= year < hi)
        beta, eta = cfg.weibull[material]
        soil = "Medium"  # paper: defaulted to medium where geospatial data are unavailable
        age = cfg.reference_year - year
        t0 = max(age - cfg.history_window_years, 0)
        km = max(pipe.length / 1000.0, 0.05)
        lam = (weibull_cum_hazard(age, beta, eta) - weibull_cum_hazard(t0, beta, eta)) \
            * km * cfg.soil_corrosivity_factor[soil] * cfg.hazard_scale
        n_fail = int(rng.poisson(lam))
        fyears = sorted(int(rng.integers(cfg.reference_year - min(age, cfg.history_window_years),
                                         cfg.reference_year)) for _ in range(n_fail))
        fmodes = [_draw(rng, cfg.failure_modes[material]) for _ in range(n_fail)]
        wear = age / cfg.design_life[material]
        grade = int(np.clip(round(1 + 2.6 * wear + 0.5 * n_fail + rng.normal(0, 0.5)), 1, 5))
        last_insp = int(cfg.reference_year - rng.integers(1, 22))
        rows.append(dict(pipe=name, install_year=year, material=material,
                         lining=cfg.lining[material], joint=cfg.joint[material],
                         soil_corrosivity=soil, condition_grade=grade,
                         last_inspection_year=max(last_insp, year),
                         failure_years=fyears, failure_modes=fmodes, material_source=source))
    return pd.DataFrame(rows, columns=ASSET_COLUMNS)


REGISTER_INT = ("install_year", "condition_grade", "last_inspection_year")
REGISTER_TEXT = ("material", "lining", "joint", "soil_corrosivity")


def load_asset_overrides(path: str, cfg: Config | None = None) -> dict[str, dict]:
    """Read an asset register as per-pipe overrides.

    Any subset of the columns in `data/asset_register_template.csv` may be present, and
    any cell may be blank: blank means "not known, keep the current value". For
    `failure_years` write years separated by ';', or `none` for a pipe with no recorded
    failures. Raises ValueError listing every invalid cell, so a register is never
    half-applied."""
    cfg = cfg or Config()
    df = pd.read_csv(path, dtype=str, keep_default_na=False).rename(columns=str.strip)
    if "pipe" not in df.columns:
        raise ValueError(f"{path}: needs a 'pipe' column")
    out, errors = {}, []
    for n, row in enumerate(df.to_dict("records"), start=2):
        pipe = row["pipe"].strip()
        if not pipe:
            continue
        rec: dict = {}
        for col in REGISTER_INT:
            v = row.get(col, "").strip()
            if v:
                try:
                    rec[col] = int(float(v))
                except ValueError:
                    errors.append(f"line {n}, {col}: '{v}' is not a number")
        for col in REGISTER_TEXT:
            v = row.get(col, "").strip()
            if v:
                rec[col] = v
        if "material" in rec and rec["material"] not in cfg.weibull:
            errors.append(f"line {n}, material: '{rec['material']}' is not one of {sorted(cfg.weibull)}")
        if "soil_corrosivity" in rec and rec["soil_corrosivity"] not in cfg.soil_corrosivity_factor:
            errors.append(f"line {n}, soil_corrosivity: use one of {sorted(cfg.soil_corrosivity_factor)}")
        if "condition_grade" in rec and not 1 <= rec["condition_grade"] <= 5:
            errors.append(f"line {n}, condition_grade: must be 1 to 5")
        fy = row.get("failure_years", "").strip()
        if fy:
            if fy.lower() in ("none", "0"):
                rec["failure_years"], rec["failure_modes"] = [], []
            else:
                try:
                    years = [int(float(x)) for x in fy.split(";") if x.strip()]
                except ValueError:
                    errors.append(f"line {n}, failure_years: use years separated by ';', or 'none'")
                    years = []
                modes = [m.strip() for m in row.get("failure_modes", "").split(";") if m.strip()]
                rec["failure_years"] = years
                rec["failure_modes"] = (modes + ["Burst"] * len(years))[: len(years)]
        if "material" in rec:
            rec["material_source"] = "register"
        out[pipe] = rec
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors[:20]) + (" ..." if len(errors) > 20 else ""))
    return out


def apply_asset_overrides(assets: pd.DataFrame, overrides: dict[str, dict], cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    rows = assets.to_dict("records")
    for row in rows:
        rec = overrides.get(row["pipe"])
        if not rec:
            continue
        row.update(rec)
        if "material" in rec:          # derived attributes follow the material unless given
            if "lining" not in rec:
                row["lining"] = cfg.lining[rec["material"]]
            if "joint" not in rec:
                row["joint"] = cfg.joint[rec["material"]]
        if row["last_inspection_year"] < row["install_year"]:
            row["last_inspection_year"] = row["install_year"]
    return pd.DataFrame(rows, columns=assets.columns)


def load_asset_table(path: str) -> pd.DataFrame:
    """Backwards-compatible: the register as a DataFrame (only rows/columns provided)."""
    recs = load_asset_overrides(path)
    return pd.DataFrame([{"pipe": p, **r} for p, r in recs.items()])


def load_facilities(path: str) -> dict[str, str]:
    """Critical customers: CSV with columns `junction,type` (e.g. Hospital, School,
    CareHome, IndustrialCustomer). Hospitals raise service criticality to Critical,
    any other listed customer to at least High."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"junction", "type"} <= set(df.columns):
        raise ValueError(f"{path}: needs columns 'junction' and 'type'")
    return {r["junction"].strip(): r["type"].strip() for r in df.to_dict("records") if r["junction"].strip()}


def load_failure_events(path: str) -> dict[str, list[tuple[int, str]]]:
    """Observed failures, e.g. converted from the BattLeDIM leakage reports.
    CSV columns: pipe, and either `year` or `date`/`start` (ISO), optional `mode`
    (defaults to Burst; leaks without a recorded mode are treated as bursts)."""
    df = pd.read_csv(path, dtype={"pipe": str})
    col = next((c for c in ("year", "date", "start", "start_time") if c in df.columns), None)
    if col is None:
        raise ValueError(f"{path}: need a 'year' or 'date'/'start' column")
    years = df[col] if col == "year" else pd.to_datetime(df[col]).dt.year
    out: dict[str, list[tuple[int, str]]] = {}
    for pipe, yr, mode in zip(df["pipe"], years, df.get("mode", pd.Series(["Burst"] * len(df)))):
        out.setdefault(str(pipe).strip(), []).append((int(yr), str(mode) if isinstance(mode, str) else "Burst"))
    return out


def apply_failure_history(assets: pd.DataFrame, events: dict[str, list[tuple[int, str]]]) -> pd.DataFrame:
    """Replace the synthetic history with observed events (treated as the complete record)."""
    df = assets.copy()
    df["failure_years"] = [sorted(y for y, _ in events.get(p, [])) for p in df["pipe"]]
    df["failure_modes"] = [[m for _, m in sorted(events.get(p, []))] for p in df["pipe"]]
    return df


def build_asset_layer(wn: wntr.network.WaterNetworkModel, kg: KnowledgeGraph,
                      assets: pd.DataFrame, cfg: Config) -> None:
    for r in assets.itertuples(index=False):
        pid = link_uid(wn, r.pipe)
        age = cfg.reference_year - int(r.install_year)
        aid = kg.add_node(f"AR:{r.pipe}", "AssetRecord", material=r.material,
                          nominalDiameter=kg.node(pid)["diameter"],
                          installationYear=int(r.install_year), age=age,
                          designLife=cfg.design_life[r.material],
                          liningType=r.lining, jointType=r.joint,
                          soilCorrosivity=r.soil_corrosivity,
                          conditionGrade=int(r.condition_grade),
                          lastInspectionYear=int(r.last_inspection_year),
                          failureCount=len(r.failure_years), recordVersion=1,
                          materialSource=getattr(r, "material_source", "prior"))
        kg.add_edge(pid, "HAS_ASSET_RECORD", aid)
        for k, (yr, mode) in enumerate(zip(r.failure_years, r.failure_modes)):
            fid = kg.add_node(f"FE:{r.pipe}:{k}", "FailureEvent", mode, year=int(yr), failureMode=mode)
            kg.add_edge(pid, "EXPERIENCED_FAILURE", fid)
        iid = kg.add_node(f"IR:{r.pipe}", "InspectionRecord",
                          year=int(r.last_inspection_year), conditionGrade=int(r.condition_grade))
        kg.add_edge(aid, "HAS_INSPECTION", iid)


def simulate_future_failures(assets: pd.DataFrame, wn: wntr.network.WaterNetworkModel,
                             cfg: Config, seed_offset: int = 1) -> pd.Series:
    """Synthetic *outcome* labels (failure within the planning horizon). Used only to
    train/evaluate the ML baseline; never written to the knowledge graph."""
    rng = np.random.default_rng(cfg.seed + seed_offset)
    out = {}
    for r in assets.itertuples(index=False):
        beta, eta = cfg.weibull[r.material]
        age = cfg.reference_year - r.install_year
        km = max(wn.get_link(r.pipe).length / 1000.0, 0.05)
        lam = (weibull_cum_hazard(age + cfg.planning_horizon_years, beta, eta)
               - weibull_cum_hazard(age, beta, eta)) * km * cfg.hazard_scale \
            * np.exp(cfg.failure_count_hazard_coef * len(r.failure_years)) * rng.lognormal(0, 0.4)
        out[r.pipe] = int(rng.random() < 1 - np.exp(-lam))
    return pd.Series(out, name="failed_in_horizon")
