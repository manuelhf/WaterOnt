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
                 "material_source", "provided"]
# attributes a user's files can supply; "provided" lists those that did, per pipe (the rest is synthetic)
PROVIDED_KEYS = ("install_year", "material", "condition_grade", "last_inspection_year", "failure_years",
                 "soil_corrosivity")


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
                         failure_years=fyears, failure_modes=fmodes, material_source=source, provided=[]))
    return pd.DataFrame(rows, columns=ASSET_COLUMNS)


REGISTER_INT = ("install_year", "condition_grade", "last_inspection_year")
REGISTER_TEXT = ("material", "lining", "joint", "soil_corrosivity")


def load_asset_overrides(path: str, cfg: Config | None = None, report: dict | None = None) -> dict[str, dict]:
    """Read an asset register or an inspection log as per-pipe overrides.

    Any subset of the columns in `data/asset_register_template.csv` may be present, under
    their template names or common alternatives (see wateront.tables), and any cell may be
    blank: blank means "not known, keep the current value". Years may be dates. For
    `failure_years` write years separated by ';', or `none` for no recorded failures.

    A pipe may appear on several rows, as in an inspection log with one row per survey: rows
    are combined in order of inspection date, so the most recent inspection (and its
    condition grade) wins; failure years from all rows are combined.

    Raises ValueError listing every invalid cell, so a register is never half-applied.
    If `report` is a dict, it is filled with what was read (rows, pipes, combined rows,
    separator, columns used and ignored)."""
    from .tables import material_of, number_of, read_table, soil_of, split_list, year_of
    cfg = cfg or Config()
    head, rows, delim = read_table(path)
    if "pipe" not in head:
        raise ValueError(f"{path}: needs a pipe column (for example 'pipe' or 'Pipe ID'); found {head}")
    known = {"pipe", *REGISTER_INT, *REGISTER_TEXT, "failure_years", "failure_modes"}
    col = {h: k for k, h in enumerate(head) if h in known and h not in {}}
    get = lambda r, c: (r[col[c]].strip() if c in col and col[c] < len(r) else "")
    ref, errors, groups = cfg.reference_year, [], {}
    for n, r in enumerate(rows, start=2):
        pipe = get(r, "pipe")
        if not pipe:
            continue
        rec: dict = {}
        for c, lo, hi in (("install_year", 1800, ref), ("last_inspection_year", 1800, ref)):
            v = get(r, c)
            if v:
                y = year_of(v)
                if y is None:
                    errors.append(f"line {n}, {c}: '{v}' is not a year or a date")
                elif not lo <= y <= hi:
                    errors.append(f"line {n}, {c}: {y} is outside {lo}-{hi}")
                else:
                    rec[c] = y
        v = get(r, "condition_grade")
        if v:
            x = number_of(v)
            if x is None:
                errors.append(f"line {n}, condition_grade: '{v}' is not a number")
            elif not 1 <= x <= 5:
                errors.append(f"line {n}, condition_grade: must be 1 to 5")
            else:
                rec["condition_grade"] = int(x)
        v = get(r, "material")
        if v:
            m = material_of(v)
            if m is None or m not in cfg.weibull:
                errors.append(f"line {n}, material: '{v}' is not recognised (use {', '.join(sorted(cfg.weibull))} "
                              f"or a name such as 'cast iron')")
            else:
                rec["material"], rec["material_source"] = m, "register"
        v = get(r, "soil_corrosivity")
        if v:
            so = soil_of(v)
            if so is None or so not in cfg.soil_corrosivity_factor:
                errors.append(f"line {n}, soil_corrosivity: use one of {sorted(cfg.soil_corrosivity_factor)}")
            else:
                rec["soil_corrosivity"] = so
        for c in ("lining", "joint"):
            if v := get(r, c):
                rec[c] = v
        fy = get(r, "failure_years")
        if fy:
            if fy.lower() in ("none", "0", "-", "no", "nil"):
                rec["failure_years"], rec["failure_modes"] = [], []
            else:
                years = [year_of(x) for x in split_list(fy)]
                if any(y is None or not 1800 <= y <= ref for y in years):
                    errors.append(f"line {n}, failure_years: use years or dates separated by ';', or 'none'")
                else:
                    modes = split_list(get(r, "failure_modes"))
                    rec["failure_years"] = years
                    rec["failure_modes"] = (modes + ["Burst"] * len(years))[: len(years)]
        groups.setdefault(pipe, []).append(rec)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors[:20]) + (" ..." if len(errors) > 20 else ""))
    out, combined = {}, 0
    for pipe, recs in groups.items():
        combined += len(recs) - 1
        # oldest inspection first, rows without an inspection date before dated ones (stable)
        recs = sorted(recs, key=lambda x: x.get("last_inspection_year", -1))
        merged, fyears, fmodes, failures_given = {}, [], [], False
        for rec in recs:
            for k, v in rec.items():
                if k in ("failure_years", "failure_modes"):
                    continue
                merged[k] = v
            if "failure_years" in rec:
                failures_given = True
                for y, m in zip(rec["failure_years"], rec["failure_modes"]):
                    if y not in fyears:
                        fyears.append(y); fmodes.append(m)
        if failures_given:
            order = sorted(range(len(fyears)), key=lambda k: fyears[k])
            merged["failure_years"] = [fyears[k] for k in order]
            merged["failure_modes"] = [fmodes[k] for k in order]
        out[pipe] = merged
    if report is not None:
        report.update(rows=sum(len(v) for v in groups.values()), pipes=len(out), combined=combined,
                      separator={",": "comma", ";": "semicolon", "\t": "tab"}[delim],
                      columns_used=[h for h in head if h in known],
                      columns_ignored=[h for h in head if h not in known])
    return out


def apply_asset_overrides(assets: pd.DataFrame, overrides: dict[str, dict], cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    rows = assets.to_dict("records")
    for row in rows:
        rec = overrides.get(row["pipe"])
        if not rec:
            continue
        row.update(rec)
        row["provided"] = sorted(set(row.get("provided") or []) | {k for k in rec if k in PROVIDED_KEYS})
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
    CareHome, IndustrialCustomer), or alternatives such as `node`. Hospitals raise service
    criticality to Critical, any other listed customer to at least High."""
    from .tables import read_table
    head, rows, _ = read_table(path)
    if "element" not in head or "type" not in head:
        raise ValueError(f"{path}: needs a junction column (for example 'junction' or 'node') and a 'type' column")
    ce, ct = head.index("element"), head.index("type")
    return {r[ce].strip(): r[ct].strip() for r in rows if len(r) > max(ce, ct) and r[ce].strip()}


def load_failure_events(path: str) -> dict[str, list[tuple[int, str]]]:
    """Observed failures, e.g. converted from the BattLeDIM leakage reports. One row per
    failure: a pipe column, and a `year` or a date column (`date`, `start`, 'Failure date'...),
    optionally `mode` (defaults to Burst). Raises ValueError listing every bad row."""
    from .tables import read_table, year_of
    head, rows, _ = read_table(path)
    if "pipe" not in head:
        raise ValueError(f"{path}: needs a pipe column (for example 'pipe' or 'Pipe ID')")
    col = next((c for c in ("year", "date", "start", "start_time", "failure_years") if c in head), None)
    if col is None:
        raise ValueError(f"{path}: needs a 'year' or a date column")
    cp, cy = head.index("pipe"), head.index(col)
    cm = next((head.index(c) for c in ("mode", "failure_modes") if c in head), None)
    out: dict[str, list[tuple[int, str]]] = {}
    errors = []
    for n, r in enumerate(rows, start=2):
        pipe = r[cp].strip() if cp < len(r) else ""
        if not pipe:
            continue
        y = year_of(r[cy]) if cy < len(r) else None
        if y is None:
            errors.append(f"line {n}: '{r[cy] if cy < len(r) else ''}' is not a year or a date")
            continue
        mode = (r[cm].strip() if cm is not None and cm < len(r) else "") or "Burst"
        out.setdefault(pipe, []).append((y, mode))
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors[:20]))
    return out


def apply_failure_history(assets: pd.DataFrame, events: dict[str, list[tuple[int, str]]]) -> pd.DataFrame:
    """Replace the synthetic history with observed events (treated as the complete record)."""
    df = assets.copy()
    df["failure_years"] = [sorted(y for y, _ in events.get(p, [])) for p in df["pipe"]]
    df["failure_modes"] = [[m for _, m in sorted(events.get(p, []))] for p in df["pipe"]]
    df["provided"] = [sorted(set(v or []) | {"failure_years"}) for v in df["provided"]]   # complete record for every pipe
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
                          materialSource=getattr(r, "material_source", "prior"),
                          providedFields=",".join(getattr(r, "provided", None) or []))
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
