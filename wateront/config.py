"""Central configuration. Every literature-derived prior lives here so it can be
audited, replaced by utility data, or varied in sensitivity analyses.

IMPORTANT: the numeric priors below are *illustrative placeholders* shaped after the
cited frameworks (AWWA M28, UKWIR, Kettler & Goulter 1985). They are not quotations
from those documents and must be calibrated before any operational use.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 42
    reference_year: int = 2026
    planning_horizon_years: int = 10        # horizon for conditional failure probability
    history_window_years: int = 20          # synthetic failure-history window
    min_service_pressure_m: float = 20.0    # below this at an end node: consequence raised one band
    pressure_margin_m: float = 10.0         # "low pressure" query = min pressure < service minimum + margin
    sim_duration_h: float | None = None     # None -> model's own duration, capped at 168 h
    n_critical_facilities: int = 8          # synthetic hospitals/schools/care homes
    oldest_install_year: int = 1930
    newest_install_year: int = 2020

    # Install-year -> material mix (AWWA-style adoption chronology, illustrative)
    material_chronology: list = field(default_factory=lambda: [
        (1900, 1950, {"CI": 0.85, "Steel": 0.15}),
        (1950, 1970, {"CI": 0.60, "AC": 0.25, "Steel": 0.15}),
        (1970, 1990, {"DI": 0.50, "AC": 0.30, "PVC": 0.20}),
        (1990, 2100, {"PVC": 0.45, "HDPE": 0.30, "DI": 0.25}),
    ])
    # Weibull (shape beta, scale eta [years]) per material, illustrative
    weibull: dict = field(default_factory=lambda: {
        "CI": (2.2, 95.0), "Steel": (2.0, 85.0), "AC": (2.4, 80.0),
        "DI": (2.0, 120.0), "PVC": (1.8, 130.0), "HDPE": (1.8, 140.0),
    })
    # Material-specific design life [years] (stand-in for AWWA M28 values)
    design_life: dict = field(default_factory=lambda: {
        "CI": 80, "Steel": 70, "AC": 60, "DI": 100, "PVC": 100, "HDPE": 100,
    })
    lining: dict = field(default_factory=lambda: {
        "CI": "Unlined", "Steel": "Bitumen", "AC": "None", "DI": "CementMortar",
        "PVC": "None", "HDPE": "None",
    })
    joint: dict = field(default_factory=lambda: {
        "CI": "LeadCaulked", "Steel": "Welded", "AC": "Coupling", "DI": "PushOn",
        "PVC": "PushOn", "HDPE": "Fusion",
    })
    failure_modes: dict = field(default_factory=lambda: {
        "CI": {"Burst": 0.5, "Corrosion": 0.3, "JointFailure": 0.15, "ThirdPartyDamage": 0.05},
        "Steel": {"Corrosion": 0.6, "Burst": 0.25, "JointFailure": 0.1, "ThirdPartyDamage": 0.05},
        "AC": {"Burst": 0.6, "JointFailure": 0.3, "ThirdPartyDamage": 0.1},
        "DI": {"Corrosion": 0.5, "JointFailure": 0.3, "ThirdPartyDamage": 0.2},
        "PVC": {"JointFailure": 0.5, "ThirdPartyDamage": 0.35, "Burst": 0.15},
        "HDPE": {"ThirdPartyDamage": 0.6, "JointFailure": 0.4},
    })
    soil_corrosivity_factor: dict = field(default_factory=lambda: {"Low": 0.8, "Medium": 1.0, "High": 1.4})
    hazard_scale: float = 3.0                # expected failures per km per unit cumulative hazard
    failure_count_hazard_coef: float = 0.35  # proportional-hazards uplift per past failure

    # Model hints: infer material/era from the hydraulic model itself (see asset.py)
    use_model_hints: bool = True
    plastic_od_series_mm: tuple = (63, 75, 90, 110, 125, 140, 160, 180, 225, 250, 280, 315, 355, 400)
    old_main_max_hw_c: float = 120.0        # Hazen-Williams C at/below this suggests an older main

    # Banding thresholds. PoF is banded PER KM so that segmentation of the hydraulic model
    # (Net3: km-long pipes; L-TOWN: ~50 m segments) does not decide the risk class.
    pof_basis: str = "per_km"               # "per_km" or "per_segment"
    pof_bands: tuple = ((0.20, "Low"), (0.40, "Medium"), (0.52, "High"), (1.01, "VeryHigh"))
    criticality_percentiles: tuple = ((0.40, "Low"), (0.70, "Medium"), (0.90, "High"), (1.01, "Critical"))
    # Inspection interval [years] by overall risk (utility policy stand-in)
    inspection_interval: dict = field(default_factory=lambda: {
        "Negligible": 20, "Tolerable": 15, "Significant": 8, "Intolerable": 4,
    })

    # LLM (optional)
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 1500
    llm_max_tokens_flat: int = 8000   # flat-text RAG writes one JSON decision per pipe (up to 40)
    llm_max_tokens_plan: int = 1500   # query plan as JSON
