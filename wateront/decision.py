"""Decision Layer (Section 3.4).

Guideline texts below are PARAPHRASED STAND-INS for clauses of a utility policy
built on AWWA M28, the UKWIR asset health framework, ISO 55000 and ISO 24510.
They are not quotations from those (copyrighted) documents. Swap in your utility's
own policy text via `load_guidelines` (JSON list with the same keys).

The policy rules are evaluated over graph facts (asset record + risk profile),
never by the LLM; each DecisionRecord is linked to the guideline nodes that
authorise it through SUPPORTED_BY, and to its evidence through BASED_ON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import Config
from .kg import KnowledgeGraph

ACTIONS = ["Replacement", "Rehabilitation", "Inspection", "Monitor"]
ACTION_RANK = {a: i for i, a in enumerate(ACTIONS)}

GUIDELINES = [
    dict(id="G-REPL-01", framework="AWWA M28", topic="replacement",
         title="Replacement at end of design life under intolerable risk",
         text="Mains whose service age exceeds the material-specific design life and whose overall "
              "risk is classified intolerable shall be programmed for replacement; renewal by "
              "rehabilitation is not appropriate once the structural design life is exhausted."),
    dict(id="G-RISK-01", framework="ISO 55000", topic="risk",
         title="Likelihood-consequence risk matrix",
         text="Asset intervention priority is set by combining the likelihood of failure with the "
              "consequence of failure in a four-band risk matrix: negligible, tolerable, significant "
              "and intolerable. Intolerable risks require action in the current planning period."),
    dict(id="G-REHAB-01", framework="UKWIR", topic="rehabilitation",
         title="Rehabilitation of deteriorated mains",
         text="Where condition assessment grades a main as poor or very poor (grade 4 or 5) and the "
              "overall risk is significant or intolerable, structural or semi-structural "
              "rehabilitation such as relining should be considered before replacement."),
    dict(id="G-INSP-01", framework="Utility policy", topic="inspection",
         title="Risk-based inspection intervals",
         text="Each main shall be inspected at an interval set by its overall risk category: 4 years "
              "for intolerable, 8 years for significant, 15 years for tolerable and 20 years for "
              "negligible risk. An exceeded interval triggers a scheduled inspection."),
    dict(id="G-INSP-02", framework="UKWIR", topic="inspection",
         title="Condition assessment before committing to renewal",
         text="Mains with intolerable risk that remain within design life and are not yet graded "
              "poor shall receive an urgent condition assessment to confirm the renewal option."),
    dict(id="G-SERV-01", framework="ISO 24510", topic="service",
         title="Critical customers and supply continuity",
         text="Mains supplying hospitals, schools and care homes carry elevated consequence of failure; "
              "interventions on these mains require supply continuity planning and customer notification."),
    dict(id="G-MON-01", framework="Utility policy", topic="monitoring",
         title="Continued monitoring of low-risk mains",
         text="Mains with negligible or tolerable risk and no exceeded inspection interval remain in "
              "routine monitoring through burst records and network telemetry."),
    dict(id="G-MON-02", framework="Utility policy", topic="inspection",
         title="Monitoring coverage of critical mains",
         text="Hydraulically critical mains of significant or intolerable risk that are not observed "
              "by any pressure sensor within two pipe segments shall be inspected and assessed for "
              "additional pressure monitoring, since failures there may go undetected."),
    # Distractors: relevant to water networks, not to pipe renewal decisions
    dict(id="G-VALV-01", framework="AWWA", topic="valves",
         title="Valve exercising programme",
         text="Isolation valves shall be exercised on a regular cycle so that sections of main can "
              "be isolated quickly during bursts and planned works."),
    dict(id="G-PUMP-01", framework="Utility policy", topic="pumps",
         title="Pump efficiency review",
         text="Pumps whose wire-to-water efficiency falls well below the manufacturer curve shall be "
              "reviewed for refurbishment or replacement of the impeller."),
    dict(id="G-PRES-01", framework="Utility policy", topic="pressure",
         title="Pressure management",
         text="Pressure management zones should reduce excess night pressure to lower leakage and "
              "burst frequency while maintaining the minimum service pressure."),
    dict(id="G-TANK-01", framework="Utility policy", topic="storage",
         title="Service reservoir cleaning",
         text="Service reservoirs and tanks shall be drained, inspected and cleaned on a fixed cycle "
              "to protect water quality."),
]


def load_guidelines(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


@dataclass
class Decision:
    pipe: str
    action: str
    guidelines: list[str]
    rule: str
    facts: dict = field(default_factory=dict)


def apply_policy(f: dict, cfg: Config) -> Decision:
    """Pure policy function over facts; reused as the evaluation reference."""
    risk, age, life, grade = f["overall"], f["age"], f["design_life"], f["condition_grade"]
    overdue = (cfg.reference_year - f["last_inspection_year"]) > cfg.inspection_interval[risk]
    extra = ["G-SERV-01"] if f.get("facilities") else []
    if risk == "Intolerable" and age > life:
        return Decision(f["pipe"], "Replacement", ["G-REPL-01", "G-RISK-01", *extra], "R1", f)
    if risk in ("Significant", "Intolerable") and grade >= 4:
        return Decision(f["pipe"], "Rehabilitation", ["G-REHAB-01", "G-RISK-01", *extra], "R2", f)
    if risk == "Intolerable":
        return Decision(f["pipe"], "Inspection", ["G-INSP-02", "G-RISK-01", *extra], "R3", f)
    if overdue:
        return Decision(f["pipe"], "Inspection", ["G-INSP-01", *extra], "R4", f)
    if (f.get("monitored") is False and risk in ("Significant", "Intolerable")
            and f["hydraulic"] in ("High", "Critical")):
        return Decision(f["pipe"], "Inspection", ["G-MON-02", *extra], "R5", f)
    return Decision(f["pipe"], "Monitor", ["G-MON-01"], "R0", f)


def pipe_facts(kg: KnowledgeGraph, pipe_uid: str) -> dict:
    ar = kg.node(kg.one(pipe_uid, "HAS_ASSET_RECORD"))
    rp_id = kg.one(pipe_uid, "HAS_RISK_PROFILE")
    rp = kg.node(rp_id)
    band = lambda rel: kg.node(kg.one(rp_id, rel))["band"]
    facilities = []
    for end in kg.out(pipe_uid, "STARTS_AT") + kg.out(pipe_uid, "ENDS_AT"):
        facilities += [kg.node(f)["facilityType"] for f in kg.out(end, "SERVES_FACILITY")]
    ms = kg.one(rp_id, "HAS_MONITORING_STATUS")
    zone = kg.one(pipe_uid, "IN_ZONE")
    return dict(pipe=kg.node(pipe_uid)["name"], uid=pipe_uid, material=ar["material"],
                material_source=ar.get("materialSource", "prior"),
                monitored=(kg.node(ms)["band"] == "Monitored") if ms else None,
                zone=kg.node(zone)["name"] if zone else None,
                age=ar["age"], design_life=ar["designLife"], condition_grade=ar["conditionGrade"],
                last_inspection_year=ar["lastInspectionYear"], failure_count=ar["failureCount"],
                pof=rp["failureProbability"], pof_km=rp.get("failureProbabilityPerKm", rp["failureProbability"]),
                likelihood_source=rp.get("likelihoodSource", "weibull"),
                pof_band=band("HAS_FAILURE_PROBABILITY"),
                hydraulic=band("HAS_HYDRAULIC_CRITICALITY"), service=band("HAS_SERVICE_CRITICALITY"),
                overall=band("HAS_OVERALL_RISK"), priority=rp["priority"],
                min_end_pressure=rp["minEndPressure"], facilities=sorted(set(facilities)),
                diameter=kg.node(pipe_uid)["diameter"], length=kg.node(pipe_uid)["length"])


def build_decision_layer(kg: KnowledgeGraph, cfg: Config,
                         guidelines: list[dict] | None = None) -> list[Decision]:
    for a in ACTIONS:
        kg.add_node(f"MA:{a}", "MaintenanceAction", a, actionType=a)
    for g in guidelines or GUIDELINES:
        kg.add_node(g["id"], "EngineeringGuideline", **g)
    decisions = []
    for pid in kg.nodes("Pipe"):
        d = apply_policy(pipe_facts(kg, pid), cfg)
        did = kg.add_node(f"DR:{d.pipe}", "DecisionRecord", rule=d.rule, action=d.action,
                          priority=d.facts["priority"], generatedBy="WaterOnt policy rules v1")
        kg.add_edge(did, "CONCERNS", pid)
        kg.add_edge(did, "RECOMMENDS", f"MA:{d.action}")
        kg.add_edge(did, "BASED_ON", kg.one(pid, "HAS_ASSET_RECORD"))
        kg.add_edge(did, "BASED_ON", kg.one(pid, "HAS_RISK_PROFILE"))
        for gid in d.guidelines:
            kg.add_edge(did, "SUPPORTED_BY", gid)
        decisions.append(d)
    return decisions
