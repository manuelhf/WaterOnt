"""Physical Layer (Section 3.1, Appendix A).

Translates an EPANET model, parsed by WNTR, into WaterOnt classes and properties.
Links are reified as graph nodes so that upper layers can attach AssetRecords,
RiskProfiles and DecisionRecords to them without modifying the physical model.

Node ids are type-prefixed (J:, R:, T:, P:, PU:, V:, PAT:, C:, CTL:, ST:) because
EPANET allows a junction and a pipe to share the same name.
"""
from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx

import wntr
from wntr.network.base import Link as WntrLink
from wntr.network.controls import Rule as WntrRule

from .kg import KnowledgeGraph

VALVE_CLASS = {
    "PRV": "PressureReducingValve", "PSV": "PressureSustainingValve",
    "PBV": "PressureBreakingValve", "FCV": "FlowControlValve",
    "TCV": "ThrottleControlValve", "GPV": "GeneralPurposeValve",
}
CURVE_CLASS = {"HEAD": "PumpCurve", "EFFICIENCY": "EfficiencyCurve",
               "VOLUME": "VolumeCurve", "HEADLOSS": "HeadlossCurve"}


def demand_by_category(junction) -> dict[str, float]:
    """Base demand per category in L/s. WNTR's `junction.base_demand` returns only the
    FIRST demand entry, which silently drops e.g. L-TOWN's commercial and industrial
    demands; always aggregate over `demand_timeseries_list` instead."""
    out: dict[str, float] = {}
    for k, ts in enumerate(junction.demand_timeseries_list):
        cat = (ts.category or ts.pattern_name or f"cat{k}")
        cat = re.sub(r"^P-", "", str(cat)).strip().lower() or f"cat{k}"
        out[cat] = out.get(cat, 0.0) + float(ts.base_value) * 1000.0
    return out


def total_demand(junction) -> float:
    return sum(demand_by_category(junction).values())


def node_id(kind: str, name: str) -> str:
    return f"{kind}:{name}"


def link_uid(wn: wntr.network.WaterNetworkModel, name: str) -> str:
    link = wn.get_link(name)
    prefix = {"Pipe": "P", "HeadPump": "PU", "PowerPump": "PU"}.get(type(link).__name__, "V")
    return node_id(prefix, name)


def node_uid(wn: wntr.network.WaterNetworkModel, name: str) -> str:
    n = wn.get_node(name)
    return node_id({"Junction": "J", "Reservoir": "R", "Tank": "T"}[type(n).__name__], name)


def build_physical_layer(wn: wntr.network.WaterNetworkModel, kg: KnowledgeGraph,
                         network_id: str) -> str:
    """Populate the Physical Layer. SI units throughout (WNTR converts on read)."""
    net = node_id("NET", network_id)
    kg.add_node(net, "WaterNetwork", name=network_id,
                flowUnits=str(wn.options.hydraulic.inpfile_units),
                headlossFormula=str(wn.options.hydraulic.headloss),
                simDuration=wn.options.time.duration / 3600.0,
                hydraulicTimestep=float(wn.options.time.hydraulic_timestep))

    # Operational elements first, so physical elements can link to them
    for name, pat in wn.patterns():
        kg.add_node(node_id("PAT", name), "DemandPattern", "OperationalElement",
                    name=name, multipliers=[float(m) for m in pat.multipliers])
    for name, curve in wn.curves():
        cls = CURVE_CLASS.get(curve.curve_type, "Curve")
        kg.add_node(node_id("C", name), cls, "Curve", "OperationalElement", name=name,
                    curveType=curve.curve_type,
                    xValues=[float(x) for x, _ in curve.points],
                    yValues=[float(y) for _, y in curve.points])

    # Nodes
    for name, j in wn.junctions():
        by_cat = demand_by_category(j)
        jid = kg.add_node(node_id("J", name), "Junction", "Node", "WaterNetworkElement",
                          name=name, elevation=float(j.elevation),
                          baseDemand=sum(by_cat.values()),                     # L/s, all categories
                          emitterCoeff=float(j.emitter_coefficient or 0.0),
                          **{f"demand_{c}": v for c, v in by_cat.items()})
        kg.add_edge(net, "CONTAINS_NODE", jid)
        for ts in j.demand_timeseries_list:
            # only patterns that actually drive demand at this junction
            if ts.pattern_name and ts.base_value > 0 and kg.has(node_id("PAT", ts.pattern_name)):
                kg.add_edge(jid, "FOLLOWS_PATTERN", node_id("PAT", ts.pattern_name))
    for name, r in wn.reservoirs():
        rid = kg.add_node(node_id("R", name), "Reservoir", "Node", "WaterNetworkElement",
                          name=name, elevation=float(r.base_head), totalHead=float(r.base_head))
        kg.add_edge(net, "CONTAINS_NODE", rid)
    for name, t in wn.tanks():
        tid = kg.add_node(node_id("T", name), "Tank", "Node", "WaterNetworkElement",
                          name=name, elevation=float(t.elevation), initLevel=float(t.init_level),
                          minLevel=float(t.min_level), maxLevel=float(t.max_level),
                          nomDiameter=float(t.diameter), minVolume=float(t.min_vol))
        kg.add_edge(net, "CONTAINS_NODE", tid)
        if t.vol_curve_name:
            kg.add_edge(tid, "HAS_CURVE", node_id("C", t.vol_curve_name))

    # Links
    for name, link in wn.links():
        uid = link_uid(wn, name)
        common = dict(name=name)
        kind = type(link).__name__
        if kind == "Pipe":
            kg.add_node(uid, "Pipe", "Link", "WaterNetworkElement", **common,
                        length=float(link.length), diameter=float(link.diameter) * 1000.0,
                        roughness=float(link.roughness), minorLoss=float(link.minor_loss),
                        checkValve=bool(link.check_valve))
        elif kind in ("HeadPump", "PowerPump"):
            kg.add_node(uid, "Pump", "Link", "WaterNetworkElement", **common, pumpType=kind)
            if kind == "HeadPump" and link.pump_curve_name:
                kg.add_edge(uid, "HAS_CURVE", node_id("C", link.pump_curve_name))
            eff = getattr(link, "efficiency_curve", None)
            if eff is not None and kg.has(node_id("C", eff.name)):
                kg.add_edge(uid, "HAS_CURVE", node_id("C", eff.name))
        else:
            vtype = link.valve_type
            kg.add_node(uid, VALVE_CLASS.get(vtype, "Valve"), "Valve", "Link",
                        "WaterNetworkElement", **common, valveType=vtype,
                        diameter=float(link.diameter) * 1000.0, minorLoss=float(link.minor_loss),
                        valveSetting=float(link.initial_setting or 0.0))
            hl = getattr(link, "headloss_curve_name", None)
            if hl:
                kg.add_edge(uid, "HAS_CURVE", node_id("C", hl))
        kg.add_edge(uid, "STARTS_AT", node_uid(wn, link.start_node_name))
        kg.add_edge(uid, "ENDS_AT", node_uid(wn, link.end_node_name))
        kg.add_edge(net, "CONTAINS_LINK", uid)

        # Reified operational status (design decision 1 in Section 3.1)
        status = link.initial_status.name if hasattr(link.initial_status, "name") else str(link.initial_status)
        if kind == "Pipe" and link.check_valve:
            status = "CV"
        sid = kg.add_node(node_id("ST", name), "OperationalStatus", "OperationalElement",
                          value=status.capitalize() if status != "CV" else "CV",
                          reason="InitialCondition", source="[STATUS]/.inp")
        kg.add_edge(uid, "HAS_STATUS", sid)

    # Controls and rules
    for cname, ctl in wn.controls():
        cls = "Rule" if isinstance(ctl, WntrRule) else "SimpleControl"
        cid = kg.add_node(node_id("CTL", cname), cls, "OperationalElement",
                          name=cname, text=str(ctl))
        for obj in ctl.requires():
            if isinstance(obj, WntrLink):
                kg.add_edge(link_uid(wn, obj.name), "GOVERNED_BY", cid)
                # Links closed by a control get the control as the status reason
                st = node_id("ST", obj.name)
                if "CLOSED" in str(ctl).upper():
                    kg.node(st).setdefault("controlledBy", []).append(cname)
    return net


# --------------------------------------------------------------------------- sensors
SENSOR_PATTERNS = [  # comment text -> WaterOnt sensor class (SOSA/SSN-aligned, see ontology)
    (re.compile(r"PRESSURE\s+SENSOR", re.I), "PressureSensor", "pressure"),
    (re.compile(r"FLOW\s+SENSOR", re.I), "FlowSensor", "flowrate"),
    (re.compile(r"LEVEL\s+SENSOR", re.I), "LevelSensor", "level"),
    (re.compile(r"\bAMR\b", re.I), "AMRMeter", "consumption"),
]
_SECTIONS = {"JUNCTIONS", "RESERVOIRS", "TANKS", "PIPES", "PUMPS", "VALVES"}


def parse_inp_annotations(path: str) -> dict[str, list[tuple[str, str]]]:
    """Read element-level ';comment' annotations, which WNTR discards. L-TOWN uses them
    to mark instrumented elements (e.g. 'n1 ... ;AMR & PRESSURE SENSOR')."""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    section = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                section = line.strip("[]").upper()
                continue
            if section not in _SECTIONS or not line or line.startswith(";") or ";" not in line:
                continue
            data, comment = line.split(";", 1)
            parts = data.split()
            if not parts:
                continue
            for rx, cls, prop in SENSOR_PATTERNS:
                if rx.search(comment):
                    out[parts[0]].append((cls, prop))
    return out


SENSOR_ALIASES = {"pressure": "PressureSensor", "pressuresensor": "PressureSensor",
                  "flow": "FlowSensor", "flowsensor": "FlowSensor",
                  "level": "LevelSensor", "levelsensor": "LevelSensor",
                  "amr": "AMRMeter", "amrmeter": "AMRMeter", "meter": "AMRMeter"}
SENSOR_PROPERTY = {cls: prop for _, cls, prop in SENSOR_PATTERNS}


def load_sensors(path: str, wn: wntr.network.WaterNetworkModel) -> dict[str, list[tuple[str, str]]]:
    """Sensor list CSV with columns `element,type` (type: pressure, flow, level or amr, or
    the class names). The file is the complete sensor layout and replaces any sensors
    marked in the .inp. Raises ValueError listing every bad row."""
    from .tables import read_table
    head, rows, _ = read_table(path)
    if "element" not in head or "type" not in head:
        raise ValueError(f"{path}: needs an element column (for example 'element' or 'node') and a 'type' column")
    ce, ct = head.index("element"), head.index("type")
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    errors = []
    for n, r in enumerate(rows, start=2):
        elem = r[ce].strip() if ce < len(r) else ""
        typ = r[ct].strip() if ct < len(r) else ""
        if not elem:
            continue
        cls = SENSOR_ALIASES.get(typ.lower().replace(" ", "").replace("_", ""))
        if cls is None:
            errors.append(f"line {n}: type '{typ}' is not pressure, flow, level or amr")
        elif elem not in wn.node_name_list and elem not in wn.link_name_list:
            errors.append(f"line {n}: '{elem}' is not a node or link in the model")
        elif (cls, SENSOR_PROPERTY[cls]) not in out[elem]:
            out[elem].append((cls, SENSOR_PROPERTY[cls]))
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors[:20]))
    return out


def build_instrumentation(wn: wntr.network.WaterNetworkModel, kg: KnowledgeGraph, inp_path: str,
                          sensors: dict[str, list[tuple[str, str]]] | None = None) -> int:
    """Sensor nodes: (element)-[:OBSERVED_BY]->(:Sensor {observedProperty}). Uses the
    .inp annotations unless a sensor layout (from `load_sensors`) is given."""
    n = 0
    source = ".inp annotation" if sensors is None else "sensor list"
    for elem, sensors_here in (parse_inp_annotations(inp_path) if sensors is None else sensors).items():
        if elem in wn.node_name_list:
            target = node_uid(wn, elem)
        elif elem in wn.link_name_list:
            target = link_uid(wn, elem)
        else:
            continue
        for cls, prop in sensors_here:
            sid = kg.add_node(f"SEN:{cls}:{elem}", cls, "Sensor", name=f"{cls} @ {elem}",
                              observedProperty=prop, source=source)
            kg.add_edge(target, "OBSERVED_BY", sid)
            n += 1
    return n


def pipes_near_sensor(wn: wntr.network.WaterNetworkModel, kg: KnowledgeGraph,
                      sensor_cls: str = "PressureSensor", hops: int = 2) -> set[str]:
    """Pipe names whose end nodes lie within `hops` of a node carrying `sensor_cls`."""
    G = wn.to_graph().to_undirected()
    sensed = [kg.node(s)["name"].split(" @ ")[1] for s in kg.nodes(sensor_cls)]
    near_nodes = set()
    for n in sensed:
        if n in G:
            near_nodes |= set(nx.single_source_shortest_path_length(G, n, cutoff=hops))
    return {p for p, pipe in wn.pipes() if pipe.start_node_name in near_nodes or pipe.end_node_name in near_nodes}


# ----------------------------------------------------------------------------- zones
def apply_dma_zones(kg: KnowledgeGraph, wn: wntr.network.WaterNetworkModel, network_id: str,
                    dma_of_pipe: dict[str, str]) -> int:
    """User-defined DMAs (from a 'dma' column in the asset register) replace the automatic
    zone of the pipes they list; other pipes keep their automatic zone. Returns the number
    of pipes moved."""
    moved = 0
    for name, dma in dma_of_pipe.items():
        dma = str(dma).strip()
        if not dma or name not in wn.link_name_list:
            continue
        zid = f"Zone:DMA:{dma}"
        if not kg.has(zid):
            kg.add_node(zid, "PressureZone", "DistrictMeteredArea", name=dma, isSupplyZone=True, nJunctions=0,
                        totalDemand=0.0, containsStorage=False, zoneSource="register")
            kg.add_edge(node_id("NET", network_id), "HAS_ZONE", zid)
        pid = link_uid(wn, name)
        for z in kg.out(pid, "IN_ZONE"):
            kg.remove_edge(pid, "IN_ZONE", z)
        kg.add_edge(pid, "IN_ZONE", zid)
        moved += 1
    for zid in kg.nodes("DistrictMeteredArea"):      # junctions reached by a DMA's pipes, for display
        ends = {e for p in kg.inc(zid, "IN_ZONE") for e in kg.out(p, "STARTS_AT") + kg.out(p, "ENDS_AT")}
        kg.node(zid)["nJunctions"] = sum(1 for e in ends if "Junction" in kg.node(e)["labels"])
        kg.node(zid)["containsStorage"] = any("Tank" in kg.node(e)["labels"] for e in ends)
    return moved


def build_pressure_zones(wn: wntr.network.WaterNetworkModel, kg: KnowledgeGraph, network_id: str) -> dict[str, str]:
    """Hydraulic supply zones = connected components after removing pumps and
    pressure/flow control valves (the boundary devices between zones). Returns
    {node name: zone id}. Zones are named by the devices/sources that feed them."""
    G = wn.to_graph().to_undirected()
    boundary = [l for l in wn.pump_name_list] + \
               [v for v, val in wn.valves() if val.valve_type in ("PRV", "PSV", "PBV", "FCV")]
    for l in boundary:
        link = wn.get_link(l)
        if G.has_edge(link.start_node_name, link.end_node_name, key=l):
            G.remove_edge(link.start_node_name, link.end_node_name, key=l)
    zone_of: dict[str, str] = {}
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    for comp in comps:
        feeders = sorted(l for l in boundary if wn.get_link(l).end_node_name in comp)
        sources = sorted(n for n in comp if n in wn.reservoir_name_list or n in wn.tank_name_list)
        junctions = [n for n in comp if n in wn.junction_name_list]
        has_reservoir = any(n in wn.reservoir_name_list for n in comp)
        if not junctions or (has_reservoir and len(junctions) <= 2):
            # source stub: reservoir plus outlet node(s) upstream of a boundary device
            label = "SourceStub:" + "+".join(sources or sorted(comp))
        else:
            label = "Zone:" + "+".join(feeders + sources) if feeders or sources else f"Zone:{len(zone_of)}"
        zid = kg.add_node(label, "PressureZone", name=label.split(":", 1)[1],
                          isSupplyZone=label.startswith("Zone:"), nJunctions=len(junctions),
                          totalDemand=round(sum(total_demand(wn.get_node(j)) for j in junctions), 3),
                          containsStorage=any(n in wn.tank_name_list for n in comp))
        kg.add_edge(node_id("NET", network_id), "HAS_ZONE", zid)
        for f in feeders:
            kg.add_edge(link_uid(wn, f), "FEEDS_ZONE", zid)
        for n in comp:
            kg.add_edge(node_uid(wn, n), "IN_ZONE", zid)
            zone_of[n] = zid
    for name, pipe in wn.pipes():
        z = zone_of.get(pipe.start_node_name)
        if z:
            kg.add_edge(link_uid(wn, name), "IN_ZONE", z)
    return zone_of
