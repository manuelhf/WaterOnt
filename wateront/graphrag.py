"""GraphRAG engine (Section 4).

query --> QueryPlan --> graph retrieval (+ vector retrieval of policy text)
      --> evidence context with citation keys --> generation --> citation check

Decisions are *read* from the Decision Layer; the generator only verbalises them.
This keeps justification stable and independent of the LLM (Section 3.4), and
lets `verify` reject any answer that cites ids absent from the retrieved evidence.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

from .config import Config
from .decision import ACTIONS
from .kg import KnowledgeGraph
from .retrieval import Evidence, GraphRetriever, VectorIndex

MATERIAL_WORDS = {"cast iron": "CI", "ductile": "DI", "asbestos": "AC", "pvc": "PVC",
                  "polyethylene": "HDPE", "hdpe": "HDPE", "steel": "Steel"}


@dataclass
class QueryPlan:
    actions: list[str] = field(default_factory=list)       # [] = any action
    exclude_monitor: bool = False
    bands: dict = field(default_factory=dict)              # dim -> [bands]
    material: list[str] = field(default_factory=list)
    installed_before: int | None = None
    min_failures: int | None = None
    beyond_design_life: bool = False
    critical_facilities: bool = False
    low_pressure: bool = False
    zones: list[str] = field(default_factory=list)        # feeding device/source, e.g. PRV-3, T1
    unmonitored: bool = False
    pipes: list[str] = field(default_factory=list)
    junctions: list[str] = field(default_factory=list)
    min_diameter: float | None = None   # mm, strict: "above 150 mm"
    max_diameter: float | None = None   # mm, strict: "below 150 mm"
    order_by: str | None = None         # "diameter_asc" | "diameter_desc"; default: urgency, then risk
    top_k: int = 25


def plan_rules(q: str) -> QueryPlan:
    """Deterministic keyword planner (no LLM). Transparent, but brittle to unusual
    phrasing; see `plan_llm` for the LLM-based alternative."""
    s = q.lower()
    p = QueryPlan()
    if re.search(r"\b(replac\w*|renew\w*|renewal)\b", s):
        p.actions.append("Replacement")
    if re.search(r"\b(rehab\w*|relin\w*|refurbish\w*)\b", s):
        p.actions.append("Rehabilitation")
    if re.search(r"\b(inspect\w*|survey\w*|condition assessment)\b", s):
        p.actions.append("Inspection")
    if re.search(r"\b(priorit\w*|intervention|need action|urgent\w*|attention|work)\b", s):
        p.exclude_monitor = True
    if re.search(r"hydraulic\w* critical|backbone|bottleneck|betweenness|trunk", s):
        p.bands["hydraulic"] = ["High", "Critical"]
    if re.search(r"(high|very high)\w* (failure|probability)|likely to fail|failure[- ]prone|"
                 r"probability of failure", s):
        p.bands["pof"] = ["High", "VeryHigh"]
    if "intolerable" in s:
        p.bands["overall"] = ["Intolerable"]
    elif re.search(r"\bsignificant risk", s):
        p.bands["overall"] = ["Significant", "Intolerable"]
    if re.search(r"(past|beyond|exceed\w*) (their |its |the )?design life|end of (their |its )?(service )?life", s):
        p.beyond_design_life = True
    if re.search(r"hospital|school|care home|critical (customer|facilit|consumer)|industrial", s):
        p.critical_facilities = True
    if re.search(r"critical service", s):
        p.bands["service"] = ["Critical"]
    if re.search(r"low (service |network )?pressure|pressure (deficit|margin)", s):
        p.low_pressure = True
    if re.search(r"unmonitored|not monitored|no (pressure )?(sensor|monitoring)|without (a |any )?"
                 r"(pressure )?(sensor|monitoring)|monitoring gap|blind spot", s):
        p.unmonitored = True
    p.zones = re.findall(r"\b(PRV-\d+|PSV-\d+|PUMP_\w+|T\d+)\b", q, flags=re.I)
    if re.search(r"tank (zone|area)|zone (fed|supplied|served) by the tank|storage zone", s):
        p.zones.append("tank")
    p.material = sorted({v for k, v in MATERIAL_WORDS.items() if k in s})
    if m := re.search(r"(before|pre-?)\s*(1[89]\d\d|20\d\d)", s):
        p.installed_before = int(m.group(2))
    if m := re.search(r"more than (\d+) failures", s):
        p.min_failures = int(m.group(1)) + 1
    p.pipes = re.findall(r"\bpipe\s+([A-Za-z0-9_\-]+)", q)
    p.junctions = re.findall(r"\bjunction\s+([A-Za-z0-9_\-]+)", q)
    if m := re.search(r"\btop\s+(\d+)", s):
        p.top_k = int(m.group(1))
    p.min_diameter, p.max_diameter, p.order_by = diameter_terms(s)
    return p


DIA_ABOVE = r"\b(?:above|over|greater than|larger than|bigger than|more than|wider than)\s*(\d+(?:\.\d+)?)\s*mm\b"
DIA_BELOW = r"\b(?:below|under|less than|smaller than|narrower than)\s*(\d+(?:\.\d+)?)\s*mm\b"


def diameter_terms(s: str) -> tuple[float | None, float | None, str | None]:
    """Diameter bounds in mm and an optional ordering; mirrored in the web explorer."""
    lo = float(m.group(1)) if (m := re.search(DIA_ABOVE, s)) else None
    hi = float(m.group(1)) if (m := re.search(DIA_BELOW, s)) else None
    order = None
    if re.search(r"\b(diameters?|bores?)\b", s):
        if re.search(r"\b(small\w*|narrow\w*|thin\w*|minimum)\b", s):
            order = "diameter_asc"
        elif re.search(r"\b(larg\w*|big\w*|wide\w*|maximum)\b", s):
            order = "diameter_desc"
    return lo, hi, order


PLANNER_PROMPT = """Convert the engineer's question into a JSON query plan for a water-network
knowledge graph. Reply with JSON only, no prose and no code fences. Keys:
actions: subset of ["Replacement","Rehabilitation","Inspection","Monitor"] ([] = any)
exclude_monitor: bool (true when the user asks what needs intervention/prioritisation)
bands: object, keys among pof|hydraulic|service|overall; values are lists from
  pof: Low, Medium, High, VeryHigh; hydraulic/service: Low, Medium, High, Critical;
  overall: Negligible, Tolerable, Significant, Intolerable
material: subset of ["CI","DI","AC","PVC","HDPE","Steel"]
installed_before: int or null; min_failures: int or null; beyond_design_life: bool
critical_facilities: bool; low_pressure: bool; unmonitored: bool
zones: list of devices/sources feeding the zone asked about, e.g. ["PRV-3"], ["T1"] or ["tank"]
pipes: list of pipe ids; junctions: list of junction ids; top_k: int (default 25)"""


def _anthropic_client():
    import anthropic  # optional dependency
    return anthropic.Anthropic()


class PlanParseError(ValueError):
    """The LLM answered, but its query plan could not be read. This is a failure of the
    planner on that question (scored), not an API problem (which aborts an experiment)."""


def plan_llm(q: str, cfg: Config) -> QueryPlan:
    msg = _anthropic_client().messages.create(
        model=cfg.llm_model, max_tokens=cfg.llm_max_tokens_plan,
        system=PLANNER_PROMPT, messages=[{"role": "user", "content": q}])
    raw = "".join(b.text for b in msg.content if b.type == "text")
    text = re.sub(r"```(json)?", "", raw).strip()
    if "{" in text and "}" in text:                      # tolerate prose around the JSON object
        text = text[text.index("{"): text.rindex("}") + 1]
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("plan is not a JSON object")
    except ValueError as exc:
        raise PlanParseError(f"{exc}; planner answer: {raw[:300]!r}") from None
    allowed = QueryPlan.__dataclass_fields__.keys()
    plan = QueryPlan(**{k: v for k, v in data.items() if k in allowed})
    plan.actions = [a for a in plan.actions if a in ACTIONS]
    return plan


@dataclass
class Answer:
    query: str
    plan: QueryPlan
    recommendations: list[Evidence]
    policy_context: list[dict]
    text: str
    unsupported_citations: list[str] = field(default_factory=list)
    generator: str = "template"

    def evidence_ids(self) -> set[str]:
        ids = {e.pipe_uid for e in self.recommendations}
        for e in self.recommendations:
            ids |= set(e.guidelines)
        return ids

    def to_dict(self) -> dict:
        return {"query": self.query, "plan": asdict(self.plan), "generator": self.generator,
                "text": self.text, "unsupported_citations": self.unsupported_citations,
                "policy_context": [g["id"] for g in self.policy_context],
                "recommendations": [{"pipe": e.facts["pipe"], "action": e.action, "rule": e.rule,
                                     "guidelines": e.guidelines, "facts": e.facts,
                                     "path": [list(t) for t in e.path]} for e in self.recommendations]}


class GraphRAG:
    def __init__(self, kg: KnowledgeGraph, cfg: Config | None = None,
                 use_llm: bool | None = None, embed=None, strict: bool = False):
        """strict=True: an LLM failure raises instead of falling back to the keyword planner or
        template. Experiments must use strict mode, so a failed call can never be recorded as
        an LLM result."""
        self.strict = strict
        self.kg, self.cfg = kg, cfg or Config()
        self.retriever = GraphRetriever(kg)
        self.guidelines = [kg.node(g) for g in kg.nodes("EngineeringGuideline")]
        self.index = VectorIndex(self.guidelines, embed=embed)
        self.use_llm = bool(os.environ.get("ANTHROPIC_API_KEY")) if use_llm is None else use_llm

    def _plan_with_llm(self, query: str) -> QueryPlan:
        """One retry if the plan cannot be read; API errors propagate unchanged."""
        try:
            return plan_llm(query, self.cfg)
        except PlanParseError as first:
            print(f"[graphrag] planner answer could not be read, retrying once: {first}")
            return plan_llm(query, self.cfg)

    # ---------------------------------------------------------------- retrieve
    def retrieve(self, plan: QueryPlan) -> list[Evidence]:
        r, kg = self.retriever, self.kg
        cand = set(kg.nodes("Pipe"))
        prefix: dict[str, list] = {}
        if plan.pipes:
            cand &= {u for n in plan.pipes if (u := r.pipe_by_name(n))}
        if plan.junctions:
            inc = set().union(*(r.pipes_incident_on(j) for j in plan.junctions))
            cand &= inc
        if plan.bands:
            hits, prefix = r.pipes_by_bands(**plan.bands)
            cand &= hits
        if plan.actions:
            cand &= r.pipes_by_action(plan.actions)
        elif plan.exclude_monitor:
            cand &= r.pipes_by_action([a for a in ACTIONS if a != "Monitor"])
        if plan.material or plan.installed_before or plan.min_failures is not None:
            cand &= r.pipes_by_asset(plan.material, plan.installed_before, plan.min_failures)
        if plan.beyond_design_life:
            cand &= r.pipes_beyond_design_life()
        if plan.critical_facilities:
            cand &= r.pipes_serving_facilities()
        if plan.low_pressure:
            cand &= r.pipes_low_pressure(self.cfg.min_service_pressure_m + self.cfg.pressure_margin_m)
        if plan.unmonitored:
            hits, mpaths = r.pipes_by_bands(monitoring=["Unmonitored"])
            cand &= hits
            for pid, path in mpaths.items():
                prefix.setdefault(pid, []).extend(t for t in path if t not in prefix.get(pid, []))
        for zref in plan.zones:
            z = r.resolve_zone(zref)
            if z is None:
                cand = set()
                break
            cand &= r.pipes_in_zone(z)
            for pid in cand:
                prefix.setdefault(pid, []).append((pid, "IN_ZONE", z))
        dia = lambda pid: round(kg.node(pid)["diameter"])     # whole mm, as in the explorer
        if plan.min_diameter is not None:
            cand = {p for p in cand if dia(p) > plan.min_diameter}
        if plan.max_diameter is not None:
            cand = {p for p in cand if dia(p) < plan.max_diameter}
        ranked = r.rank([r.evidence(p, prefix.get(p)) for p in cand])
        if plan.order_by in ("diameter_asc", "diameter_desc"):     # stable: urgency breaks ties
            ranked.sort(key=lambda e: dia(e.pipe_uid), reverse=plan.order_by == "diameter_desc")
        return ranked[: plan.top_k]

    # -------------------------------------------------------------------- ask
    def ask(self, query: str, plan: QueryPlan | None = None) -> Answer:
        if plan is None:
            plan = plan_rules(query)
            if self.use_llm:
                try:
                    plan = self._plan_with_llm(query)
                except PlanParseError as exc:
                    if self.strict:   # experiments: a plan that cannot be read is a scored failure
                        print(f"[graphrag] planner failed twice on {query!r}: {exc}")
                        ans = Answer(query, QueryPlan(top_k=0), [], [], "", generator="llm (planning failed)")
                        ans.planning_failed = True
                        return ans
                    print(f"[graphrag] LLM planner failed ({exc}); using rule planner")
                except Exception as exc:  # API errors: abort experiments, fall back interactively
                    if self.strict:
                        raise
                    print(f"[graphrag] LLM planner failed ({exc}); using rule planner")
        recs = self.retrieve(plan)
        policy = [d for d, _ in self.index.search(query, k=3)]
        context = self.build_context(recs, policy)
        if self.use_llm:
            try:
                text, gen = self.generate_llm(query, context), "llm"
            except Exception as exc:
                if self.strict:
                    raise
                print(f"[graphrag] LLM generation failed ({exc}); using template")
                text, gen = self.generate_template(query, recs, policy), "template"
        else:
            text, gen = self.generate_template(query, recs, policy), "template"
        ans = Answer(query, plan, recs, policy, text, generator=gen)
        ans.unsupported_citations = self.verify(ans)
        return ans

    # ---------------------------------------------------------------- context
    def build_context(self, recs: list[Evidence], policy: list[dict]) -> str:
        lines = ["## Graph evidence (authoritative; decisions computed by policy rules)"]
        for e in recs:
            f = e.facts
            lines.append(
                f"[{e.pipe_uid}] action={e.action} rule={e.rule} | material={f['material']} "
                f"age={f['age']}y design_life={f['design_life']}y grade={f['condition_grade']} "
                f"failures={f['failure_count']} pof_segment={f['pof']:.3f} pof_per_km={f['pof_km']:.3f}({f['pof_band']}) "
                f"hydraulic={f['hydraulic']} service={f['service']} overall={f['overall']} "
                f"min_pressure={f['min_end_pressure']}m facilities={f['facilities'] or 'none'} "
                f"zone={f.get('zone')} monitored={f.get('monitored')} material_source={f.get('material_source')} "
                f"likelihood_source={f.get('likelihood_source', 'weibull')} "
                f"| supported_by={','.join(f'[{g}]' for g in e.guidelines)}")
        ids = {g for e in recs for g in e.guidelines} | {g["id"] for g in policy}
        lines.append("\n## Policy clauses")
        for gid in sorted(ids):
            g = self.kg.node(gid)
            lines.append(f"[{gid}] ({g['framework']}) {g['title']}: {g['text']}")
        return "\n".join(lines)

    def generate_template(self, query: str, recs: list[Evidence], policy: list[dict]) -> str:
        if not recs:
            consulted = ", ".join("[" + g["id"] + "]" for g in policy) or "none"
            return f"No pipe in the knowledge graph satisfies this query. Policy passages consulted: {consulted}."
        out = [f"{len(recs)} pipe(s) match. Ranked by action urgency, then risk priority:"]
        for i, e in enumerate(recs, 1):
            f = e.facts
            why = {
                "R1": f"age {f['age']}y exceeds the {f['material']} design life of {f['design_life']}y "
                      f"and overall risk is {f['overall']}",
                "R2": f"condition grade {f['condition_grade']} with {f['overall']} overall risk",
                "R3": f"{f['overall']} risk within design life; condition must be confirmed",
                "R4": "the risk-based inspection interval is exceeded",
                "R5": f"{f['overall']} risk on a {f['hydraulic']}-criticality main with no pressure "
                      f"sensor within two segments",
                "R0": f"{f['overall']} risk and inspection up to date",
            }[e.rule]
            fac = f"; serves {', '.join(f['facilities'])}" if f["facilities"] else ""
            if f.get("zone"):
                fac += f"; zone {f['zone']}"
            if f.get("material_source", "prior") != "prior":
                fac += f"; material from {f['material_source']}"
            out.append(f"{i}. [{e.pipe_uid}] {e.action}: {why}. PoF {f['pof_km']:.2f}/km ({f['pof_band']}; "
                       f"{f['pof']:.2f} for this {f['length']:.0f} m segment), "
                       f"hydraulic {f['hydraulic']}, service {f['service']}, min pressure "
                       f"{f['min_end_pressure']} m{fac}. Authorised by "
                       + ", ".join(f"[{g}]" for g in e.guidelines) + ".")
        return "\n".join(out)

    def generate_llm(self, query: str, context: str) -> str:
        system = ("You are a water-utility asset-management assistant. Answer ONLY from the evidence "
                  "provided. Do not change any recommended action. Cite every pipe and policy clause "
                  "with its bracketed id exactly as given, e.g. [P:101] and [G-REPL-01]. If the "
                  "evidence is empty or insufficient, say so.")
        msg = _anthropic_client().messages.create(
            model=self.cfg.llm_model, max_tokens=self.cfg.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": f"{context}\n\n## Question\n{query}"}])
        return "".join(b.text for b in msg.content if b.type == "text")

    # ----------------------------------------------------------------- verify
    def verify(self, ans: Answer) -> list[str]:
        """Citations in the text that are not in the retrieved evidence set."""
        allowed = ans.evidence_ids() | {g["id"] for g in ans.policy_context}
        cited = set(re.findall(r"\[([A-Z]{1,3}[:\-][^\]\s]+)\]", ans.text))
        return sorted(c for c in cited if c not in allowed)
