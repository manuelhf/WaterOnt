"""The web demonstrator re-implements the policy rules and the keyword planner in
JavaScript. These tests run that JavaScript in Node and require it to reproduce the
Python decisions and query answers exactly."""
import json
import os
import re
import shutil
import subprocess

import pytest

from wateront.graphrag import GraphRAG
from wateront.web_export import export_payload

ROOT = os.path.dirname(os.path.dirname(__file__))
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

QUERIES = ["Which pipes should be replaced?", "Which hydraulically critical pipes also have a high failure probability?",
           "Which mains serving hospitals, schools or care homes need intervention?",
           "Which cast iron mains at low service pressure need intervention?",
           "Which cast iron pipes are past their design life?"]
LTOWN_QUERIES = ["Which unmonitored hydraulically critical mains have intolerable risk?",
                 "Which pipes downstream of PRV-3 need intervention?",
                 "Which mains in the tank zone should be replaced? top 5"]


def run_js(payload, queries, tmp_path):
    tpl = open(os.path.join(ROOT, "wateront", "web_template.html"), encoding="utf-8").read()
    logic = re.search(r"// LOGIC START[^\n]*\n(.*?)// LOGIC END", tpl, re.S).group(1)
    (tmp_path / "data.json").write_text(json.dumps(payload))
    script = logic + """
const D = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
const pol = defaultPolicy(D.config);
const dec = D.pipes.map(p => decide(p, pol));
const qs = JSON.parse(process.argv[3]);
console.log(JSON.stringify({actions: dec.map(d => d.action),
  answers: qs.map(q => retrieve(planRules(q), D.pipes, dec, D).map(r => r.pipe.id))}));
"""
    (tmp_path / "t.js").write_text(script)
    out = subprocess.run(["node", str(tmp_path / "t.js"), str(tmp_path / "data.json"), json.dumps(queries)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def check(res, queries, tmp_path):
    payload = export_payload(res)
    js = run_js(payload, queries, tmp_path)
    assert js["actions"] == [p["pyAction"] for p in payload["pipes"]]
    rag = GraphRAG(res.kg, use_llm=False)
    for q, ids in zip(queries, js["answers"]):
        assert ids == [e.facts["pipe"] for e in rag.ask(q).recommendations], q


def test_web_parity_net3(built, tmp_path):
    check(built, QUERIES, tmp_path)


def test_web_parity_ltown(tmp_path):
    path = os.path.join(ROOT, "data", "networks", "L-TOWN.inp")
    if not os.path.exists(path):
        pytest.skip("L-TOWN.inp not present")
    from wateront.pipeline import build_knowledge_graph
    check(build_knowledge_graph(path, verbose=False), QUERIES + LTOWN_QUERIES, tmp_path)


def test_network_file_export(built, tmp_path):
    from wateront.web_export import write_json
    d = json.load(open(write_json(built, str(tmp_path / "n.json"))))
    assert d["format"] == "wateront-web/1" and len(d["pipes"]) == 117
    assert "EUPL" not in d["notice"]          # Net3 is not L-TOWN-derived


def test_ltown_export_carries_eupl_notice():
    path = os.path.join(ROOT, "data", "networks", "L-TOWN.inp")
    if not os.path.exists(path):
        pytest.skip("L-TOWN.inp not present")
    from wateront.pipeline import build_knowledge_graph
    assert "EUPL-1.2" in export_payload(build_knowledge_graph(path, verbose=False))["notice"]


def run_js_edits(payload, edits, tmp_path):
    tpl = open(os.path.join(ROOT, "wateront", "web_template.html"), encoding="utf-8").read()
    logic = re.search(r"// LOGIC START[^\n]*\n(.*?)// LOGIC END", tpl, re.S).group(1)
    (tmp_path / "data.json").write_text(json.dumps(payload))
    (tmp_path / "edits.json").write_text(json.dumps(edits))
    script = logic + """
const fs = require('fs');
const D = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const edits = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const maxDiff = Math.max(...D.pipes.map(p => Math.abs(pofKmOf(p, D.config) - p.pofKm)));
const idx = Object.fromEntries(D.pipes.map((p, i) => [p.id, i]));
for (const [id, e] of Object.entries(edits)) D.pipes[idx[id]] = applyEdit(D.pipes[idx[id]], e, D.config);
const pol = defaultPolicy(D.config);
console.log(JSON.stringify({maxDiff, register: toRegisterCsv(D.pipes, D.config),
  actions: D.pipes.map(p => decide(p, pol).action),
  parsed: parseRegister(toRegisterCsv(D.pipes, D.config), D.pipes.map(p => p.id), D.config)}));
"""
    (tmp_path / "e.js").write_text(script)
    out = subprocess.run(["node", str(tmp_path / "e.js"), str(tmp_path / "data.json"), str(tmp_path / "edits.json")],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_browser_edits_match_python_rebuild(tmp_path):
    """Edit pipes in the browser logic, export the asset register, rebuild in Python with
    --assets: every pipe's recommended action must agree."""
    from wateront.pipeline import build_knowledge_graph
    res = build_knowledge_graph("Net3", verbose=False)
    payload = export_payload(res)
    ids = [p["id"] for p in payload["pipes"]]
    edits = {ids[0]: {"installYear": 1925, "mat": "CI", "grade": 5, "failYears": [2010, 2015, 2021]},
             ids[5]: {"installYear": 2010, "mat": "HDPE", "grade": 1, "failYears": [], "insp": 2025},
             ids[9]: {"soil": "High", "insp": 1990}}
    js = run_js_edits(payload, edits, tmp_path)
    assert js["maxDiff"] <= 5e-7          # JS PoF formula == Python, up to the 6-dp rounding in the export
    assert not js["parsed"]["errors"] and len(js["parsed"]["edits"]) == len(ids)   # round trip parses
    reg = tmp_path / "register.csv"
    reg.write_text(js["register"])
    rebuilt = build_knowledge_graph("Net3", asset_csv=str(reg), verbose=False)
    py = {d.facts["pipe"]: d.action for d in rebuilt.decisions}
    assert js["actions"] == [py[i] for i in ids]
    assert any(a != p["pyAction"] for a, p in zip(js["actions"], payload["pipes"]))  # edits mattered


def test_partial_register_keeps_blank_cells(tmp_path):
    from wateront.pipeline import build_knowledge_graph
    base = build_knowledge_graph("Net3", verbose=False).assets.set_index("pipe")
    reg = tmp_path / "r.csv"
    reg.write_text("pipe,install_year,material,condition_grade,failure_years\n101,1930,CI,,\n105,,,2,none\n")
    a = build_knowledge_graph("Net3", asset_csv=str(reg), verbose=False).assets.set_index("pipe")
    assert a.loc["101", "install_year"] == 1930 and a.loc["101", "condition_grade"] == base.loc["101", "condition_grade"]
    assert a.loc["105", "install_year"] == base.loc["105", "install_year"] and a.loc["105", "failure_years"] == []
    bad = tmp_path / "bad.csv"
    bad.write_text("pipe,material,condition_grade\n101,Wood,9\n")
    with pytest.raises(ValueError, match="Wood.*condition_grade|condition_grade.*Wood|material"):
        build_knowledge_graph("Net3", asset_csv=str(bad), verbose=False)
