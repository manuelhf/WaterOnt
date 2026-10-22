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
