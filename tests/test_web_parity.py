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
           "Which cast iron pipes are past their design life?",
           "Which pipes have the smallest diameters?", "Which pipes have the biggest diameters?",
           "Which mains above 150 mm need intervention?", "Cast iron pipes below 150 mm that should be replaced",
           "Large diameter mains above 100 mm with intolerable risk"]
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


def run_js_sensors(payload, add, remove, tmp_path):
    tpl = open(os.path.join(ROOT, "wateront", "web_template.html"), encoding="utf-8").read()
    logic = re.search(r"// LOGIC START[^\n]*\n(.*?)// LOGIC END", tpl, re.S).group(1)
    (tmp_path / "data.json").write_text(json.dumps(payload))
    (tmp_path / "chg.json").write_text(json.dumps({"add": add, "remove": remove}))
    script = logic + """
const fs = require('fs');
const D = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const chg = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const monOk = monitoringOf(D).every((m, i) => m === D.pipes[i].mon);
for (const nd of D.nodes) {
  if (chg.remove.includes(nd[0])) nd[4] = nd[4].filter(c => c !== 'PressureSensor');
  if (chg.add.includes(nd[0]) && !nd[4].includes('PressureSensor')) nd[4].push('PressureSensor');
}
const m = monitoringOf(D);
D.pipes.forEach((p, i) => { p.mon = m[i]; });
const pol = defaultPolicy(D.config);
const csv = sensorCsv(D);
console.log(JSON.stringify({monOk, csv, mon: m, actions: D.pipes.map(p => decide(p, pol).action),
  reparsed: parseSensors(csv, D).errors}));
"""
    (tmp_path / "s.js").write_text(script)
    out = subprocess.run(["node", str(tmp_path / "s.js"), str(tmp_path / "data.json"), str(tmp_path / "chg.json")],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_monitoring_parity_and_sensor_round_trip(tmp_path):
    path = os.path.join(ROOT, "data", "networks", "L-TOWN.inp")
    if not os.path.exists(path):
        pytest.skip("L-TOWN.inp not present")
    from wateront.pipeline import build_knowledge_graph
    res = build_knowledge_graph(path, verbose=False)
    payload = export_payload(res)
    sensed = [n[0] for n in payload["nodes"] if "PressureSensor" in n[4]]
    unsensed = [n[0] for n in payload["nodes"] if n[3] == "J" and "PressureSensor" not in n[4]]
    js = run_js_sensors(payload, add=unsensed[100:104], remove=sensed[:3], tmp_path=tmp_path)
    assert js["monOk"]                      # browser coverage == Python on the original layout
    assert not js["reparsed"]               # exported layout re-imports cleanly
    f = tmp_path / "sensors.csv"
    f.write_text(js["csv"])
    rebuilt = build_knowledge_graph(path, sensors_csv=str(f), verbose=False)
    py_mon = rebuilt.risk["monitored"].to_dict()
    ids = [p["id"] for p in payload["pipes"]]
    assert js["mon"] == [bool(py_mon[i]) for i in ids]
    py = {d.facts["pipe"]: d.action for d in rebuilt.decisions}
    assert js["actions"] == [py[i] for i in ids]
    assert js["mon"] != [p["mon"] for p in payload["pipes"]]   # the change mattered


def test_net3_without_sensors_is_all_monitored(built, tmp_path):
    js = run_js_sensors(export_payload(built), add=[], remove=[], tmp_path=tmp_path)
    assert js["monOk"] and all(js["mon"])


def test_register_reading_matches_python(built, tmp_path):
    """The explorer reads asset registers and inspection logs exactly as asset.py does."""
    from wateront.asset import load_asset_overrides
    from wateront.config import Config
    fixtures = ["register_excel_semicolon.csv", "inspection_log.csv", "register_tabs.csv", "register_bad.csv",
                "register_dma.csv"]
    tpl = open(os.path.join(ROOT, "wateront", "web_template.html"), encoding="utf-8").read()
    logic = re.search(r"// LOGIC START[^\n]*\n(.*?)// LOGIC END", tpl, re.S).group(1)
    cfg = Config()
    jscfg = {"referenceYear": cfg.reference_year, "weibull": cfg.weibull, "soilFactor": cfg.soil_corrosivity_factor}
    pipes = [built.kg.node(p)["name"] for p in built.kg.nodes("Pipe")]
    (tmp_path / "r.js").write_text(logic + """
const fs = require('fs'), [cfg, pipes, files] = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
console.log(JSON.stringify(files.map(f => parseRegister(fs.readFileSync(f, 'utf8'), pipes, cfg))));
""")
    paths = [os.path.join(ROOT, "tests", "fixtures", f) for f in fixtures]
    (tmp_path / "in.json").write_text(json.dumps([jscfg, pipes, paths]))
    js = json.loads(subprocess.run(["node", str(tmp_path / "r.js"), str(tmp_path / "in.json")],
                                   capture_output=True, text=True, check=True).stdout)
    key = {"install_year": "installYear", "condition_grade": "grade", "last_inspection_year": "insp",
           "material": "mat", "soil_corrosivity": "soil", "failure_years": "failYears", "dma": "dma"}
    for f, path, j in zip(fixtures, paths, js):
        try:
            py = load_asset_overrides(path, cfg)
        except ValueError as e:
            assert j["errors"], f
            assert len(j["errors"]) == str(e).count("line "), f      # same problems found
            continue
        assert not j["errors"], (f, j["errors"])
        known = set(pipes)
        assert sorted(j["unknown"]) == sorted(p for p in py if p not in known), f
        for pipe, rec in py.items():
            if pipe not in known:
                continue
            assert {key[k]: v for k, v in rec.items() if k in key} == j["edits"][pipe], (f, pipe)


def test_sensor_file_reading_matches_python(built, tmp_path):
    import wntr
    from wateront.physical import load_sensors
    path = os.path.join(ROOT, "tests", "fixtures", "sensors_semicolon.csv")
    py = load_sensors(path, built.wn)
    tpl = open(os.path.join(ROOT, "wateront", "web_template.html"), encoding="utf-8").read()
    logic = re.search(r"// LOGIC START[^\n]*\n(.*?)// LOGIC END", tpl, re.S).group(1)
    (tmp_path / "data.json").write_text(json.dumps(export_payload(built)))
    (tmp_path / "s.js").write_text(logic + """
const fs = require('fs'), D = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
console.log(JSON.stringify(parseSensors(fs.readFileSync(process.argv[3], 'utf8'), D)));
""")
    js = json.loads(subprocess.run(["node", str(tmp_path / "s.js"), str(tmp_path / "data.json"), path],
                                   capture_output=True, text=True, check=True).stdout)
    assert not js["errors"], js["errors"]
    assert {k: sorted(v) for k, v in js["layout"].items()} == {k: sorted(c for c, _ in v) for k, v in py.items()}


def test_colab_code_from_the_page_converts_the_users_model(tmp_path):
    """The explorer's 'Use my EPANET model' code (files embedded, gzip + base64) restores the
    files byte for byte in Colab and yields the same network file as a direct conversion."""
    import glob
    import sys
    import types
    import wntr
    from wateront.convert import convert
    inp = glob.glob(os.path.join(os.path.dirname(wntr.__file__), "library", "**", "Net3.inp"), recursive=True)[0]
    log = os.path.join(ROOT, "tests", "fixtures", "inspection_log.csv")
    tpl = open(os.path.join(ROOT, "wateront", "web_template.html"), encoding="utf-8").read()
    logic = re.search(r"// LOGIC START[^\n]*\n(.*?)// LOGIC END", tpl, re.S).group(1)
    (tmp_path / "g.js").write_text(logic + """
const fs = require('fs');
(async () => {
  const files = process.argv.slice(3).map(p => ({ name: p.split('/').pop(), bytes: new Uint8Array(fs.readFileSync(p)) }));
  fs.writeFileSync(process.argv[2], await colabCode(files));
})();
""")
    cell = tmp_path / "cell.py"
    subprocess.run(["node", str(tmp_path / "g.js"), str(cell), inp, log], check=True)
    code = cell.read_text()
    assert len(code) < os.path.getsize(inp)                       # compressed
    work = tmp_path / "colab"
    code = code.replace("INSTALL = True", "INSTALL = False").replace('work = "/content/wateront"', f'work = r"{work}"')
    downloaded = []
    colab = types.ModuleType("google.colab"); colab.files = types.SimpleNamespace(download=downloaded.append)
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules["google"], sys.modules["google.colab"] = types.ModuleType("google"), colab
    env_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = ROOT + (os.pathsep + env_path if env_path else "")
    try:
        exec(compile(code, "colab_cell", "exec"), {})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if env_path is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = env_path
    assert open(work / "Net3.inp", "rb").read() == open(inp, "rb").read()
    ref, _ = convert([inp, log], out_dir=str(tmp_path / "ref"))
    assert downloaded and json.load(open(downloaded[0])) == json.load(open(ref))


def test_web_parity_with_user_dmas(tmp_path):
    """DMAs from a register: the explorer places pipes and answers DMA questions as Python does."""
    from wateront.pipeline import build_knowledge_graph
    res = build_knowledge_graph("Net3", asset_csv=os.path.join(ROOT, "tests", "fixtures", "register_dma.csv"), verbose=False)
    check(res, ["Which pipes in DMA North-1 need attention?", "Show the pipes in zone South-2",
                "Which pipes in district North-1 should be replaced?", "Which pipes in the zone fed by the tank need intervention?",
                "Which pipes should be replaced?"], tmp_path)
