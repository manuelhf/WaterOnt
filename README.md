# WaterOnt GraphRAG — code companion

[![tests](https://github.com/manuelhf/WaterOnt/actions/workflows/tests.yml/badge.svg)](https://github.com/manuelhf/WaterOnt/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manuelhf/WaterOnt/blob/main/demo/WaterOnt_Colab.ipynb)
[![Licence: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

**[Open the live explorer](https://manuelhf.github.io/WaterOnt/)**: the L-TOWN network as a knowledge graph you can question, audit pipe by pipe, and re-run under your own maintenance policy, all in the browser.

Reference implementation for *Explainable Asset Management in Water Networks using GraphRAG and Engineering Knowledge Graphs*, Herrera et al. (in preparation).

It builds the four-layer WaterOnt knowledge graph from any EPANET `.inp` file, answers engineering questions through a GraphRAG engine whose every recommendation carries a verifiable path back to asset evidence and the authorising policy clause, and compares it against an ML failure-prediction baseline and a flat-document RAG baseline.

## Use your own network (no Python needed)

WaterOnt is research software under active development: it runs on any EPANET model, but its
recommendations follow an illustrative policy and, unless you supply your own records,
synthetic asset histories. Use it to explore and to give feedback, not for operational decisions.

1. **Convert your model.** Open the [one-click converter](https://colab.research.google.com/github/manuelhf/WaterOnt/blob/main/demo/WaterOnt_convert.ipynb) in Google Colab, run step 1, then step 2, and select your `.inp` together with any of: an asset register or inspection log, a failure list, a sensor layout, critical customers. The `.json` downloads automatically, with a summary of what came from your files and what is still synthetic.
2. **Open it** in the [explorer](https://manuelhf.github.io/WaterOnt/) with **Open network file (.json)…**. The file is read in your browser and not uploaded anywhere.
3. **Add or update data later** in the explorer's Data tab: *Import asset register* also accepts inspection logs, and sensor layouts can be imported or edited on the map.

Files do not have to follow the templates in [`data/`](data/) exactly: column names such as "Pipe ID", "Last inspection" or "Installation year" are recognised; dates can stand in for years; materials can be written as names ("cast iron", "uPVC", "MDPE"); Excel CSV files with commas, semicolons or tabs all work. An inspection log may list a pipe several times: the most recent inspection wins. A file with any invalid value is rejected as a whole, with every problem listed, so a register is never half-applied.

## Paper → code map

| Paper | Module | What it does |
|---|---|---|
| §3.1 Physical Layer, App. A | `wateront/physical.py` | `.inp` → Junction/Reservoir/Tank, Pipe/Pump/6 valve classes, DemandPattern, typed Curves, SimpleControl/Rule, reified OperationalStatus, sensors from `.inp` annotations, pressure zones |
| §3.2 Asset Layer | `wateront/asset.py` | AssetRecord, FailureEvent (UKWIR modes), InspectionRecord; literature-shaped priors or a real register CSV |
| §3.3 Risk Layer | `wateront/risk.py` | Weibull conditional PoF, betweenness + simulated-flow hydraulic criticality, ISO 24510-style service criticality with pressure margin, ISO 55000-style matrix; risk bands are shared nodes |
| §3.4 Decision Layer | `wateront/decision.py` | Guideline nodes, rules R1–R4, DecisionRecord `-[:SUPPORTED_BY]->` guideline, `-[:BASED_ON]->` evidence |
| §3.5 Instantiation, Fig. 1 | `wateront/pipeline.py`, `wateront/kg.py` | In-memory property graph; optional Neo4j push with the Listing A.2 constraints |
| §4 GraphRAG | `wateront/retrieval.py`, `wateront/graphrag.py` | Query planner (rules or Claude), graph traversal with explicit triples, TF-IDF/dense policy retrieval, template or LLM generation, citation verifier |
| §5 Evaluation | `wateront/evaluation.py`, `wateront/baselines.py` | 6 intents × 3 paraphrases; Decision F1, Policy Recall, Traceability, Evidence Consistency, unsupported citations |
| App. A OWL + SPARQL | `ontology/wateront.ttl`, `wateront/rdf_export.py` | OWL 2 axioms for all four layers; instance export; CQ1–CQ8 and upper-layer CQs |
| Neo4j | `cypher/competency_questions.cypher` | Cypher versions of the competency questions |

## Quick start

```bash
pip install -r requirements.txt
python -m wateront build --inp Net3 --out kg.pkl --ttl kg.ttl
python -m wateront ask --kg kg.pkl "Which hydraulically critical pipes also have a high failure probability? top 5"
python -m wateront ask --kg kg.pkl "What is recommended for pipe 131?" --json    # full audit trail
python -m wateront evaluate --inp Net3 --out results/
python -m pytest -q tests
```

`Net1`, `Net3` and `Net6` ship with WNTR; L-TOWN is included in `data/networks/` (see the L-TOWN section).

Install as a package with `pip install -e .` (extras: `.[llm]`, `.[neo4j]`, `.[demo]`). Optional extras: `--neo4j bolt://localhost:7687` on `build` (credentials from `NEO4J_USER`/`NEO4J_PASSWORD`; each push replaces the previous WaterOnt graph, leaving any other data in the database untouched, so a rebuilt graph never mixes with an old one; `--keep-existing` merges instead, `--wipe` clears the whole database), and `--llm` on `ask`/`evaluate` to use Claude for planning, generation and an additional LLM-based flat-RAG baseline (needs `ANTHROPIC_API_KEY`; model set in `Config.llm_model`). Real data plugs in through `--assets` (see `data/asset_register_template.csv`, which overrides priors pipe by pipe) and `--guidelines` (see `data/guidelines_template.json`).

## Results produced by this code (deterministic mode, seed 42)

| Network | System | Decision F1 | Policy recall | Traceability | Evidence consistency |
|---|---|---|---|---|---|
| L-TOWN (905 pipes, 8 intents) | WaterOnt GraphRAG | 0.937 | 0.955 | 1.000 | 0.862 |
| | Flat-document RAG | 0.010 | 0.010 | 0.000 | 0.391 |
| | ML (GBM rank) | 0.050 | 0.000 | 0.000 | 0.917 |
| Net3 (117 pipes, 6 intents) | WaterOnt GraphRAG | 0.958 | 0.984 | 1.000 | 0.883 |
| | Flat-document RAG | 0.018 | 0.014 | 0.000 | 0.403 |
| | ML (GBM rank) | 0.168 | 0.000 | 0.000 | 0.876 |
| Net6 (3,829 pipes, 6 intents) | WaterOnt GraphRAG | 0.944 | 0.944 | 1.000 | 0.891 |
| | Flat-document RAG | 0.002 | 0.001 | 0.000 | 0.404 |
| | ML (GBM rank) | 0.121 | 0.000 | 0.000 | 0.889 |

Per-query detail is in `results/`. GraphRAG's shortfall from 1.0 comes entirely from the keyword planner misreading paraphrases (e.g. "relined rather than replaced" triggers both actions; "What maintenance is needed in the PRV-3 zone?" does not exclude Monitor), which is a useful honest finding about the planner rather than the graph.

## L-TOWN

`data/networks/L-TOWN.inp` (L-TOWN v2, KIOS/BattLeDIM) is the paper's case study. Running on it exposed issues that the WNTR example networks hid, and added capabilities Net3 cannot exercise:

**Fixed because of L-TOWN.** Each L-TOWN junction has three demand categories (residential, commercial, industrial), and WNTR's `junction.base_demand` returns only the first, so junction n1 (industrial only) appeared to have zero demand; demand is now summed over all categories and stored per category. L-TOWN models mains as ~50 m segments, so per-segment failure probability never exceeded "Medium" and the replacement query had an empty reference; PoF is now banded per km (`Config.pof_basis`), which gives near-identical PoF distributions on Net3 and L-TOWN. Evaluation intents with an empty reference are now skipped and reported instead of scored.

**Used from the model.** Element comments, which WNTR discards, mark 33 pressure sensors, 82 AMR meters, 3 flow sensors (both reservoir outlets and the pump) and the tank level sensor; they become SOSA-aligned `Sensor` nodes, and each pipe gets a `MonitoringStatus` (pressure sensor within two segments: 325 of 905 pipes). Removing pumps and PRVs yields three supply zones (`PRV-1+PRV-2` main area, `PRV-3` sub-zone, `PUMP_1+T1` tank zone) as `PressureZone` nodes. The four junctions with industrial demand become real `IndustrialCustomer` consequence evidence. Plastic OD diameters (63/75/160/225 mm, 33 pipes) and Hazen-Williams C = 120 (117 pipes) inform material and age priors, with provenance stored in `materialSource`. A new rule R5 and guideline G-MON-02 flag critical, high-risk mains with no nearby pressure sensor.

New queries this enables:

```bash
python -m wateront build --inp data/networks/L-TOWN.inp --out ltown.pkl
python -m wateront ask --kg ltown.pkl "Which unmonitored hydraulically critical mains have intolerable risk?"
python -m wateront ask --kg ltown.pkl "Which pipes downstream of PRV-3 need intervention?"
python -m wateront ask --kg ltown.pkl "Which mains in the tank zone should be replaced? top 5"
python -m wateront ask --kg ltown.pkl "Which pipes at low pressure need intervention?"
```

All 46 pipes whose minimum pressure is within 10 m of the 20 m service level lie in the pump/tank zone.

**Non-circular validation with BattLeDIM leaks.** The BattLeDIM dataset includes the leak records for L-TOWN. Convert them to `pipe,date[,mode]` CSVs and use 2018 as history and 2019 as a withheld test set:

```bash
python -m wateront evaluate --inp data/networks/L-TOWN.inp \
    --failures leaks_2018.csv --holdout-failures leaks_2019.csv
```

This trains the ML baseline on the 2019 labels (out-of-fold) and reports capture of 2019 failures within 5/10/20% of network length for the WaterOnt risk ranking, a WaterOnt likelihood-only ranking, and ML. Two design choices matter here. Budgets are in km, not pipe counts, because failures scale with segment length while WaterOnt ranks per km. And failure capture measures likelihood only, so the risk ranking (likelihood × consequence) is not expected to win it; the likelihood-only ranking is the fair comparator. Leakages are hydraulic events rather than structural pipe failures, so frame this as a proxy validation in the paper.

## Demonstrators

**Web explorer** (`demo/index.html`, published by GitHub Pages). One self-contained page with the L-TOWN graph embedded: a pannable map coloured by risk, action, zone, monitoring or material; an Ask tab using the same keyword planner as Python; a Pipe tab showing each recommendation's audit trail from asset record to authorising clause; and a Policy sandbox where changing design lives, PoF bands, inspection intervals or rule R5 recomputes every decision live. The policy rules and planner are re-implemented in JavaScript, and `tests/test_web_parity.py` runs them in Node to check they reproduce the Python decisions and answers exactly. Links like `demo/#p257` open that pipe directly.

**Saving work.** The explorer's **Save…** menu writes four files, all generated in the browser:

| File | Contents | Use it for |
|---|---|---|
| Network file (`.json`) | The whole network, your pipe edits (with their original values) and your policy settings | Reopening your work later with **Open network…** |
| Decisions table (`.csv`) | Every pipe with its facts, risk bands, action, rule, reason and clauses, plus the action under the default policy | Excel, reports, comparing scenarios |
| Asset register (`.csv`) | Asset facts in the format Python reads with `--assets` | Feeding browser edits back into a full Python rebuild |
| Map picture (`.svg`) | The map as currently coloured, with its legend | Papers and slides |

The Ask tab can also save a question's results as CSV, and the Pipe tab can save one pipe's audit trail as a Markdown report.

**Editing pipes.** In the Pipe tab, **Edit asset record** changes install year, material, condition grade, last inspection, soil corrosivity and failure years. The failure probability and decision are recomputed at once with the same formulas as Python; `tests/test_web_parity.py` checks that browser edits, exported as an asset register and rebuilt in Python, give identical recommendations for every pipe. Hydraulic and service criticality come from network simulation and do not change in the browser. Edits stay in the page until saved in a network file, and the page warns before you leave with unsaved edits.

**Asking Claude.** The Ask tab understands questions built from its known concepts (actions, risk bands, zones, materials, monitoring, pressure, dates and failure counts) and always shows how it read the question. For anything else, such as "why", comparisons or explanations, **Ask Claude about this** (on a pipe or on a question's results) copies your question together with the evidence, the policy in force and a summary of the rules, and opens claude.ai in a new tab: paste and send. Claude is asked to answer only from that evidence and to cite pipe and clause ids. This needs no API key and costs the site nothing; each person uses their own Claude account.

**Pressure sensors and monitoring.** A pipe counts as monitored when a pressure sensor sits within two links of either end; monitoring feeds rule R5 and the questions about unmonitored mains. In the Data tab, **Place sensors on the map** lets you click junctions to add or remove pressure sensors and watch coverage and decisions update, which makes the page a simple sensor-placement what-if tool. Sensor layouts can be imported or saved as CSV (`element,type`), and Python reads the same file with `--sensors`, replacing the sensors marked in the `.inp`. The tests check that browser coverage matches Python before and after moving sensors.

**Scope: pipes only.** Pipes are the assets assessed. Pumps, valves, tanks and reservoirs take part in the hydraulics and appear on the map, but receive no condition, risk or maintenance decision: they need different deterioration models (pump reliability and efficiency, valve operability, tank structural and water-quality inspection) and are left for future work.

**Your data.** The Data tab explains what comes from the `.inp` and what is synthetic, and imports an asset register CSV straight into the page. Any subset of columns can be given, blank cells keep their current value, `none` in `failure_years` means no recorded failures, and invalid values (unknown materials, grades outside 1 to 5) stop the import with line-by-line reasons. The same rules apply in Python. Data that change the hydraulics or the network need a rebuild in Python or Colab:

| Data | Python option | Template |
|---|---|---|
| Network model | `--inp mynet.inp` | any EPANET 2 file; sensors can be marked in element comments as in L-TOWN |
| Asset register | `--assets register.csv` | `data/asset_register_template.csv` |
| Dated failures or leaks | `--failures failures.csv` | columns `pipe,date[,mode]` |
| Critical customers | `--facilities facilities.csv` | `data/facilities_template.csv`; replaces the synthetic hospitals and schools |
| Sensor layout | `--sensors sensors.csv` | `data/sensors_template.csv`; replaces the sensors marked in the `.inp` |
| Policy text | `--guidelines policy.json` | `data/guidelines_template.json` |

Then `python -m wateront web --inp mynet.inp --assets register.csv --json --out mynet.json` and open the file with **Open network…**. An opened file is read in the visitor's browser and never uploaded, so a utility can view its own network on the public page without sharing any data. With real data, the main practical step is matching register IDs to model pipe IDs; registers often use GIS asset IDs that do not match EPANET link names.

**Live app on Colab** (`demo/WaterOnt_Colab.ipynb`). Builds the graph, downloads the static page, and launches `wateront/gradio_app.py` with `share=True`, which gives a temporary public link to the real Python engine. Add `ANTHROPIC_API_KEY` in Colab Secrets to switch on Claude for planning and generation.

## Working with synthetic data

The network, hydraulics, sensors and demand categories are real model data; only asset histories are synthetic. To make results defensible:

- **Report across many synthetic utilities.** `python -m wateront evaluate --inp <net> --seeds 10` draws 10 independent asset histories on the same network and reports mean and sd. On L-TOWN, GraphRAG decision F1 is 0.927 ± 0.006, flat RAG 0.013 ± 0.005 and ML 0.041 ± 0.017, so the conclusions do not depend on one draw.
- **Document the generator as a method.** Every prior is in `config.py`, every attribute records its provenance (`materialSource`), and the generator is seeded and versioned.
- **Validate against the one observed signal available.** That is the BattLeDIM leak records (`--holdout-failures`).
- **Use synthetic data for what it is good at.** It suits testing whether the pipeline retrieves, reasons and cites correctly. It does not show real-world decision quality, and the paper should say so.

## Read before reporting these numbers

The reference decisions are produced by applying the same encoded policy to the same recorded asset data, so Decision F1 measures faithful, query-correct application of policy, not real-world decision quality. To claim the latter, supply expert-labelled decisions with `evaluate --expert refs.csv` (columns `intent,pipe,action,guidelines`, guidelines `;`-separated).

The lexical flat-RAG baseline is weak by construction, because TF-IDF similarity cannot evaluate conditions such as "age > design life". It is still informative, but reviewers will expect the stronger `--llm` variant (`Flat-document RAG + LLM`), where an LLM reasons over the top-k retrieved asset cards and policy text.

All numeric priors in `config.py` (Weibull parameters, design lives, chronology, inspection intervals) and the guideline texts in `decision.py` are illustrative placeholders shaped after AWWA M28, UKWIR, ISO 55000 and ISO 24510. They are not quotations or published values from those documents and should be replaced or cited precisely before publication. Hospitals, schools and care homes are synthetic and seeded; industrial customers come from the model's demand categories.

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)). Ontology: CC BY 4.0 ([`ontology/LICENSE.md`](ontology/LICENSE.md)). The L-TOWN model (`data/networks/L-TOWN.inp`) is © KIOS Research and Innovation Center of Excellence, distributed under the European Union Public Licence v. 1.2 ([full text](data/networks/LICENSE-L-TOWN-EUPL-1.2.txt)); data derived from it, including the explorer page, remain under EUPL-1.2. See [`NOTICE.md`](NOTICE.md) for details, and cite this repository using [`CITATION.cff`](CITATION.cff) ("Cite this repository" on GitHub).

## Notes on the manuscript draft

While implementing, a few points in the LaTeX surfaced that you may want to fix:

1. The Table A.5 footnote references `\cref{subsec:properties}`, which is not defined anywhere.
2. A.3 says "the three upper-level classes are mutually disjoint", but the equation states two separate pairwise axioms. The `.ttl` uses `owl:AllDisjointClasses` over WaterNetwork, WaterNetworkElement and OperationalElement.
3. `hasCurve` is marked functional with range `Curve`, but head pumps can also carry an efficiency curve, tanks a volume curve and GPVs a head-loss curve. The `.ttl` makes it non-functional with a union domain. The "exactly one PumpCurve" axiom also does not hold for EPANET power pumps.
4. CQ7 uses `wno:value` and `wno:reason`, which are not declared in A.5. Table 1 lists `InspectionRecord` and `DecisionRecord`, which the text does not describe.
5. §3.1 claims instantiation was verified on C-Town; this repository tests Net3, Net6 and L-TOWN.
6. The bibliography file is `Demand_decay` and the commented-out highlights belong to a different paper (network right-sizing).
7. Section 3.2 says asset attributes come from literature priors. L-TOWN supports a stronger claim: part of them is inferred from the model itself (pipe series and roughness), with provenance recorded per attribute, and the sensor layout supports the SOSA/SSN link your literature review motivates.
