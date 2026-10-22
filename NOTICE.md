# Licensing and attribution

| Part | Licence | Holder |
|---|---|---|
| Code (`wateront/`, `tests/`, `demo/WaterOnt_Colab.ipynb`, scripts) | MIT, see `LICENSE` | Manuel Herrera et al. |
| Ontology (`ontology/`) | CC BY 4.0, see `ontology/LICENSE.md` | Manuel Herrera et al. |
| L-TOWN network model (`data/networks/L-TOWN.inp`) | EUPL-1.2, see `data/networks/LICENSE-L-TOWN-EUPL-1.2.txt` | KIOS Research and Innovation Center of Excellence, University of Cyprus |
| Web demonstrator (`demo/index.html`) | MIT for the page code; the embedded network data are derived from L-TOWN and remain under EUPL-1.2 | as above |

## L-TOWN (BattLeDIM)

L-TOWN was developed by the KIOS Research and Innovation Center of Excellence for the
Battle of the Leakage Detection and Isolation Methods (BattLeDIM 2020) and is
distributed under the European Union Public Licence v. 1.2 (EUPL © the European Union
2007, 2016). Please cite the BattLeDIM dataset and its accompanying paper when using it.

`data/networks/L-TOWN.inp` is the original `L-TOWN_v2_Model.inp`, renamed and otherwise
unmodified. Its own header and notices are kept intact.

**Modification notice (EUPL-1.2, Article 5).** `demo/index.html` and any page produced by
`python -m wateront web` or `--json` contain data derived from L-TOWN: node coordinates,
pipe topology and dimensions, sensor locations, demand categories and simulated pressures
and flows, converted to JSON and combined with synthetic asset data.
Modified 22 October 2026 by Manuel Herrera et al.

## Synthetic data and policy text

All asset histories (materials, ages, condition, failures, inspections) and critical
facilities other than industrial customers are synthetic. The engineering guideline texts
are paraphrased stand-ins written for this project, not quotations from AWWA, UKWIR or ISO
documents.

## Interpretation

This summary is not legal advice. In case of doubt the licence texts prevail.
