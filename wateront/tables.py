"""Tolerant reading of the tables utilities actually produce (Excel exports, inspection logs).

The same rules are implemented in the web explorer (`parseTable`, `parseRegister` in
web_template.html) and checked for agreement by tests/test_web_parity.py:

* separator detected from the header line: comma, semicolon (European Excel) or tab;
* Excel's byte-order mark is ignored;
* headers are matched case-insensitively, ignoring spaces and punctuation, with common
  alternative names (e.g. "Pipe ID", "Last inspection", "Installation year");
* years may be written as dates ("03/05/2019", "2019-05-03"): the four-digit year is used;
* materials may be written as names ("Cast iron", "uPVC", "MDPE");
* decimal commas ("4,0") are accepted in numbers.
"""
from __future__ import annotations

import re

ALIASES = {
    "pipe": ["pipe_id", "pipeid", "id", "asset_id", "assetid", "link", "link_id", "pipe_name", "name", "main", "main_id"],
    "install_year": ["installation_year", "year_installed", "installed", "install_date", "installation_date",
                     "year_laid", "laid", "install", "date_installed", "year_of_installation"],
    "material": ["pipe_material", "mat", "material_type"],
    "condition_grade": ["condition", "grade", "condition_score", "condition_rating", "cond_grade"],
    "last_inspection_year": ["last_inspection", "last_inspected", "inspection_year", "inspection_date",
                             "date_inspected", "inspected", "inspection", "date_of_inspection", "survey_date",
                             "survey_year", "last_survey"],
    "soil_corrosivity": ["soil", "corrosivity", "soil_corrosion", "soil_aggressivity"],
    "failure_years": ["failure_year", "failure_dates", "failure_date", "burst_years", "burst_dates"],
    "failure_modes": ["failure_mode", "burst_modes", "burst_mode"],
    "lining": ["liner", "lining_type"],
    "joint": ["joint_type", "joints"],
    "dma": ["zone", "supply_zone", "district", "district_metered_area", "dma_id", "dma_name", "dma_code",
            "zone_id", "zone_name", "pressure_zone", "sector"],
    "element": ["node", "junction", "location", "sensor_location", "asset"],
    "type": ["sensor_type", "kind", "customer_type", "facility_type", "category"],
}
_ALIAS = {a: canon for canon, names in ALIASES.items() for a in names}

MATERIAL_NAMES = {   # keys compared without case, spaces or punctuation
    "ci": "CI", "castiron": "CI", "greyiron": "CI", "grayiron": "CI", "spuniron": "CI", "iron": "CI",
    "di": "DI", "ductileiron": "DI", "ductile": "DI", "sgiron": "DI",
    "ac": "AC", "asbestoscement": "AC", "asbestos": "AC", "fibrecement": "AC", "fibercement": "AC",
    "pvc": "PVC", "upvc": "PVC", "pvcu": "PVC", "mpvc": "PVC", "pvco": "PVC", "polyvinylchloride": "PVC",
    "hdpe": "HDPE", "pe": "HDPE", "mdpe": "HDPE", "polyethylene": "HDPE", "polyethene": "HDPE",
    "pe80": "HDPE", "pe100": "HDPE", "alkathene": "HDPE",
    "steel": "Steel", "st": "Steel", "galvanisedsteel": "Steel", "galvanizedsteel": "Steel",
}
SOIL_NAMES = {"low": "Low", "medium": "Medium", "med": "Medium", "moderate": "Medium", "high": "High"}
NONE_WORDS = {"none", "0", "-", "no", "nil"}


def norm_header(h: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", h.replace("\ufeff", "").strip().lower()).strip("_")
    return _ALIAS.get(s, s)


def detect_delimiter(first_line: str) -> str:
    counts, q = {",": 0, ";": 0, "\t": 0}, False
    for c in first_line:
        if c == '"':
            q = not q
        elif not q and c in counts:
            counts[c] += 1
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] else ","


def parse_csv(text: str) -> tuple[list[str], list[list[str]], str]:
    """Header (normalised), data rows, and the separator used. Handles quotes and CRLF."""
    text = text.lstrip("\ufeff")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    delim = detect_delimiter(first)
    rows, row, cell, q, i = [], [], "", False, 0
    while i < len(text):
        c = text[i]
        if q:
            if c == '"' and i + 1 < len(text) and text[i + 1] == '"':
                cell += '"'; i += 1
            elif c == '"':
                q = False
            else:
                cell += c
        elif c == '"':
            q = True
        elif c == delim:
            row.append(cell); cell = ""
        elif c in "\r\n":
            if c == "\r" and i + 1 < len(text) and text[i + 1] == "\n":
                i += 1
            row.append(cell); rows.append(row); row, cell = [], ""
        else:
            cell += c
        i += 1
    if cell or row:
        row.append(cell); rows.append(row)
    rows = [r for r in rows if any(x.strip() for x in r)]
    if not rows:
        return [], [], delim
    return [norm_header(h) for h in rows[0]], rows[1:], delim


def read_table(path: str) -> tuple[list[str], list[list[str]], str]:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        return parse_csv(fh.read())


def year_of(v: str) -> int | None:
    """'2019', '2019.0', '2019-05-03', '03/05/2019' -> 2019; None if no single 4-digit year."""
    v = v.strip()
    if re.fullmatch(r"\d{4}(\.0*)?", v):
        return int(float(v))
    years = re.findall(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", v)
    return int(years[0]) if len(years) == 1 else None


def number_of(v: str) -> float | None:
    v = v.strip()
    if "," in v and "." not in v:
        v = v.replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return None


def material_of(v: str) -> str | None:
    return MATERIAL_NAMES.get(re.sub(r"[^a-z0-9]", "", v.lower()))


def soil_of(v: str) -> str | None:
    return SOIL_NAMES.get(v.strip().lower())


def split_list(v: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;,|]", v) if x.strip()]
