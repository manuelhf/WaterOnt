"""Figures for the paper, drawn at the final printed width (elsarticle preprint,
12 pt: \\textwidth = 390 pt) with LaTeX fonts, so they are included at 100% scale.

    python paper/experiments.py   # first, to produce paper/results/
    python paper/figures.py       # writes paper/figures/*.pdf

Figures carry no titles; titles belong in the LaTeX captions.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wateront.config import Config  # noqa: E402
from wateront.pipeline import build_knowledge_graph  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures")
LTOWN = os.path.join(os.path.dirname(HERE), "data", "networks", "L-TOWN.inp")
TEXTWIDTH_IN = 390 / 72.27

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
RISK = {"Negligible": "#A6BDB7", "Tolerable": "#E1B941", "Significant": "#D9822B", "Intolerable": "#A8231C"}
ACTION = {"Monitor": "#C9CFD2", "Inspection": "#2F6FB0", "Rehabilitation": "#7A4FB5", "Replacement": "#A8231C"}
ZONE = {"PRV-1+PRV-2": "#56B4E9", "PRV-3": "#E69F00", "PUMP_1+T1": "#009E73"}
ZONE_LABEL = {"PRV-1+PRV-2": "Main zone (fed by PRV-1, PRV-2)", "PRV-3": "Sub-zone (fed by PRV-3)",
              "PUMP_1+T1": "Tank zone (fed by PUMP\\_1, T1)"}


def panel(ax, s, x=0.0, y=1.0):
    """Panel label just above the top-left corner of the axes, clear of the data."""
    ax.text(x, y + 0.035, s, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


def network_segments(res):
    wn = res.wn
    segs, names, widths = [], [], []
    for name, link in wn.links():
        a, b = link.start_node.coordinates, link.end_node.coordinates
        segs.append([a, b]); names.append(name)
        widths.append(0.35 + link.diameter * 1000 / 220 if type(link).__name__ == "Pipe" else 1.6)
    return segs, names, np.array(widths)


def map_axes(ax, res):
    ax.set_aspect("equal"); ax.axis("off")
    xs = [n.coordinates[0] for _, n in res.wn.nodes()]; ys = [n.coordinates[1] for _, n in res.wn.nodes()]
    pad = 0.01 * (max(xs) - min(xs))
    ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(min(ys) - pad, max(ys) + pad)


def map_aspect(res):
    xs = [n.coordinates[0] for _, n in res.wn.nodes()]; ys = [n.coordinates[1] for _, n in res.wn.nodes()]
    return (max(xs) - min(xs)) / (max(ys) - min(ys))


def map_figure(res, panels=1, legend_in=0.42, gap_in=0.12):
    """Figure whose map axes span the full text width, sized from the network's extent."""
    W = TEXTWIDTH_IN
    mh = W / map_aspect(res) * 1.02
    H = panels * (mh + legend_in) + (panels - 1) * gap_in
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k in range(panels):
        bottom = H - (k + 1) * mh - k * (legend_in + gap_in)
        axes.append(fig.add_axes([0, bottom / H, 1, mh / H]))
    return fig, axes


def draw_sources(ax, res, size=16):
    wn = res.wn
    for n in wn.reservoir_name_list:
        x, y = wn.get_node(n).coordinates
        ax.scatter([x], [y], marker="s", s=size, color="black", zorder=6, linewidths=0)
    for n in wn.tank_name_list:
        x, y = wn.get_node(n).coordinates
        ax.scatter([x], [y], marker="D", s=size * 0.9, color="black", zorder=6, linewidths=0)


def fig_case_study(res):
    kg, wn = res.kg, res.wn
    segs, names, widths = network_segments(res)
    zone_of = {}
    for z in kg.nodes("PressureZone"):
        for p in kg.inc(z, "IN_ZONE"):
            if "Pipe" in kg.node(p)["labels"]:
                zone_of[kg.node(p)["name"]] = kg.node(z)["name"]
    colors = [ZONE.get(zone_of.get(n), "#7F7F7F") for n in names]
    fig, (ax,) = map_figure(res, 1, legend_in=0.5)
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, capstyle="round", zorder=2))
    devices = [(n, l) for n, l in wn.links() if type(l).__name__ != "Pipe"]
    for n, l in devices:
        a, b = l.start_node.coordinates, l.end_node.coordinates
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=2.4, solid_capstyle="butt", zorder=4)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx, dy, ha = {"PRV-2": (-6, -2, "right"), "PUMP_1": (-5, 5, "right"), "PRV-1": (-6, 6, "right")}.get(n, (5, 3, "left"))
        ax.annotate(n.replace("_", "\\_"), (mx, my), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha=ha, zorder=7, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8))
    ps = [kg.node(s)["name"].split(" @ ")[1] for s in kg.nodes("PressureSensor")]
    pxy = np.array([wn.get_node(n).coordinates for n in ps])
    ax.scatter(pxy[:, 0], pxy[:, 1], s=14, facecolors="white", edgecolors="black", linewidths=0.7, zorder=5)
    draw_sources(ax, res)
    map_axes(ax, res)
    handles = [Line2D([], [], color=ZONE[z], lw=2.2, label=ZONE_LABEL[z]) for z in ZONE]
    handles += [Line2D([], [], color="black", lw=2.4, label="PRV or pump"),
                Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=0.7, ms=4, label="Pressure sensor"),
                Line2D([], [], marker="s", ls="", color="black", ms=4, label="Reservoir"),
                Line2D([], [], marker="D", ls="", color="black", ms=3.6, label="Tank")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, columnspacing=1.0,
              handlelength=1.6, borderaxespad=0, fontsize=7.5)
    fig.savefig(os.path.join(OUT, "fig_ltown_case.pdf"))
    plt.close(fig)


def fig_risk_action(res):
    kg = res.kg
    segs, names, widths = network_segments(res)
    pipes = {n for n in res.wn.pipe_name_list}
    act = {d.facts["pipe"]: d.action for d in res.decisions}
    risk = res.risk.overall_risk.to_dict()
    fig, axes = map_figure(res, 2, legend_in=0.3, gap_in=0.1)
    for ax, values, palette, lab in [(axes[0], risk, RISK, "(a)"), (axes[1], act, ACTION, "(b)")]:
        order = {k: i for i, k in enumerate(palette)}
        idx = sorted(range(len(names)), key=lambda i: order.get(values.get(names[i]), -1))
        ax.add_collection(LineCollection([segs[i] for i in idx],
                                         colors=[palette.get(values.get(names[i]), "black") for i in idx],
                                         linewidths=[widths[i] * (1.0 if names[i] in pipes else 1.0) for i in idx],
                                         capstyle="round"))
        draw_sources(ax, res, size=12)
        map_axes(ax, res)
        counts = pd.Series([values[n] for n in names if n in values]).value_counts()
        ax.legend(handles=[Line2D([], [], color=c, lw=2.4, label=f"{k} ({counts.get(k, 0)})") for k, c in palette.items()],
                  loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=4, handlelength=1.6, borderaxespad=0)
        ax.text(1.0, 1.0, lab, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_ltown_risk_action.pdf"))
    plt.close(fig)


def fig_banding():
    b = pd.read_csv(os.path.join(RES, "banding.csv"))
    cfg = Config()
    limits = [t for t, _ in cfg.pof_bands[:3]]
    style = {"Net3": dict(color="#9AA5A9", ls="--"), "L-TOWN": dict(color="#1F6F8B", ls="-")}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.15), sharey=True)
    for ax, col, lab, xl in [(axes[0], "pof_segment", "(a)", "Failure probability of the segment"),
                             (axes[1], "pof_km", "(b)", "Failure probability per km")]:
        for net, g in b.groupby("network"):
            v = np.sort(g[col].values)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", lw=1.2,
                    label=f"{net} (median length {g.length_m.median():.0f} m)", **style[net])
        for t in limits:
            ax.axvline(t, color="black", lw=0.5, ls=":")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.set_xlabel(xl)
        panel(ax, lab)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Share of pipes")
    for t, name in zip([0.0] + limits, ["Low", "Medium", "High", "Very high"]):
        nxt = ([*limits, 1.0])[[0.0, *limits].index(t)]
        axes[1].text((t + nxt) / 2, 0.035, name, ha="center", fontsize=6.5, color="#444444",
                     rotation=90 if name in ("Medium", "High") else 0, va="bottom")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.04), handlelength=1.8, fontsize=7)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(os.path.join(OUT, "fig_banding.pdf"))
    plt.close(fig)


def fig_sensitivity():
    d = pd.read_csv(os.path.join(RES, "sensitivity.csv"))
    import json
    base = json.load(open(os.path.join(RES, "meta.json")))["base_replacements"]
    groups = [("Weibull scale", "Weibull scale $\\eta$", "#A8231C", "o"),
              ("Design life", "Design life", "#7A4FB5", "s"),
              ("Inspection intervals", "Inspection intervals", "#2F6FB0", "^"),
              ("PoF band limits", "Failure-probability band limits", "#D9822B", "D"),
              ("Hazard scale", "Hazard scale $\\kappa$", "#7F7F7F", "v")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.3))
    for g, lab, c, mk in groups:
        q = d[d.group == g].sort_values("factor")
        x = [0.8, 0.9, 1.0, 1.1, 1.2]
        y1 = list(q.share_changed.values[:2]) + [0.0] + list(q.share_changed.values[2:])
        y2 = list(q.replacements.values[:2]) + [base] + list(q.replacements.values[2:])
        axes[0].plot(x, np.array(y1) * 100, marker=mk, ms=3.2, lw=1.0, color=c, label=lab)
        axes[1].plot(x, y2, marker=mk, ms=3.2, lw=1.0, color=c, label=lab)
    axes[0].set_ylabel("Pipes whose action changes (\\%)")
    axes[1].set_ylabel("Pipes recommended for replacement")
    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.set_xlabel("Parameter multiplier")
        ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
        ax.axvline(1.0, color="black", lw=0.5, ls=":")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        panel(ax, lab)
    axes[0].set_ylim(0, None); axes[1].set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(os.path.join(OUT, "fig_sensitivity.pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = build_knowledge_graph(LTOWN, Config(seed=42), verbose=False)
    fig_case_study(res)
    fig_risk_action(res)
    fig_banding()
    fig_sensitivity()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
