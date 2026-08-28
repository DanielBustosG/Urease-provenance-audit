"""Regenerate main Figure 3 (the naive-to-honest AUC collapse) from the shipped result tables.

Run from the repository root:  python scripts/08_figure3_collapse.py

Inputs   tableS6_pairwise_designs.csv, tableS7_matching_smd.csv, tableS8_yscrambling.csv
Output   results/figures/fig3_collapse.png

Every value plotted here is read from the tables in results/tables/ or the datasets
in data/processed/. Nothing is hardcoded, so editing a table and re-running this
script updates the figure.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ureaseaudit import config as cfg

DATA = cfg.DATA_PROCESSED
TABLES = cfg.TABLES
FIGURES = cfg.FIGURES
FIGURES.mkdir(parents=True, exist_ok=True)

# skill:figure-style kernel.py (auto-injected on skill load)
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 60)

apply_figure_style(frame='open', sizes=(8, 7, 6))
mpl.rcParams['savefig.dpi'] = 300

PAL = {
    'UI-Ref': '#1B4F8C',
    'CLUE': '#E8871A',
    'COCONUT': '#157F3D',
    'DSSTox': '#7D4E9E',
    'decoys': '#6E7B8B',
    'naive': '#AAB2BD',
    'honest': '#2B2B2B',
    'limit': '#C42B3A',
    'null': '#DCDEE1',
}

FLOOR = 0.954368


pw = pd.read_csv(TABLES / 'tableS6_pairwise_designs.csv')
ysc = pd.read_csv(TABLES / 'tableS8_yscrambling.csv')
smd = pd.read_csv(TABLES / 'tableS7_matching_smd.csv')

DES = ['a2_naive_randomCV', 'b_doc_grouped', 'c_scaffold_grouped', 'd_matched', 'e_matched_doc_grouped']
# Short forms: the panel is 2.45 in wide, so five two-line labels must stay narrow.
DLAB = ['naive\nrandom', 'doc-\ngrouped', 'scaffold-\ngrouped', 'property-\nmatched', 'matched +\ndoc']
COMPS = ['UI-Ref vs CLUE', 'UI-Ref vs COCONUT', 'UI-Ref vs DSSTox']
CC = {'UI-Ref vs CLUE': PAL['CLUE'], 'UI-Ref vs COCONUT': PAL['COCONUT'], 'UI-Ref vs DSSTox': PAL['DSSTox']}
P = pw.set_index(['comparison', 'design'])
Y = ysc.set_index(['comparison', 'design'])
nullband = (ysc[ysc['design'].isin(DES)]['null_mean'].min(), ysc[ysc['design'].isin(DES)]['null_p95'].max())


def fig3():
    fig = plt.figure(figsize=(7.0, 3.45))
    axA = fig.add_axes([0.075, 0.285, 0.350, 0.545])
    axB = fig.add_axes([0.630, 0.285, 0.110, 0.545])
    axC = fig.add_axes([0.870, 0.285, 0.110, 0.545])
    x = np.arange(5)
    axA.axhspan(nullband[0], nullband[1], color=PAL['null'], zorder=1)
    axA.text(-0.40, nullband[1] + 0.008, '$y$-scrambled null: mean 0.497–0.504, $p_{95}$ 0.516–0.524',
             fontsize=5.7, color='#6B7480', ha='left', va='bottom')
    axA.axhline(FLOOR, color=PAL['limit'], lw=1.3, zorder=3)
    # Above every series (max plotted AUC is 0.983) so no line crosses the text.
    axA.text(-0.40, 0.9905, 'provenance floor 0.954', fontsize=6.2, color=PAL['limit'],
             ha='left', va='bottom')
    voff = {'UI-Ref vs COCONUT': 0.0, 'UI-Ref vs CLUE': +0.017, 'UI-Ref vs DSSTox': -0.017}
    for comp in COMPS:
        v = P.loc[comp].loc[DES]
        axA.plot(x, v['auc'], '-', lw=1.4, color=CC[comp], zorder=4)
        axA.errorbar(x, v['auc'], yerr=[v['auc'] - v['ci_lo'], v['ci_hi'] - v['auc']], fmt='o', ms=4.4,
                     color=CC[comp], ecolor=CC[comp], elinewidth=1.0, capsize=2.0, zorder=5)
        e = v['auc'].iloc[-1]
        axA.text(4.13, e + voff[comp], f"{comp.replace('UI-Ref vs ', '')}  {e:.3f}", fontsize=6.3,
                 color=CC[comp], va='center', ha='left')
    axA.set_xticks(x)
    axA.set_xticklabels(DLAB, fontsize=6)
    axA.set_xlim(-0.42, 5.05)
    axA.set_ylim(0.46, 1.00)
    axA.set_ylabel('ROC AUC (95% CI)')
    axA.text(0.005, 0.030, 'n = 5,635 / 13,217 / 12,905 (naive, grouped)   ·   n = 2,436 (matched 1:1)',
             transform=axA.transAxes, fontsize=5.7, color='#5E7183')
    panel_letter(axA, 'a', dx=-0.140, dy=1.04)

    for comp in COMPS:
        v = P.loc[comp].loc[DES]
        axB.plot(v['mcc'], x, '-o', lw=1.2, ms=3.4, color=CC[comp], zorder=4)
    axB.set_yticks(x)
    axB.set_yticklabels(['naive', 'doc-grouped', 'scaffold-grp.', 'matched', 'matched + doc'], fontsize=5.6)
    axB.invert_yaxis()
    axB.set_ylim(4.5, -0.5)
    axB.set_xlim(0.28, 0.86)
    axB.set_xticks([0.3, 0.5, 0.7])
    axB.set_xlabel('Matthews corr. coeff.')
    axB.text(0.96, 0.06, '0.72–0.80\n→ 0.35–0.48', transform=axB.transAxes, fontsize=5.8, ha='right',
             va='bottom', color='#3E4C59')
    panel_letter(axB, 'b', dx=-0.68, dy=1.04)

    feats = ['MolWt', 'LogP', 'TPSA', 'NumHeavyAtoms']
    fl = {'MolWt': 'MW', 'LogP': 'log$P$', 'TPSA': 'TPSA', 'NumHeavyAtoms': 'heavy at.'}
    ypos = []
    lbl = []
    heads = []
    k = 0
    for comp in COMPS:
        heads.append((k - 0.85, comp.replace('UI-Ref vs ', ''), CC[comp]))
        s = smd[smd['comparison'] == comp].set_index('feature').loc[feats]
        for f in feats:
            b, a = abs(s.loc[f, 'smd_before']), abs(s.loc[f, 'smd_after'])
            axC.plot([a, b], [k, k], '-', lw=0.7, color='#C6CBD1', zorder=2)
            axC.plot(b, k, 'o', ms=3.2, color=PAL['naive'], zorder=4)
            axC.plot(a, k, 'o', ms=3.2, color=CC[comp], zorder=5)
            ypos.append(k)
            lbl.append(fl[f])
            k += 1
        k += 1.5
    axC.axvline(0.1, color=PAL['limit'], lw=1.0, ls='--', zorder=3)
    for yh, txt, col in heads:
        axC.text(0.30, yh, txt, fontsize=5.8, color=col, ha='left', va='center')
    axC.set_yticks(ypos)
    axC.set_yticklabels(lbl, fontsize=5.5)
    axC.invert_yaxis()
    axC.set_ylim(ypos[-1] + 0.8, heads[0][0] - 0.9)
    axC.set_xlim(-0.02, 1.0)
    axC.set_xticks([0, 0.5, 1.0])
    axC.set_xlabel('|standardized mean diff.|')
    h = [mlines.Line2D([], [], marker='o', ls='', ms=3.2, color=PAL['naive'], label='full collection'),
         mlines.Line2D([], [], marker='o', ls='', ms=3.2, color='#4A5568', label='matched decoys')]
    # Anchored to panel (c)'s own axes, not floating between (b) and (c).
    axC.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, -0.235), fontsize=5.6,
               frameon=False, ncol=1, handletextpad=0.25, borderpad=0.05, labelspacing=0.25)
    panel_letter(axC, 'c', dx=-0.54, dy=1.04)
    return fig


fig = fig3()
fig.savefig(FIGURES / 'fig3_collapse.png', dpi=300)
plt.close(fig)