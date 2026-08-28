"""Regenerate Figure S2 (per-model benchmark detail) from the shipped result tables.

Run from the repository root:  python scripts/12_figureS2_benchmark.py

Inputs   tableS24_cascade_benchmark.csv
Output   results/figures/figS2_benchmark_detail.png

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

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---- skill:figure-style helpers (auto-injected) ----
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

# ---- global style ----
apply_figure_style(frame='open', sizes=(8, 7, 6))
mpl.rcParams['savefig.dpi'] = 300

# ---- palette ----
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

# ---- load data ----
cb = pd.read_csv(TABLES / 'tableS24_cascade_benchmark.csv')

# ---- constants ----
STAGE_ORDER = ['S1B_decoy', 'S1A_measured', 'S2_potency', 'S3_toxicophore', 'S4_clinical']
STAGE_SCHEME = {
    'S1B_decoy': 'document',
    'S1A_measured': 'document',
    'S2_potency': 'document',
    'S3_toxicophore': 'scaffold',
    'S4_clinical': 'scaffold',
}
MODELS = ['RF', 'XGBoost', 'LightGBM', 'logreg', 'kNN_Tanimoto', 'majority', 'RF_yscrambled']
MLAB = {
    'RF': 'random forest',
    'XGBoost': 'XGBoost',
    'LightGBM': 'LightGBM',
    'logreg': 'logistic regression',
    'kNN_Tanimoto': '$k$NN Tanimoto',
    'majority': 'majority class',
    'RF_yscrambled': '$y$-scrambled RF',
}
FEATC = {'desc': '#9FB3C8', 'ECFP4': '#6BA3D6', 'union': PAL['UI-Ref']}

# ---- figS2 ----
def figS2():
    fig = plt.figure(figsize=(7.0, 5.30))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.52, left=0.135, right=0.980, top=0.905, bottom=0.150)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    ttl = ['1B inhibitor vs decoy', '1A active vs inactive', '2 potent vs weak',
           '3 alert-flagged vs not', '4 CLUE launched vs not']
    for i, st in enumerate(STAGE_ORDER):
        ax = axes[i]
        sch = STAGE_SCHEME[st]
        s = cb[(cb['stage'] == st) & (cb['scheme'] == sch)]
        ys = []; ylab = []; k = 0
        for m in MODELS:
            sm = s[s['model'] == m]
            for feat in ['union', 'ECFP4', 'desc']:
                r = sm[sm['features'] == feat]
                if len(r) == 0:
                    continue
                r = r.iloc[0]
                ax.errorbar(r['AUC'], k,
                            xerr=[[max(r['AUC'] - r['AUC_lo'], 0)], [max(r['AUC_hi'] - r['AUC'], 0)]],
                            fmt='o', ms=3.0, color=FEATC[feat], ecolor=FEATC[feat], elinewidth=0.7,
                            capsize=1.2, zorder=4)
                ys.append(k)
                ylab.append(MLAB[m] if (feat == 'union' or len(sm) == 1) else '')
                k += 1
            k += 0.55
        ax.axvline(0.5, color='#B0B7BE', lw=0.7, ls=':', zorder=2)
        ax.set_yticks(ys)
        ax.set_yticklabels(ylab, fontsize=5.2)
        ax.invert_yaxis()
        ax.set_ylim(k - 0.4, -0.7)
        ax.set_xlim(0.42, 1.02)
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.set_xlabel('ROC AUC (95% CI)')
        # Panel identifier only: names which cascade stage this panel shows.
        # The CV scheme moved to the caption (it is stated in panel (f) as well).
        ax.set_title(ttl[i], fontsize=6.8, loc='left', pad=4)
        panel_letter(ax, 'abcdef'[i], dx=-0.44, dy=1.13)
    # The sixth slot held a prose block (feature-block definitions and CV rationale);
    # that is caption material and has moved there. What must stay in the figure is
    # the colour key, because it is the only way to tell the three marks apart.
    ax = axes[5]
    ax.set_axis_off()
    ax.legend(handles=[mlines.Line2D([], [], marker='o', ls='', ms=4.2, color=FEATC[f], label=l)
                       for f, l in [('union', '25 descriptors + ECFP4'),
                                    ('ECFP4', 'ECFP4 only (2,048-bit)'),
                                    ('desc', '25 descriptors only')]],
              loc='center left', bbox_to_anchor=(0.02, 0.62), fontsize=6.2, frameon=False,
              handletextpad=0.4, labelspacing=0.45)
    return fig

_S2 = figS2
def figS2():
    fig = _S2()
    for i, ax in enumerate(fig.axes[:5]):
        ax.set_title(["1B inhibitor vs decoy", "1A active vs inactive", "2 potent vs weak",
                      "3 alert-flagged vs not", "4 CLUE launched vs not"][i],
                     fontsize=6.8, loc='left', pad=4)
    return fig

fig = figS2()
fig.savefig(FIGURES / 'figS2_benchmark_detail.png', dpi=300)
plt.close(fig)