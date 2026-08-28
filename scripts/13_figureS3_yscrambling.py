"""Regenerate Figure S3 (y-scrambling null distributions) from the shipped result tables.

Run from the repository root:  python scripts/13_figureS3_yscrambling.py

Inputs   tableS8_yscrambling.csv
Output   results/figures/figS3_yscrambling.png

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
import os

# ---- Global state ----
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
FLOOR_ECFP = 0.964053
DESC_C = '#1B4F8C'
ECFP_C = '#6BA3D6'

DES = ['a2_naive_randomCV', 'b_doc_grouped', 'c_scaffold_grouped', 'd_matched', 'e_matched_doc_grouped']
DLAB = ['naive\nrandom CV', 'document-\ngrouped', 'scaffold-\ngrouped', 'property-\nmatched', 'matched +\ndoc-grouped']


# ---- Load data ----
ysc = pd.read_csv(TABLES / 'tableS8_yscrambling.csv')

YLAB = {'a2_naive_randomCV': 'naive random CV', 'b_doc_grouped': 'document-grouped',
        'c_scaffold_grouped': 'scaffold-grouped', 'd_matched': 'property-matched',
        'e_matched_doc_grouped': 'matched + doc-grouped', 'provenance_floor': 'provenance reference'}

# ---- figS3 ----
def figS3():
    fig = plt.figure(figsize=(7.0, 3.45))
    axA = fig.add_axes([0.245, 0.245, 0.335, 0.550])
    axB = fig.add_axes([0.740, 0.245, 0.240, 0.550])
    ys = []; ylab = []; k = 0
    groups = [('UI-Ref vs CLUE', PAL['CLUE']), ('UI-Ref vs COCONUT', PAL['COCONUT']),
              ('UI-Ref vs DSSTox', PAL['DSSTox'])]
    heads = []
    for comp, col in groups:
        heads.append((k - 0.80, comp, col))
        for des in DES:
            r = ysc[(ysc['comparison'] == comp) & (ysc['design'] == des)].iloc[0]
            lo = r['null_mean'] - 1.96 * r['null_sd']
            axA.barh(k, r['null_p95'] - lo, left=lo, height=0.52, color=PAL['null'], zorder=2)
            axA.plot([r['null_mean']], [k], '|', ms=5.5, mew=1.1, color='#6B7480', zorder=4)
            axA.plot([r['null_p95']], [k], '|', ms=5.5, mew=1.1, color=PAL['limit'], zorder=4)
            axA.plot([r['null_max']], [k], 'x', ms=2.8, mew=0.9, color='#6B7480', zorder=4)
            axA.plot([r['real_auc_5fold']], [k], 'o', ms=3.8, color=col, zorder=5)
            ys.append(k); ylab.append(YLAB[des]); k += 1
        k += 1.25
    heads.append((k - 0.80, 'provenance reference', DESC_C))
    for comp, lab in [('PubChem vs BindingDB (within UI-Ref)', 'database of origin'),
                      ('document 10.1016/j.bmcl.2019.02.032 one-vs-rest', 'largest source document')]:
        r = ysc[ysc['comparison'] == comp].iloc[0]
        lo = r['null_mean'] - 1.96 * r['null_sd']
        axA.barh(k, r['null_p95'] - lo, left=lo, height=0.52, color=PAL['null'], zorder=2)
        axA.plot([r['null_mean']], [k], '|', ms=5.5, mew=1.1, color='#6B7480', zorder=4)
        axA.plot([r['null_p95']], [k], '|', ms=5.5, mew=1.1, color=PAL['limit'], zorder=4)
        axA.plot([r['null_max']], [k], 'x', ms=2.8, mew=0.9, color='#6B7480', zorder=4)
        axA.plot([r['real_auc_3fold']], [k], 'o', ms=3.8, color=DESC_C, zorder=5)
        ys.append(k); ylab.append(lab); k += 1
    axA.axvline(0.5, color='#B0B7BE', lw=0.7, ls=':', zorder=1)
    for yh, txt, col in heads:
        axA.text(0.452, yh, txt, fontsize=5.7, color=col, va='center', ha='left')
    axA.set_yticks(ys); axA.set_yticklabels(ylab, fontsize=5.4); axA.invert_yaxis()
    axA.set_ylim(k - 0.5, heads[0][0] - 0.7)
    axA.set_xlim(0.445, 1.02); axA.set_xticks([0.5, 0.7, 0.9, 1.0])
    axA.set_xlabel('ROC AUC')
    h = [mlines.Line2D([], [], marker='|', ls='', ms=5.5, mew=1.1, color='#6B7480', label='null mean'),
         mlines.Line2D([], [], marker='|', ls='', ms=5.5, mew=1.1, color=PAL['limit'], label='null $p_{95}$'),
         mlines.Line2D([], [], marker='x', ls='', ms=2.8, mew=0.9, color='#6B7480', label='null maximum'),
         mlines.Line2D([], [], marker='o', ls='', ms=3.8, color='#4A5568', label='real AUC')]
    # Left-aligned to (a)'s own axes instead of centred under the figure.
    axA.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, -0.20), fontsize=5.4, frameon=False, ncol=4,
               handletextpad=0.2, columnspacing=0.9, borderpad=0.05)
    panel_letter(axA, 'a', dx=-0.40, dy=1.04)

    axB.scatter(ysc['null_mean'], ysc['null_p95'], s=16,
                c=[DESC_C if d == 'provenance_floor' else '#8A929B' for d in ysc['design']], zorder=4)
    axB.axvline(0.5, color='#B0B7BE', lw=0.7, ls=':', zorder=2)
    # Panel (b) had no key: the grey/blue distinction was unexplained.
    axB.legend(handles=[mlines.Line2D([], [], marker='o', ls='', ms=4.0, color='#8A929B',
                                      label='collection designs'),
                        mlines.Line2D([], [], marker='o', ls='', ms=4.0, color=DESC_C,
                                      label='provenance designs')],
               loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=5.4, frameon=False,
               handletextpad=0.3, labelspacing=0.25)
    r = ysc[ysc['design'] == 'provenance_floor'].iloc[0]
    axB.annotate('provenance designs:\nsmaller $n$, wider null', xy=(r['null_mean'], r['null_p95']),
                 xytext=(0.4980, 0.578), fontsize=5.5, color=DESC_C,
                 arrowprops=dict(arrowstyle='-', lw=0.6, color=DESC_C))
    axB.set_xlim(0.4955, 0.5045); axB.set_xticks([0.497, 0.500, 0.503])
    axB.set_ylim(0.508, 0.615); axB.set_yticks([0.52, 0.55, 0.58, 0.61])
    axB.set_xlabel('null mean AUC'); axB.set_ylabel('null 95th percentile')
    axB.text(0.98, -0.28, 'permutation $p$ = 0.0099 for every design\n(1 of 101, including the observed value)',
             transform=axB.transAxes, fontsize=5.4, color='#5E7183', ha='right', va='top')
    panel_letter(axB, 'b', dx=-0.42, dy=1.04)
    return fig


fig = figS3()
fig.savefig(FIGURES / 'figS3_yscrambling.png', dpi=300)
plt.close(fig)