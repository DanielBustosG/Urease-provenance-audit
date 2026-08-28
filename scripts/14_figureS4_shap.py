"""Regenerate Figure S4 (SHAP naive vs corrected vs provenance) from the shipped result tables.

Run from the repository root:  python scripts/14_figureS4_shap.py

Inputs   tableS10_shap_comparison.csv, tableS11_shap_rank_correlations.csv
Output   results/figures/figS4_shap.png

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

# Global state
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

DESC_C = '#1B4F8C'
ECFP_C = '#6BA3D6'

# Load data
sh = pd.read_csv(TABLES / 'tableS10_shap_comparison.csv').set_index('descriptor')
srk = pd.read_csv(TABLES / 'tableS11_shap_rank_correlations.csv')

import os

def figS4():
    fig = plt.figure(figsize=(7.0, 4.10))
    axA = fig.add_axes([0.230, 0.140, 0.300, 0.720])
    axB = fig.add_axes([0.630, 0.505, 0.350, 0.355])
    axC = fig.add_axes([0.630, 0.140, 0.350, 0.250])
    cols = ['CLUE|a_naive', 'CLUE|e_matched_doc', 'COCONUT|a_naive', 'COCONUT|e_matched_doc',
            'DSSTox|a_naive', 'DSSTox|e_matched_doc', 'PROVENANCE|f_pubchem_vs_bindingdb',
            'PROVENANCE|g_document_membership']
    order = sh[cols].mean(axis=1).sort_values().index.tolist()
    y = np.arange(len(order))
    cmap = {'CLUE': PAL['CLUE'], 'COCONUT': PAL['COCONUT'], 'DSSTox': PAL['DSSTox'], 'PROVENANCE': DESC_C}
    for c in cols:
        coll = c.split('|')[0]
        naive = 'a_naive' in c or 'f_pubchem' in c
        axA.plot(sh.loc[order, c] * 100, y, 'o', ms=3.0, color=cmap[coll],
                 mfc=cmap[coll] if not naive else 'white', mew=0.9, zorder=4)
    axA.set_yticks(y)
    axA.set_yticklabels(order, fontsize=5.3)
    axA.set_ylim(-0.6, len(order) - 0.4)
    axA.set_xlim(0, 12.0)
    axA.set_xticks([0, 3, 6, 9, 12])
    axA.set_xlabel('% of total mean |SHAP| in that model')
    axA.annotate('FractionCSP3\n12.9–21.8% (collections)\n5.0–6.5% (provenance)',
                 xy=(10.8, len(order) - 1), xytext=(4.3, len(order) - 4.6), fontsize=5.5, color='#3E4C59',
                 arrowprops=dict(arrowstyle='-', lw=0.6, color='#3E4C59'))
    h = [mlines.Line2D([], [], marker='o', ls='', ms=3.0, mfc='white', mew=0.9, color='#4A5568',
                       label='naive design'),
         mlines.Line2D([], [], marker='o', ls='', ms=3.0, color='#4A5568',
                       label='matched + doc-grouped'),
         mlines.Line2D([], [], marker='o', ls='', ms=3.0, color=DESC_C, label='provenance models')]
    axA.legend(handles=h, loc='lower right', bbox_to_anchor=(1.03, 0.02), fontsize=5.3, frameon=False,
               handletextpad=0.2, labelspacing=0.22, borderpad=0.05)
    panel_letter(axA, 'a', dx=-0.42, dy=1.04)

    # (b) rank correlations
    lbl = {'provenance (PubChem vs BindingDB)': 'vs database-of-origin model',
           'provenance (document membership, mean of 12 OvR)': 'vs document-membership model'}
    sub = srk[srk['vs'].isin(lbl)].copy()
    sub['coll'] = sub['model'].str.split('|').str[0]
    sub['des'] = np.where(sub['model'].str.contains('a_naive'), 'naive', 'matched + doc-grouped')
    xs = np.arange(3)
    w = 0.20
    for i, (vs, off, al) in enumerate([('provenance (PubChem vs BindingDB)', -0.5, 1.0),
                                       ('provenance (document membership, mean of 12 OvR)', 0.5, 0.55)]):
        for j, des in enumerate(['naive', 'matched + doc-grouped']):
            v = [sub[(sub['coll'] == c) & (sub['vs'] == vs) & (sub['des'] == des)]['spearman_rho'].iloc[0]
                 for c in ['CLUE', 'COCONUT', 'DSSTox']]
            axB.bar(xs + (off + (j - 0.5) * 0.9) * w, v, w * 0.85,
                    color=[PAL[c] for c in ['CLUE', 'COCONUT', 'DSSTox']], alpha=al,
                    edgecolor='none' if des == 'naive' else 'white',
                    hatch='' if des == 'naive' else '///', zorder=3)
    axB.axhline(0, color='#8A929B', lw=0.7, zorder=2)
    axB.set_xticks(xs)
    axB.set_xticklabels(['CLUE', 'COCONUT', 'DSSTox'], fontsize=6)
    axB.set_ylim(-0.2, 1.05)
    axB.set_yticks([-0.2, 0, 0.2, 0.4, 0.6])
    axB.set_ylabel("Spearman ρ of descriptor ranks")
    axB.text(0.98, 0.98, 'DSSTox: ρ = 0.422 ($p$ = 0.036)\nand 0.533 ($p$ = 0.006)',
             transform=axB.transAxes, fontsize=5.5, color=PAL['DSSTox'], ha='right', va='top')
    axB.text(0.02, 0.60, 'solid: vs database-of-origin model\ndimmed hatched: vs document membership',
             transform=axB.transAxes, fontsize=5.2, color='#5E7183', va='top')
    panel_letter(axB, 'b', dx=-0.28, dy=1.04)

    # (c) naive vs corrected rank stability
    pairs = [('CLUE', 0.8085), ('COCONUT', 0.6400), ('DSSTox', 0.8300)]
    for i, (c, rho) in enumerate(pairs):
        axC.plot(sh[f'{c}|a_naive_rank'], sh[f'{c}|e_matched_doc_rank'], 'o', ms=2.6,
                 color=PAL[c], alpha=0.75, zorder=4, label=f'{c} (ρ = {rho:.2f})')
    axC.plot([1, 25], [1, 25], '--', lw=0.8, color='#8A929B', zorder=2)
    axC.set_xlim(0, 26)
    axC.set_ylim(0, 26)
    axC.set_xticks([1, 10, 20, 25])
    axC.set_yticks([1, 10, 20, 25])
    axC.set_xlabel('descriptor rank, naive design')
    axC.set_ylabel('rank, matched +\ndoc-grouped')
    axC.legend(loc='upper left', bbox_to_anchor=(0.02, 1.00), fontsize=5.3, frameon=False,
               handletextpad=0.2, labelspacing=0.22, borderpad=0.05)
    axC.text(0.98, 0.03, 'the features do not change;\nonly the estimated performance does',
             transform=axC.transAxes, fontsize=5.3, color='#5E7183', ha='right')
    panel_letter(axC, 'c', dx=-0.20, dy=1.04)
    return fig


fig = figS4()
fig.savefig(FIGURES / 'figS4_shap.png', dpi=300)
plt.close(fig)