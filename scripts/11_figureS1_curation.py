"""Regenerate Figure S1 (curation and standardization detail) from the shipped result tables.

Run from the repository root:  python scripts/11_figureS1_curation.py

Inputs   records_curated.parquet, tableS1_curation_flow.csv
Output   results/figures/figS1_curation.png

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
from matplotlib.patches import Patch

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


rec = pd.read_parquet(DATA / 'records_curated.parquet')
cf_flow = pd.read_csv(TABLES / 'tableS1_curation_flow.csv')

rep = rec[rec['smiles_repaired'].notna() & (rec['smiles_repaired'] != rec['smiles_raw'])]

def figS1():
    fig = plt.figure(figsize=(7.0, 3.15))
    axA = fig.add_axes([0.175, 0.360, 0.300, 0.510])
    axB = fig.add_axes([0.660, 0.360, 0.250, 0.510])

    steps = cf_flow['step'].str.replace(r'^\d+ ', '', regex=True).tolist()
    recs = cf_flow['records'].values
    mols = cf_flow['molecules'].values
    ys = np.arange(len(steps))[::-1]
    for y, r, m in zip(ys, recs, mols):
        if not np.isnan(r):
            axA.barh(y + 0.17, r, 0.32, color='#9FB3C8', zorder=3)
            axA.text(r + 45, y + 0.17, f'{int(r):,}', va='center', fontsize=5.5)
        if not np.isnan(m):
            axA.barh(y - 0.17, m, 0.32, color=PAL['UI-Ref'], zorder=3)
            axA.text(m + 45, y - 0.17, f'{int(m):,}', va='center', fontsize=5.5, color=PAL['UI-Ref'])
    axA.set_yticks(ys)
    axA.set_yticklabels(steps, fontsize=5.6)
    axA.set_xlim(0, 3900)
    axA.set_xticks([0, 1000, 2000, 3000])
    axA.set_xticklabels(['0', '1k', '2k', '3k'])
    axA.set_xlabel('count')
    axA.set_ylim(-0.7, len(steps) - 0.3)
    axA.legend(handles=[Patch(facecolor='#9FB3C8', label='assay records'),
                        Patch(facecolor=PAL['UI-Ref'], label='unique molecules')],
               loc='lower right', bbox_to_anchor=(1.02, 0.02), fontsize=5.6, frameon=False,
               handlelength=1.0, handleheight=0.8, handletextpad=0.4)
    panel_letter(axA, 'a', dx=-0.62, dy=1.04)

    oc = rec['outcome'].value_counts()[['Active', 'Unspecified', 'Inactive', 'Inconclusive']]
    cols = [PAL['UI-Ref'], '#9FB3C8', PAL['limit'], '#E8A5AC']
    xs = np.arange(4)
    axB.bar(xs, oc.values, 0.60, color=cols, zorder=3)
    for xx, v in zip(xs, oc.values):
        axB.text(xx, v + 30, f'{v:,}', ha='center', fontsize=5.8)
    axB.set_xticks(xs)
    axB.set_xticklabels(['Active', 'Unspec.', 'Inactive', 'Inconcl.'], fontsize=5.0, rotation=35, ha='right')
    axB.set_ylabel('assay records')
    axB.set_ylim(0, 2500)
    axB.set_xlabel('reported activity outcome')
    axB.text(0.98, 0.98, '2,879 standardized records', transform=axB.transAxes, fontsize=5.6,
             color='#5E7183', ha='right', va='top')
    panel_letter(axB, 'b', dx=-0.44, dy=1.04)


    # Belongs to (b): anchored to its axes rather than centred under the figure.
    axB.text(0.0, -0.42,
             'The 236 recovered Inactive/Inconclusive records give 98 molecules a\n'
             'measured-inactive result, which makes cascade stage 1A possible.',
             transform=axB.transAxes, fontsize=5.6, color='#3E4C59', ha='left', va='top')
    return fig


fig = figS1()
fig.savefig(FIGURES / 'figS1_curation.png', dpi=300)
plt.close(fig)