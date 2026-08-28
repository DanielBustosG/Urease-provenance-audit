"""Regenerate Figure S5 (near-duplicate leakage by split type) from the shipped result tables.

Run from the repository root:  python scripts/15_figureS5_leakage.py

Inputs   tableS9_leakage.csv
Output   results/figures/figS5_leakage.png

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


apply_figure_style(frame='open', sizes=(8,7,6))
mpl.rcParams['savefig.dpi']=300

PAL = {
 'UI-Ref' : '#1B4F8C',
 'CLUE'   : '#E8871A',
 'COCONUT': '#157F3D',
 'DSSTox' : '#7D4E9E',
 'decoys' : '#6E7B8B',
 'naive'  : '#AAB2BD',
 'honest' : '#2B2B2B',
 'limit'  : '#C42B3A',
 'null'   : '#DCDEE1',
}


lk = pd.read_csv(TABLES / 'tableS9_leakage.csv')

SPLC={'random':PAL['naive'],'document-grouped':PAL['UI-Ref'],'scaffold-grouped':'#6BA3D6'}

def figS5():
    fig=plt.figure(figsize=(7.0,3.30))
    axA=fig.add_axes([0.230,0.245,0.310,0.545])
    axB=fig.add_axes([0.665,0.245,0.315,0.545])
    sides=['UI-Ref (within)','UI-Ref vs CLUE [UI-Ref side]','UI-Ref vs COCONUT [UI-Ref side]',
           'UI-Ref vs DSSTox [UI-Ref side]','UI-Ref vs CLUE [CLUE side]',
           'UI-Ref vs COCONUT [COCONUT side]','UI-Ref vs DSSTox [DSSTox side]']
    slab=['UI-Ref, within-set','UI-Ref side, vs CLUE','UI-Ref side, vs COCONUT','UI-Ref side, vs DSSTox',
          'CLUE side','COCONUT side','DSSTox side']
    y=np.arange(len(sides))
    for spl,off in [('random',0.24),('document-grouped',0.0),('scaffold-grouped',-0.24)]:
        s=lk[(lk['split']==spl)&(lk['dataset'].isin(sides))].set_index('dataset').reindex(sides)
        axA.plot(s['frac_ge_0_7']*100,y+off,'o',ms=3.6,color=SPLC[spl],zorder=4,label=spl)
    axA.set_yticks(y); axA.set_yticklabels(slab,fontsize=5.6); axA.invert_yaxis()
    axA.set_ylim(len(sides)-0.45,-0.55)
    axA.set_xlim(0,72); axA.set_xticks([0,20,40,60])
    axA.set_xlabel('% of test molecules with a training neighbour at Tanimoto ≥ 0.7')
    axA.legend(loc='lower right',bbox_to_anchor=(1.03,0.02),fontsize=5.5,frameon=False,
               handletextpad=0.2,labelspacing=0.22,borderpad=0.05)
    panel_letter(axA,'a',dx=-0.40,dy=1.04)

    for spl in ['random','document-grouped','scaffold-grouped']:
        s=lk[(lk['split']==spl)&(lk['dataset'].str.contains(r'\[pooled\]',regex=True))]
        axB.plot(s['median_max_sim'],s['frac_ge_0_9']*100,'o',ms=4.2,color=SPLC[spl],zorder=4,label=spl)
    s=lk[(lk['dataset']=='UI-Ref (within)')]
    for _,r in s.iterrows():
        axB.plot(r['median_max_sim'],r['frac_ge_0_9']*100,'s',ms=4.2,color=SPLC[r['split']],zorder=5)
    axB.set_xlim(0.40,0.78); axB.set_xticks([0.45,0.55,0.65,0.75])
    axB.set_ylim(0,13); axB.set_yticks([0,4,8,12])
    axB.set_xlabel('median maximum Tanimoto to training set')
    axB.set_ylabel('% at Tanimoto ≥ 0.9')
    # One key in the lower-right corner resolving BOTH encodings: colour = split type,
    # shape = which test set. Previously shape was explained in free text at the top
    # and colour in a legend at the bottom left, so the squares had no entry at all.
    h5 = [mlines.Line2D([], [], marker='o', ls='', ms=4.2, color=SPLC[s], label=s)
          for s in ['random', 'document-grouped', 'scaffold-grouped']]
    h5 += [mlines.Line2D([], [], marker='o', ls='', ms=4.2, color='#8A929B',
                         label='pooled test sets'),
           mlines.Line2D([], [], marker='s', ls='', ms=4.2, color='#8A929B',
                         label='UI-Ref alone')]
    axB.legend(handles=h5, loc='lower right', bbox_to_anchor=(1.02, 0.02), fontsize=5.4,
               frameon=False, handletextpad=0.3, labelspacing=0.28, borderpad=0.05)
    panel_letter(axB,'b',dx=-0.28,dy=1.04)
    return fig

fig=figS5(); fig.savefig(FIGURES / 'figS5_leakage.png', dpi=300); plt.close(fig)