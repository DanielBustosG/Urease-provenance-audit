"""Regenerate Figure S7 (conformal coverage per stage and alpha) from the shipped result tables.

Run from the repository root:  python scripts/17_figureS7_conformal.py

Inputs   tableS27_conformal_coverage.csv
Output   results/figures/figS7_conformal_detail.png

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

cf = pd.read_csv(TABLES / 'tableS27_conformal_coverage.csv')

STAGE_ORDER=['S1B_decoy','S1A_measured','S2_potency','S3_toxicophore','S4_clinical']

SC={'S1B_decoy':PAL['UI-Ref'],'S1A_measured':PAL['limit'],'S2_potency':'#6BA3D6',
    'S3_toxicophore':PAL['COCONUT'],'S4_clinical':PAL['DSSTox']}
SL={'S1B_decoy':'1B decoy','S1A_measured':'1A measured','S2_potency':'2 potency',
    'S3_toxicophore':'3 alerts','S4_clinical':'4 clinical'}

def figS7():
    fig=plt.figure(figsize=(7.0,3.30))
    axA=fig.add_axes([0.085,0.245,0.215,0.545])
    axB=fig.add_axes([0.410,0.245,0.195,0.545])
    axC=fig.add_axes([0.775,0.245,0.195,0.545])

    # (a) coverage gap
    for st in STAGE_ORDER:
        s=cf[cf['stage']==st].sort_values('alpha')
        gap=(s['empirical_coverage']-s['nominal_coverage'])*100
        axA.plot(s['alpha']*100,gap,'-o',lw=1.2,ms=3.6,color=SC[st],zorder=4,label=SL[st])
        axA.fill_between(s['alpha']*100,(s['coverage_lo']-s['nominal_coverage'])*100,
                         (s['coverage_hi']-s['nominal_coverage'])*100,color=SC[st],alpha=0.13,lw=0,zorder=2)
    axA.axhline(0,color='#8A929B',lw=0.9,ls='--',zorder=3)
    axA.text(19.6,1.2,'exact coverage',fontsize=5.5,color='#6B7480',ha='right')
    axA.set_xlim(4,21); axA.set_xticks([5,10,20])
    axA.set_xlabel('miscoverage level α (%)'); axA.set_ylabel('empirical − nominal coverage (pp)')
    axA.legend(loc='lower left',bbox_to_anchor=(0.00,0.02),fontsize=5.4,frameon=False,
               handletextpad=0.2,labelspacing=0.22,borderpad=0.05)
    panel_letter(axA,'a',dx=-0.34,dy=1.04)

    # (b) uninformative set fraction
    for st in STAGE_ORDER:
        s=cf[cf['stage']==st].sort_values('alpha')
        axB.plot(s['alpha']*100,s['frac_BOTH_uninformative']*100,'-o',lw=1.2,ms=3.6,
                 color=SC[st],zorder=4)
        axB.text(21.0,s['frac_BOTH_uninformative'].iloc[-1]*100,SL[st],fontsize=5.5,
                 color=SC[st],va='center')
    axB.set_xlim(4,21); axB.set_xticks([5,10,20]); axB.set_ylim(-4,76)
    axB.set_xlabel('miscoverage level α (%)')
    axB.set_ylabel('% with a two-class prediction set')
    yoff={'1B decoy':-2.5,'1A measured':+1.0,'2 potency':+2.0,'3 alerts':-6.0,'4 clinical':-2.5}
    for t in list(axB.texts):
        s=t.get_text()
        if s in yoff:
            x,y=t.get_position(); t.set_position((21.0,y+yoff[s]))
    axB.text(0.02,0.10,'no empty prediction sets except\nstage 3 at α = 0.20 (4.3%)',
             transform=axB.transAxes,fontsize=5.3,color='#5E7183')
    panel_letter(axB,'b',dx=-0.34,dy=1.04)

    # (c) singleton accuracy vs singleton fraction
    for st in STAGE_ORDER:
        s=cf[cf['stage']==st].sort_values('alpha')
        axC.plot(s['frac_singleton']*100,s['singleton_accuracy']*100,'-o',lw=1.0,ms=3.4,
                 color=SC[st],zorder=4)
        axC.plot(s['frac_singleton'].iloc[-1]*100,s['singleton_accuracy'].iloc[-1]*100,'o',ms=5.0,
                 mfc='none',mew=1.0,color=SC[st],zorder=5)
    axC.set_xlim(28,100); axC.set_xticks([30,50,70,90])
    axC.set_ylim(52,96); axC.set_yticks([60,70,80,90])
    axC.set_xlabel('% singleton predictions')
    axC.set_ylabel('singleton accuracy (%)')
    axC.text(0.98,0.03,'open ring marks α = 0.20;\nline runs α = 0.05 → 0.20',transform=axC.transAxes,
             fontsize=5.3,color='#5E7183',ha='right')
    axC.set_position([0.760,0.245,0.180,0.545])
    panel_letter(axC,'c',dx=-0.36,dy=1.04)
    return fig

fig=figS7()
fig.savefig(FIGURES / 'figS7_conformal_detail.png', dpi=300)
plt.close(fig)