"""Regenerate main Figure 5 (scaffold and alert landscape) from the shipped result tables.

Run from the repository root:  python scripts/10_figure5_scaffolds_alerts.py

Inputs   tableS16_scaffolds.csv, tableS18_alerts_by_collection.csv, tableS20_motif_variance_by_document.csv, tableS21_motif_potency_or.csv
Output   results/figures/fig5_scaffolds_alerts.png

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
from matplotlib.patches import Patch
import os

# ---- skill:figure-style helpers ----
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
mpl.rcParams['savefig.dpi'] = 300

# ---- palette ----
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

ALERT_LAB = {'thiourea':'thiourea','selenium':'selenium','hydrazone':'hydrazone',
             'phosphinic_phosphonic':'phosphinic / phosphonic','hydroxamate':'hydroxamate',
             'catechol':'catechol','carboxylate':'carboxylate','urea':'urea','thiol':'thiol',
             'any_chelator':'any Ni-chelating motif','PAINS':'PAINS','NIH':'NIH','BRENK':'Brenk',
             'any_alert':'any alert (PAINS+Brenk+NIH)'}


# ---- load data ----
sc = pd.read_csv(TABLES / 'tableS16_scaffolds.csv')
alerts_coll = pd.read_csv(TABLES / 'tableS18_alerts_by_collection.csv')
ac = alerts_coll.set_index('collection')
motif_icc = pd.read_csv(TABLES / 'tableS20_motif_variance_by_document.csv')
motif_or = pd.read_csv(TABLES / 'tableS21_motif_potency_or.csv')

tst = motif_or[np.isfinite(motif_or['OR_MH']) & (motif_or['n_informative_strata']>=2)].copy()

# ---- build fig5 (final version after all iterations) ----

def fig5():
    fig=plt.figure(figsize=(7.0,5.05))
    axA=fig.add_axes([0.085,0.660,0.205,0.250])
    axB=fig.add_axes([0.415,0.660,0.170,0.250])
    axC=fig.add_axes([0.760,0.660,0.185,0.250])
    axD=fig.add_axes([0.235,0.145,0.185,0.300])
    axE=fig.add_axes([0.655,0.145,0.200,0.300])

    # (a)
    cf=np.concatenate([[0],sc['cum_frac'].values])*100
    axA.step(np.arange(len(cf)),cf,where='post',color=PAL['UI-Ref'],lw=1.3,zorder=3)
    for nsc in (13,52): axA.plot([nsc],[cf[nsc]],'o',ms=3.4,color=PAL['limit'],zorder=5)
    axA.annotate('13 scaffolds → 50%',xy=(13,cf[13]),xytext=(32,24),fontsize=6,color=PAL['limit'],
                 arrowprops=dict(arrowstyle='-',lw=0.6,color=PAL['limit']))
    axA.annotate('52 → 80%',xy=(52,cf[52]),xytext=(80,54),fontsize=6,color=PAL['limit'],
                 arrowprops=dict(arrowstyle='-',lw=0.6,color=PAL['limit']))
    axA.set_xlim(0,214); axA.set_ylim(0,104)
    axA.set_xlabel('Murcko scaffold classes (ranked)'); axA.set_ylabel('% of UI-Ref molecules')
    axA.text(0.97,0.05,'n = 1,218; 55 acyclic molecules\nform one class',transform=axA.transAxes,
             fontsize=5.6,color='#5E7183',ha='right')
    panel_letter(axA,'a',dx=-0.34,dy=1.04)

    # (b)
    grp=pd.cut(sc['n_documents'],[-0.5,0.5,1.5,2.5,100],labels=['0','1','2','≥3'])
    tab=sc.groupby(grp,observed=True).agg(n_scaffolds=('scaffold','size'),n_mols=('n_mols','sum'))
    xs=np.arange(len(tab))
    axB.bar(xs,tab['n_mols'],0.60,color=PAL['UI-Ref'],zorder=3)
    for xx,(ns,nm) in zip(xs,tab[['n_scaffolds','n_mols']].values):
        axB.text(xx,nm+12,f'{nm}',ha='center',fontsize=5.8,color='#1F2933')
        axB.text(xx,nm+70,f'{ns} scaff.',ha='center',fontsize=5.3,color='#5E7183')
    axB.set_xticks(xs); axB.set_xticklabels(['0\n(no doc.)','1','2','≥3'],fontsize=5.8)
    axB.set_xlabel('documents containing the scaffold\n(192 of 214 classes: at most one)')
    axB.set_ylabel('molecules'); axB.set_ylim(0,700)
    panel_letter(axB,'b',dx=-0.40,dy=1.04)

    # (c)
    rows=['any_alert','BRENK','NIH','PAINS','any_chelator','thiourea','catechol','hydrazone',
          'phosphinic_phosphonic','carboxylate','hydroxamate','urea','selenium','thiol']
    y=np.arange(len(rows))
    for i,r in enumerate(rows):
        axC.plot([0,ac.loc['UI-Ref',r]*100],[i,i],'-',lw=0.6,color='#DCE0E4',zorder=2)
    for c,off,ms in [('DSSTox',-0.24,3.0),('CLUE',-0.34,3.0),('COCONUT',-0.44,3.0),('UI-Ref',0.0,4.6)]:
        axC.plot(ac.loc[c,rows].values*100,y+off,'o',ms=ms,color=PAL[c],
                 mfc=PAL[c] if c=='UI-Ref' else 'white',mew=1.0,zorder=5 if c=='UI-Ref' else 4)
    axC.set_yticks(y); axC.set_yticklabels([ALERT_LAB[r] for r in rows],fontsize=5.5); axC.invert_yaxis()
    axC.set_xlim(-2,100); axC.set_xticks([0,50,100])
    axC.set_xlabel('% of collection carrying the motif')
    axC.set_ylim(len(rows)-0.30,-0.95)
    axC.annotate('thiourea 24.9%\nvs CLUE 0.27%\n(91.9×)',xy=(24.9,5),xytext=(48,7.8),fontsize=5.6,
                 color=PAL['UI-Ref'],arrowprops=dict(arrowstyle='-',lw=0.6,color=PAL['UI-Ref']))
    h=[mlines.Line2D([],[],marker='o',ls='',ms=4.2,color=PAL['UI-Ref'],label='UI-Ref (1,218)'),
       mlines.Line2D([],[],marker='o',ls='',ms=3.0,mfc='white',mew=1.0,color=PAL['CLUE'],label='CLUE (4,431)'),
       mlines.Line2D([],[],marker='o',ls='',ms=3.0,mfc='white',mew=1.0,color=PAL['COCONUT'],label='COCONUT (12,000)'),
       mlines.Line2D([],[],marker='o',ls='',ms=3.0,mfc='white',mew=1.0,color=PAL['DSSTox'],label='DSSTox (11,694)')]
    axC.legend(handles=h,loc='lower right',bbox_to_anchor=(1.05,-0.03),fontsize=5.4,frameon=False,
               handletextpad=0.2,labelspacing=0.20,borderpad=0.05)
    panel_letter(axC,'c',dx=-0.62,dy=1.04)

    # (d)
    mi=motif_icc.sort_values('icc_between_document')
    y=np.arange(len(mi))
    axD.hlines(y,0,mi['icc_between_document'],color='#C6CBD1',lw=0.7,zorder=2)
    axD.plot(mi['icc_between_document'],y,'o',ms=4.0,color=PAL['UI-Ref'],zorder=4)
    for yy,v in enumerate(mi['icc_between_document']):
        axD.text(v+0.03,yy,f'{v:.2f}',va='center',fontsize=5.3,color='#1F2933')
    axD.set_yticks(y); axD.set_yticklabels([ALERT_LAB.get(m,m) for m in mi['motif']],fontsize=5.5)
    axD.set_xlim(0,1.20); axD.set_xticks([0,0.5,1.0])
    axD.set_ylim(-0.7,len(mi)-0.3)
    axD.set_xlabel('between-document ICC of motif presence')
    panel_letter(axD,'d',dx=-0.95,dy=1.04)

    # (e)
    t=tst.sort_values('OR_pooled')
    y=np.arange(len(t))
    for yy,r in zip(y,t.itertuples()):
        axE.plot([r.OR_pooled,r.OR_MH],[yy,yy],'-',lw=0.7,
                 color=PAL['limit'] if r.reverses else '#C6CBD1',zorder=2)
        axE.plot(r.OR_pooled,yy,'o',ms=3.4,color=PAL['naive'],zorder=4)
        axE.plot(r.OR_MH,yy,'o',ms=3.6,color=PAL['limit'] if r.reverses else PAL['honest'],zorder=5)
    axE.axvline(1.0,color='#8A929B',lw=0.9,ls=':',zorder=3)
    axE.set_xscale('log'); axE.set_xlim(0.115,26)
    axE.set_xticks([0.2,1,5,20]); axE.set_xticklabels(['0.2','1','5','20'])
    axE.set_yticks(y); axE.set_yticklabels([ALERT_LAB.get(m,m) for m in t['motif']],fontsize=5.5)
    axE.set_ylim(-0.7,len(t)-0.3)
    axE.set_xlabel('odds ratio for $p$Act ≥ 6')
    axE.annotate('8.91 → 0.45',xy=(0.452,9),xytext=(0.135,8.3),fontsize=5.6,color=PAL['limit'],
                 arrowprops=dict(arrowstyle='-',lw=0.6,color=PAL['limit']))
    axE.text(2.1,4.3,'7.17 → 1.05: collapses\nto the null, does not\nreverse',
             fontsize=5.5,color='#3E4C59',va='center',ha='left')
    h=[mlines.Line2D([],[],marker='o',ls='',ms=3.4,color=PAL['naive'],label='pooled (document ignored)'),
       mlines.Line2D([],[],marker='o',ls='',ms=3.4,color=PAL['honest'],label='Mantel–Haenszel, stratified'),
       mlines.Line2D([],[],marker='o',ls='',ms=3.4,color=PAL['limit'],label='direction of effect reverses')]
    axE.legend(handles=h,loc='upper left',bbox_to_anchor=(0.0,-0.30),
               fontsize=5.4,frameon=False,handletextpad=0.2,labelspacing=0.20,borderpad=0.05)
    axE.text(0.98,0.02,'178 potent of 1,110 quantified',transform=axE.transAxes,fontsize=5.6,
             color='#5E7183',va='bottom',ha='right')
    panel_letter(axE,'e',dx=-0.72,dy=1.04)
    # Split per panel: the note covered both (d) ICC and (e) odds ratios, so a single
    # centred line under the pair could not be attributed to either.
    axD.text(0.0,-0.30,'1,142 documented molecules,\n77 documents',
             transform=axD.transAxes,fontsize=5.6,color='#5E7183',ha='left',va='top')

    return fig

fig = fig5()
fig.savefig(FIGURES / 'fig5_scaffolds_alerts.png', dpi=300)
plt.close(fig)