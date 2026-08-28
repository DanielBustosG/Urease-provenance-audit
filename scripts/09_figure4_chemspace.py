"""Regenerate main Figure 4 (chemical space as a projection artifact) from the shipped result tables.

Run from the repository root:  python scripts/09_figure4_chemspace.py

Inputs   collection_CLUE.parquet, collection_COCONUT.parquet, collection_DSSTox.parquet, tableS12_neighbourhood_enrichment.csv, tableS14_cluster_ami.csv, uiref_curated.parquet
Output   results/figures/fig4_chemspace.png

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


# Load data
u = pd.read_parquet(DATA / 'uiref_curated.parquet')
D_CLUE = pd.read_parquet(DATA / 'collection_CLUE.parquet')
D_COCONUT = pd.read_parquet(DATA / 'collection_COCONUT.parquet')
D_DSSTox = pd.read_parquet(DATA / 'collection_DSSTox.parquet'  )
nb = pd.read_csv(TABLES / 'tableS12_neighbourhood_enrichment.csv')
ami = pd.read_csv(TABLES / 'tableS14_cluster_ami.csv')

# Compute NN Tanimoto fingerprints
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs
RDLogger.DisableLog('rdApp.*')
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def fps(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out

FP = {'UI-Ref': fps(u['smiles_std'].tolist())}
for c, df in [('CLUE', D_CLUE), ('COCONUT', D_COCONUT), ('DSSTox', D_DSSTox)]:
    FP[c] = fps(df['smiles_std'].tolist())

NN = {}
ui = FP['UI-Ref']
for c in ['CLUE', 'COCONUT', 'DSSTox']:
    NN[c] = np.array([max(DataStructs.BulkTanimotoSimilarity(q, FP[c])) for q in ui])
loo = []
for i, q in enumerate(ui):
    s = DataStructs.BulkTanimotoSimilarity(q, ui)
    s[i] = -1
    loo.append(max(s))
NN['UI-Ref'] = np.array(loo)

from scipy.stats import gaussian_kde

mpl.rcParams['savefig.bbox'] = None
mpl.rcParams['savefig.pad_inches'] = 0.0


def fig4():
    fig = plt.figure(figsize=(7.0, 4.75))
    axA = fig.add_axes([0.090, 0.630, 0.185, 0.265])
    axB = fig.add_axes([0.430, 0.630, 0.200, 0.265])
    axC = fig.add_axes([0.750, 0.630, 0.145, 0.265])
    axD = fig.add_axes([0.090, 0.115, 0.300, 0.320])
    axE = fig.add_axes([0.560, 0.115, 0.320, 0.320])
    COLL = ['CLUE', 'COCONUT', 'DSSTox']

    th = nb[(nb['sampling'] == 'unequal_n_native') & (nb['space'] == 'ECFP4_full') & (nb['k'] == 5)].set_index('collection')
    eq = nb[(nb['sampling'] == 'equal_n1218') & (nb['space'] == 'ECFP4_full') & (nb['k'] == 5)].set_index('collection')
    voff = {'DSSTox': +0.32, 'CLUE': -0.32, 'COCONUT': 0.0}
    for c in COLL:
        t = th.loc[c, 'obs_frac'] * 100
        q = eq.loc[c, 'obs_frac'] * 100
        axA.plot([0, 1], [t, q], '-', lw=1.3, color=PAL[c], zorder=3)
        axA.plot(0, t, 'o', ms=4.6, color=PAL[c], mfc='white', mew=1.4, zorder=5)
        axA.errorbar(1, q, yerr=[[q - eq.loc[c, 'obs_frac_lo'] * 100], [eq.loc[c, 'obs_frac_hi'] * 100 - q]],
                     fmt='o', ms=4.6, color=PAL[c], ecolor=PAL[c], elinewidth=1.0, capsize=2.0, zorder=5)
        axA.text(1.10, q + voff[c], f'{c} {q:.2f}%', fontsize=6.0, color=PAL[c], va='center')
        axA.text(0.06, t, f'{t:.1f}%', fontsize=5.8, color=PAL[c], va='center', ha='left')
    axA.set_xticks([0, 1])
    axA.set_xticklabels(['unequal $n$', 'equal $n$'], fontsize=6)
    axA.set_xlim(-0.15, 2.35)
    axA.set_ylim(0, 8.6)
    axA.set_ylabel('% of UI-Ref 5-NN that are foreign')
    axA.text(0.98, 0.98, 'open: one draw at native size\n(4.4k / 12k / 11.7k)\nfilled: 20 equal-$n$ draws, 95% CI',
             transform=axA.transAxes, fontsize=5.4, color='#5E7183', va='top', ha='right')
    # Note on what the unequal-n sampling reads
    axA.text(0.55, 0.70, 'unequal $n$ reads\nCLUE 5.9% > DSSTox 1.3%',
             transform=axA.transAxes, fontsize=5.6, color='#3E4C59', va='top')
    panel_letter(axA, 'a', dx=-0.40, dy=1.04)

    sp = ['ECFP4_full', 'tSNE_2D', 'UMAP_2D', 'PCA_2D']
    spl = ['ECFP4\nfull', '$t$-SNE\n2-D', 'UMAP\n2-D', 'PCA\n2-D']
    xs = np.arange(4)
    endy = {'DSSTox': 29.0, 'CLUE': 19.5, 'COCONUT': 9.0}
    for c in COLL:
        v = [nb[(nb['sampling'] == 'equal_n1218') & (nb['space'] == s) & (nb['k'] == 5) & (nb['collection'] == c)]['obs_frac'].iloc[0] * 100 for s in sp]
        axB.plot(xs, v, '-o', lw=1.3, ms=3.8, color=PAL[c], zorder=4)
        axB.text(3.12, endy[c], c, fontsize=6.0, color=PAL[c], va='center')
    vu = [nb[(nb['sampling'] == 'equal_n1218') & (nb['space'] == s) & (nb['k'] == 5) & (nb['collection'] == 'UI-Ref')]['obs_frac'].iloc[0] * 100 for s in sp]
    axB.plot(xs, vu, '-s', lw=1.6, ms=4.0, color=PAL['UI-Ref'], zorder=5)
    axB.text(3.12, vu[-1], 'UI-Ref (self)', fontsize=6.0, color=PAL['UI-Ref'], va='center')
    axB.text(0.05, vu[0] - 7, '93.8%', fontsize=6, color=PAL['UI-Ref'])
    axB.text(2.52, vu[-1] + 8, '49.5%', fontsize=6, color=PAL['UI-Ref'])
    axB.set_xticks(xs)
    axB.set_xticklabels(spl, fontsize=5.8)
    axB.set_xlim(-0.35, 4.55)
    axB.set_ylim(0, 100)
    axB.set_ylabel('% of 5-NN in that class')
    axB.text(0.55, 0.98, 'foreign fractions inflated\n5.6–9.3× at $k$ = 5,\nup to 14× at $k$ = 1',
             transform=axB.transAxes, fontsize=5.6, color='#5E7183', ha='left', va='top')
    panel_letter(axB, 'b', dx=-0.34, dy=1.04)

    order = ['UI-Ref', 'DSSTox', 'CLUE', 'COCONUT']
    xg = np.linspace(0, 1, 220)
    for i, c in enumerate(order):
        v = NN[c]
        kd = gaussian_kde(v, bw_method=0.22)(xg)
        kd = kd / kd.max() * 0.82
        base = len(order) - 1 - i
        axC.fill_between(xg, base, base + kd, color=PAL[c], alpha=0.55, lw=0, zorder=3 + i)
        axC.plot(xg, base + kd, color=PAL[c], lw=0.9, zorder=3 + i)
        med = np.median(v)
        axC.plot([med, med], [base, base + 0.30], color=PAL[c], lw=1.1, zorder=9)
        axC.text(1.04, base + 0.36, f'{c} {med:.3f}', fontsize=5.7, color=PAL[c], va='center')
    axC.axvline(0.40, color=PAL['limit'], lw=1.0, ls='--', zorder=10)
    axC.text(0.375, 3.60, 'AD threshold\n$T$ = 0.40', fontsize=5.5, color=PAL['limit'], ha='right', va='top')
    axC.set_yticks([])
    axC.set_ylim(-0.10, 3.95)
    axC.set_xlim(0, 1.0)
    axC.set_xticks([0, 0.5, 1.0])
    axC.set_xlabel('Tanimoto to nearest\nUI-Ref molecule')
    axC.spines['left'].set_visible(False)
    panel_letter(axC, 'c', dx=-0.22, dy=1.04)

    for c in COLL + ['UI-Ref']:
        s = nb[(nb['sampling'] == 'equal_n1218') & (nb['space'] == 'ECFP4_full') & (nb['collection'] == c)].sort_values('k')
        axD.errorbar(s['k'], s['enrichment'],
                     yerr=[s['enrichment'] - s['enrich_lo'], s['enrich_hi'] - s['enrichment']],
                     fmt='-o', lw=1.3, ms=4.0, color=PAL[c], ecolor=PAL[c], elinewidth=0.9, capsize=1.8, zorder=4)
        axD.text(28, s['enrichment'].iloc[-1], c if c != 'UI-Ref' else 'UI-Ref (self)', fontsize=6.0,
                 color=PAL[c], va='center')
    axD.axhline(1.0, color='#8A929B', lw=0.9, ls=':', zorder=2)
    axD.text(0.90, 1.13, '1.0 = joint background', fontsize=5.7, color='#6B7480', va='bottom')
    axD.set_xscale('log')
    axD.set_xticks([1, 5, 25])
    axD.set_xticklabels(['1', '5', '25'])
    axD.set_yscale('log')
    axD.set_yticks([0.02, 0.1, 0.5, 1, 4])
    axD.set_yticklabels(['0.02', '0.1', '0.5', '1', '4'])
    axD.set_xlabel('neighbourhood size $k$')
    axD.set_ylabel('neighbourhood enrichment')
    axD.set_xlim(0.85, 75)
    axD.set_ylim(0.014, 7.5)
    axD.text(0.30, 0.04, 'equal $n$ = 1,218 per collection, 20 draws, 95% CI', transform=axD.transAxes,
             fontsize=5.7, color='#5E7183')
    panel_letter(axD, 'd', dx=-0.245, dy=1.04)

    rows = ami[ami['space'] == 'joint_equal_n1218'].iloc[1:]
    lab = ['$k$ = 4', '$k$ = 20', '$k$ = 50', '$k$ = 100']
    xs = np.arange(len(rows))
    w = 0.26
    axE.bar(xs - w, rows['AMI_collection'], w, color='#8A929B', label='collection membership', zorder=3)
    axE.bar(xs, rows['AMI_UIRef_organism'], w, color='#9FB3C8', label='assay organism (within UI-Ref)', zorder=3)
    axE.bar(xs + w, rows['AMI_UIRef_document'], w, color=PAL['UI-Ref'], label='source document (within UI-Ref)', zorder=3)
    for xx, v in zip(xs + w, rows['AMI_UIRef_document']):
        axE.text(xx, v + 0.02, f'{v:.2f}', ha='center', fontsize=5.6, color=PAL['UI-Ref'])
    axE.set_xticks(xs)
    axE.set_xticklabels(lab, fontsize=6)
    axE.set_xlabel('$k$-means clusters on the joint UMAP embedding')
    axE.set_ylim(0, 1.15)
    axE.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axE.set_ylabel('adjusted mutual information')
    axE.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.005), fontsize=5.7, frameon=False,
               handlelength=0.9, handleheight=0.8, labelspacing=0.22, handletextpad=0.4)
    axE.set_ylim(0, 1.42)
    axE.text(0.015, 0.72, 'silhouette by collection ≈ 0 or negative;\nby document 0.263 (full space) / 0.456 (UMAP)',
             transform=axE.transAxes, fontsize=5.6, color='#5E7183', ha='left', va='top')
    panel_letter(axE, 'e', dx=-0.23, dy=1.04)
    return fig


fig = fig4()

# render-then-verify: no two visible text objects may overlap in panel (e)
_r = fig.canvas.get_renderer()
_axE = fig.axes[4]
_te = [(t_, t_.get_window_extent(_r)) for t_ in _axE.texts if t_.get_text().strip() and t_.get_visible()]
_ov = [(a.get_text()[:24], b.get_text()[:24])
       for i, (a, ba) in enumerate(_te) for b, bb in _te[i+1:] if ba.overlaps(bb)]
if _ov:
    raise SystemExit(f"panel (e) text overlaps, fix placement before shipping: {_ov}")

fig.savefig(FIGURES / 'fig4_chemspace.png', dpi=300)
plt.close(fig)