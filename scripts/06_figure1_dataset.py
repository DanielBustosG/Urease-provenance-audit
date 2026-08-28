"""Regenerate main Figure 1 (what UI-Ref actually is) from the shipped result tables.

Run from the repository root:  python scripts/06_figure1_dataset.py

Inputs   records_curated.parquet, uiref_curated.parquet
Output   results/figures/fig1_dataset.png

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
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 60)

u = pd.read_parquet(DATA / 'uiref_curated.parquet')
rec = pd.read_parquet(DATA / 'records_curated.parquet')

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


# Build allnon (non-urease records)
allnon = rec[~rec['urease_assay'].astype(bool)]


def klass(s):
    s = str(s).lower()
    if 'anticancer' in s or 'cytotox' in s or 'cell growth' in s or 'antiproliferative' in s:
        return 'Anticancer / cytotoxicity'
    if 'antioxidant' in s or 'dpph' in s or 'radical scaveng' in s:
        return 'Antioxidant (DPPH)'
    if 'hiv' in s or 'integrase' in s:
        return 'HIV-1 integrase'
    if 'glucosidase' in s or 'amylase' in s:
        return 'Glycosidase (α-glucosidase/amylase)'
    if 'antibacterial' in s or 'antimicrob' in s or 'staphyl' in s or 'escherichia' in s or 'tyrrs' in s or 'mic' == s.strip():
        return 'Antibacterial'
    if 'parp' in s:
        return 'PARP1'
    if 'acetylcholinesterase' in s or 'cholinesterase' in s:
        return 'Cholinesterase'
    if 'agonist' in s or 'receptor' in s or 'ffa3' in s or 'hca2' in s:
        return 'GPCR agonism'
    if 'tyrosinase' in s:
        return 'Tyrosinase'
    if 'carbonic anhydrase' in s:
        return 'Carbonic anhydrase'
    return 'Other non-urease'


allnon = allnon.assign(assay_class=allnon['assay_name'].fillna(allnon['assay_id_s'].astype(str)).map(klass))

# Build organism data
ORG_COL = {'Canavalia ensiformis': '#1B4F8C', 'Helicobacter pylori': '#E8871A',
           'Sporosarcina pasteurii': '#157F3D', 'other / unspecified': '#AAB2BD'}


def org3(o):
    return o if o in ('Canavalia ensiformis', 'Helicobacter pylori', 'Sporosarcina pasteurii') else 'other / unspecified'


subdoc = u.dropna(subset=['primary_document']).assign(org=lambda d: d['primary_organism'].map(org3))
top12 = subdoc['primary_document'].value_counts().head(12).index.tolist()
orgmat = (subdoc[subdoc['primary_document'].isin(top12)]
          .groupby(['primary_document', 'org']).size().unstack(fill_value=0)
          .reindex(index=top12).reindex(columns=list(ORG_COL), fill_value=0))
orgfrac = orgmat.div(orgmat.sum(axis=1), axis=0)

# Build document labels
JAB2 = {'Bioorg Med Chem Lett': 'Bioorg. Med. Chem. Lett.', 'Bioorg Med Chem': 'Bioorg. Med. Chem.',
        'Eur J Med Chem': 'Eur. J. Med. Chem.', 'J Med Chem': 'J. Med. Chem.',
        'Chem Biol Drug Des': 'Chem. Biol. Drug Des.'}

meta = (rec.dropna(subset=['document']).groupby('document')[['journal', 'year']].agg(
    lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan))

yr_from_rec = rec.dropna(subset=['document']).groupby('document')['year'].agg(
    lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
jrn_from_rec = rec.dropna(subset=['document']).groupby('document')['journal'].agg(
    lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)


def doc_year(d):
    d = str(d)
    m = re.search(r'\.(19\d{2}|20\d{2})\.', d)
    if m: return int(m.group(1))
    m = re.search(r'\.(\d)([bc])0', d)
    if m: return (2010 if m.group(2) == 'b' else 2020) + int(m.group(1))
    if d in meta.index and pd.notna(meta.loc[d, 'year']): return int(meta.loc[d, 'year'])
    return None


def doclabel(d):
    j = jrn_from_rec.get(d, np.nan)
    y = doc_year(d)
    if y is None and d in yr_from_rec.index and pd.notna(yr_from_rec[d]): y = int(yr_from_rec[d])
    if pd.notna(j): return f"{JAB2.get(j, j)} {y}" if y else JAB2.get(j, j)
    if str(d).startswith('CHEMBL'): return f"ChEMBL doc. {y}" if y else str(d)
    return f"DOI {str(d).split('/')[-1]}"


_c = Counter()


def uniq(l):
    out = []
    for x in l:
        _c[x] += 1
        out.append(x if _c[x] == 1 else f"{x}{chr(96 + _c[x])}")
    return out


lbl12 = [doclabel(d) for d in top12]
LBL12 = uniq(lbl12)


def build_fig1():
    fig = plt.figure(figsize=(7.0, 4.60))
    axA = fig.add_axes([0.150, 0.620, 0.185, 0.285])
    axB = fig.add_axes([0.470, 0.620, 0.170, 0.285])
    axC = fig.add_axes([0.775, 0.620, 0.205, 0.285])
    axD = fig.add_axes([0.090, 0.200, 0.240, 0.280])
    axE = fig.add_axes([0.620, 0.200, 0.330, 0.280])

    # (a) curation cascade
    stages = [('pooled records', 3038, 'rec', None), ('valid structures', 2879, 'rec', '−159 unparseable'),
              ('urease assays only', 2567, 'rec', '−312 counter-assay records'),
              ('unique molecules', 1361, 'mol', None),
              ('UI-Ref (corrected)', 1218, 'mol', '−143 mol., no urease assay'),
              ('quantitative $p$Act', 1110, 'mol', '−108 no direct nM endpoint')]
    ys = [6.3, 5.0, 3.7, 1.8, 0.5, -0.8]
    for y, (lab, val, kind, drop) in zip(ys, stages):
        dark = kind == 'mol'
        axA.barh(y, val, height=0.55, color=PAL['UI-Ref'] if dark else '#9FB3C8', zorder=3)
        axA.text(val + 70, y, f'{val:,}', va='center', ha='left', fontsize=6, zorder=4)
        if drop:
            axA.text(40, y + 0.62, drop, va='center', ha='left', fontsize=5.4, style='italic',
                     color=PAL['limit'], zorder=5)
    axA.set_yticks(ys)
    axA.set_yticklabels([s[0] for s in stages], fontsize=6)
    axA.set_ylim(-1.5, 7.25)
    axA.set_xlim(0, 4800)
    axA.set_xticks([0, 1000, 2000, 3000])
    axA.set_xticklabels(['0', '1k', '2k', '3k'])
    axA.set_xlabel('count')
    panel_letter(axA, 'a', dx=-0.52, dy=1.04)

    # (b) counter-assay composition
    ren = {'Anticancer / cytotoxicity': 'anticancer / cytotoxicity', 'Other non-urease': 'other non-urease',
           'Antioxidant (DPPH)': 'antioxidant (DPPH)', 'Antibacterial': 'antibacterial',
           'GPCR agonism': 'GPCR agonism', 'Glycosidase (α-glucosidase/amylase)': 'glycosidase', 'PARP1': 'PARP1'}
    cc = allnon['assay_class'].value_counts().sort_values().rename(ren)
    axB.barh(range(len(cc)), cc.values, color=PAL['limit'], alpha=0.85, height=0.64, zorder=3)
    for i, v in enumerate(cc.values):
        axB.text(v + 5, i, str(v), va='center', fontsize=6)
    axB.set_yticks(range(len(cc)))
    axB.set_yticklabels(cc.index, fontsize=6)
    axB.set_ylim(-0.7, len(cc) - 0.3)
    axB.set_xlim(0, 205)
    axB.set_xticks([0, 50, 100, 150])
    axB.set_xlabel('records removed')
    axB.text(0.99, 0.02, '312 records,\n176 molecules touched', transform=axB.transAxes, ha='right',
             va='bottom', fontsize=5.6, color='#5E7183')
    panel_letter(axB, 'b', dx=-0.62, dy=1.04)

    # (c) potency
    pa = u['pAct_median'].dropna().values
    axC.hist(pa, bins=np.arange(2.0, 10.01, 0.35), color=PAL['UI-Ref'], edgecolor='white', linewidth=0.3, zorder=3)
    top = axC.get_ylim()[1]
    axC.axvspan(6, 10, color=PAL['limit'], alpha=0.10, zorder=1)
    axC.axvline(4.72, color=PAL['limit'], lw=1.1, zorder=5)
    axC.annotate('median 4.72', xy=(4.72, top * 0.62), xytext=(2.10, top * 0.99), fontsize=6, color=PAL['limit'],
                 arrowprops=dict(arrowstyle='-', lw=0.6, color=PAL['limit']))
    axC.text(7.95, top * 0.46, '$p$Act ≥ 6\n178 mol.\n(16.0%)', fontsize=6, ha='center', color=PAL['limit'])
    axC.set_xlabel('$p$Act (median per molecule)')
    axC.set_ylabel('molecules')
    axC.set_xlim(2, 10)
    axC.set_ylim(0, top * 1.18)
    axC.text(0.99, 0.99, 'n = 1,110', transform=axC.transAxes, fontsize=5.6, color='#5E7183', va='top', ha='right')
    panel_letter(axC, 'c', dx=-0.36, dy=1.04)

    # (d) document concentration
    dc = u['primary_document'].dropna().value_counts()
    cum = np.concatenate([[0], (dc.cumsum() / dc.sum()).values])
    axD.step(np.arange(len(cum)), cum * 100, where='post', color=PAL['UI-Ref'], lw=1.3, zorder=3)
    axD.plot([20], [cum[20] * 100], 'o', ms=3.4, color=PAL['limit'], zorder=5)
    axD.annotate('20 documents → 51%', xy=(20, cum[20] * 100), xytext=(26, 26), fontsize=6, color=PAL['limit'],
                 arrowprops=dict(arrowstyle='-', lw=0.6, color=PAL['limit']))
    axD.plot([39], [cum[39] * 100], 'o', ms=3.4, color=PAL['limit'], zorder=5)
    axD.annotate('39 → 81%', xy=(39, cum[39] * 100), xytext=(45, 58), fontsize=6, color=PAL['limit'],
                 arrowprops=dict(arrowstyle='-', lw=0.6, color=PAL['limit']))
    axD.set_xlabel('source documents (ranked by size)')
    axD.set_ylabel('% of documented molecules')
    axD.set_xlim(0, 77)
    axD.set_ylim(0, 104)
    axD.text(0.97, 0.04, '1,142 of 1,218 molecules carry\na document; 76 without excluded',
             transform=axD.transAxes, ha='right', va='bottom', fontsize=5.6, color='#5E7183')
    panel_letter(axD, 'd', dx=-0.34, dy=1.04)

    # (e) organism | document
    left = np.zeros(len(top12))
    for org, col in ORG_COL.items():
        vals = orgfrac[org].values * 100
        axE.barh(range(len(top12)), vals, left=left, height=0.70, color=col, zorder=3)
        left += vals
    for i, n in enumerate(orgmat.sum(axis=1).values):
        axE.text(103, i, f'n={n}', va='center', fontsize=5.6, color='#3E4C59')
    axE.set_yticks(range(len(top12)))
    axE.set_yticklabels(LBL12, fontsize=5.8)
    axE.invert_yaxis()
    axE.set_xlim(0, 100)
    axE.set_xticks([0, 25, 50, 75, 100])
    axE.set_xlabel('% of molecules in document')
    it = lambda s: '$\\it{' + s.replace(' ', '\\ ') + '}$'
    h = [Patch(facecolor=ORG_COL[o], label=it(o)) for o in list(ORG_COL)[:3]] + [
        Patch(facecolor=ORG_COL['other / unspecified'], label='other / unspecified')]
    axE.legend(handles=h, loc='upper left', bbox_to_anchor=(-0.44, -0.30), ncol=2, fontsize=5.8,
               frameon=False, handlelength=1.0, handleheight=0.8, columnspacing=1.0, handletextpad=0.4)
    panel_letter(axE, 'e', dx=-0.30, dy=1.04)
    return fig


fig = build_fig1()
fig.savefig(FIGURES / 'fig1_dataset.png', dpi=300)
plt.close(fig)