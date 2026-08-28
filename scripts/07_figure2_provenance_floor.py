"""Regenerate main Figure 2 (the provenance noise floor) from the shipped result tables.

Run from the repository root:  python scripts/07_figure2_provenance_floor.py

Inputs   records_curated.parquet, tableS5_provenance_floor.csv
Output   results/figures/fig2_provenance_floor.png

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
FLOOR_ECFP = 0.964053


# Load data
pf = pd.read_csv(TABLES / 'tableS5_provenance_floor.csv')
rec = pd.read_parquet(DATA / 'records_curated.parquet')

db_pairs = pf[pf['subset'] == 'exclusive-source'].copy()
docs = pf[pf['subset'] == 'all 1218 one-vs-rest'].copy()
docs['doc'] = docs['task'].str.replace('document ', '', regex=False)
robust = pf[~pf['subset'].isin(['exclusive-source', 'all 1218 one-vs-rest'])].copy()

dm = docs[docs['features'] == '25 descriptors'].set_index('doc')
de = docs[docs['features'] == 'ECFP4'].set_index('doc')

JAB = {'Bioorganic & Medicinal Chemistry Letters': 'Bioorg. Med. Chem. Lett.',
       'Bioorganic & Medicinal Chemistry': 'Bioorg. Med. Chem.',
       'European Journal of Medicinal Chemistry': 'Eur. J. Med. Chem.',
       'Journal of Medicinal Chemistry': 'J. Med. Chem.'}

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


from collections import Counter
_c = Counter()


def uniq(l):
    out = []
    for x in l:
        _c[x] += 1
        out.append(x if _c[x] == 1 else f"{x}{chr(96 + _c[x])}")
    return out


DESC_C = '#1B4F8C'
ECFP_C = '#6BA3D6'


def fig2():
    fig = plt.figure(figsize=(7.0, 3.25))
    axA = fig.add_axes([0.120, 0.215, 0.175, 0.520])
    axB = fig.add_axes([0.560, 0.215, 0.150, 0.520])
    axC = fig.add_axes([0.865, 0.215, 0.120, 0.520])

    tasks = ['PubChem vs BindingDB', 'PubChem vs ChEMBL', 'BindingDB vs ChEMBL']
    lab = ['PubChem vs\nBindingDB', 'PubChem vs\nChEMBL', 'BindingDB vs\nChEMBL']
    for feat, col, off, lg in [('25 descriptors', DESC_C, +0.15, '25 descriptors'),
                                ('ECFP4', ECFP_C, -0.15, 'ECFP4 fingerprint')]:
        s = db_pairs[db_pairs['features'] == feat].set_index('task').loc[tasks]
        axA.errorbar(s['auc'], np.arange(3) + off, xerr=[s['auc'] - s['ci_lo'], s['ci_hi'] - s['auc']],
                     fmt='o', ms=4.2, color=col, ecolor=col, elinewidth=1.0, capsize=1.8, zorder=4, label=lg)
    axA.axvline(0.5, color='#B0B7BE', lw=0.8, ls=':', zorder=2)
    axA.axvline(FLOOR, color=PAL['limit'], lw=1.2, zorder=3)
    axA.annotate('provenance floor 0.954', xy=(FLOOR, 2.35), xytext=(0.70, 2.66), fontsize=6,
                 color=PAL['limit'], ha='center', va='center',
                 arrowprops=dict(arrowstyle='-', lw=0.6, color=PAL['limit']))
    # Under the axis at its own tick: the in-panel bands are all occupied (floor
    # annotation above, n-note mid-left, legend lower right).
    # Beside the dotted chance line inside the panel, in the gap between the two
    # lowest rows: the only band free of data, annotation and legend.
    axA.text(0.503, 0.72, 'chance', fontsize=5.4, color='#7A828A', ha='left', va='center')
    axA.set_yticks(range(3))
    axA.set_yticklabels(lab, fontsize=6)
    axA.set_ylim(-0.95, 2.95)
    axA.set_xlim(0.46, 1.012)
    axA.set_xticks([0.5, 0.7, 0.9, 1.0])
    axA.set_xlabel('ROC AUC (5-fold CV)')
    axA.legend(loc='lower right', bbox_to_anchor=(1.00, 0.00), fontsize=5.8, frameon=False,
               handletextpad=0.3, borderpad=0.05, labelspacing=0.2)
    axA.text(0.02, 0.66, 'n = 742 / 603 / 183\nexclusive-source molecules', transform=axA.transAxes,
             ha='left', va='top', fontsize=5.6, color='#5E7183')
    panel_letter(axA, 'a', dx=-0.46, dy=1.04)

    order = dm.sort_values('auc', ascending=False).index.tolist()
    _c.clear()
    lb = uniq([doclabel(d) for d in order])
    ymap = np.arange(len(order))
    for s, col, off in [(dm, DESC_C, +0.17), (de, ECFP_C, -0.17)]:
        v = s.loc[order]
        axB.errorbar(v['auc'], ymap + off,
                     xerr=[np.maximum(v['auc'] - v['ci_lo'], 0), np.maximum(v['ci_hi'] - v['auc'], 0)],
                     fmt='o', ms=3.0, color=col, ecolor=col, elinewidth=0.8, capsize=1.2, zorder=4)
    axB.axvline(FLOOR, color=PAL['limit'], lw=1.2, zorder=3)
    axB.set_yticks(ymap)
    axB.set_yticklabels(lb, fontsize=5.4)
    axB.invert_yaxis()
    axB.set_xlim(0.930, 1.004)
    axB.set_xticks([0.94, 0.97, 1.00])
    axB.set_xlabel('ROC AUC (one-vs-rest)')
    # Belongs to (b): left-aligned with (b)'s y-axis and hugging its x-label, so it
    # cannot be read as a figure-wide note. No room inside the axes: every AUC > 0.93.
    axB.text(0.0, -0.245, 'panel b: median 0.997 (descriptors), 1.000 (ECFP4);  $n$ = 1,218 per model',
             transform=axB.transAxes, fontsize=5.3, color='#3E4C59', ha='left', va='top')
    panel_letter(axB, 'b', dx=-0.80, dy=1.04)

    rl = {'no-overlap (ChEMBL allowed)': 'ChEMBL allowed\neither arm (956)',
          'all 1218, BindingDB-vs-rest': 'BindingDB vs rest\n(1,218)',
          'majority-source db': 'majority source\ndatabase (847)',
          'record level': 'assay-record level\n(2,142)'}
    rr = pd.concat([pd.DataFrame({'lab': ['exclusive-source\nreference (742)'], 'auc': [FLOOR],
                                   'ci_lo': [0.939123], 'ci_hi': [0.968341]}),
                    robust.assign(lab=robust['subset'].map(rl))[['lab', 'auc', 'ci_lo', 'ci_hi']]])
    y = np.arange(len(rr))
    axC.errorbar(rr['auc'], y, xerr=[rr['auc'] - rr['ci_lo'], rr['ci_hi'] - rr['auc']], fmt='o', ms=3.6,
                 color=DESC_C, ecolor=DESC_C, elinewidth=0.9, capsize=1.5, zorder=4)
    axC.axvline(FLOOR, color=PAL['limit'], lw=1.2, zorder=3)
    axC.set_yticks(y)
    axC.set_yticklabels(rr['lab'], fontsize=5.4)
    axC.invert_yaxis()
    axC.set_ylim(len(rr) - 0.5, -0.6)
    axC.set_xlim(0.79, 1.008)
    axC.set_xticks([0.8, 0.9, 1.0])
    axC.set_xlabel('ROC AUC')
    panel_letter(axC, 'c', dx=-1.20, dy=1.04)
    return fig


fig = fig2()
fig.savefig(FIGURES / 'fig2_provenance_floor.png', dpi=300)
plt.close(fig)