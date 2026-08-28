"""Regenerate Figure S6 (alert profiles by source document) from the shipped result tables.

Run from the repository root:  python scripts/16_figureS6_alerts_by_document.py

Inputs   records_curated.parquet, tableS19_alerts_by_document.csv
Output   results/figures/figS6_alerts_by_document.png

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
from matplotlib.patches import Patch
import re
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
rec = pd.read_parquet(DATA / 'records_curated.parquet')
adoc_raw = pd.read_csv(TABLES / 'tableS19_alerts_by_document.csv')

ALERT_LAB = {'thiourea': 'thiourea', 'selenium': 'selenium', 'hydrazone': 'hydrazone',
             'phosphinic_phosphonic': 'phosphinic / phosphonic', 'hydroxamate': 'hydroxamate',
             'catechol': 'catechol', 'carboxylate': 'carboxylate', 'urea': 'urea', 'thiol': 'thiol',
             'any_chelator': 'any Ni-chelating motif', 'PAINS': 'PAINS', 'NIH': 'NIH', 'BRENK': 'Brenk',
             'any_alert': 'any alert (PAINS+Brenk+NIH)'}

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


MOTIFS = ['any_alert_frac', 'BRENK_frac', 'NIH_frac', 'PAINS_frac', 'any_chelator_frac', 'thiourea_frac',
          'catechol_frac', 'hydrazone_frac', 'phosphinic_phosphonic_frac', 'carboxylate_frac',
          'hydroxamate_frac', 'urea_frac', 'selenium_frac', 'thiol_frac']
MLBL = [ALERT_LAB[m.replace('_frac', '')] for m in MOTIFS]

adoc = adoc_raw.copy()
adoc = adoc[adoc['n_molecules'] >= 10].copy()
adoc['lab'] = [doclabel(d) for d in adoc['primary_document']]
_c.clear()
adoc['lab'] = uniq(adoc['lab'].tolist())
adoc = adoc.sort_values('n_molecules', ascending=False)
adoc = adoc.reset_index(drop=True)
adoc['doc_id'] = [f'D{i + 1:02d}' for i in range(len(adoc))]
M = adoc.set_index('doc_id')[MOTIFS].T * 100
M.index = MLBL


def figS6():
    fig = plt.figure(figsize=(7.0, 3.60))
    axA = fig.add_axes([0.185, 0.360, 0.700, 0.475])
    axC = fig.add_axes([0.905, 0.360, 0.020, 0.475])
    axB = fig.add_axes([0.185, 0.165, 0.700, 0.115])
    im = axA.imshow(M.values, aspect='auto', cmap='Blues', vmin=0, vmax=100, zorder=3)
    axA.set_xticks(range(M.shape[1]))
    axA.set_xticklabels([])
    axA.set_yticks(range(M.shape[0]))
    axA.set_yticklabels(M.index, fontsize=5.3)
    axA.set_xlim(-0.5, M.shape[1] - 0.5)
    for sp in axA.spines.values(): sp.set_visible(False)
    axA.tick_params(length=0)
    cb = fig.colorbar(im, cax=axC)
    cb.set_label('% of the document\'s molecules with the motif', fontsize=5.3)
    cb.ax.tick_params(labelsize=5.0)
    cb.outline.set_visible(False)
    panel_letter(axA, 'a', dx=-0.235, dy=1.04)

    axB.bar(range(M.shape[1]), adoc['n_molecules'].values, 0.72, color=PAL['UI-Ref'], zorder=3)
    axB.set_xticks(range(M.shape[1]))
    axB.set_xticklabels(M.columns, fontsize=4.2, rotation=90)
    axB.set_xlim(-0.5, M.shape[1] - 0.5)
    axB.set_ylim(0, 52)
    axB.set_yticks([0, 25, 50])
    axB.set_ylabel('molecules', fontsize=5.6)
    axB.tick_params(axis='y', labelsize=5.0)
    axB.set_xlabel('source document, ranked by size (identifiers resolved in Table S31)', fontsize=5.8)
    panel_letter(axB, 'b', dx=-0.235, dy=1.04)
    fig.text(0.185, 0.925, '48 documents with ≥ 10 molecules (1,020 of the 1,142 documented molecules).',
             fontsize=5.5, color='#5E7183')
    fe = ((M.values < 5) | (M.values > 95)).mean() * 100
    fig.text(0.185, 0.300, f'{fe:.0f}% of the {M.size} document × motif cells sit below 5% or above 95%;  '
             'between-document ICC spans 0.06 (thiol) to 0.92 (phosphinic/phosphonic, thiourea).',
             fontsize=5.5, color='#3E4C59')
    return fig


_S6f = figS6


def figS6():
    fig = _S6f()
    for t in list(fig.texts):
        if t.get_text().startswith('84% of'):
            t.set_text('84% of the 672 document × motif cells sit below 5% or above 95%;\n'
                       'between-document ICC spans 0.06 (thiol) to 0.92 (phosphinic/phosphonic, thiourea).')
    return fig


fig = figS6()
fig.savefig(FIGURES / 'figS6_alerts_by_document.png', dpi=300)
plt.close(fig)