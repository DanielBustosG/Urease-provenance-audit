"""Source for main Figure 6 (the honest predictor).

Run from the repository root:  python scripts/99_figure6_source.py

Reads the shipped result tables in results/tables/ and writes
results/figures/fig6_honest_predictor.png at 300 dpi. Panel (e) reads
tableS29_example_compound_trace.csv, which carries the corrected worked example
(see finding F11-selfaudit in results/corrections_registry.csv).
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


apply_figure_style(frame='open', sizes=(8,7,6))
mpl.rcParams['savefig.dpi'] = 300

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


# Load data
u = pd.read_parquet(DATA / 'uiref_curated.parquet')
ad = pd.read_csv(TABLES / 'tableS26_applicability_domain.csv')
calibration_df = pd.read_csv(TABLES / 'tableS25_calibration.csv')
conformal_df = pd.read_csv(TABLES / 'tableS27_conformal_coverage.csv'  )
cascade_bench_df = pd.read_csv(TABLES / 'tableS24_cascade_benchmark.csv')
corrected_trace_df = pd.read_csv(TABLES / 'tableS29_example_compound_trace.csv')

cb = cascade_bench_df

STAGE_ORDER = ['S1B_decoy', 'S1A_measured', 'S2_potency', 'S3_toxicophore', 'S4_clinical']
STAGE_LAB = {
    'S1B_decoy': '1B  inhibitor vs\nmatched decoy',
    'S1A_measured': '1A  active vs\nmeasured-inactive',
    'S2_potency': '2  potent vs weak\n($p$Act ≥ 6)',
    'S3_toxicophore': '3  alert-flagged\nvs not',
    'S4_clinical': '4  CLUE launched\nvs not'
}
STAGE_SCHEME = {
    'S1B_decoy': 'document',
    'S1A_measured': 'document',
    'S2_potency': 'document',
    'S3_toxicophore': 'scaffold',
    'S4_clinical': 'scaffold'
}
MODEL_C = {
    'best tree ensemble': PAL['UI-Ref'],
    'nearest-neighbour Tanimoto': '#6BA3D6',
    'logistic regression': '#9FB3C8',
    'majority class': PAL['naive'],
    '$y$-scrambled RF': PAL['limit']
}

MODELS = ['RF', 'XGBoost', 'LightGBM', 'logreg', 'kNN_Tanimoto', 'majority', 'RF_yscrambled']

rowsB = []
for st in STAGE_ORDER:
    s = cb[(cb['stage'] == st) & (cb['scheme'] == STAGE_SCHEME[st])]
    tree = s[s['model'].isin(['RF', 'LightGBM', 'XGBoost'])].sort_values('AUC', ascending=False).iloc[0]
    for lbl, r in [('best tree ensemble', tree),
                   ('nearest-neighbour Tanimoto', s[s['model'] == 'kNN_Tanimoto'].iloc[0]),
                   ('logistic regression', s[(s['model'] == 'logreg')].sort_values('AUC', ascending=False).iloc[0]),
                   ('majority class', s[s['model'] == 'majority'].iloc[0]),
                   ('$y$-scrambled RF', s[s['model'] == 'RF_yscrambled'].iloc[0])]:
        rowsB.append(dict(stage=st, label=lbl, model=r['model'], features=r['features'],
                          AUC=r['AUC'], lo=r['AUC_lo'], hi=r['AUC_hi'], MCC=r['MCC'], BalAcc=r['BalAcc']))
BM = pd.DataFrame(rowsB)


def fig6():
    fig = plt.figure(figsize=(7.0, 5.65))
    axA = fig.add_axes([0.230, 0.710, 0.235, 0.225])
    axB = fig.add_axes([0.565, 0.710, 0.130, 0.225])
    axC = fig.add_axes([0.825, 0.710, 0.135, 0.225])
    axD = fig.add_axes([0.230, 0.255, 0.185, 0.215])
    axE = fig.add_axes([0.520, 0.230, 0.440, 0.240])
    axE.set_axis_off()

    ys = np.arange(len(STAGE_ORDER))
    offs = {'best tree ensemble': 0.28, 'nearest-neighbour Tanimoto': 0.14, 'logistic regression': 0.0,
            'majority class': -0.16, '$y$-scrambled RF': -0.30}
    for lbl, off in offs.items():
        s = BM[BM['label'] == lbl].set_index('stage').loc[STAGE_ORDER]
        axA.errorbar(s['AUC'], ys + off, xerr=[s['AUC'] - s['lo'], s['hi'] - s['AUC']], fmt='o',
                     ms=4.0 if lbl == 'best tree ensemble' else 3.0, color=MODEL_C[lbl],
                     ecolor=MODEL_C[lbl], elinewidth=0.8, capsize=1.4,
                     zorder=5 if lbl == 'best tree ensemble' else 4, label=lbl)
    axA.axvline(0.5, color='#B0B7BE', lw=0.8, ls=':', zorder=2)
    axA.set_yticks(ys)
    axA.set_yticklabels([STAGE_LAB[s] for s in STAGE_ORDER], fontsize=5.7)
    axA.invert_yaxis()
    axA.set_ylim(len(ys) - 0.45, -0.55)
    axA.set_xlim(0.42, 1.0)
    axA.set_xticks([0.5, 0.7, 0.9, 1.0])
    axA.set_xlabel('ROC AUC under the honest split (95% CI)')
    h = [mlines.Line2D([], [], marker='o', ls='', ms=3.8, color=PAL['UI-Ref'], label='best tree ensemble'),
         mlines.Line2D([], [], marker='o', ls='', ms=3.8, color='#6BA3D6', label='nearest-neighbour Tanimoto'),
         mlines.Line2D([], [], marker='o', ls='', ms=3.8, color='#9FB3C8', label='logistic regression'),
         mlines.Line2D([], [], marker='o', ls='', ms=3.8, color='#AAB2BD', label='majority class'),
         mlines.Line2D([], [], marker='o', ls='', ms=3.8, color=PAL['limit'], label='$y$-scrambled RF')]
    axA.legend(handles=h, loc='upper left', bbox_to_anchor=(-0.02, -0.30), fontsize=5.5, frameon=False,
               ncol=2, handletextpad=0.25, columnspacing=1.1, labelspacing=0.25, borderpad=0.05)
    panel_letter(axA, 'a', dx=-0.63, dy=1.04)

    cal = calibration_df
    raw = cal[cal['calibration'] == 'none (raw RF)'].set_index('stage').loc[STAGE_ORDER]
    pl = cal[cal['calibration'] == 'Platt (sigmoid)'].set_index('stage').loc[STAGE_ORDER]
    for yy, (r, p) in enumerate(zip(raw['ECE'], pl['ECE'])):
        axB.plot([r, p], [yy, yy], '-', lw=0.7, color='#C6CBD1', zorder=2)
    axB.plot(raw['ECE'], ys, 'o', ms=3.4, color=PAL['naive'], zorder=4, label='raw random forest')
    axB.plot(pl['ECE'], ys, 'o', ms=3.6, color=PAL['UI-Ref'], zorder=5, label='Platt-calibrated')
    axB.set_yticks(ys)
    axB.set_yticklabels(['1B', '1A', '2', '3', '4'], fontsize=6)
    axB.invert_yaxis()
    axB.set_ylim(len(ys) - 0.45, -0.55)
    axB.set_xlim(0, 0.185)
    axB.set_xticks([0, 0.05, 0.10, 0.15])
    axB.set_xlabel('expected calib. error')
    axB.set_ylabel('cascade stage')
    lgB = axB.legend(loc='upper left', bbox_to_anchor=(-0.10, -0.42), fontsize=5.5, frameon=False,
                     handletextpad=0.25, labelspacing=0.25, borderpad=0.05)
    panel_letter(axB, 'b', dx=-0.52, dy=1.04)

    cf = conformal_df
    for st in STAGE_ORDER:
        s = cf[cf['stage'] == st].sort_values('nominal_coverage')
        col = PAL['limit'] if st == 'S1A_measured' else PAL['UI-Ref']
        axC.plot(s['nominal_coverage'] * 100, s['empirical_coverage'] * 100, '-o', lw=1.0, ms=3.0,
                 color=col, alpha=0.9, zorder=5 if st == 'S1A_measured' else 4)
    axC.plot([78, 97], [78, 97], '--', lw=0.9, color='#8A929B', zorder=3)
    axC.text(96.5, 86, 'exact\ncoverage', fontsize=5.3, color='#6B7480', ha='right')
    axC.text(88, 60, 'stage 1A', fontsize=5.5, color=PAL['limit'])
    axC.set_xlim(78, 97)
    axC.set_ylim(55, 100)
    axC.set_xticks([80, 90, 95])
    axC.set_xlabel('nominal coverage (%)')
    axC.set_ylabel('empirical coverage (%)')
    axC.text(0.02, -0.34, 'uninformative two-class sets:\n3.9–71.3% across stages',
             transform=axC.transAxes, fontsize=5.4, color='#5E7183', va='top')
    panel_letter(axC, 'c', dx=-0.78, dy=1.04)

    cov = ad[ad['table'] == 'collection_coverage'].copy()
    order = ['UI-Ref (self, leave-one-out)', 'DSSTox', 'CLUE', 'COCONUT', 'matched decoys (Stage 1B negatives)']
    lab = ['UI-Ref, leave-one-out\n(n = 1,218)', 'DSSTox (n = 11,694)', 'CLUE (n = 4,431)',
           'COCONUT (n = 12,000)', 'matched decoys\n(n = 1,218)']
    cov = cov.set_index('collection').loc[order]
    colr = [PAL['UI-Ref'], PAL['DSSTox'], PAL['CLUE'], PAL['COCONUT'], PAL['decoys']]
    yd = np.arange(len(cov))
    axD.barh(yd, cov['frac_OUTSIDE_either'] * 100, 0.62, color=colr, zorder=3)
    for yy, v in enumerate(cov['frac_OUTSIDE_either'] * 100):
        axD.text(v + 2, yy, f'{v:.1f}%', va='center', fontsize=5.8, color='#1F2933')
    axD.set_yticks(yd)
    axD.set_yticklabels(lab, fontsize=5.6)
    axD.invert_yaxis()
    axD.set_xlim(0, 118)
    axD.set_xticks([0, 50, 100])
    axD.set_xlabel('% outside UI-Ref applicability domain')
    axD.set_ylim(len(cov) - 0.4, -0.6)
    axD.text(0.02, -0.42, 'preregistered domain: ECFP4 Tanimoto ≥ 0.40 to the\n'
             'nearest training molecule AND Mahalanobis within\nthe $\\chi^2_{0.95}$ cutoff',
             transform=axD.transAxes, fontsize=5.4, color='#5E7183', va='top')
    axD.set_position(axD.get_position())
    panel_letter(axD, 'd', dx=-0.70, dy=1.04)

    t10 = corrected_trace_df.set_index('stage').loc[STAGE_ORDER]
    axE.set_xlim(0, 1)
    axE.set_ylim(0, 1)
    # The claim-title moved to the caption, like every other panel in the figure set.
    # What stays is a one-line orientation note: the row order is the cascade's own
    # execution order (1B runs before 1A), which is not guessable from the labels.
    axE.text(0, 1.20, 'rows follow the cascade\'s execution order: stage 1B runs before 1A',
             fontsize=5.5, color='#5E7183', va='top', transform=axE.transAxes)
    # Column centres spaced so no header touches its neighbour. 'why refused' replaces
    # 'governing refusal reason', which was wide enough to collide with 'verdict'.
    cols = [(0.00, 'stage', 'left'), (0.375, 'calibrated\n$P$', 'center'),
            (0.495, 'nearest\n$T$', 'center'),
            (0.685, 'why refused', 'center'), (0.925, 'verdict', 'center')]
    for xx, h2, al in cols:
        axE.text(xx, 0.93, h2, fontsize=5.5, color='#5E7183', va='bottom', ha=al)
    axE.plot([0, 1], [0.90, 0.90], '-', lw=0.6, color='#B0B7BE')
    rows = [('1B  inhibitor vs decoy', 'convention conflict'),
            ('1A  active vs inactive', 'convention conflict'),
            ('2  potent vs weak', 'uninformative set'),
            ('3  alert-flagged vs not', 'not a model'),
            ('4  CLUE launched vs not', 'outside domain')]
    for i, (st, (lbl, cset)) in enumerate(zip(STAGE_ORDER, rows)):
        yy = 0.78 - i * 0.165
        r = t10.loc[st]
        axE.text(0.00, yy, lbl, fontsize=5.6, va='center')
        axE.text(0.375, yy, f"{r['p_calibrated']:.3f}", fontsize=5.6, va='center', ha='center')
        tcol = PAL['limit'] if r['nn_tanimoto_frozen'] < 0.40 else '#1F2933'
        axE.text(0.495, yy, f"{r['nn_tanimoto_frozen']:.2f}", fontsize=5.6, va='center', ha='center', color=tcol)
        axE.text(0.685, yy, cset, fontsize=5.3, va='center', ha='center')
        # Sentence case: the colour already carries the emphasis, so caps only shout.
        axE.text(0.925, yy, 'refused', fontsize=5.7, va='center', ha='center',
                 color=PAL['limit'], fontweight='bold')
    axE.plot([0, 1], [0.05, 0.05], '-', lw=0.6, color='#B0B7BE')
    # The five-line explanation moved to the caption: it is prose, and at 5.3 pt inside
    # the figure it was neither legible nor reusable. Only the input structure stays,
    # because the panel is meaningless without knowing which molecule was scored.
    axE.text(0, -0.12, 'input: CN(C)c1ccc(/C=N/N=C2\\NC(=O)CS2)cc1',
             fontsize=5.4, color='#3E4C59', va='top', family='monospace')
    panel_letter(axE, 'e', dx=-0.11, dy=1.04)
    return fig


fig = fig6()
fig.savefig(FIGURES / 'fig6_honest_predictor.png', dpi=300, bbox_inches='tight')
plt.close(fig)