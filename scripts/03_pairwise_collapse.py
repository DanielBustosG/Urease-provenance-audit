"""Reproduce the naive-to-honest AUC collapse for all three comparisons.

Run:  python scripts/03_pairwise_collapse.py

Published honest values (property-matched negatives + document-grouped CV):
  CLUE 0.784 [0.765-0.801], COCONUT 0.838 [0.823-0.853], DSSTox 0.760 [0.741-0.778]
all of which sit BELOW the 0.954 provenance floor from script 02.
"""
import sys
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ureaseaudit import config as cfg
from ureaseaudit.validation import greedy_property_match, document_groups, bootstrap_ci, summarize

PUBLISHED = {c: {"naive": cfg.PUBLISHED["naive_auc"][c], "honest": cfg.PUBLISHED["honest_auc"][c]}
             for c in ("CLUE", "COCONUT", "DSSTox")}
FLOOR = cfg.PUBLISHED["provenance_floor_descriptors"]


def rf():
    return RandomForestClassifier(n_estimators=cfg.RF_N_ESTIMATORS, class_weight="balanced",
                                  random_state=cfg.SEED, n_jobs=-1)


def main():
    ui = pd.read_parquet(cfg.UIREF)
    rows = []
    for coll in ["CLUE", "COCONUT", "DSSTox"]:
        neg = pd.read_parquet(cfg.COLLECTION[coll])
        neg = neg[neg[cfg.DESCRIPTORS].notna().all(axis=1)]

        # (a) naive: all negatives, random stratified CV
        X = np.vstack([ui[cfg.DESCRIPTORS].to_numpy(), neg[cfg.DESCRIPTORS].to_numpy()])
        y = np.r_[np.ones(len(ui), int), np.zeros(len(neg), int)]
        p = cross_val_predict(rf(), X, y, method="predict_proba",
                              cv=StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.SEED))[:, 1]
        naive = summarize(y, p)
        lo, hi = bootstrap_ci(y, p, n=cfg.N_BOOTSTRAP, seed=cfg.SEED)

        # (e) honest: property-matched negatives AND document-grouped folds
        matched, smd_b, smd_a = greedy_property_match(ui, neg, cfg.MATCH_FEATURES, seed=cfg.SEED)
        Xh = np.vstack([ui[cfg.DESCRIPTORS].to_numpy(), matched[cfg.DESCRIPTORS].to_numpy()])
        yh = np.r_[np.ones(len(ui), int), np.zeros(len(matched), int)]
        # each decoy is its own group; positives group by source publication
        groups = np.r_[document_groups(ui), np.array([f"DECOY::{i}" for i in range(len(matched))])]
        ph = cross_val_predict(rf(), Xh, yh, groups=groups, method="predict_proba",
                               cv=StratifiedGroupKFold(cfg.CV_FOLDS, shuffle=True,
                                                       random_state=cfg.SEED))[:, 1]
        honest = summarize(yh, ph)
        hlo, hhi = bootstrap_ci(yh, ph, n=cfg.N_BOOTSTRAP, seed=cfg.SEED)

        print(f"\nUI-Ref vs {coll}   (n_pos={len(ui)}, n_neg_full={len(neg)}, n_matched={len(matched)})")
        print(f"  naive  AUC {naive['AUC']:.4f} [{lo:.4f}-{hi:.4f}]  MCC {naive['MCC']:.4f}"
              f"   published {PUBLISHED[coll]['naive']:.3f}")
        print(f"  honest AUC {honest['AUC']:.4f} [{hlo:.4f}-{hhi:.4f}]  MCC {honest['MCC']:.4f}"
              f"   published {PUBLISHED[coll]['honest']:.3f}")
        print(f"  collapse {naive['AUC'] - honest['AUC']:+.4f} | below the {FLOOR} floor: "
              f"{honest['AUC'] < FLOOR}")
        print(f"  max |SMD| before matching {max(abs(v) for v in smd_b.values()):.4f}"
              f" -> after {max(abs(v) for v in smd_a.values()):.4f}")

        rows.append({"comparison": coll, "naive_AUC": naive["AUC"], "naive_MCC": naive["MCC"],
                     "honest_AUC": honest["AUC"], "honest_AUC_lo": hlo, "honest_AUC_hi": hhi,
                     "honest_MCC": honest["MCC"], "collapse": naive["AUC"] - honest["AUC"],
                     "below_provenance_floor": bool(honest["AUC"] < FLOOR),
                     "max_abs_SMD_before": max(abs(v) for v in smd_b.values()),
                     "max_abs_SMD_after": max(abs(v) for v in smd_a.values()),
                     "published_naive": PUBLISHED[coll]["naive"],
                     "published_honest": PUBLISHED[coll]["honest"]})

    out = pd.DataFrame(rows)
    out["divergence_honest"] = out.honest_AUC - out.published_honest
    dest = cfg.TABLES / "reproduction_pairwise_collapse.csv"
    out.to_csv(dest, index=False)
    print(f"\nwritten: {dest.relative_to(cfg.ROOT)}")
    print(f"all honest AUCs below the floor: {bool(out.below_provenance_floor.all())}")
    print(f"largest divergence from published honest AUC: {out.divergence_honest.abs().max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
