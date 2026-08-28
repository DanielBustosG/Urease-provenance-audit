"""Recompute the provenance noise floor from the shipped derived data.

Run:  python scripts/02_provenance_floor.py

This is the paper's central control: within UI-Ref alone -- where every molecule is a
urease inhibitor -- how well can a model identify which database, or which publication,
a molecule came from? Published values: 0.954 [0.939-0.968] for PubChem vs BindingDB on
25 descriptors, 0.964 on ECFP4, and a median 0.997 for document membership.

The script prints the recomputed value beside the published one and states the
divergence. It does not adjust anything to match.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ureaseaudit import config as cfg
from ureaseaudit.chem import fingerprint_matrix
from ureaseaudit.validation import bootstrap_ci

PUBLISHED = {"descriptors": cfg.PUBLISHED["provenance_floor_descriptors"],
             "ECFP4": cfg.PUBLISHED["provenance_floor_ecfp4"],
             "document_median": cfg.PUBLISHED["document_membership_median"]}


def rf():
    return RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS, class_weight="balanced",
        random_state=cfg.SEED, n_jobs=-1)


def main():
    ui = pd.read_parquet(cfg.UIREF)
    print(f"UI-Ref: {len(ui)} molecules\n")

    # --- database provenance: molecules exclusive to PubChem vs exclusive to BindingDB
    excl = ui[ui.n_source_dbs == 1].copy()
    sub = excl[excl.source_dbs.isin(["PubChem", "BindingDB"])]
    y = (sub.source_dbs == "PubChem").astype(int).to_numpy()
    print(f"exclusive-source molecules: PubChem {int(y.sum())}, BindingDB {int((1 - y).sum())}")

    cv = StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.SEED)
    rows = []
    for name, X in [("descriptors", sub[cfg.DESCRIPTORS].to_numpy()),
                    ("ECFP4", fingerprint_matrix(sub.smiles_std.tolist()))]:
        p = cross_val_predict(rf(), X, y, cv=cv, method="predict_proba")[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
        lo, hi = bootstrap_ci(y, p, n=cfg.N_BOOTSTRAP, seed=cfg.SEED)
        rows.append({"analysis": f"PubChem vs BindingDB ({name})", "AUC": auc,
                     "lo": lo, "hi": hi, "published": PUBLISHED[name]})
        print(f"  {name:12s} AUC {auc:.4f} [{lo:.4f}-{hi:.4f}]  published {PUBLISHED[name]:.4f}"
              f"   divergence {auc - PUBLISHED[name]:+.4f}")

    # --- publication provenance: one-vs-rest for the largest documents
    docs = ui.primary_document.value_counts().head(12).index
    aucs = []
    for doc in docs:
        yd = (ui.primary_document == doc).astype(int).to_numpy()
        if yd.sum() < 10:
            continue
        p = cross_val_predict(rf(), ui[cfg.DESCRIPTORS].to_numpy(), yd,
                              cv=StratifiedKFold(3, shuffle=True, random_state=cfg.SEED),
                              method="predict_proba")[:, 1]
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(yd, p))
    med = float(np.median(aucs))
    print(f"\n  document one-vs-rest, n={len(aucs)} documents: median AUC {med:.4f}"
          f"   published {PUBLISHED['document_median']:.4f}   divergence {med - PUBLISHED['document_median']:+.4f}")
    rows.append({"analysis": f"document one-vs-rest (median of {len(aucs)})", "AUC": med,
                 "lo": float(np.min(aucs)), "hi": float(np.max(aucs)),
                 "published": PUBLISHED["document_median"]})

    out = pd.DataFrame(rows)
    out["divergence"] = out.AUC - out.published
    dest = cfg.TABLES / "reproduction_provenance_floor.csv"
    out.to_csv(dest, index=False)
    print(f"\nwritten: {dest.relative_to(cfg.ROOT)}")
    worst = out.divergence.abs().max()
    print(f"largest divergence from published: {worst:.4f}")
    if worst > 0.01:
        print("NOTE: divergence exceeds 0.01. Record it in the README known-differences "
              "section rather than adjusting the analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
