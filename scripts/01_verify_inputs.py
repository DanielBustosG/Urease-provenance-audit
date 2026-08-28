"""Verify the analysis inputs before anything is computed from them.

Run:  python scripts/01_verify_inputs.py

Three checks, and the script exits non-zero if any fails, so later steps cannot
silently analyse a different data snapshot:

1. The derived datasets exist and have the expected shape.
2. Descriptor completeness is exactly 1.000. If it is 0.9926 instead, the
   phosphinic-acid repair was skipped and nine inhibitors are missing every
   descriptor (see the standardization notes in README.md).
3. If the primary database exports are present in data/raw/, their SHA-256
   checksums match data/raw/CHECKSUMS.csv. They are not redistributed, so
   absence is expected and is reported rather than treated as an error.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ureaseaudit import config as cfg

EXPECTED = {
    "uiref_molecules": 1218,
    "uiref_columns_min": 47,
    "records": 2879,
    "collections": {"CLUE": 4431, "COCONUT": 12000, "DSSTox": 11694},
    "with_quantitative_pact": 1110,
    "with_measured_inactive": 98,
    "named_documents": 77,
}


def main() -> int:
    failures = []

    # ---------------------------------------------------------------- derived data
    ui = pd.read_parquet(cfg.UIREF)
    print(f"UI-Ref                 {len(ui):>6d} molecules x {ui.shape[1]} columns")
    if len(ui) != EXPECTED["uiref_molecules"]:
        failures.append(f"UI-Ref has {len(ui)} molecules, expected {EXPECTED['uiref_molecules']}")
    if ui.shape[1] < EXPECTED["uiref_columns_min"]:
        failures.append(f"UI-Ref has {ui.shape[1]} columns, expected at least {EXPECTED['uiref_columns_min']}")

    rec = pd.read_parquet(cfg.RECORDS)
    print(f"assay records          {len(rec):>6d}")
    if len(rec) != EXPECTED["records"]:
        failures.append(f"records table has {len(rec)} rows, expected {EXPECTED['records']}")

    for name, path in cfg.COLLECTION.items():
        d = pd.read_parquet(path)
        print(f"collection {name:<10s} {len(d):>6d}")
        if len(d) != EXPECTED["collections"][name]:
            failures.append(f"{name} has {len(d)} molecules, expected {EXPECTED['collections'][name]}")

    # ------------------------------------------------------- descriptor completeness
    completeness = float(ui[cfg.DESCRIPTORS].notna().all(axis=1).mean())
    print(f"\ndescriptor completeness  {completeness:.4f}")
    if completeness < 1.0:
        failures.append(
            f"descriptor completeness is {completeness:.4f}, not 1.000 — the "
            "phosphinic-acid repair ([P+](=O)O -> [PH](=O)O) was probably skipped"
        )

    # ------------------------------------------------------------- label composition
    n_pact = int((ui.n_direct > 0).sum())
    n_inact = int(ui.any_inactive.sum())
    n_docs = int(ui.primary_document.nunique())
    print(f"with quantitative pAct   {n_pact}")
    print(f"with measured-inactive   {n_inact}")
    print(f"named source documents   {n_docs}")
    for got, want, label in [(n_pact, EXPECTED["with_quantitative_pact"], "quantitative pAct"),
                             (n_inact, EXPECTED["with_measured_inactive"], "measured-inactive"),
                             (n_docs, EXPECTED["named_documents"], "named documents")]:
        if got != want:
            failures.append(f"{label}: {got}, expected {want}")

    # ----------------------------------------------------------------- raw checksums
    print()
    if cfg.RAW_CHECKSUM_FILE.exists():
        import hashlib

        sums = pd.read_csv(cfg.RAW_CHECKSUM_FILE)
        present = missing = 0
        for r in sums.itertuples():
            f = cfg.DATA_RAW / r.filename
            if not f.exists():
                missing += 1
                continue
            present += 1
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            if digest != r.sha256:
                failures.append(
                    f"{r.filename}: checksum mismatch. The upstream database is "
                    "versioned; a different snapshot will not reproduce the paper."
                )
            else:
                print(f"raw checksum OK          {r.filename}")
        print(f"raw exports              {present} present, {missing} not shipped "
              "(expected — see data/raw/README.md)")
    else:
        print("raw exports              CHECKSUMS.csv absent")

    # ------------------------------------------------------------------------ verdict
    print()
    if failures:
        print(f"FAILED — {len(failures)} check(s) did not pass:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All input checks passed. Steps 02 onward are safe to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
