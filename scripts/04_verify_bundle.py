"""Verify the frozen model bundle and re-run the worked example end to end.

Run:  python scripts/04_verify_bundle.py

Confirms the shipped bundle loads and matches its recorded checksum, then puts the
prototype interface's own compound through the corrected cascade and prints exactly
what the system says about it. The published result is a refusal at every stage.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
from ureaseaudit import config as cfg

PROTOTYPE_COMPOUND = r"CN(C)c1ccc(/C=N/N=C2\NC(=O)CS2)cc1"
PROTOTYPE_CLAIM = 'NON-INHIBITOR, "Yes 0.462 / No 0.538"'
EXPECTED_BUNDLE_SHA16 = cfg.BUNDLE_SHA256_PREFIX


def main():
    from cascade import predict, BUNDLE_SHA256_PREFIX

    digest = hashlib.sha256(cfg.BUNDLE.read_bytes()).hexdigest()[:16]
    print(f"bundle:          {cfg.BUNDLE.relative_to(ROOT)}")
    print(f"sha256[:16]:     {digest}")
    print(f"expected:        {EXPECTED_BUNDLE_SHA16}   match: {digest == EXPECTED_BUNDLE_SHA16}")
    print(f"module constant: {BUNDLE_SHA256_PREFIX}")

    print(f"\nprototype compound: {PROTOTYPE_COMPOUND}")
    print(f"prototype interface reported: {PROTOTYPE_CLAIM}\n")

    res = predict(PROTOTYPE_COMPOUND, alpha=0.10)
    if not res["parse_ok"]:
        print(f"FAILED to parse: {res.get('error')}")
        return 1

    rows = []
    for name, st in res["stages"].items():
        d = st["diagnostics"]
        rows.append({
            "stage": name,
            "p_calibrated": d.get("p_positive_reported"),
            "nn_tanimoto": d.get("nn_tanimoto"),
            "in_applicability_domain": d.get("in_applicability_domain"),
            "conformal_set": str(d.get("conformal_set")),
            "refused": st["refused"],
            "answer": st["answer"],
            "refusal_reasons": " | ".join(st["refusal_reasons"] or []),
        })
        verdict = "REFUSED" if st["refused"] else str(st["answer"])
        print(f"  {name:16s} P={d['p_positive_reported']:.3f}  T_nn={d['nn_tanimoto']:.2f}"
              f"  in_domain={str(d['in_applicability_domain']):5s}"
              f"  set={str(d.get('conformal_set')):22s} {verdict}")

    out = pd.DataFrame(rows)
    dest = cfg.TABLES / "reproduction_example_trace.csv"
    out.to_csv(dest, index=False)

    n_ref = int(out.refused.sum())
    print(f"\nrefused at {n_ref} of {len(out)} stages")
    print(f"phosphinic repair applied to this input: {res['phosphinic_repair_applied']}")
    print(f"standardization conventions agree:       {res['conventions_agree']}")
    if n_ref == len(out):
        print("\nThe corrected system refuses at every stage. A documented refusal, with the"
              "\nnearest measured analogue attached, replaces the prototype's 0.538 verdict.")
    print(f"\nwritten: {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
