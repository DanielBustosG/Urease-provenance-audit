"""CLI for the corrected urease cascade.

    python -m app.predict "CN(C)c1ccc(/C=N/N=C2\\NC(=O)CS2)cc1"
    python -m app.predict --alpha 0.05 --presets
    python -m app.predict "OCCO" --quiet | jq .cascade_summary

Emits JSON on stdout. This is a thin argument parser over
``app.cascade.predict`` — no inference logic lives here, so the CLI and the
Streamlit UI can never disagree (enforced by
``test_app.py::test_cli_agrees_with_library``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Work both as `python -m app.predict` from the repository root and as
# `python app/predict.py` from anywhere: put the repository root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import cascade
except ImportError:  # invoked with app/ itself on sys.path
    import cascade


def _json_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_one(smiles: str, alpha: float, bundle_file: str | None,
            with_neighbour_activity: bool = True) -> dict:
    res = cascade.predict(smiles, alpha=alpha, bundle_file=bundle_file)
    if with_neighbour_activity and res.get("parse_ok"):
        for sv in res["stages"].values():
            sv["diagnostics"]["nearest_neighbours"] = cascade.annotate_neighbours(
                sv["diagnostics"]["nearest_neighbours"])
    return res


def human_summary(res: dict) -> str:
    """One-screen text rendering that obeys the same refusal contract as the UI."""
    lines: list[str] = []
    if not res.get("parse_ok"):
        return f"INPUT REJECTED: {res.get('error')}  ({res['input_smiles']})"
    lines.append(f"input                : {res['input_smiles']}")
    lines.append(f"standardized (frozen): {res['standardization']['frozen_curation']['smiles']}")
    lines.append(f"InChIKey             : {res['standardization']['frozen_curation']['inchikey']}")
    if res["phosphinic_repair_applied"]:
        lines.append("phosphinic repair    : APPLIED ([P+](=O)O -> [PH](=O)O)")
    if not res["conventions_agree"]:
        lines.append("standardization      : CONVENTIONS DISAGREE "
                     f"(bundle note gives {res['standardization']['bundle_featurization_note']['smiles']})")
    al = res["structural_alerts_exact_rule"]
    lines.append(f"structural alerts    : {al['n_alerts']}"
                 + (f"  [{', '.join(al['alerts'])}]" if al["alerts"] else ""))
    lines.append("")
    for name, sv in res["stages"].items():
        d = sv["diagnostics"]
        if sv["refused"]:
            lines.append(f"{name:<16} REFUSED")
            for r in sv["refusal_reasons"]:
                lines.append(f"                 - {r}")
            lines.append(f"                 (suppressed diagnostic: p={d['p_positive_reported']}, "
                         f"set={d['conformal_set']}, nn_T={d['nn_tanimoto']})")
        else:
            lines.append(f"{name:<16} {sv['answer']}  p={d['p_positive_reported']}  "
                         f"set={d['conformal_set']}  nn_T={d['nn_tanimoto']}")
            lines.append(f"                 means: {sv['answer_means']}")
            if d["is_training_molecule"]:
                lines.append("                 NOTE: nearest training molecule at Tanimoto "
                             "1.0 — this stage trained on this molecule. The number is "
                             "label recall, not prediction.")
        lines.append("")
    cs = res["cascade_summary"]
    lines.append(f"summary: {cs['n_answered']} answered / {cs['n_refused']} refused "
                 f"of {cs['n_stages']} stages at alpha={res['alpha']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m app.predict",
        description="Applicability-domain-aware urease cascade. Refuses outside its domain.")
    ap.add_argument("smiles", nargs="*", help="one or more SMILES strings")
    ap.add_argument("--alpha", type=float, default=0.10, choices=list(cascade.ALPHAS),
                    help="conformal miscoverage level (default 0.10)")
    ap.add_argument("--bundle", default=None, help="path to cascade_bundle.joblib")
    ap.add_argument("--presets", action="store_true", help="run the built-in preset molecules")
    ap.add_argument("--quiet", action="store_true", help="JSON only, no human summary on stderr")
    ap.add_argument("--no-neighbour-activity", action="store_true",
                    help="skip the uiref_curated.parquet activity lookup")
    args = ap.parse_args(argv)

    queries = list(args.smiles)
    if args.presets:
        queries += [p["smiles"] for p in cascade.PRESETS]
    if not queries:
        ap.error("give at least one SMILES, or --presets")

    out = [run_one(s, args.alpha, args.bundle,
                   with_neighbour_activity=not args.no_neighbour_activity)
           for s in queries]
    if not args.quiet:
        for res in out:
            print(human_summary(res), file=sys.stderr)
            print("-" * 78, file=sys.stderr)
    print(json.dumps(out[0] if len(out) == 1 else out, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
