"""Inference library for the corrected urease cascade — the single source of truth.

Everything the Streamlit UI and the CLI display comes from :func:`predict`.
Neither reimplements a featurization, a calibration, a conformal set, or a
domain test. If this module refuses, both refuse.

Design commitments
------------------
1. **Standardization is the frozen curation pipeline** — largest fragment,
   Normalizer, canonical tautomer, and *no* Reionizer — preceded by the
   phosphinic-acid repair ``[P+](=O)O -> [PH](=O)O``. Without the repair, 9
   known phosphinic/phosphonic urease inhibitors are silently rewritten into a
   +1 species matching nothing in the dataset — RDKit raises no error at all.

2. **The bundle's own featurization note specifies a *different*, weaker
   convention** (bare ``Chem.MolToSmiles(Chem.MolFromSmiles(smi))``, no
   tautomer canonicalization). For most inputs the two agree. When they do
   not, the molecule's identity — and therefore its domain verdict — is
   tautomer-convention-dependent, and *no* answer is defensible. Both
   conventions are always evaluated; see :data:`STANDARDIZATION_CONVENTIONS`,
   the top-level ``conventions_agree`` flag, and the
   ``diagnostics_by_convention`` map on every stage result.

3. **A prediction is presented as an answer only when it is usable under BOTH
   conventions.** This is strictly more conservative than the bundle's own
   refusal contract and never contradicts it.

4. **Stage 3 is not a model.** Its label is an exact SMARTS function of the
   structure, so the RDKit FilterCatalog is evaluated directly and the fitted
   model is never consulted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any

import numpy as np

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------
# Frozen constants — must match cascade_bundle.joblib. Never tune these.
# --------------------------------------------------------------------------

NBITS = 2048
MORGAN_RADIUS = 2
AD_TANIMOTO_THRESHOLD = 0.40          # preregistered, never tuned
ALPHAS = (0.05, 0.10, 0.20)
STAGE_ORDER = ("S1B_decoy", "S1A_measured", "S2_potency", "S3_toxicophore", "S4_clinical")
STANDARDIZATION_CONVENTIONS = ("frozen_curation", "bundle_featurization_note")

BUNDLE_SHA256_PREFIX = "90c177393e154f25"

DESCRIPTOR_NAMES = (
    "MolWt", "ExactMolWt", "NumHeavyAtoms", "NumAtoms", "LogP", "MolMR", "TPSA",
    "HBA", "HBD", "RotatableBonds", "RingCount", "AromaticRings", "AliphaticRings",
    "FractionCSP3", "BertzCT", "FormalCharge", "NumAmideBonds", "NumHeteroAtoms",
    "NumAromaticHeterocycles", "NumAliphaticHeterocycles", "NumAromaticCarbocycles",
    "NumAliphaticCarbocycles", "HeavyAtomMolWt", "NHOHCount", "NOCount",
)

# Stages the bundle marks usable_for_prediction=False are never shown as models.
NOT_A_MODEL = ("S3_toxicophore",)


def bundle_path() -> str:
    """Locate ``cascade_bundle.joblib``.

    Search order: ``$UREASE_CASCADE_BUNDLE``, then ``app/``, then the repo
    root, then the current working directory.
    """
    env = os.environ.get("UREASE_CASCADE_BUNDLE")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"UREASE_CASCADE_BUNDLE={env} does not exist")
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "cascade_bundle.joblib"),
        os.path.join(os.path.dirname(here), "cascade_bundle.joblib"),
        os.path.join(os.getcwd(), "cascade_bundle.joblib"),
    ):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "cascade_bundle.joblib not found. Put it in app/ or set "
        "UREASE_CASCADE_BUNDLE to its path."
    )


# --------------------------------------------------------------------------
# Standardization
# --------------------------------------------------------------------------

_lfc = rdMolStandardize.LargestFragmentChooser()
_norm = rdMolStandardize.Normalizer()
_taut = rdMolStandardize.TautomerEnumerator()
# NOTE: Reionizer is deliberately ABSENT. The curation pipeline that produced
# uiref_curated.parquet dropped it, because reionization undid the phosphinic
# acid repair below.

_PHOSPHINIC_REPAIRS = (("[P+](=O)O", "[PH](=O)O"),)


def repair_phosphinic(smi: str) -> tuple[str, bool]:
    """Apply the frozen phosphinic-acid repair.

    Nine urease inhibitors in the source records carry a spurious ``[P+](=O)O``
    from database export. The failure this repair prevents is **silent**: RDKit
    parses ``[P+](=O)O`` happily as tetravalent cationic phosphorus, and the
    Normalizer then rewrites it to ``[PH+](=O)=O`` — a distinct species with a
    net +1 charge and a different InChIKey that matches nothing in UI-Ref. No
    exception is raised; the molecule simply becomes unrecognizable.

    Returns the repaired SMILES and whether anything changed.
    """
    if not isinstance(smi, str):
        return smi, False
    out = smi
    for bad, good in _PHOSPHINIC_REPAIRS:
        out = out.replace(bad, good)
    return out, out != smi


@dataclass
class Standardized:
    """Result of one standardization convention."""

    convention: str
    ok: bool
    smiles: str | None = None
    inchikey: str | None = None
    mol: Any = field(default=None, repr=False)
    phosphinic_repair_applied: bool = False
    status: str = "ok"


def standardize(smi: str, convention: str = "frozen_curation") -> Standardized:
    """Standardize one SMILES under the named convention.

    ``frozen_curation``
        largest fragment -> Normalizer -> canonical tautomer (no Reionizer).
        This is the pipeline that defined dataset identity, so it is the only
        convention under which "is this molecule in UI-Ref?" is well posed.

    ``bundle_featurization_note``
        ``Chem.MolToSmiles(Chem.MolFromSmiles(smi))`` — the weaker convention
        the bundle's own ``featurization`` block prescribes, and the one
        ``example_compound_trace.csv`` was generated under.

    Both paths apply the phosphinic-acid repair first.
    """
    if convention not in STANDARDIZATION_CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}")
    if not isinstance(smi, str) or not smi.strip():
        return Standardized(convention, False, status="empty_input")

    repaired, did_repair = repair_phosphinic(smi.strip())
    try:
        mol = Chem.MolFromSmiles(repaired, sanitize=False)
        if mol is None:
            return Standardized(convention, False, status="parse_failed",
                                phosphinic_repair_applied=did_repair)
        Chem.SanitizeMol(mol)
        if convention == "frozen_curation":
            mol = _lfc.choose(mol)
            mol = _norm.normalize(mol)
            mol = _taut.Canonicalize(mol)
        cs = Chem.MolToSmiles(mol, canonical=True)
        remol = Chem.MolFromSmiles(cs)
        if remol is None:
            return Standardized(convention, False, status="invalid_after_standardization",
                                phosphinic_repair_applied=did_repair)
        return Standardized(convention, True, cs, Chem.MolToInchiKey(remol), remol,
                            did_repair, "ok")
    except Exception as exc:  # noqa: BLE001 - report, never crash the UI
        return Standardized(convention, False, status=f"error: {type(exc).__name__}",
                            phosphinic_repair_applied=did_repair)


# --------------------------------------------------------------------------
# Featurization
# --------------------------------------------------------------------------

_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=NBITS)


def _alert_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    for cat in (FilterCatalogParams.FilterCatalogs.PAINS,
                FilterCatalogParams.FilterCatalogs.BRENK,
                FilterCatalogParams.FilterCatalogs.NIH):
        params.AddCatalog(cat)
    return FilterCatalog(params)


@lru_cache(maxsize=1)
def alert_catalog() -> FilterCatalog:
    return _alert_catalog()


def fingerprint(mol) -> tuple[Any, np.ndarray]:
    """ECFP4 as (RDKit bit vector, uint8 numpy row)."""
    fp = _fpgen.GetFingerprint(mol)
    arr = np.zeros((NBITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return fp, arr


def descriptors(mol) -> np.ndarray:
    """The 25 physicochemical descriptors, in frozen bundle order."""
    return np.array([
        Descriptors.MolWt(mol), Descriptors.ExactMolWt(mol), mol.GetNumHeavyAtoms(),
        mol.GetNumAtoms(), Crippen.MolLogP(mol), Crippen.MolMR(mol), rdmd.CalcTPSA(mol),
        rdmd.CalcNumHBA(mol), rdmd.CalcNumHBD(mol), rdmd.CalcNumRotatableBonds(mol),
        rdmd.CalcNumRings(mol), rdmd.CalcNumAromaticRings(mol), rdmd.CalcNumAliphaticRings(mol),
        rdmd.CalcFractionCSP3(mol), Descriptors.BertzCT(mol), Chem.GetFormalCharge(mol),
        rdmd.CalcNumAmideBonds(mol), rdmd.CalcNumHeteroatoms(mol),
        rdmd.CalcNumAromaticHeterocycles(mol), rdmd.CalcNumAliphaticHeterocycles(mol),
        rdmd.CalcNumAromaticCarbocycles(mol), rdmd.CalcNumAliphaticCarbocycles(mol),
        Descriptors.HeavyAtomMolWt(mol), Descriptors.NHOHCount(mol), Descriptors.NOCount(mol),
    ], dtype=float)


def structural_alerts(mol) -> list[str]:
    """Exact PAINS/BRENK/NIH SMARTS matches — the deterministic Stage 3 rule."""
    return [m.GetDescription() for m in alert_catalog().GetMatches(mol)]


# --------------------------------------------------------------------------
# Bundle loading
# --------------------------------------------------------------------------

_CACHE: dict[str, Any] = {}


def load_bundle(path: str | None = None) -> dict[str, Any]:
    """Load and index the frozen bundle. Cached per path."""
    import joblib

    p = path or bundle_path()
    key = os.path.abspath(p)
    if key in _CACHE:
        return _CACHE[key]

    bundle = joblib.load(p)
    ref = {}
    for name, stage in bundle["stages"].items():
        n_rows, n_bits = stage["reference_fingerprint_shape"]
        bits = np.unpackbits(stage["reference_fingerprints"], axis=1)[:, :n_bits]
        ref[name] = {
            "bits": bits.astype(np.uint8),
            "popcount": bits.sum(axis=1).astype(np.int64),
            "labels": np.asarray(stage["reference_labels"]),
            "smiles": list(stage["reference_smiles"]),
            "inchikeys": list(stage["reference_inchikeys"]),
            "mahal_mean": np.asarray(stage["ad_mahalanobis"]["mean"], dtype=float).ravel(),
            "mahal_inv_cov": np.asarray(stage["ad_mahalanobis"]["inv_cov"], dtype=float),
            "mahal_cutoff": float(stage["ad_mahalanobis"]["cutoff"]),
        }
        assert bits.shape == (n_rows, n_bits), f"{name}: reference fingerprint shape mismatch"
    _CACHE[key] = {"bundle": bundle, "ref": ref, "path": key}
    return _CACHE[key]


# --------------------------------------------------------------------------
# Calibration / conformal / applicability domain
# --------------------------------------------------------------------------

def _platt(calibrator, p_raw: float) -> float:
    """Apply the frozen Platt calibrator on the logit of the RF score."""
    clipped = float(np.clip(p_raw, 1e-6, 1 - 1e-6))
    z = np.log(clipped / (1.0 - clipped))
    return float(calibrator.predict_proba(np.array([[z]]))[0, 1])


def tanimoto_to_references(query_bits: np.ndarray, ref: dict) -> np.ndarray:
    """Exact integer Tanimoto of one query against every reference fingerprint.

    Integer arithmetic on the unpacked bit matrix, so this is bit-for-bit
    identical to ``DataStructs.BulkTanimotoSimilarity`` (verified in
    ``test_app.py::test_numpy_tanimoto_matches_rdkit``) and much faster.
    """
    q = query_bits.astype(np.int64)
    inter = ref["bits"].astype(np.int64) @ q
    union = ref["popcount"] + int(q.sum()) - inter
    out = np.zeros(inter.shape, dtype=float)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def mahalanobis(desc: np.ndarray, ref: dict) -> float:
    d = desc.ravel() - ref["mahal_mean"]
    return float(np.sqrt(max(float(d @ ref["mahal_inv_cov"] @ d), 0.0)))


def conformal_set(p_platt: float, quantiles: dict) -> tuple[bool, bool, int, str]:
    """Label-conditional (Mondrian) conformal set from the frozen quantiles.

    Nonconformity is ``1 - p(class)`` on Platt-calibrated scores, matching
    ``bundle['conformal']['nonconformity']``.
    """
    in_pos = (1.0 - p_platt) <= quantiles[1]
    in_neg = (1.0 - (1.0 - p_platt)) <= quantiles[0]
    size = int(in_pos) + int(in_neg)
    if size == 2:
        label = "{both classes} — UNINFORMATIVE"
    elif size == 0:
        label = "{} — EMPTY: atypical even for its own predicted class"
    else:
        label = "{positive}" if in_pos else "{negative}"
    return in_neg, in_pos, size, label


# --------------------------------------------------------------------------
# Per-stage prediction
# --------------------------------------------------------------------------

def _stage_once(name: str, std: Standardized, alpha: float, loaded: dict) -> dict[str, Any]:
    """Evaluate one stage for one standardization convention."""
    bundle, ref = loaded["bundle"], loaded["ref"][name]
    stage = bundle["stages"][name]

    fp, bits = fingerprint(std.mol)
    desc = descriptors(std.mol)
    X = np.hstack([bits.reshape(1, -1), desc.reshape(1, -1)])
    assert X.shape[1] == len(stage["feature_names"]), (
        f"{name}: built {X.shape[1]} features, bundle expects {len(stage['feature_names'])}"
    )

    p_raw = float(stage["pipeline"].predict_proba(X)[0, 1])
    p_platt = _platt(stage["calibrator_platt"], p_raw)
    recommended = stage["calibrator_recommended"]
    if recommended == "platt":
        p_reported = p_platt
    elif recommended == "isotonic":
        p_reported = float(stage["calibrator_isotonic"].predict([p_raw])[0])
    else:
        p_reported = p_raw

    # Conformal scores always use the Platt-calibrated probability, because the
    # frozen quantiles were fitted on Platt-calibrated calibration scores.
    in_neg, in_pos, set_size, set_label = conformal_set(
        p_platt, stage["conformal_quantiles"][alpha])

    sims = tanimoto_to_references(bits, ref)
    nn_tanimoto = float(sims.max())
    mahal = mahalanobis(desc, ref)
    ad_tanimoto_pass = nn_tanimoto >= float(stage["ad_tanimoto_threshold"])
    ad_mahalanobis_pass = mahal <= ref["mahal_cutoff"]
    in_domain = bool(ad_tanimoto_pass and ad_mahalanobis_pass)

    # Stable sort so Tanimoto ties break on ascending reference index. Ties are
    # common (near-duplicate training molecules), and an unstable sort would make
    # the displayed neighbour depend on NumPy's internals rather than the data.
    order = np.argsort(-sims, kind="stable")[:3]
    n_tied_at_max = int((sims >= sims.max() - 1e-12).sum())
    neighbours = [
        {"rank": int(i + 1), "tanimoto": round(float(sims[j]), 4),
         "smiles": ref["smiles"][j], "inchikey": ref["inchikeys"][j],
         "training_label": int(ref["labels"][j]),
         "reference_index": int(j)}
        for i, j in enumerate(order)
    ]

    return {
        "stage": name,
        "convention": std.convention,
        "standardized_smiles": std.smiles,
        "inchikey": std.inchikey,
        "alpha": alpha,
        "p_positive_reported": round(p_reported, 4),
        "p_positive_raw_rf": round(p_raw, 4),
        "p_positive_platt": round(p_platt, 4),
        "calibrator_used": recommended,
        "nn_tanimoto": round(nn_tanimoto, 4),
        "ad_tanimoto_threshold": float(stage["ad_tanimoto_threshold"]),
        "ad_tanimoto_pass": bool(ad_tanimoto_pass),
        "mahalanobis": round(mahal, 4),
        "mahalanobis_cutoff": round(ref["mahal_cutoff"], 4),
        "ad_mahalanobis_pass": bool(ad_mahalanobis_pass),
        "in_applicability_domain": in_domain,
        "conformal_set": set_label,
        "conformal_set_size": set_size,
        "conformal_includes_positive": bool(in_pos),
        "conformal_includes_negative": bool(in_neg),
        "usable_prediction": bool(in_domain and set_size == 1),
        "is_training_molecule": bool(nn_tanimoto >= 0.9999),
        "n_references_tied_at_max_tanimoto": n_tied_at_max,
        "nearest_neighbours": neighbours,
    }


def _refusal_reasons(views: dict[str, dict], stage_name: str, stage: dict) -> list[str]:
    """Why this stage may not be reported as an answer. Empty list == usable."""
    reasons: list[str] = []
    if stage_name in NOT_A_MODEL or not stage.get("usable_for_prediction", True):
        reasons.append(
            "This stage is NOT a model. Its label is an exact SMARTS function of the "
            "structure; the alert list below IS the answer. The fitted model exists "
            "only for audit reproducibility and is never consulted."
        )
        return reasons

    ordered = [views[c] for c in STANDARDIZATION_CONVENTIONS if c in views]
    for v in ordered:
        tag = v["convention"]
        if not v["ad_tanimoto_pass"]:
            reasons.append(
                f"OUTSIDE the applicability domain under the {tag} convention: nearest "
                f"training molecule is only Tanimoto {v['nn_tanimoto']:.4f}, below the "
                f"preregistered {v['ad_tanimoto_threshold']:.2f} floor."
            )
        if not v["ad_mahalanobis_pass"]:
            reasons.append(
                f"OUTSIDE the applicability domain under the {tag} convention: "
                f"Mahalanobis distance {v['mahalanobis']:.4f} exceeds the cutoff "
                f"{v['mahalanobis_cutoff']:.4f} in 25-descriptor space."
            )
        if v["in_applicability_domain"] and v["conformal_set_size"] == 2:
            reasons.append(
                f"Conformal set at alpha={v['alpha']:.2f} is {v['conformal_set']} under the "
                f"{tag} convention — the predictor cannot separate the classes for this "
                "molecule at the requested confidence."
            )
        if v["in_applicability_domain"] and v["conformal_set_size"] == 0:
            reasons.append(
                f"Conformal set at alpha={v['alpha']:.2f} is EMPTY under the {tag} "
                "convention — the molecule is atypical even for its own predicted class."
            )

    if len(ordered) == 2:
        a, c = ordered
        if a["inchikey"] != c["inchikey"]:
            differing = [
                k for k in ("in_applicability_domain", "conformal_set_size")
                if a[k] != c[k]
            ]
            if differing:
                reasons.append(
                    "The two standardization conventions disagree about this molecule "
                    f"({a['convention']} -> {a['standardized_smiles']}; "
                    f"{c['convention']} -> {c['standardized_smiles']}), and the "
                    f"disagreement changes {', '.join(differing)}. A verdict that flips "
                    "with tautomer convention is not a verdict."
                )
    return reasons


def stage_semantics(stage: dict) -> dict[str, str]:
    ls = stage["label_semantics"]
    return {
        "positive_class": ls["positive_class"],
        "negative_class": ls["negative_class"],
        "negative_class_means": ls["negative_class_MEANS"],
        "honest_verdict": ls["honest_verdict"],
        "design": ls["design"],
        "grouping": ls["grouping"],
        "n_positive": ls["n_positive"],
        "n_negative": ls["n_negative"],
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def predict(smiles: str, alpha: float = 0.10, bundle_file: str | None = None,
            stages: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Run the corrected cascade on one SMILES.

    Returns a fully self-describing dict: per-stage probabilities, conformal
    sets, domain verdicts, nearest training neighbours, label semantics and —
    for every stage — either ``answer`` or a non-empty ``refusal_reasons``.

    A stage's probability is *never* the answer when ``refused`` is True. The
    caller may show it as subordinate diagnostic detail; it must not be
    rendered as a verdict.
    """
    if alpha not in ALPHAS:
        raise ValueError(f"alpha must be one of {ALPHAS}, got {alpha}")
    loaded = load_bundle(bundle_file)
    bundle = loaded["bundle"]
    names = tuple(stages) if stages else STAGE_ORDER

    std = {}
    for conv in STANDARDIZATION_CONVENTIONS:
        std[conv] = standardize(smiles, conv)

    primary = std["frozen_curation"]
    if not primary.ok:
        return {
            "input_smiles": smiles,
            "parse_ok": False,
            "error": primary.status,
            "phosphinic_repair_applied": primary.phosphinic_repair_applied,
            "standardization": {c: asdict(s) | {"mol": None} for c, s in std.items()},
            "stages": {},
        }

    alerts = structural_alerts(primary.mol)
    result: dict[str, Any] = {
        "input_smiles": smiles,
        "parse_ok": True,
        "alpha": alpha,
        "bundle_path": loaded["path"],
        "bundle_created_utc": bundle["created_utc"],
        "bundle_schema_version": bundle["schema_version"],
        "seed": bundle["seed"],
        "phosphinic_repair_applied": primary.phosphinic_repair_applied,
        "standardization": {
            c: {"convention": c, "ok": s.ok, "smiles": s.smiles, "inchikey": s.inchikey,
                "status": s.status, "phosphinic_repair_applied": s.phosphinic_repair_applied}
            for c, s in std.items()
        },
        "conventions_agree": (std["frozen_curation"].inchikey
                              == std["bundle_featurization_note"].inchikey),
        "structural_alerts_exact_rule": {
            "n_alerts": len(alerts),
            "alerts": alerts,
            "rule": "PAINS + BRENK + NIH SMARTS via rdkit.Chem.FilterCatalog",
            "caveat": ("Zero alerts is NOT 'non-toxic'. No toxicity assay of any kind "
                       "enters this label."),
        },
        "stages": {},
    }

    for name in names:
        stage = bundle["stages"][name]
        views = {}
        for conv, s in std.items():
            if s.ok:
                views[conv] = _stage_once(name, s, alpha, loaded)
        reasons = _refusal_reasons(views, name, stage)
        primary_view = views["frozen_curation"]
        refused = bool(reasons)
        result["stages"][name] = {
            "stage": name,
            "usable_for_prediction_per_bundle": bool(stage.get("usable_for_prediction", True)),
            "refused": refused,
            "refusal_reasons": reasons,
            "answer": None if refused else (
                "POSITIVE" if primary_view["conformal_includes_positive"] else "NEGATIVE"),
            "answer_means": None if refused else (
                stage["label_semantics"]["positive_class"]
                if primary_view["conformal_includes_positive"]
                else stage["label_semantics"]["negative_class"]),
            "semantics": stage_semantics(stage),
            "diagnostics": primary_view,
            "diagnostics_by_convention": views,
            "validated_performance": stage["validated_performance"],
        }

    result["cascade_summary"] = {
        "n_stages": len(names),
        "n_refused": sum(1 for v in result["stages"].values() if v["refused"]),
        "n_answered": sum(1 for v in result["stages"].values() if not v["refused"]),
        "any_stage_out_of_domain": any(
            not v["diagnostics"]["in_applicability_domain"] for v in result["stages"].values()),
    }
    result["limitations"] = limitations(bundle)
    return result


def limitations(bundle: dict | None = None) -> dict[str, Any]:
    """The honest-limitations panel, drawn from bundle metadata — not prose."""
    b = bundle if bundle is not None else load_bundle()["bundle"]
    ad = b["applicability_domain"]
    cov = ad["collection_coverage"]
    return {
        "refusal_policy": ad["refusal_policy"],
        "domain_rule": ad["rule"],
        "domain_excludes_screening_libraries": {
            k: round(float(v["frac_outside_either"]) * 100, 1) for k, v in cov.items()
        },
        "library_membership_warning": b["library_membership_warning"],
        "provenance_noise_floor": b["provenance"]["noise_floor"],
        "what_ui_ref_membership_means": b["provenance"]["ui_ref_nature"],
        "genuine_negatives": b["provenance"]["genuine_negatives"],
        "conformal_caveat": b["conformal"]["note"],
        "stage_3_is_not_a_model": b["dropped_stages"]["S3_toxicophore"],
        "environment": b["environment"],
    }


def training_record(inchikey: str, table_path: str | None = None) -> dict[str, Any] | None:
    """Look up a molecule's measured activity in ``uiref_curated.parquet``, if present."""
    import pandas as pd

    if table_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        for cand in (os.path.join(here, "uiref_curated.parquet"),
                     os.path.join(root, "uiref_curated.parquet"),
                     os.path.join(root, "data", "processed", "uiref_curated.parquet"),
                     os.path.join(root, "data", "derived", "uiref_curated.parquet")):
            if os.path.exists(cand):
                table_path = cand
                break
    if table_path is None or not os.path.exists(table_path):
        return None
    df = pd.read_parquet(table_path)
    hit = df[df.inchikey == inchikey]
    if hit.empty:
        return None
    r = hit.iloc[0]
    def _num(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
    return {
        "row_id": str(r.row_id),
        "smiles_std": str(r.smiles_std),
        "pAct_median": _num(r.pAct_median),
        "pAct_min": _num(r.pAct_min),
        "pAct_max": _num(r.pAct_max),
        "n_records": int(r.n_records),
        "n_direct_measurements": int(r.n_direct),
        "act_types": None if r.act_types is None else str(r.act_types),
        "source_dbs": str(r.source_dbs),
        "primary_document": None if r.primary_document is None else str(r.primary_document),
        "primary_organism": None if r.primary_organism is None else str(r.primary_organism),
        "any_active": bool(r.any_active),
        "any_inactive": bool(r.any_inactive),
    }


def annotate_neighbours(neighbours: list[dict], table_path: str | None = None) -> list[dict]:
    """Attach measured activity to nearest-neighbour records where available."""
    out = []
    for nb in neighbours:
        rec = training_record(nb["inchikey"], table_path)
        out.append(nb | {"measured": rec})
    return out


PRESETS = [
    {
        "label": "Worked standardization example",
        "smiles": "CN(C)c1ccc(/C=N/N=C2\\NC(=O)CS2)cc1",
        "note": ("This reported urease-active molecule receives different identities under "
                 "the two standardization conventions. The application evaluates both and "
                 "refuses unsupported predictions."),
    },
    {
        "label": "UIREF-0126 — potent inhibitor, pAct 9.12 (held out of Stage 2)",
        "smiles": "O=c1c2ccccc2[se]n1-c1ccc(C(F)(F)F)cc1",
        "note": ("Benzisoselenazolone from 10.1021/acs.jmedchem.2c01799, measured "
                 "pAct_median 9.12 against Proteus mirabilis urease. Not in the Stage 2 "
                 "training references, so Stage 2 is a genuine held-out prediction."),
    },
    {
        "label": "Sucrose — obviously foreign",
        "smiles": "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
        "note": "A disaccharide. Nothing in UI-Ref resembles it; the domain must reject it.",
    },
    {
        "label": "Phosphinic acid needing the [P+](=O)O repair",
        "smiles": "COC(=O)/C(=C/C1=CC=C(C=C1)F)/C[P+](=O)O",
        "note": ("One of 9 UI-Ref inhibitors exported with a spurious [P+](=O)O. RDKit "
                 "parses it without complaint, then the Normalizer turns it into "
                 "[PH+](=O)=O — a +1 species with a different InChIKey that matches "
                 "nothing in UI-Ref. The repair is what keeps these 9 molecules "
                 "identifiable; without it they vanish silently."),
    },
    {
        "label": "Polyethylene glycol (decamer) — polymer, out of domain",
        "smiles": "OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO",
        "note": "A flexible polymer, chemically unlike anything in a urease inhibitor set.",
    },
]
