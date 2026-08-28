"""Smoke tests for the applicability-domain-aware urease cascade.

    pytest app/test_app.py -v

Covers the six required behaviours plus the invariants that keep the app from
silently drifting away from the frozen bundle:

  (i)   a known UI-Ref potent inhibitor is in-domain and positive at Stage 2
  (ii)  the prototype compound is out-of-domain / uninformative -> REFUSAL
  (iii) sucrose is out-of-domain
  (iv)  the phosphinic-acid repair path works
  (v)   reloading the bundle reproduces the stored probabilities exactly
  (vi)  the CLI and the library agree
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem, DataStructs

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import cascade  # noqa: E402
from app import predict as predict_cli  # noqa: E402

PROTOTYPE = "CN(C)c1ccc(/C=N/N=C2\\NC(=O)CS2)cc1"
POTENT = "O=c1c2ccccc2[se]n1-c1ccc(C(F)(F)F)cc1"          # UIREF-0126, pAct 9.12
SUCROSE = "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"
PHOSPHINIC_BROKEN = "COC(=O)/C(=C/C1=CC=C(C=C1)F)/C[P+](=O)O"
PHOSPHINIC_EXPECTED_STD = "COC(=O)/C(=C/c1ccc(F)cc1)C[PH](=O)O"
PEG = "OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO"


@pytest.fixture(scope="module")
def loaded():
    return cascade.load_bundle()


# --------------------------------------------------------------------- (i)
def test_known_potent_inhibitor_in_domain_and_positive_at_stage2():
    """UIREF-0126 (measured pAct 9.12) is in-domain and predicted potent.

    It is deliberately absent from Stage 2's training references, so this is a
    genuine held-out prediction rather than label recall.
    """
    res = cascade.predict(POTENT, alpha=0.10)
    assert res["parse_ok"]
    s2 = res["stages"]["S2_potency"]
    d = s2["diagnostics"]

    assert d["in_applicability_domain"] is True, d
    assert d["ad_tanimoto_pass"] and d["ad_mahalanobis_pass"]
    assert d["nn_tanimoto"] >= cascade.AD_TANIMOTO_THRESHOLD
    assert d["conformal_set_size"] == 1, d["conformal_set"]
    assert d["conformal_includes_positive"] is True
    assert s2["refused"] is False, s2["refusal_reasons"]
    assert s2["answer"] == "POSITIVE"
    assert d["p_positive_reported"] > 0.5
    # not a training molecule of this stage
    assert d["is_training_molecule"] is False, "should be held out of Stage 2 references"

    rec = cascade.training_record(res["standardization"]["frozen_curation"]["inchikey"])
    assert rec is not None and rec["pAct_median"] >= 6.0


# -------------------------------------------------------------------- (ii)
def test_prototype_compound_is_refused():
    """The prototype's 'NON-INHIBITOR 0.462/0.538' claim must not be reproducible."""
    res = cascade.predict(PROTOTYPE, alpha=0.10)
    assert res["parse_ok"]

    # Every stage refuses; none offers an answer.
    for name, sv in res["stages"].items():
        assert sv["refused"] is True, f"{name} should refuse: {sv}"
        assert sv["answer"] is None
        assert sv["refusal_reasons"], f"{name} refused with no stated reason"
    assert res["cascade_summary"]["n_answered"] == 0
    assert res["cascade_summary"]["n_refused"] == len(cascade.STAGE_ORDER)

    # The bundle's own convention places it outside the domain at T = 0.34.
    bn = res["stages"]["S1B_decoy"]["diagnostics_by_convention"]["bundle_featurization_note"]
    assert bn["nn_tanimoto"] == pytest.approx(0.3400, abs=1e-4)
    assert bn["in_applicability_domain"] is False
    assert bn["conformal_set_size"] == 2  # {both classes}: UNINFORMATIVE

    # And the two standardization conventions disagree about its identity.
    assert res["conventions_agree"] is False
    joined = " ".join(res["stages"]["S1B_decoy"]["refusal_reasons"]).lower()
    assert "convention" in joined


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_prototype_conformal_set_uninformative_under_bundle_convention(alpha):
    """At alpha 0.05 and 0.10 the bundle-convention set is {both classes}."""
    res = cascade.predict(PROTOTYPE, alpha=alpha)
    for name in ("S1B_decoy", "S1A_measured", "S4_clinical"):
        v = res["stages"][name]["diagnostics_by_convention"]["bundle_featurization_note"]
        assert v["conformal_set_size"] == 2, (name, alpha, v["conformal_set"])
        assert v["usable_prediction"] is False


# ------------------------------------------------------------------- (iii)
@pytest.mark.parametrize("smi,label", [(SUCROSE, "sucrose"), (PEG, "PEG decamer")])
def test_foreign_molecule_out_of_domain(smi, label):
    """A sugar and a polymer must be rejected by the domain on the real stages."""
    res = cascade.predict(smi, alpha=0.10)
    assert res["parse_ok"]
    for name in ("S1B_decoy", "S1A_measured", "S2_potency"):
        sv = res["stages"][name]
        d = sv["diagnostics"]
        assert d["in_applicability_domain"] is False, f"{label}/{name}: {d}"
        assert sv["refused"] is True
        assert sv["answer"] is None
        assert any("OUTSIDE the applicability domain" in r for r in sv["refusal_reasons"])


def test_sucrose_is_a_clue_library_member_but_still_not_biology():
    """Sucrose IS in the CLUE library, so Stage 4 is in-domain for it.

    This is exactly the trap the honest-limitations panel exists for: an
    in-domain Stage 4 answer is a library-membership statement, never biology.
    """
    res = cascade.predict(SUCROSE, alpha=0.10)
    s4 = res["stages"]["S4_clinical"]
    assert s4["diagnostics"]["in_applicability_domain"] is True
    assert s4["diagnostics"]["is_training_molecule"] is True
    warn = res["limitations"]["library_membership_warning"]["S4_clinical"]
    assert "LIBRARY MEMBERSHIP" in warn and "not approval prediction" in warn
    # meanwhile the biology stages still refuse
    assert res["stages"]["S2_potency"]["refused"] is True


# -------------------------------------------------------------------- (iv)
def test_phosphinic_acid_repair_path():
    """[P+](=O)O -> [PH](=O)O must fire.

    The failure mode without the repair is NOT a parse error — RDKit accepts
    ``[P+](=O)O`` as a tetravalent cationic phosphorus. The Normalizer then
    rewrites it to ``[PH+](=O)=O``, a different species with a net +1 charge,
    which yields a different InChIKey and matches nothing in UI-Ref. The
    molecule does not error; it silently becomes a stranger to its own dataset.
    """
    raw = Chem.MolFromSmiles(PHOSPHINIC_BROKEN)
    assert raw is not None, "RDKit does parse [P+](=O)O — the corruption is silent"
    assert Chem.GetFormalCharge(raw) == 1, "unrepaired form carries a spurious +1"
    assert cascade.training_record(Chem.MolToInchiKey(raw)) is None, (
        "the unrepaired InChIKey must NOT be found in UI-Ref")

    repaired, changed = cascade.repair_phosphinic(PHOSPHINIC_BROKEN)
    assert changed is True
    assert "[PH](=O)O" in repaired and "[P+](=O)O" not in repaired

    std = cascade.standardize(PHOSPHINIC_BROKEN, "frozen_curation")
    assert std.ok is True, std.status
    assert std.phosphinic_repair_applied is True
    assert std.smiles == PHOSPHINIC_EXPECTED_STD, std.smiles
    assert Chem.GetFormalCharge(std.mol) == 0, "repaired form must be neutral"

    # Without the repair, the frozen pipeline produces [PH+](=O)=O and loses it.
    unrep = Chem.MolFromSmiles(PHOSPHINIC_BROKEN, sanitize=False)
    Chem.SanitizeMol(unrep)
    unrep = cascade._taut.Canonicalize(
        cascade._norm.normalize(cascade._lfc.choose(unrep)))
    assert "[PH+](=O)=O" in Chem.MolToSmiles(unrep)
    assert cascade.training_record(Chem.MolToInchiKey(unrep)) is None

    res = cascade.predict(PHOSPHINIC_BROKEN, alpha=0.10)
    assert res["parse_ok"] is True
    assert res["phosphinic_repair_applied"] is True
    # It is a real UI-Ref molecule, so it must be found in the curated table.
    rec = cascade.training_record(std.inchikey)
    assert rec is not None, "repaired phosphinic acid should exist in uiref_curated"
    # and it must be in-domain for the UI-Ref-membership stage
    assert res["stages"]["S1B_decoy"]["diagnostics"]["in_applicability_domain"] is True


def test_repair_is_a_noop_on_clean_input():
    out, changed = cascade.repair_phosphinic("CCO")
    assert out == "CCO" and changed is False


# --------------------------------------------------------------------- (v)
def test_reload_reproduces_stored_probabilities_exactly(loaded):
    """A fresh bundle load must give bit-identical probabilities.

    Also pins the values recorded in ``example_compound_trace.csv`` for the
    prototype under the bundle's own featurization convention.
    """
    stored = {          # from example_compound_trace.csv, alpha=0.10
        "S1B_decoy":    {"p": 0.6253, "p_raw": 0.3567, "nn": 0.3400, "mahal": 3.7525},
        "S1A_measured": {"p": 0.9300, "p_raw": 0.9300, "nn": 0.3400, "mahal": 3.3773},
        "S2_potency":   {"p": 0.0933, "p_raw": 0.0933, "nn": 0.3673, "mahal": 4.1132},
        "S4_clinical":  {"p": 0.5458, "p_raw": 0.5133, "nn": 0.2222, "mahal": 3.8223},
    }
    res = cascade.predict(PROTOTYPE, alpha=0.10)
    for name, exp in stored.items():
        v = res["stages"][name]["diagnostics_by_convention"]["bundle_featurization_note"]
        assert v["p_positive_reported"] == pytest.approx(exp["p"], abs=1e-4), name
        assert v["p_positive_raw_rf"] == pytest.approx(exp["p_raw"], abs=1e-4), name
        assert v["nn_tanimoto"] == pytest.approx(exp["nn"], abs=1e-4), name
        assert v["mahalanobis"] == pytest.approx(exp["mahal"], abs=1e-4), name

    # Force a genuine re-read from disk and re-check.
    cascade._CACHE.clear()
    again = cascade.predict(PROTOTYPE, alpha=0.10)
    for name in stored:
        a = res["stages"][name]["diagnostics_by_convention"]["bundle_featurization_note"]
        b = again["stages"][name]["diagnostics_by_convention"]["bundle_featurization_note"]
        for k in ("p_positive_reported", "p_positive_raw_rf", "p_positive_platt",
                  "nn_tanimoto", "mahalanobis", "conformal_set_size",
                  "in_applicability_domain"):
            assert a[k] == b[k], f"{name}.{k} changed across reload: {a[k]} vs {b[k]}"


def test_stored_trace_csv_matches_shipped_table():
    """The app must agree with the published worked example, row for row.

    Reads results/tables/tableS29_example_compound_trace.csv — the corrected trace
    behind main Figure 6e (see finding F11-selfaudit in
    results/corrections_registry.csv). Every value the paper prints for this compound
    is re-derived here from the bundle; if the app and the table ever diverge, this
    test is the one that fails.
    """
    import pandas as pd

    path = None
    for cand in (os.path.join(_ROOT, "results", "tables",
                              "tableS29_example_compound_trace.csv"),
                 os.path.join(_HERE, "tableS29_example_compound_trace.csv")):
        if os.path.exists(cand):
            path = cand
            break
    assert path is not None, "tableS29_example_compound_trace.csv must ship with the repository"

    df = pd.read_csv(path)
    assert len(df) == 5, f"expected 5 cascade stages, got {len(df)}"

    # The table is the alpha = 0.10 trace under the frozen curation convention.
    res = cascade.predict(PROTOTYPE, alpha=0.10)
    assert res["parse_ok"]

    # Structural identity: the compound IS a UI-Ref member under the frozen pipeline.
    std = res["standardization"]["frozen_curation"]
    assert std["smiles"] == df.smiles_frozen_curation.iloc[0]
    assert std["inchikey"] == df.inchikey_frozen_curation.iloc[0]
    assert res["conventions_agree"] is bool(df.conventions_agree.iloc[0]) is False
    rec = cascade.training_record(std["inchikey"])
    assert rec is not None, "UIREF-1160 must be findable in uiref_curated"
    assert rec["row_id"] == df.UIRef_row_id.iloc[0]
    assert rec["pAct_median"] == pytest.approx(float(df.measured_pAct.iloc[0]), abs=1e-3)

    checked = 0
    for _, row in df.iterrows():
        st = res["stages"][row.stage]
        d = st["diagnostics"]
        assert d["p_positive_reported"] == pytest.approx(row.p_calibrated, abs=1e-4), row.stage
        assert d["nn_tanimoto"] == pytest.approx(row.nn_tanimoto_frozen, abs=1e-4), row.stage
        assert d["in_applicability_domain"] == bool(row.in_AD_frozen), row.stage
        assert st["refused"] == bool(row.refused), row.stage
        checked += 1

    assert checked == len(df)
    # The published claim: refusal at every stage, for a documented reason.
    assert bool(df.refused.all()), "the published trace refuses at all five stages"
    assert res["cascade_summary"]["n_answered"] == 0


# -------------------------------------------------------------------- (vi)
@pytest.mark.parametrize("smi", [PROTOTYPE, POTENT, SUCROSE])
def test_cli_agrees_with_library(smi, tmp_path):
    """`python -m app.predict` must emit exactly what the library computes."""
    env = dict(os.environ, PYTHONPATH=_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "app.predict", smi, "--alpha", "0.10", "--quiet"],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    cli = json.loads(proc.stdout)

    lib = cascade.predict(smi, alpha=0.10)
    assert cli["standardization"]["frozen_curation"]["inchikey"] == \
        lib["standardization"]["frozen_curation"]["inchikey"]
    assert cli["cascade_summary"] == lib["cascade_summary"]
    for name in cascade.STAGE_ORDER:
        c, l = cli["stages"][name], lib["stages"][name]
        assert c["refused"] == l["refused"], name
        assert c["answer"] == l["answer"], name
        assert c["refusal_reasons"] == l["refusal_reasons"], name
        for k in ("p_positive_reported", "p_positive_raw_rf", "p_positive_platt",
                  "nn_tanimoto", "mahalanobis", "conformal_set",
                  "conformal_set_size", "in_applicability_domain", "usable_prediction"):
            assert c["diagnostics"][k] == l["diagnostics"][k], f"{name}.{k}"


def test_cli_human_summary_never_leads_with_a_refused_probability():
    """The text renderer must not present a suppressed number as the verdict."""
    res = cascade.predict(PROTOTYPE, alpha=0.10)
    text = predict_cli.human_summary(res)
    for line in text.splitlines():
        if "REFUSED" in line:
            assert "p=" not in line, line
    assert "suppressed diagnostic" in text
    assert "0 answered / 5 refused" in text


# ------------------------------------------------------- invariants
def test_numpy_tanimoto_matches_rdkit(loaded):
    """The fast integer Tanimoto must equal RDKit's, or the domain test is wrong."""
    ref = loaded["ref"]["S2_potency"]
    mol = cascade.standardize(POTENT, "frozen_curation").mol
    _, bits = cascade.fingerprint(mol)
    mine = cascade.tanimoto_to_references(bits, ref)

    fp = cascade._fpgen.GetFingerprint(mol)
    reffps = [DataStructs.CreateFromBitString("".join(map(str, r))) for r in ref["bits"]]
    theirs = np.array(DataStructs.BulkTanimotoSimilarity(fp, reffps))
    assert np.allclose(mine, theirs, atol=1e-12), np.abs(mine - theirs).max()


def test_feature_order_is_fingerprint_then_descriptors(loaded):
    names = loaded["bundle"]["stages"]["S2_potency"]["feature_names"]
    assert len(names) == cascade.NBITS + len(cascade.DESCRIPTOR_NAMES) == 2073
    assert tuple(names[cascade.NBITS:]) == cascade.DESCRIPTOR_NAMES


def test_stage3_is_never_used_as_a_model():
    res = cascade.predict(POTENT, alpha=0.20)
    s3 = res["stages"]["S3_toxicophore"]
    assert s3["usable_for_prediction_per_bundle"] is False
    assert s3["refused"] is True
    assert s3["answer"] is None
    assert any("NOT a model" in r for r in s3["refusal_reasons"])
    # the exact rule still answers
    assert isinstance(res["structural_alerts_exact_rule"]["n_alerts"], int)


def test_no_reionizer_in_standardization():
    """Reionization would undo the phosphinic repair; it must be absent."""
    src = open(os.path.join(_HERE, "cascade.py")).read()
    assert "Reionizer()" not in src.replace("# NOTE: Reionizer is deliberately ABSENT.", "")


def test_alpha_must_be_one_of_the_frozen_levels():
    with pytest.raises(ValueError):
        cascade.predict(POTENT, alpha=0.15)


def test_invalid_smiles_is_rejected_not_guessed():
    res = cascade.predict("not_a_molecule((", alpha=0.10)
    assert res["parse_ok"] is False
    assert res["stages"] == {}
    assert "fail" in res["error"] or "error" in res["error"]


def test_every_preset_runs_and_is_self_consistent():
    for p in cascade.PRESETS:
        res = cascade.predict(p["smiles"], alpha=0.10)
        assert res["parse_ok"], p["label"]
        for name, sv in res["stages"].items():
            # the contract: refused <=> no answer, and answered => usable
            assert (sv["answer"] is None) == sv["refused"], (p["label"], name)
            if not sv["refused"]:
                assert sv["diagnostics"]["usable_prediction"] is True, (p["label"], name)
                assert sv["diagnostics"]["in_applicability_domain"] is True
                assert sv["diagnostics"]["conformal_set_size"] == 1


def test_training_molecule_is_flagged_as_label_recall():
    """An answered stage that trained on the molecule must say so."""
    res = cascade.predict(POTENT, alpha=0.10)
    s1b = res["stages"]["S1B_decoy"]
    assert s1b["refused"] is False
    assert s1b["diagnostics"]["is_training_molecule"] is True
    assert s1b["diagnostics"]["nn_tanimoto"] == pytest.approx(1.0, abs=1e-9)
    text = predict_cli.human_summary(res)
    assert "label recall, not prediction" in text
    # Stage 2 held it out, so that one must NOT carry the flag.
    assert res["stages"]["S2_potency"]["diagnostics"]["is_training_molecule"] is False


def test_limitations_panel_is_drawn_from_bundle_metadata():
    lim = cascade.limitations()
    pct = lim["domain_excludes_screening_libraries"]
    assert pct["CLUE"] == pytest.approx(94.6, abs=0.05)
    assert pct["COCONUT"] == pytest.approx(97.0, abs=0.05)
    assert pct["DSSTox"] == pytest.approx(92.8, abs=0.05)
    assert "MUST refuse" in lim["refusal_policy"]
    assert "not toxicity" in lim["stage_3_is_not_a_model"].lower() or \
        "not a model" in lim["stage_3_is_not_a_model"].lower()


def test_app_module_imports_without_a_streamlit_server():
    """The UI module must at least import cleanly (catches syntax/name errors)."""
    import importlib

    spec = importlib.util.find_spec("app.app")
    assert spec is not None
    src = open(os.path.join(_HERE, "app.py")).read()
    compile(src, "app/app.py", "exec")
