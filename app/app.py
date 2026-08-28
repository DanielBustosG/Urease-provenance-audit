"""Streamlit view over ``app.cascade``.

    streamlit run app/app.py

This module contains no inference logic. Every number, verdict and refusal it
renders comes from ``app.cascade.predict``, which the CLI calls identically.
The one rule the layout enforces: **a refused stage never shows a probability
as its answer.** The number appears only inside a collapsed diagnostics
expander, labelled as suppressed.
"""

from __future__ import annotations

import os
import sys

# Allow `streamlit run app/app.py` from the repo root without installing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from app import cascade

st.set_page_config(page_title="Urease prediction framework",
                   page_icon=None, layout="wide")

STAGE_TITLES = {
    "S1B_decoy": "Stage 1B — resembles an assayed molecule?",
    "S1A_measured": "Stage 1A — measured Active vs measured Inactive",
    "S2_potency": "Stage 2 — potent (pAct >= 6) vs weak",
    "S3_toxicophore": "Stage 3 — structural alerts (exact rule, not a model)",
    "S4_clinical": "Stage 4 — resembles a launched CLUE drug? (library membership)",
}


@st.cache_resource(show_spinner="Loading frozen cascade bundle...")
def _bundle():
    return cascade.load_bundle()


@st.cache_data(show_spinner="Running the cascade...")
def _predict(smiles: str, alpha: float):
    res = cascade.predict(smiles, alpha=alpha)
    if res.get("parse_ok"):
        for sv in res["stages"].values():
            sv["diagnostics"]["nearest_neighbours"] = cascade.annotate_neighbours(
                sv["diagnostics"]["nearest_neighbours"])
    return res


def draw(smiles: str, size=(420, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Input")
labels = [p["label"] for p in cascade.PRESETS]
choice = st.sidebar.selectbox("Preset example", ["— type my own —"] + labels, index=1)
if choice == "— type my own —":
    default_smiles, preset_note = "", None
else:
    p = cascade.PRESETS[labels.index(choice)]
    default_smiles, preset_note = p["smiles"], p["note"]

smiles = st.sidebar.text_area("SMILES", value=default_smiles, height=90,
                              key=f"smi_{choice}")
alpha = st.sidebar.select_slider(
    "Conformal miscoverage alpha", options=list(cascade.ALPHAS), value=0.10,
    help="Smaller alpha = stronger coverage guarantee = more refusals.")
go = st.sidebar.button("Run cascade", type="primary", width='stretch')

st.sidebar.divider()
B = _bundle()["bundle"]
st.sidebar.caption(
    f"Bundle schema {B['schema_version']} · built {B['created_utc'][:19]}Z\n\n"
    f"SEED {B['seed']} · scikit-learn {B['environment']['python_sklearn']} · "
    f"RDKit {B['environment']['rdkit']}")

# ---------------------------------------------------------------- header
st.title("Urease prediction framework: applicability-domain-aware")
st.markdown(
    "This predictor is **built to refuse.** Every stage carries a prespecified "
    "applicability domain (ECFP4 Tanimoto >= 0.40 to the nearest training molecule "
    "**and** Mahalanobis <= 6.1362 in 25-descriptor space) and a label-conditional "
    "conformal predictor. When either says the molecule is out of scope, the app "
    "reports **the reason, not a number**. Refusal is the correct behaviour, not a bug: "
    "92.8–97.0% of every external screening library falls outside this domain."
)

if preset_note:
    st.info(f"**{choice}:** {preset_note}")

if not smiles.strip():
    st.stop()
if not go and choice == "— type my own —":
    st.caption("Enter a SMILES and press **Run cascade**.")
    st.stop()

res = _predict(smiles.strip(), alpha)

if not res.get("parse_ok"):
    st.error(f"**Input rejected** — {res.get('error')}. Nothing was predicted.")
    st.stop()

# ------------------------------------------------- structure & identity
c1, c2 = st.columns([1, 1.6])
with c1:
    fz = res["standardization"]["frozen_curation"]
    img = draw(fz["smiles"])
    if img is not None:
        st.image(img, caption=f"{fz['smiles']}")
with c2:
    st.subheader("Standardization and identity")
    if res["phosphinic_repair_applied"]:
        st.success(
            "**Phosphinic-acid repair applied** — `[P+](=O)O` → `[PH](=O)O`. "
            "This failure is silent, not loud: RDKit *accepts* `[P+](=O)O` as a "
            "tetravalent cationic phosphorus, and the Normalizer then rewrites it to "
            "`[PH+](=O)=O` — a different species with a net +1 charge and a different "
            "InChIKey that matches nothing in UI-Ref. Without this repair all 9 "
            "affected inhibitors become strangers to their own dataset with no error "
            "raised anywhere.")
    st.markdown(
        f"- **Frozen curation pipeline** (largest fragment → Normalizer → canonical "
        f"tautomer, no Reionizer): `{fz['smiles']}`\n"
        f"- **InChIKey**: `{fz['inchikey']}`"
    )
    bn = res["standardization"]["bundle_featurization_note"]
    if res["conventions_agree"]:
        st.caption("Both standardization conventions agree on this molecule.")
    else:
        st.warning(
            "**The two standardization conventions disagree.** The frozen curation "
            f"pipeline gives `{fz['smiles']}` (`{fz['inchikey']}`); the bundle's own "
            f"featurization note gives `{bn['smiles']}` (`{bn['inchikey']}`). "
            "This molecule's *identity*, and therefore its domain verdict, is "
            "tautomer-convention-dependent. Both are evaluated below, and any stage "
            "whose verdict flips between them is refused."
        )

    rec = cascade.training_record(fz["inchikey"])
    if rec:
        st.error(
            f"**This molecule is in the UI-Ref training set** as `{rec['row_id']}` "
            f"(measured pAct_median **{rec['pAct_median']}**, "
            f"{rec['n_direct_measurements']} direct measurement(s), "
            f"{rec['act_types']}, {rec['primary_organism']}, "
            f"source {rec['primary_document']}). Any stage that trained on it is "
            "recalling a label, not predicting one; read those numbers as memorisation."
        )
    else:
        st.caption("Not present in UI-Ref under the frozen convention — "
                   "predictions below are out-of-sample.")

# ------------------------------------------------- Stage 3 exact rule
al = res["structural_alerts_exact_rule"]
st.subheader("Structural alerts — exact rule, computed not predicted")
if al["n_alerts"]:
    st.markdown(f"**{al['n_alerts']} alert(s)**: " + ", ".join(f"`{a}`" for a in al["alerts"]))
else:
    st.markdown("**0 alerts.**")
st.caption(f"{al['rule']}. {al['caveat']}")

st.divider()

# ------------------------------------------------- per-stage cards
answered = res["cascade_summary"]["n_answered"]
refused = res["cascade_summary"]["n_refused"]
st.header(f"Per-stage verdicts at alpha = {alpha}")
if answered == 0:
    st.error(f"**All {refused} stages refuse to answer for this molecule.** "
             "No usable prediction exists. The diagnostics below explain why.")
else:
    st.markdown(f"**{answered} stage(s) answered, {refused} refused.**")

for name in cascade.STAGE_ORDER:
    sv = res["stages"][name]
    d = sv["diagnostics"]
    sem = sv["semantics"]
    with st.container(border=True):
        st.markdown(f"### {STAGE_TITLES[name]}")

        # --- the verdict, or the refusal in its place
        if sv["refused"]:
            st.error("#### REFUSED — no prediction is reported for this molecule")
            for r in sv["refusal_reasons"]:
                st.markdown(f"- {r}")
        else:
            verdict = sv["answer"]
            (st.success if verdict == "POSITIVE" else st.warning)(
                f"#### {verdict} — calibrated p(positive) = {d['p_positive_reported']}, "
                f"conformal set {d['conformal_set']}")
            st.markdown(f"**This means:** {sv['answer_means']}")
            if d["is_training_molecule"]:
                st.warning(
                    "**This stage trained on this exact molecule** (nearest reference at "
                    "Tanimoto 1.0). The number above is label recall, not an "
                    "out-of-sample prediction — do not quote it as predictive performance.")

        # --- what the labels actually mean (always, refused or not)
        st.markdown("**Label semantics — what the classes mean in plain words**")
        st.markdown(
            f"- **Positive class**: {sem['positive_class']} (n = {sem['n_positive']})\n"
            f"- **Negative class**: {sem['negative_class']} (n = {sem['n_negative']})\n"
            f"- **What the negative class actually means**: {sem['negative_class_means']}"
        )
        st.info(f"**Honest verdict on this stage:** {sem['honest_verdict']}")

        # --- domain panel
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Nearest-neighbour Tanimoto", f"{d['nn_tanimoto']:.4f}",
                   delta=f"threshold {d['ad_tanimoto_threshold']:.2f}"
                         f" — {'PASS' if d['ad_tanimoto_pass'] else 'FAIL'}",
                   delta_color="normal" if d["ad_tanimoto_pass"] else "inverse")
        dc2.metric("Mahalanobis distance", f"{d['mahalanobis']:.4f}",
                   delta=f"cutoff {d['mahalanobis_cutoff']:.4f}"
                         f" — {'PASS' if d['ad_mahalanobis_pass'] else 'FAIL'}",
                   delta_color="normal" if d["ad_mahalanobis_pass"] else "inverse")
        dc3.metric("Applicability domain",
                   "IN DOMAIN" if d["in_applicability_domain"] else "OUT OF DOMAIN")

        # --- nearest training molecules as the evidence
        st.markdown("**Nearest training molecules — the evidence behind (or against) "
                    "any prediction**")
        rows = []
        for nb in d["nearest_neighbours"]:
            meas = nb.get("measured")
            rows.append({
                "rank": nb["rank"],
                "Tanimoto": nb["tanimoto"],
                "training SMILES": nb["smiles"],
                "training label": ("positive" if nb["training_label"] == 1 else "negative"),
                "measured pAct (UI-Ref)": (meas["pAct_median"] if meas else None),
                "measured Active record": (meas["any_active"] if meas else None),
                "measured Inactive record": (meas["any_inactive"] if meas else None),
                "source document": (meas["primary_document"] if meas else None),
            })
        nbdf = pd.DataFrame(rows)
        st.dataframe(nbdf, hide_index=True, width='stretch')
        imgs = [draw(nb["smiles"], (240, 180)) for nb in d["nearest_neighbours"]]
        icols = st.columns(3)
        for col, nb, im in zip(icols, d["nearest_neighbours"], imgs):
            if im is not None:
                col.image(im, caption=f"T = {nb['tanimoto']:.3f}")
        st.caption("A blank 'measured pAct' means the reference molecule is not a UI-Ref "
                   "member — Stage 1B negatives are property-matched library molecules "
                   "that were never assayed, and Stages 3/4 train on library collections.")

        # --- subordinate diagnostics, collapsed
        with st.expander("Suppressed numeric diagnostics (not the answer)"
                         if sv["refused"] else "Numeric diagnostics"):
            if sv["refused"]:
                st.caption("These numbers are shown for audit only. This stage produced "
                           "no reportable prediction for this molecule; do not quote "
                           "them as one.")
            diag = pd.DataFrame([
                {"convention": v["convention"],
                 "standardized SMILES": v["standardized_smiles"],
                 "p(positive) reported": v["p_positive_reported"],
                 "p raw RF": v["p_positive_raw_rf"],
                 "p Platt": v["p_positive_platt"],
                 "calibrator": v["calibrator_used"],
                 "nn Tanimoto": v["nn_tanimoto"],
                 "Mahalanobis": v["mahalanobis"],
                 "in domain": v["in_applicability_domain"],
                 "conformal set": v["conformal_set"],
                 "usable": v["usable_prediction"]}
                for v in sv["diagnostics_by_convention"].values()])
            st.dataframe(diag, hide_index=True, width='stretch')
            vp = sv["validated_performance"]
            gcv = vp.get("grouped_cv_RF_union", {})
            if "document" in gcv:
                g = gcv["document"]
                st.markdown(
                    f"**Document-grouped CV**: AUC {g['AUC']:.3f} "
                    f"[{g['AUC_CI'][0]:.3f}–{g['AUC_CI'][1]:.3f}], "
                    f"MCC {g['MCC']:.3f} [{g['MCC_CI'][0]:.3f}–{g['MCC_CI'][1]:.3f}]")
            ho = vp.get("heldout_test", {})
            if ho:
                st.markdown(
                    f"**Held-out test** (n = {ho.get('n')}): AUC {ho.get('AUC', float('nan')):.3f}, "
                    f"Brier {ho.get('Brier_uncalibrated', float('nan')):.4f}, "
                    f"ECE {ho.get('ECE_uncalibrated', float('nan')):.4f}")
            conf = vp.get("conformal", [])
            if conf:
                st.dataframe(pd.DataFrame(conf), hide_index=True, width='stretch')

st.divider()

# ------------------------------------------------- limitations panel
lim = res["limitations"]
st.header("Honest limitations — read before quoting anything above")
st.error(f"**Refusal policy (from the bundle):** {lim['refusal_policy']}")

lc1, lc2 = st.columns(2)
with lc1:
    st.subheader("The domain excludes almost every screening library")
    st.dataframe(
        pd.DataFrame([{"collection": k, "% outside UI-Ref domain": v}
                      for k, v in lim["domain_excludes_screening_libraries"].items()]),
        hide_index=True, width='stretch')
    st.caption("Virtual screening with this bundle will refuse most inputs. "
               "That is the designed behaviour.")
with lc2:
    st.subheader("Stages that are NOT biology predictors")
    for k, v in lim["library_membership_warning"].items():
        st.warning(f"**{k}** — {v}")

st.subheader("What membership and the negatives actually mean")
st.markdown(
    f"- **UI-Ref membership**: {lim['what_ui_ref_membership_means']}\n"
    f"- **Genuine negatives**: {lim['genuine_negatives']}\n"
    f"- **Provenance noise floor**: {lim['provenance_noise_floor']}\n"
    f"- **Conformal caveat**: {lim['conformal_caveat']}\n"
    f"- **Stage 3**: {lim['stage_3_is_not_a_model']}"
)

st.subheader("Not a toxicity or approval predictor")
st.markdown(
    "Stage 3's label is a deterministic SMARTS match, and zero alerts is **not** "
    "\"non-toxic\" — no toxicity assay of any kind enters it. Stage 4's label is CLUE "
    "library membership; every molecule in its training set had already reached human "
    "trials, so its base rate is a library artifact and it can never be read as "
    "P(approval). Neither stage was built on, or validated against, any toxicity or "
    "regulatory outcome. This tool supports hypothesis generation for laboratory "
    "follow-up; it is not evidence of safety, efficacy, or clinical utility."
)

with st.expander("Full JSON result (identical to `python -m app.predict`)"):
    st.json(res)
