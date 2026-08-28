# `cascade_bundle.joblib` — frozen urease cascade, honest-label rebuild

SHA-256 (first 16): `90c177393e154f25`  ·  size 16.74 MB  ·  SEED = 42
Built 2026-07-28T17:07:23.321816+00:00 · scikit-learn 1.9.0 · RDKit 2026.03.4

## What this bundle is for

It serves the corrected urease cascade **and it is built to refuse.** Every stage carries an
applicability domain and a conformal predictor, and the app is required to suppress the probability
when either says the molecule is out of scope. A probability from this bundle without its domain and
conformal verdict attached is a misreport.

## Load and predict

```python
import joblib
B = joblib.load("cascade_bundle.joblib")
```

Featurization the caller must reproduce exactly (`B['featurization']`):
ECFP4 via `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`, then the 25 RDKit
descriptors in `B['featurization']['descriptor_names']` order, concatenated as
`hstack([ecfp4_bits, descriptors])` — **2048 fingerprint bits first, 25 descriptors second.**
Canonicalize input with `Chem.MolToSmiles(Chem.MolFromSmiles(smi))` before featurizing.

## Top-level keys

| key | contents |
|---|---|
| `schema_version`, `created_utc`, `seed` | provenance |
| `featurization` | fingerprint spec, descriptor names, feature order, standardization note |
| `applicability_domain` | preregistered Tanimoto threshold 0.4, Mahalanobis χ² quantile 0.95, and the **refusal policy** |
| `conformal` | conformal type, nonconformity function, alphas, and the Stage-1A coverage caveat |
| `stages` | one entry per stage (below) |
| `dropped_stages` | why Stage 3 must not be used as a model |
| `provenance` | the provenance noise floor, organism/document entropy, and what UI-Ref membership means |
| `library_membership_warning` | the two stages that must never be reported as biology |
| `environment` | library versions |

## Per-stage contents — `B['stages'][name]`

| field | meaning |
|---|---|
| `pipeline` | fitted RandomForest (300 trees, `class_weight='balanced'`) on the union feature space |
| `calibrator_platt`, `calibrator_isotonic` | both calibrators, fitted on a **group-disjoint** calibration split |
| `calibrator_recommended` | which one to apply (`'platt'` or `'none'`) — chosen by held-out Brier/ECE |
| `conformal_quantiles` | `{alpha: {class: quantile}}` — label-conditional (Mondrian) quantiles for α ∈ [0.05, 0.1, 0.2] |
| `ad_tanimoto_threshold` | 0.40, **preregistered and never tuned** |
| `ad_mahalanobis` | `mean`, `inv_cov`, `cutoff` = √χ²₀.₉₅(df=25) = 6.1362 |
| `reference_fingerprints` | training ECFP4 matrix, `np.packbits`-compressed — unpack with `np.unpackbits(..., axis=1)[:, :2048]` |
| `reference_fingerprint_shape`, `reference_labels`, `reference_smiles`, `reference_inchikeys` | nearest-neighbour lookup and audit trail |
| `feature_names` | 2073 names in feature order |
| `label_semantics` | positive class, negative class, **what the negative class actually means**, design, honest verdict, grouping, class counts |
| `validated_performance` | grouped-CV AUC/MCC with CIs, held-out AUC/Brier/ECE, conformal coverage per α |
| `usable_for_prediction` | `False` for Stage 3 — evaluate the SMARTS rule instead |

## The refusal contract the app must honour

A stage prediction is **usable** only when both hold:

1. **In domain** — ECFP4 Tanimoto to the nearest reference molecule ≥ 0.40 **and** Mahalanobis ≤ cutoff.
2. **Informative conformal set** — exactly one class in the set. `{both classes}` means "cannot
   determine"; `{}` means the molecule is atypical even for its own predicted class.

When either fails, show the reason, not the number. Reference implementation: `bundle_predict()` in
this track's code returns `REFUSE: True` alongside every field needed to explain the refusal.

## Stage semantics — read before displaying anything

- **`S1B_decoy`** — UI-Ref membership vs **property-matched presumed negatives** from COCONUT/DSSTox.
  The negatives were **never assayed against urease**. This probability is
  P(resembles a molecule someone published a urease assay for), **not** P(inhibits urease).
  Document-grouped RF-union AUC 0.904
  (95% CI 0.882–0.928),
  MCC 0.586.
- **`S1A_measured`** — measured-Active vs measured-**Inactive**. The only genuine negatives that exist
  (59 Inactive-only molecules from 6 publications). AUC looks strong
  (0.840 document-grouped) but MCC is
  0.071
  (95% CI -0.027–0.202)
  — **statistically indistinguishable from the majority classifier.** Its conformal coverage also
  undershoots nominal (0.899 vs 0.950). Report as exploratory only.
- **`S2_potency`** — potent (pAct ≥ 6) vs weak, both measured. **The most defensible stage**: the only
  one whose negative class was actually tested. Document-grouped AUC
  0.782, MCC 0.504.
- **`S3_toxicophore`** — **not a model.** The label is a deterministic PAINS/BRENK/NIH SMARTS match, so
  the app must call `rdkit.Chem.FilterCatalog` and report alert count and names. Zero alerts is
  **not** "non-toxic" — no toxicity was ever measured. The fitted model is kept for audit only.
- **`S4_clinical`** — CLUE `Launched` vs not. **Library membership, not approval prediction.** Every
  molecule in that training set already reached human trials, so the 53.9% base rate is a library
  artifact. Never render as P(approval).

## Known limits, stated plainly

- **The domain excludes almost every screening library.** 92.8-97.0% of each external collection falls
  outside UI-Ref's domain (CLUE 94.6%, COCONUT 97.0%, DSSTox 92.8%; Stage-1B matched decoys 90.4%;
  UI-Ref itself 13.6% leave-one-out). Virtual screening with this bundle will refuse most inputs —
  that is the correct behaviour, not a bug.
- **Stage 1A's conformal guarantee does not hold** across publications (exchangeability fails).
- **Cascade survival is low by design.** Of 244 held-out UI-Ref molecules, 128 pass Stage 1B, and only
  25 reach the end with usable, in-domain, labelled predictions at α=0.10.
- **Between-collection discrimination must be judged against a 0.979 provenance noise floor**, not 0.5.

## Files produced alongside this bundle

`cascade_label_semantics.csv`, `cascade_benchmark.csv`, `calibration.csv`,
`applicability_domain.csv`, `conformal_coverage.csv`, `cascade_propagation.csv`,
`example_compound_trace.csv`, and figures `fig_cascade_benchmark.png`, `fig_calibration_ad.png`,
`fig_conformal.png`.
