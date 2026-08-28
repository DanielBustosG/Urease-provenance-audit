# Urease cascade — applicability-domain-aware predictor

A local app that serves the corrected urease cascade and **is built to refuse.**
Every stage carries a preregistered applicability domain and a conformal
predictor; when either says a molecule is out of scope, the app reports the
reason instead of a number.

It exists because of a specific failure. An earlier prototype, given
`CN(C)c1ccc(/C=N/N=C2\NC(=O)CS2)cc1`, printed:

> **NON-INHIBITOR — Yes: 0.462 / No: 0.538**

with no uncertainty, no domain check, and a negative class that meant nothing in
particular. This app reports **five refusals** for that molecule, and explains
each one. That is the whole point.

## Install and run

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Two data files must sit beside the code (or be pointed at by
`UREASE_CASCADE_BUNDLE`):

| file | role | required |
|---|---|---|
| `app/cascade_bundle.joblib` | the frozen models, calibrators, conformal quantiles, domain parameters and reference fingerprints (SHA-256 prefix `90c177393e154f25`, 16.7 MB) | yes |
| `app/uiref_curated.parquet` | the 1218-molecule UI-Ref table, used to attach *measured* activity to nearest neighbours | optional but strongly recommended |
| `app/example_compound_trace.csv` | the stored reference trace the test suite pins against | optional (a test skips without it) |

`scikit-learn` must stay pinned at **1.9.0** — the bundle is a pickle of
1.9.0 estimators, and a different minor version may warn or fail outright.

### Command line

```bash
# one molecule, human summary on stderr + full JSON on stdout
python -m app.predict "O=c1c2ccccc2[se]n1-c1ccc(C(F)(F)F)cc1"

# all built-in presets, JSON only
python -m app.predict --presets --quiet > out.json

# tighter conformal guarantee => more refusals
python -m app.predict "CCO" --alpha 0.05
```

`app/app.py` (UI) and `app/predict.py` (CLI) are both thin views over
`app/cascade.py`. No inference logic is duplicated, and
`test_cli_agrees_with_library` fails the build if they ever diverge.

### Tests

```bash
PYTHONPATH=. pytest app/test_app.py -v
```

25 tests, all passing. They cover a known potent inhibitor (in-domain,
positive at Stage 2), the prototype compound (refused), sucrose and a PEG
decamer (out of domain), the phosphinic-acid repair, exact reproduction of the
stored bundle probabilities across a fresh reload, and CLI/library agreement.

## What each stage means

The cascade has five stages. **Only one of them is a defensible biology
predictor**, and the app says so on every screen.

| stage | positive class | negative class | what the negative class *actually* means | honest status |
|---|---|---|---|---|
| **1B** `S1B_decoy` | UI-Ref member (reported urease activity of any magnitude) | property-matched COCONUT/DSSTox molecule | **never assayed against urease.** Absence of evidence. Some are certainly actives. | ranking/enrichment only. Its probability is P(resembles a molecule someone published a urease assay for), **not** P(inhibits urease). Document-grouped AUC 0.904 |
| **1A** `S1A_measured` | measured Active | measured **Inactive** | genuinely tested and found inactive — the only real negatives that exist for this target (59 molecules, 6 publications) | exploratory only. AUC 0.840 but MCC 0.071 [−0.027–0.202] — indistinguishable from the majority classifier. Its conformal coverage undershoots nominal (0.899 vs 0.950); the guarantee does not hold |
| **2** `S2_potency` | potent, pAct ≥ 6 | weak, pAct < 6 | measured and quantitatively weak — a real, earned negative | **the most defensible stage.** The only one where both classes were tested on the same footing. Document-grouped AUC 0.782, MCC 0.504 |
| **3** `S3_toxicophore` | ≥1 PAINS/BRENK/NIH alert | 0 alerts | **no published SMARTS pattern matched.** No toxicity assay of any kind enters this label | **not a model.** The label is an exact function of the structure, so the app calls `rdkit.Chem.FilterCatalog` directly and reports alert names. The fitted model is never consulted |
| **4** `S4_clinical` | CLUE Phase == Launched | CLUE Phase ∈ {clinical, withdrawn} | present in the CLUE library with a non-launched annotation. Every molecule here already reached human trials | **library membership, not approval prediction.** The 53.9% base rate is a library artifact. Never render as P(approval) |

## What the app will refuse to answer

A stage reports a verdict only when **all** of the following hold. Otherwise it
refuses and names the failing condition.

1. **In the applicability domain** — ECFP4 Tanimoto ≥ **0.40** to the nearest
   training molecule (preregistered, never tuned) **and** Mahalanobis distance
   ≤ **6.1362** = √χ²₀.₉₅(df=25) in 25-descriptor space.
2. **Informative conformal set** — exactly one class. `{both classes}` means
   "cannot determine"; `{}` means the molecule is atypical even for its own
   predicted class.
3. **The stage is a model at all** — Stage 3 always refuses as a predictor; its
   exact SMARTS alert list is reported instead.
4. **The two standardization conventions agree** — see below.

When a stage refuses, its probability is still computed and shown inside a
collapsed *"Suppressed numeric diagnostics"* expander, explicitly labelled as
not-the-answer. It is never rendered as a verdict, and the CLI's text summary
is tested to never place a probability on a `REFUSED` line.

### The domain excludes almost every screening library

| collection | % outside UI-Ref's domain |
|---|---|
| CLUE | 94.6% |
| COCONUT | 97.0% |
| DSSTox | 92.8% |
| Stage-1B matched decoys | 90.4% |
| UI-Ref itself (leave-one-out) | 13.6% |

Virtual screening with this bundle will refuse most inputs. That is the correct
behaviour. The uncorrected analysis this work audits ran its classifiers almost
entirely outside their own applicability domain.

## Two things this app does that the bundle alone does not

Both were found while building it, and both are load-bearing.

### 1. Standardization convention is not settled, so both are evaluated

The frozen curation pipeline that defined dataset identity is *largest fragment
→ Normalizer → canonical tautomer* (no Reionizer). The bundle's own
`featurization` block prescribes something weaker — bare
`Chem.MolToSmiles(Chem.MolFromSmiles(smi))`, no tautomer canonicalization —
and that is the convention `example_compound_trace.csv` was generated under.

For most molecules the two agree. For the prototype compound they do not, and
the difference is not cosmetic:

| convention | standardized SMILES | InChIKey | Stage 1A nearest-neighbour T |
|---|---|---|---|
| frozen curation | `CN(C)c1ccc(/C=N/Nc2nc(O)cs2)cc1` | `YXXKRHZFVGZYRJ-NTUHNPAUSA-N` | **1.0000** |
| bundle note | `CN(C)c1ccc(/C=N/N=C2\NC(=O)CS2)cc1` | `AOFVYOCIKCHIBD-NTUHNPAUSA-N` | 0.3400 |

Under the frozen convention the prototype **is UI-Ref molecule `UIREF-1160`**
(measured pAct 5.14, IC50, *Sporosarcina pasteurii*,
DOI 10.1016/j.bioorg.2015.10.005) — a *measured active*, in-domain, and the app
flags it as a training molecule whose Stage 1A "prediction" is label recall.
Under the bundle's convention it is a stranger at T = 0.34, out of domain.

Same input, opposite verdicts, decided entirely by a thiazoline↔thiazole
tautomer convention. The app therefore evaluates **both**, and refuses any
stage whose domain or conformal verdict flips between them. This is strictly
more conservative than the bundle's own contract and never contradicts it. The
prototype's five refusals are the right answer for a reason the original
prototype could not have articulated: not merely "we don't know", but "the
question is not yet well posed."

### 2. The phosphinic-acid repair prevents a silent loss, not a crash

`[P+](=O)O → [PH](=O)O` must be applied before standardization or 9 known
inhibitors are lost. The failure mode is worth stating precisely, because it is
easy to describe wrongly: RDKit **does** parse `[P+](=O)O` — as tetravalent
cationic phosphorus, formal charge +1. Nothing errors. The Normalizer then
rewrites it to `[PH+](=O)=O`, a different species with a different InChIKey.

Measured over all 9 affected molecules: **0/9** are recoverable in
`uiref_curated.parquet` without the repair; **9/9** with it. No exception is
raised at any point. A pipeline missing this line does not fail loudly — it
quietly drops 9 phosphinic/phosphonic inhibitors and reports success.

## Not a toxicity or approval predictor

Stage 3's label is a deterministic SMARTS match; **zero alerts is not
"non-toxic"** — no toxicity assay of any kind enters it. Stage 4's label is
CLUE library membership; every training molecule had already reached human
trials, so its base rate is a library artifact and it can never be read as
P(approval). Neither stage was built on or validated against any toxicity or
regulatory outcome.

Note that Stage 4 can be *in-domain* for molecules the biology stages reject —
sucrose is a CLUE member, so Stage 4 answers "resembles a launched drug"
(p = 0.8597) while Stages 1B/1A/2 all refuse. That contrast is exactly the trap
the limitations panel exists to close: an in-domain Stage 4 answer is a
statement about a library, never about urease.

Between-collection discrimination must be judged against a **0.954 provenance
noise floor** (a Random Forest separates PubChem- from BindingDB-sourced UI-Ref
molecules — all the same chemical class — at ROC AUC 0.954), not against 0.5.

This tool supports hypothesis generation for laboratory follow-up. It is not
evidence of safety, efficacy, or clinical utility, and nothing it outputs
should inform a decision about a human subject.

## Files

| file | role |
|---|---|
| `cascade.py` | the inference library — standardization, featurization, calibration, conformal sets, domain tests, refusal logic, presets |
| `app.py` | Streamlit UI. Renders only what `cascade.predict` returns |
| `predict.py` | CLI (`python -m app.predict`). JSON on stdout |
| `test_app.py` | 24 pytest smoke tests |
| `requirements.txt` | pinned versions matching the bundle's build environment |

Reproducibility: SEED = 42, scikit-learn 1.9.0, RDKit 2026.03.4, NumPy 2.4.6.
