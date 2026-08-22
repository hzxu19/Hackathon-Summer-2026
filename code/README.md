# MERFISH spinal-cord cell-type classification: final pipeline

Reproduction package for the submitted `prediction.csv` (team "Biostat Ted", Hackathon Summer 2026): predict
the MERFISH cell-type annotation (60 classes) of 5,000 test cells from a 200-gene count matrix and per-cell
metadata, given 5,000 labelled training cells. The 10,000 cells come from 108 spinal-cord sections of 10 mice;
train and test are a random split within every section, so spatial neighbours of a test cell are in the
training set. The metadata column `Excitatory_vs_Inhibitory` is given for test cells and is a deterministic
function of the class, so it partitions the 60 classes into three disjoint groups (24 excitatory and 15
inhibitory neuron subtypes, 21 other classes).

## 1. Method

Seventeen level-1 classifiers, all trained on the same fixed 5-fold split with fold-safe
spatial-neighbourhood label features, are combined by the average of (i) a greedy forward-selection weighted
average and (ii) a per-class ridge stacker; the excitatory / inhibitory / other constraint is applied and the
arg-max class is written.

**Features.** Raw and depth-normalised log counts, 19 numeric metadata columns, one-hot categorical metadata,
the mean log expression of the 10 nearest same-section cells ("niche"); the class proportions among the k
nearest *labelled* same-section cells (k = 10/30, also 5/15/50 for some members; self excluded; pool = the
training part of the current fold, for the test set all 5,000 training cells); a label-free spatial block
(within-section relative / polar / PCA-rotated coordinates, 30 expression PCs and their multi-scale spatial
niche means, local density, neighbourhood E/I and Region composition); a slice-edge block (distance to the
section's convex hull, edge indicators, radial rank); kNN-smoothed expression and expression PCs; a 60-column
class-conditional negative-binomial log-posterior block of the raw counts (Dirichlet prior on the class gene
composition, per-gene dispersion, estimated fold-wise with inner folds); and, for the spatial xgboost members,
a label-derived spatial block (kNN and Gaussian-kernel class proportions, distance to the nearest cell of
every class, section composition, a dorsal-ventral axis from the pool centroids; self always excluded).

**Level-1 members** (identifiers = the directories written under `work/models/`):

| # | Member | Learner |
|---|---|---|
| 1 | `glm3_h3` | Spliced sparse multinomial lasso: exc/inh cells from a single 60-class glmnet, "other" cells from a per-group glmnet (1,083 z-scored columns). |
| 2 | `nb_glm2` | Same plus the 60-column negative-binomial log-posterior block. |
| 3 | `bag_glm` | 32 bootstrap replicates of the member-1 pipeline at the per-fold lambda, averaged. |
| 4 | `hier2_a` | Two-level lasso: 12 coarse groups (per E/I group) x subtypes within each group. |
| 5 | `xgb3_a` | xgboost, 60 classes, merged feature set (880 label-free columns + neighbour-label block); eta 0.04, depth 6. |
| 6 | `expr_c` | xgboost on base + smoothed expression + expression PCs + neighbour-label block; eta 0.05. |
| 7 | `spatial_d` | Spatial xgboost, round 1, with the slice-edge block. |
| 8 | `spatial_e` | Spatial xgboost, round 2: the label-derived features also use the soft round-1 predictions of the *test* cells as pseudo labels (built so that no training cell ever sees a prediction derived from its own label). |
| 9 | `hier_e` | Average of two grouped xgboost models (one per E/I group, two label-block scales), then within-family expert models (oligodendrocyte lineage, astrocytes, vascular, motoneurons) redistribute each family's total probability. |
| 10 | `diverse_glmnet` | Elastic net (alpha 0.5) on the standardized 699-column fold-wise matrix F. |
| 11-17 | `diverse_hgb`, `diverse_et`, `diverse_rf`, `diverse_mlp`, `diverse_mlp_b`, `diverse_lr`, `diverse_svc` | scikit-learn: HistGradientBoosting, ExtraTrees (800), RandomForest (600), MLP (256,128), bagged MLP (5 seeds), LogisticRegression (C = 0.5), RBF SVC (C = 5, Platt), all on F (standardized where appropriate). |

**Ensemble.** Every member's probability matrix is E/I-masked and row-normalised. (i) Greedy forward
selection with replacement (60 iterations, patience 12, criterion = masked accuracy), nested: weights
selected on four folds' out-of-fold rows, evaluated on the fifth; the weights refitted on all 5,000 rows are
applied to the test probabilities. (ii) Per-class ridge stacker after a correlation de-duplication of the
members (greedy by accuracy, keep if correlation < 0.97 with every kept member; 13 of 17 remain): the score
of class c is a closed-form ridge regression of 1[y = c] on the members' P(c) columns, penalty chosen per
outer fold on an inner fold; scores are clipped, masked, normalised; nested the same way. Final
probabilities = (greedy + stacker) / 2, E/I mask, arg-max.

## 2. Cross-validation protocol

* **Fixed folds.** A stratified 5-fold assignment of the 5,000 training cells (`folds/folds_5_seed2.csv`,
  sizes 994 / 1003 / 998 / 1003 / 1002, generated with `caret::createFolds`, seed 20260819). Every member,
  every hyper-parameter selection and the ensemble use these same folds.
* **Fold safety.** All label-dependent features (neighbour-label composition, label-derived spatial features,
  count-likelihood block) are recomputed inside each fold from the training part only. glmnet penalties, the
  stacker's ridge penalty and the Dirichlet pseudo-count are chosen on inner splits / inner folds of the
  training part. Caveat: the xgboost members select their number of boosting rounds by early stopping on the
  held-out fold itself (standard practice, but a mild optimism for those members' individual out-of-fold
  numbers; their test models use 1.1 x the mean best round); the scikit-learn early stopping uses an internal
  validation split. The test set uses a label pool of all 5,000 training cells.
* **Nested ensemble evaluation.** Greedy weights and stacker coefficients are fitted on four folds'
  out-of-fold rows and evaluated on the fifth, so the reported ensemble accuracy contains no selection
  optimism from the out-of-fold rows themselves.
* **Two fold splits.** The method was developed on a first split (seed 20260818); the complete pipeline with
  all settings frozen was then re-fitted from scratch on the second split shipped here (seed 20260819), and
  the submission is taken from that run (nested blend accuracy 0.7830 on the first split, 0.7804 on the
  second, i.e. the development split carried about half a point of selection optimism).

## 3. Results

Masked out-of-fold accuracy on the 5,000 training cells (this package, fully seeded pipeline; exact ties in a
tree ensemble's arg-max are broken arbitrarily when the summary is printed, which can move those printed
numbers by 1-2 cells):

| Model | OOF acc |
|---|---|
| `nb_glm2` | 0.7744 |
| `hier2_a` | 0.7738 |
| `glm3_h3` / `bag_glm` | 0.7724 |
| `xgb3_a` | 0.7570 |
| `hier_e` | 0.7554 |
| `expr_c` | 0.7544 |
| `spatial_e` | 0.7468 |
| `spatial_d` | 0.7424 |
| `diverse_hgb` | 0.7320 |
| `diverse_glmnet` | 0.7296 |
| `diverse_mlp_b` | 0.7268 |
| `diverse_et` | 0.7262 |
| `diverse_lr` | 0.7086 |
| `diverse_svc` | 0.7036 |
| `diverse_rf` | 0.7032 |
| `diverse_mlp` | 0.6910 |
| **greedy weighted average (nested)** | **0.7858** |
| **per-class ridge stacker (nested)** | **0.7744** |
| **final blend (nested)** | **0.7804** |

Final blend by group: excitatory 0.853, inhibitory 0.953, other 0.710; per fold
0.7777 / 0.7846 / 0.7655 / 0.7807 / 0.7934. Greedy weights: `hier2_a` 0.40, `hier_e` 0.20, `nb_glm2` 0.20,
`xgb3_a` 0.20. Public interim scoreboard: 0.7780 for an earlier version (members 1 and 5-17 only), 0.7786 after adding
the last three members, 0.7806 for the seeded pipeline shipped here. The remaining errors are concentrated in the non-neuronal group
(oligodendrocyte progenitor vs. mature classes, astrocyte subtypes, meninges subtypes).

## 4. How to reproduce

**Environment used.** Windows 11, 32 cores; R 4.4.1 with data.table 1.17.0, FNN 1.1.4.1, glmnet 4.1-8,
xgboost 1.7.9.1, Matrix 1.7.0; Python 3.12.5 with numpy 2.5.1 and scikit-learn 1.9.0. The scripts are plain
R / Python and run unchanged on Linux or macOS (set the paths at the top of `main.R`).

**Data.** The four organiser files `counts_train.csv`, `counts_test.csv`, `meta_train.csv`, `meta_test.csv`,
expected in `../Hackathon-Summer-2026/data/` relative to this directory (edit `data_dir` at the top of
`main.R` to point elsewhere). Nothing else is needed: every intermediate file is rebuilt from these four.

```
cd final_pipeline
Rscript main.R        # -> output/prediction.csv   (~95 min sequential on the machine above; work/ ~0.9 GB)
```

`main.R` runs everything in order: feature files -> 17 members -> ensemble -> `output/prediction.csv`
(header `Cell_ID,MERFISH_cell_type_annotation.y`, 5,000 rows in `meta_test.csv` order). Members whose
`work/models/<name>/test_prob.rds` already exists are skipped, so an interrupted run can simply be restarted.
The two Python scripts are called by `main.R` through the `python` path set at its top.
`output/final_summary.txt` reports the nested accuracies, the greedy weights and the agreement with the
shipped prediction files.

**File map.**

| Path | Purpose |
|---|---|
| `main.R` | The complete pipeline (paths and thread count at the top). |
| `sk_models.py`, `sk_mlp_bag.py` | The seven scikit-learn members (called by `main.R`). |
| `folds/folds_5_seed2.csv` | The fixed fold assignment. |
| `reference/prediction_submitted.csv` | The submitted prediction file. |
| `candidate_seeded/` | Prediction + run summary of the seeded pipeline's verification run (byte-identical to the submitted file). |
| `output/` | Prediction, `final_summary.txt` and the final probability matrices of the latest run. |
| `work/` (created) | Intermediate files: features, per-member probability matrices, scratch. |

## 5. Randomness and reproducibility

Every random step is seeded: the glmnet members' inner splits / inner cv folds (`set.seed(100 + fold)`; the
two-level model's subtype level 200 + fold), the bootstrap bags (7000/8000 + 100 x fold + bag), the count
model's inner folds (500 + fold), fixed `random_state` for every scikit-learn model, and the xgboost members'
row / column subsampling (`xgb3_a` 1000 + fold, `expr_c` 21000 + fold, `spatial_a` 22000 + fold, `spatial_d`
22100 + fold, `spatial_e` 23000 + fold, `hier_b` / `hier_d` 24000/25000 + 100 x group + fold, family experts
26000 + 10 x family + fold; fold 0 = the full-training test model). The greedy average, the stacker and the
blend are closed form; the final arg-max has no ties on the test set. Keep `nthread = 5`: a different thread
count can change xgboost's floating-point summation order and hence the result.

With these seeds the pipeline is exactly reproducible: independent runs from the four organiser csv files
produce bit-identical probability matrices for every member and a byte-identical `prediction.csv`
(md5 `459c99284607b13eb570448d411f9e59` = `candidate_seeded/prediction.csv` =
`reference/prediction_submitted.csv`; verified across three independent runs, the last one with this
`main.R`). Development history: in early runs the four xgboost members `expr_c`, `spatial_d`, `spatial_e`,
`hier_e` were unseeded and drew from R's default random stream, so each run gave a slightly different version
of them (about 98% of test cells agreed between runs; blend 0.780-0.782); explicit seeds were added on
2026-08-19 and the submission is the seeded pipeline's output.
