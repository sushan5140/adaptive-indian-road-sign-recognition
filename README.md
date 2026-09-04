# Adaptive Indian Road Sign Recognition Using Open-Set and Few-Shot Learning

This internship project builds a computer-vision system that recognizes Indian
road signs while allowing new sign classes to be registered from a few examples.
The base dataset remains unchanged, and registering a sign never retrains the
full base classifier.

## Results summary

**Reference model: Baseline V2.** Measured once on a locked, group-aware
63-image test split held out from Dataset B:

| Metric | Value |
| --- | ---: |
| Top-1 accuracy | **0.6032** (38/63) |
| Macro F1 | **0.6756** |
| Weighted F1 | 0.5861 |
| Classes | 17 |
| Training images | 287 |

Full artifacts: `outputs/v2_results/`. Checkpoint SHA-256
`0f990f21c7f844f5611e91f867740b7f980e851426681c69deb2fefadbea8ff4`.
Accuracy on 63 images carries a wide interval: Wilson 95% CI [0.4798, 0.7147],
bootstrap 95% CI [0.4762, 0.7302] (`outputs/v2_results/test_accuracy_confidence.json`).

**Open-set and few-shot layer.** `inference/` returns one of three verdicts —
known base class, registered incremental class, or unknown — and registering a
new sign never updates a model weight, which is asserted at runtime and in tests.
The claim has been measured by leave-one-class-out proxy, holding out a known
class and registering it from five reference images:

| Held-out class | Recognition rate | False-accept rate | Calibrated alone |
| --- | ---: | ---: | --- |
| `filling_station` | 31.3% | 0.0% | 93.75% TPR at 3.33% FPR (Youden J 0.904) |
| `school_ahead` | 42.4% | 22.8% | 54.4% FPR (Youden J 0.335) |

The mechanism works when a new sign is visually distinct from the base classes
and fails when it resembles them; that is class-dependent, and no single global
threshold fixes it. Thresholds are measured rather than assumed
(`base_confidence_threshold` 0.1747 from the validation split,
`prototype_similarity_threshold` 0.5044 from pooled proxy negatives) and
`calibrated: true` in `configs/config.yaml` records that. No result on a sign
genuinely absent from every dataset in the project is claimed.

**Four independent attempts to beat 60.32% all failed**, and they converge:

| Attempt | Lever pulled | Test top-1 | vs V2 |
| --- | --- | ---: | ---: |
| Dataset A | more images | rejected before training | — |
| V3 | data volume (+771 real photos) | 0.5079 | -0.0952 |
| V4 | stronger augmentation, higher weight decay | 0.5079 | -0.0952 |
| V5 | 5-fold cross-validation ensemble | 0.5714 | -0.0317 |

Dataset A was rejected before training as template-plus-augmentation rather than
independent photographs. Each of the other three pulled a different, standard
lever — more data, better regularisation, model averaging — and every one landed
below the baseline. V5 came closest and is the most informative: ensembling
demonstrably worked on its own terms, beating the mean of its five members by
+0.073 macro F1, yet every member scored below V2 because each trained on a
smaller effective pool.

That four-way convergence is the substantive finding of this project. When
several unrelated techniques all fail in the same direction, the limit is not the
methodology. It is roughly 17-20 training images per class across 17 classes, on
a 63-image test split too small to resolve differences under about ten points.
All four results are documented below rather than discarded.

## Licensing status

**This repository is currently unlicensed.** No `LICENSE` file is present and
`pyproject.toml` asserts no licence, deliberately rather than by oversight.

Two of the candidate training datasets have unresolved provenance. Dataset A
(the manually supplied Kaggle archive) ships no licence, README, attribution or
citation of any kind — only a class-map CSV. Dataset C (the HuggingFace
supplement) declares `cc-by-4.0`, but its README contains that single line and
names no author, source or collection methodology, so the attribution that CC BY
requires cannot be satisfied. Dataset B, the Zenodo record this project actually
trains on, is the one source with clear terms.

Asserting an open-source licence over a codebase whose data inputs carry unknown
or unsatisfiable terms would claim more than the evidence supports. Until those
questions are resolved, no licence is asserted here. See
`outputs/open_set_acquisition/acquisition_source_assessment.md` for the full
source-by-source assessment.

## Why a standard classifier is not enough

A conventional supervised classifier learns a fixed output vocabulary. Its
softmax layer must select among those classes even when an image belongs to a
class that was completely absent during training. A low score can suggest
uncertainty, but the classifier cannot invent the semantic name of an unseen
sign. Adding a new output normally requires changing the classification head and
training again.

This project separates those concerns. A MobileNetV3-Small base classifier names
the classes present during supervised training. The same model's feature
extractor maps images into an embedding space. Low-confidence or
out-of-distribution inputs can be rejected as unknown, and user-registered signs
are represented by prototypes in a separate registry.

## Proposed architecture

1. Train `RoadSignClassifier` on the supplied base classes.
2. Reuse `FeatureExtractor` to produce normalized image embeddings.
3. Calibrate classifier-confidence and out-of-distribution thresholds using
   held-out data.
4. For a new sign, embed a small set of reference images, normalize each vector,
   average them, and normalize the resulting class prototype.
5. Persist the prototype, label, and JSON metadata in the separate NPZ registry.
6. At inference, combine base-class confidence with nearest-prototype cosine
   similarity and return a base class, registered incremental class, or unknown.

The incremental registry never writes to the original dataset. Its prototypes
can be added and removed without changing or fully retraining the base model.

## Repository layout

```text
.
├── api/                     # FastAPI package (deferred)
├── artifacts/
│   ├── checkpoints/         # Generated base-model checkpoints
│   └── prototypes/          # Generated incremental registry files
├── configs/config.yaml      # Paths, model, runtime, and threshold defaults
├── data/                    # Ignored local data mount; no automatic downloads
├── evaluation/              # Closed-set metrics and measured reports
├── inference/               # Open-set decision policy, registration, pipeline
├── models/
│   ├── classifier.py
│   ├── feature_extractor.py
│   └── prototype_registry.py
├── scripts/                 # Entry points: audit, train, evaluate, calibrate, experiments
├── tests/                   # Unit and infrastructure smoke tests
├── training/                # Loaders, transforms, trainer, history, checkpoints
├── ui/                      # Streamlit interface (deferred)
└── utils/                   # Shared utilities (deferred)
```

## Dataset inspection and adapter

The manually supplied Kaggle archive is retained unchanged under the ignored
`data/downloads/` directory and extracted under ignored `data/raw/` storage. A
full read-only audit on 30 August 2026 decoded 13,971 PNG images in 58 populated
numeric class directories. The class lookup CSV contains IDs 0 through 58, but
class 39 has no image directory. Measured reports are written to
`outputs/dataset_audit/`; deterministic numeric-label metadata is written to
`outputs/manifests/`.

Every decoded source image is 32x32. Source files remain unchanged; the model
pipeline dynamically converts them to RGB and resizes tensors to 224x224 for
MobileNetV3. The low native resolution is a dataset limitation, and permanent
upscaled copies are not created.

This archive cannot currently support a legitimate held-out evaluation. Each
populated class contains exactly one original template and otherwise consists of
derived augmentations/copies. The audit also found 344 exact-duplicate content
groups, 1,041 redundant files, and five duplicate groups spanning different
labels. An image-level split would therefore leak source templates and even
identical content across partitions. No training split or real checkpoint is
created from this archive until an independent-image dataset or defensible group
metadata is available.

Dataset A is therefore classified as a controlled, augmentation-derived source.
It may be useful for limited representation learning or prototype experiments,
but its images cannot provide an independent random holdout.

### Dataset B: Indian Traffic VQA

Dataset B was manually acquired from the public Zenodo record
`10.5281/zenodo.17300841`; the repository never downloads it automatically. The
three retained source files match Zenodo's published MD5 checksums, and their
SHA-256 hashes are recorded in `outputs/dataset_b_audit/source_files.json`.
Extraction is isolated under `data/raw/indian_traffic_vqa/`; Dataset A and all of
its prior audit artifacts remain unchanged.

The source is a VQA dataset, not a ready-made image-classification dataset. The
measured full table contains 4,341 question-answer rows for 1,085 real-world JPEG
photographs: 1,084 photographs have four questions and one has five. All 1,085
images decode successfully as 512x512 RGB JPEGs. The audit found no byte-identical
files and flagged 87 dHash near-duplicate pairs for review. It also found one
conflicting answer for a repeated image/question, two photographs with multiple
sign-category annotations, and two compact-CSV rows absent from the full CSV.

Classification labels are derived only from direct sign-identity questions, not
from shape, colour, driver-action, consequence, or location questions. Generic
speed-limit answers are refined using the corresponding numeric speed answer.
Unsupported or semantically ambiguous answers are excluded. A class must have at
least 10 photographs and five perceptual-hash leakage groups. Any duplicate group
with conflicting derived labels is excluded in full. Under these conservative
rules, the review manifest contains 530 unique photographs across 20 candidate
classes. This is a derived manifest requiring human label review; it is not
claimed as source ground truth.

Run the reproducible audit with:

```bash
python scripts/audit_vqa_dataset.py
```

Measured JSON reports, frequency tables, duplicate screening, and exclusions are
written under `outputs/dataset_b_audit/`. The auditable classification manifest,
class mapping, and explicit Dataset A/B semantic alignment are written under
`outputs/manifests/`. Dataset B was manually reviewed and then retained as one
external held-out evaluation set. It was not used for model selection, and a
random Dataset A holdout is not presented as evidence of real-world
generalization.

### Dataset C: HuggingFace Indian-Traffic-Sign-Classification (training supplement)

Source: `kannanwisen/Indian-Traffic-Sign-Classification` on HuggingFace, imported
2026-09-04. Extracted under the ignored `data/raw/hf_indian_traffic_sign/`; the
manifest of references is `outputs/manifests/hf_supplement.csv`.

**Licence caveat, unresolved.** The upstream repository declares `cc-by-4.0`, and
its README contains that one line and nothing else: no author, no original source,
no collection methodology, no citation. CC BY 4.0 requires attribution to the
licensor, and no licensor is named. This is better than Dataset A, which declares
nothing at all, but it is not clean. Resolve the upstream attribution before any
public release, publication, or redistribution of work derived from it.

The upstream archive holds 5,726 images across 85 classes. Only 1,948 survived
import:

| Filter | Removed |
| --- | ---: |
| Not 50x50 (clean clipart/template renders on white) | 1,923 |
| 50x50 but near-white border, template-like | 19 |
| Duplicate of another kept image (difference hash) | 1,836 |
| Near-duplicate of a locked validation/test image | 0 |
| **Kept** | **1,948** |

The duplicate count is the significant one. Of the 3,803 genuinely photographic
50x50 crops, only 1,958 have a distinct difference hash and only 2,326 a distinct
SHA-256; 2,717 images sit in 1,240 byte-identical groups. Those duplicates also
cross the upstream archive's own `train`/`test` boundary, so the split shipped
with it leaks and is not used. Every kept image is assigned `split: train` here.

Of the 1,948 kept, 771 map to one of the 17 base classes and 1,177 belong to
classes outside them. `maps_to_base_class` in the manifest marks which, so the
base class set is not silently widened.

Nothing from this source may enter validation or test. Those splits are locked and
group-aware, and were fixed before this dataset existed; the import verified that
no kept image is a near-duplicate of any locked image.

#### Outcome: training on this supplement made the model worse

This was tried and the result is negative. `scripts/run_v3_hf_supplement.py`
retrained the exact Baseline V2 recipe — same architecture, optimizer, cosine
schedule, augmentation, class-weighted loss, seed 42, 30 epochs, selection on
validation macro F1 — changing only the training pool, from 287 images to 1,058
(287 Dataset B plus the 771 base-class-mapped supplement rows). It was then
evaluated once on the same locked `v2_test` split.

| Metric (locked v2_test, 63 images) | V2 | V3 | Delta |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | 0.6032 | 0.5079 | -0.0952 |
| Macro F1 | 0.6756 | 0.5133 | -0.1623 |
| Weighted F1 | 0.5861 | 0.4983 | -0.0878 |
| Macro precision | 0.7353 | 0.5951 | -0.1402 |
| Macro recall | 0.6725 | 0.5255 | -0.1471 |

38 of 63 correct became 32 of 63. Validation macro F1 peaked at 0.4594 at epoch
16, against V2's 0.6223. Full artifacts are in `outputs/v3_results/`.

The control-group check is what makes this interpretable. Five classes received
no supplement images at all (`filling_station`, `hairpin_bend_ahead`,
`major_road_ahead`, `side_road_right`, and `school_ahead` with 3). Their mean F1
change was **-0.214**, against **-0.141** for the twelve classes that did receive
data. Untouched classes degraded *more* than supplemented ones. Had the
supplement been merely unhelpful, the control group would have stayed near zero.
It did not, so the damage is not confined to the classes that received images:
the shared backbone representation itself was altered.

The cause is the resolution mismatch. The supplement is 50x50 crops upscaled to
224, and it formed 73% of the training pool; the original pool is 512x512
downscaled to 224, and the test split is 512x512. Training accuracy reached 0.985
while validation stalled near 0.39 — the model fitted the upscaled crops without
that transferring to sharp photographs. This is structurally the same failure as
Baseline V1, which scored 14.53% after training on 32x32 augmented images.

The conclusion is that image *count* was never the bottleneck; image
*characteristics* are.

**Baseline V2 remains the reference model.** This dataset is not usable for
training as-is. Two approaches might work and are future work, not done: two-stage
fine-tuning, pre-training on the supplement then fine-tuning on the 287 Dataset B
images alone; or a higher-resolution source, since every real crop here is 50x50.

Caveat on the per-class figures in `outputs/v3_results/per_class_metrics.csv`:
eleven of seventeen classes have three or fewer test images, so individual
per-class deltas are extremely noisy and must not be quoted on their own. The
aggregate direction is consistent across every headline metric and both groups.

### V4 (stronger augmentation): also did not beat V2

A second attempt to improve on Baseline V2, this time changing only how the same
287 Dataset B images are augmented. No HF supplement, no Dataset A, same
MobileNetV3-Small, same 30 epochs, seed 42, batch size, learning rate, cosine
schedule, class-weighted cross-entropy with label smoothing, and the same
selection on validation macro F1. `scripts/run_v4_strong_aug.py`.

Changed: random resized crop (scale 0.70-1.0), rotation to 15 degrees, wider
colour jitter (0.30/0.30/0.20, hue 0.02), random erasing at p=0.25, and weight
decay raised from 0.0001 to 0.0005. Flips stay disabled, since mirroring changes
a sign's meaning. Test-time augmentation over five deterministic
semantics-preserving views was also evaluated (`scripts/eval_tta.py`).

Selection on validation, 62 images, before the test split was read:

| Config | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: |
| V2 plain | 0.5484 | 0.6223 |
| V2 + TTA | 0.5645 | 0.6254 |
| **V4 plain (selected)** | 0.5484 | **0.6493** |
| V4 + TTA | 0.5484 | 0.6454 |

V4 plain won on validation, so it alone was evaluated once on the locked test
split (`outputs/v4_results/`):

| Metric (locked v2_test, 63 images) | V2 | V4 | Delta |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | 0.6032 | 0.5079 | -0.0952 |
| Macro F1 | 0.6756 | 0.5405 | -0.1351 |
| Weighted F1 | 0.5861 | 0.4860 | -0.1001 |

38 of 63 correct became 32 of 63. Two classes improved, five were unchanged, ten
got worse.

**The finding worth keeping is the validation/test disagreement.** V4 beat V2 on
validation by +0.027 macro F1 and lost on test by -0.135. The selection protocol
was followed correctly — validation-only selection, the test split read exactly
once, by one configuration — and it still chose the worse model. A +0.027 margin
on 62 images across 17 classes is about two images, which was never enough to
survive transfer to a different 63-image sample. Had all four configurations been
run against the test split, the apparent winner would have been chosen by noise.
This is an argument for a larger evaluation set, not against the protocol: the
discipline is what makes the disagreement visible and interpretable instead of
producing a lucky pick presented as a result.

Test-time augmentation is not worth adopting on this evidence. It moved macro F1
by +0.0031 on V2 and -0.0039 on V4 — inconsistent in sign, and smaller than one
image on a 62-image split.

**Baseline V2 remains the reference model.** No configuration points at V4.

### V5 (5-fold cross-validation ensemble): closer, still short of V2

The final experiment, and the closest any attempt came. `scripts/run_v5_kfold_ensemble.py`
merges `v2_train.csv` (287) and `v2_validation.csv` (62) into one 349-image
Dataset B pool, splits it into five stratified folds seeded at 42, trains one
model per fold on the other four fifths (279-280 images each), and averages the
five softmax vectors at inference. The locked test split was read once, by the
assembled ensemble.

Two choices were made in reaction to how V4 went wrong. V2's original
augmentation was used rather than V4's stronger set, because V2's recipe is the
one with evidence of transferring to test. And each fold ran a fixed 21 epochs
with no validation pass and no checkpoint selection, since per-fold
validation-based selection is the exact mechanism that picked the worse model in
V4 — and merging validation into the pool leaves nothing to select on anyway.

| Metric (locked v2_test, 63 images) | V2 | V5 ensemble | Delta |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | 0.6032 | 0.5714 | -0.0317 |
| Macro F1 | 0.6756 | 0.6049 | -0.0707 |

38 of 63 correct became 36 of 63.

**Ensembling worked; the members were too weak.** This is the most technically
interesting number in the whole sequence of experiments:

| | Top-1 | Macro F1 |
| --- | ---: | ---: |
| fold 0 | 0.5079 | 0.5433 |
| fold 1 | 0.4762 | 0.4917 |
| fold 2 | 0.4921 | 0.5320 |
| fold 3 | 0.4921 | 0.4727 |
| fold 4 | 0.5873 | 0.6199 |
| mean of folds | 0.5111 | 0.5319 |
| best single fold | 0.5873 | 0.6199 |
| **ensemble** | **0.5714** | **0.6049** |

Against the mean of its members the ensemble gained **+0.0603 top-1 and +0.0730
macro F1** — a large, textbook ensembling lift, and evidence the folds were not
simply making correlated errors. Against the best single member it lost 0.0159
top-1. Every member scored below V2's 0.6032, so averaging five weak models
reached 0.5714 and no further.

The fold-to-fold spread is itself a result: top-1 ranged from 0.4762 to 0.5873, an
eleven-point swing caused by nothing but which 70 images were held out. At this
pool size, fold assignment moves test accuracy more than any intervention
attempted in this project.

Per class the profile is the most balanced of any attempt — **6 improved, 6
unchanged, 5 worse**. Gains concentrate in the larger classes
(`pedestrian_crossing_ahead` +0.1146 at support 10, `maximum_speed_limit_30_km_h`
+0.1333, and `school_ahead` +0.0364, its only improvement across all
experiments). Losses concentrate in the smallest (`gap_in_median` -0.6667 at
support 2, `side_road_right` -0.4667 at support 3), where a single image moves F1
by a large fraction.

**Honest caveat on the epoch count.** The fixed 21 epochs is inherited from V2's
best epoch, which was itself selected on a validation set that is inside this
training pool. That is a mild optimism leak: 21 is not an independent choice. It
is far better than tuning against the test split, and it was the most defensible
number available without a held-out set, but it is not clean and should not be
described as such.

**Baseline V2 remains the reference model.** No configuration points at V5.

### Manual review and five-class baseline

The proposed overlaps were `filling_station`, `give_way`, `no_entry`,
`no_right_turn`, `road_hump`, and `y_junction`. The 147-photo review CSV is
`outputs/manual_review/dataset_b_six_class_review.csv`; 15 class-specific
contact sheets are under `outputs/manual_review/contact_sheets/`. Reviewers edit
only `review_status`, `review_notes`, and, for a `relabel` decision,
`review_label`. Valid statuses are `pending`, `approved`, `rejected`, and
`relabel`; relabel targets are restricted to the same six-class vocabulary.

Apply a completed review with:

```bash
python scripts/apply_manual_review.py
```

The command rejects pending, missing, duplicated, path-altered, or otherwise
malformed rows. Only after every decision is complete does it create
`outputs/manifests/dataset_b_external_test.csv` and its JSON decision summary.
Those files are external-test artifacts and must never be used for model
selection.

The Dataset A training pool contains references only—no images are copied. It
records exact hashes, source-template IDs, and augmentation generations. All six
selected classes have exactly one source template, so there is no independent
Dataset A validation set. Three exact-content groups also cross class labels,
which further demonstrates why an ordinary image-level split is invalid.

Manual review rejected all 16 proposed Y-junction photographs because they show
side-road junctions or no reliably visible sign. Dataset A class 41 was verified
as a correctly labelled true Y-junction and remains unchanged globally. The
experiment was relocked to five classes before training.

The precommitted five-class protocol used 30 fixed training epochs on all 1,420
selected Dataset A references, with training loss used only as an optimization
diagnostic. Its configuration is `configs/five_class_baseline.yaml`, and the
scientific rationale is in
`outputs/experiment_protocol/five_class_baseline_protocol.md`. The run saved
`last.pt` only and did not create a validation-selected `best.pt`.

Implemented dataset modes are:

- `directory`: immediate child directories are class labels;
- `split_directory`: `train`, `val`/`validation`, and `test` directories contain
  class directories;
- `csv_manifest`: configurable image-path, label, and optional split columns;
- `json_manifest`: a list of sample objects or an object containing `samples`;
- `auto`: infer one of the modes above from an explicit manifest or directory
  structure.

The inspector additionally reports a flat directory as `flat`, but flat,
unlabelled data cannot be used to train the supervised adapter. Labels are sorted
alphabetically unless a contiguous, zero-based JSON class mapping is supplied.

Example directory layouts:

```text
dataset/                    dataset/
├── no_entry/               ├── train/
│   └── 001.jpg             │   ├── no_entry/
└── stop/                   │   └── stop/
    └── 002.png             ├── val/
                            └── test/
```

Example CSV manifest:

```csv
image_path,label,split
images/001.jpg,stop,train
images/002.jpg,no_entry,validation
```

Paths may be relative or absolute, but every image must resolve inside the
configured dataset root. Supported extensions, columns, RGB conversion,
requested split, corrupt-image policy, unknown-label policy, metadata return,
and mapping path are configurable in YAML.

### Read-only inspection

From the repository root:

```bash
python scripts/inspect_dataset.py --config configs/config.yaml
python scripts/inspect_dataset.py --config configs/config.yaml \
  --dataset-root D:/datasets/indian-road-signs \
  --output-json outputs/dataset-report.json \
  --output-csv outputs/dataset-report.csv
```

Use `--mode`, `--manifest`, `--max-images`, or `--no-verify-images` when needed.
Warnings are written separately from fatal errors, and fatal configuration
errors return exit status 2.

Inspection reports layout, counts, classes, splits, imbalance, dimensions,
source colour modes, unreadable images, duplicate paths, empty class folders,
unsupported files, and likely annotation files. Images are decoded one at a
time; `--max-images` can bound validation work.

### Generated splits

`data.split.stratified_split` produces deterministic train/validation/test
assignments without copying images. `save_split_manifest` writes only relative
paths, labels, and split names:

```csv
image_path,label,split
stop/001.jpg,stop,train
```

Generated manifests must be outside the source dataset. The configured default
is `outputs/manifests`. The implementation rejects attempts to write a generated
manifest into the dataset root.

The adapter and inspection tools never rename, move, copy, rewrite, or delete
source images. Incremental registration images must also remain separate from
the immutable base dataset; only derived prototypes and JSON metadata belong in
`artifacts/prototypes/`.

## Setup and verification

Python 3.11 or 3.12 is required. Select the PyTorch wheel appropriate for the target CPU
or CUDA runtime, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Verification commands:

```bash
black --check .
isort --check-only .
mypy models data training evaluation utils inference
pytest
```

The checked-in experiment configuration sets `model.pretrained: true` so the
first real baseline fine-tunes ImageNet-initialized MobileNetV3-Small weights.
On first use, timm may need network access to obtain those weights. Unit and
synthetic smoke tests explicitly use `pretrained=False`, so verification remains
offline and deterministic. Automatic device selection uses CUDA when available
and CPU otherwise.

## Base-model training infrastructure

The closed-set base model uses the existing timm-backed
`mobilenetv3_small_100` feature extractor and linear `RoadSignClassifier` head.
`model.num_classes: auto` derives the output count from the training class
mapping; an incompatible explicit count is rejected. Enabling
`model.pretrained` may cause timm to download weights. It is enabled in the real
baseline YAML but is never enabled by tests.

Reproducibility covers Python `random`, NumPy, PyTorch CPU, every available CUDA
device, deterministic algorithms, deterministic cuDNN behavior, and seeded
DataLoader workers. `device.type` accepts `auto`, `cpu`, or `cuda`; an explicit
unavailable CUDA request fails rather than silently using CPU.

Training augmentation is intentionally conservative because mirroring or large
rotations can change traffic-sign meaning. Defaults are no horizontal flip, no
vertical flip, at most 5 degrees of affine rotation, and mild brightness and
contrast jitter. Evaluation uses deterministic resize, tensor conversion, and
ImageNet mean/standard-deviation normalization. The same preprocessing
configuration is stored in checkpoints and reconstructed during evaluation.

Train with CLI values overriding YAML values:

```bash
python scripts/train.py --config configs/config.yaml
python scripts/train.py --config configs/config.yaml \
  --resume outputs/checkpoints/<run_id>/last.pt \
  --epochs 40
```

The command validates the dataset root before importing or constructing the
model. It does not generate a dataset. Training requires predefined split
directories or a CSV/JSON generated manifest; using one unsplit class directory
for both training and validation is rejected as leakage. Resolved image paths are
also checked for overlap across train, validation, and test splits.

Evaluate a measured checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/<run_id>/best.pt
```

### Training outputs and checkpoints

Every execution receives a microsecond-resolution identifier such as
`20260830_153500_123456_mobilenetv3_small_100`. Existing run directories are
never overwritten.

```text
outputs/
├── checkpoints/<run_id>/
│   ├── best.pt             # only when validation-based selection is configured
│   └── last.pt             # fixed-epoch six-class baseline uses this only
├── runs/<run_id>/
│   ├── environment.json
│   ├── resolved_config.yaml
│   ├── history.csv
│   └── history.json
└── evaluations/<run_id>/
    ├── metrics.json
    ├── per_class_metrics.csv
    ├── confusion_matrix.csv
    └── predictions.csv
```

Checkpoints contain model, optimizer, and optional scheduler state; completed
epoch; optional best validation accuracy; exact class mapping; model and
preprocessing configuration; random seed; training configuration; UTC timestamp;
and project metadata. Saves use atomic replacement. Resume validates both
architecture and class mapping and continues at the next epoch.

Measured history records train/validation loss and top-1 accuracy, learning rate,
and elapsed seconds. Evaluation reports top-1 accuracy; macro and weighted
precision, recall, and F1; per-class precision, recall, F1, and support; a
confusion matrix; and individual predictions with maximum-softmax confidence.

Maximum softmax probability is only closed-set classifier confidence. It is not
an unknown-sign or open-set confidence score; open-set scoring belongs to the
next phase.

### First measured five-class baseline

The fixed run `20260831_162238_646352_mobilenetv3_small_100` completed all 30
epochs on Dataset A and was then evaluated exactly once on the 117 manually
approved Dataset B photographs. It measured 14.5299% top-1 accuracy, 6.5393%
macro F1, and 5.1905% weighted F1. The poor result is retained without tuning or
retraining. The classifier predicted most real-world images as `give_way`, which
demonstrates the severe domain gap between single-template augmented 32×32
training images and independent photographs.

The final training accuracy was 99.1549%, but it is an optimization diagnostic,
not a generalization estimate. Dataset A's single-template-per-class structure
still cannot support an independent internal validation partition.

### Open-set decision and few-shot registration

`inference/` is what allows a sign that was never trained on to be recognized
rather than forced into the nearest trained class. Nothing here retrains
anything: registering a sign embeds a few reference photographs with the frozen
backbone, averages the normalized embeddings into a unit-norm prototype, and
stores it in the separate NPZ registry.

`inference/decision.py` holds the policy and consumes only already-computed
numbers, so it is testable without torch, a checkpoint, or a dataset. One image
produces two independent pieces of evidence, and the decision records which rule
fired:

1. base-classifier maximum softmax probability, and
2. cosine similarity to the nearest registered prototype.

Three arbitration strategies are configurable. `conservative` is the configured
default: when both the base classifier and a registered prototype clear their
thresholds, the input is rejected as ambiguous rather than assigned to either.
The two raw scores have never been calibrated against each other, so a conflict
is not something this system can currently arbitrate honestly; the reasoning is
recorded in `outputs/open_set/open_set_protocol.md`. `classifier_first` accepts
a confident base class, then consults the registry, then returns `unknown`.
`prototype_priority` reverses the first two steps, which suits a sign the base
model is known to misread. A registered class is accepted only when its
similarity clears `prototype_similarity_threshold` and, when a runner-up exists,
beats it by at least `prototype_margin`. Each decision also reports the
equivalent nearest-prototype L2 distance, since open-set work often thresholds a
distance rather than a similarity.

`inference/registration.py` applies registration policy. It enforces the
reference-count bounds, refuses to write the registry inside any configured
protected dataset root, and measures the mean and minimum pairwise cosine
similarity of the reference set. A set whose mean falls below `min_coherence` is
rejected, because photographs of several different signs would otherwise average
into a prototype that matches nothing. The measured coherence is stored in the
prototype metadata.

`inference/pipeline.py` provides `OpenSetRecognizer`, rebuilt from a training
checkpoint so that the class ordering and input normalization cannot disagree
with training. One forward pass yields both the probabilities and the embedding,
and every parameter is explicitly frozen after loading.

Four invariants are enforced at runtime rather than by convention:

1. **The frozen model cannot drift.** `model_state_sha256()` digests every
   tensor name, dtype, shape and value. The registrar samples it before and
   after each registration and refuses the result if it changed.
2. **Checkpoint provenance is recorded.** The checkpoint file's SHA-256 is
   computed on load, exposed through `info()`, and written into every
   prototype's metadata, so a stored prototype always names the model that
   produced it. `from_checkpoint(..., expected_sha256=...)` pins it.
3. **Incremental labels cannot collide with base classes.** Registering a name
   the frozen classifier already owns is refused, because one label would
   otherwise have two independent sources of truth.
4. **The registry cannot be written into a dataset root.** Checked on
   construction and again on every save.

```python
from inference.decision import OpenSetThresholds
from inference.pipeline import OpenSetRecognizer

recognizer = OpenSetRecognizer.from_checkpoint(
    "outputs/checkpoints/<run_id>/best.pt",
    registry_path="artifacts/prototypes/registry.npz",
    thresholds=OpenSetThresholds.from_config(config["open_set"]),
)
recognizer.register_sign_from_paths("school_ahead", reference_paths)
decision = recognizer.predict_paths(["query.jpg"])[0]
print(decision.verdict, decision.label, decision.reason)
```

**The thresholds are not calibrated.** `OpenSetThresholds` carries an explicit
`calibrated` flag, defaulting to `false`, and every decision reports it, so a
placeholder can never be presented as a measured value. Calibration needs a
validation split plus classes held out as unknown, and has not been run. No
open-set accuracy, false-unknown rate, or latency figure is claimed anywhere in
this repository.

## Current status

The **Results summary** at the top of this file is the authoritative statement of
project state; this section records scope rather than repeating those numbers.

Built and verified:

- timm-backed MobileNetV3-Small feature extractor and reusable linear head;
- pickle-free NPZ prototype registry with normalized multi-shot prototypes, add,
  overwrite, remove, lookup and cosine search, schema-validated loading;
- dataset adapter covering directory, split-directory, CSV and JSON manifests,
  with path-containment, extension and corrupt-image validation;
- reproducible group-aware splitting, deterministic class mappings, and
  audit tooling for both datasets including duplicate and leakage screening;
- deterministic CPU/CUDA training with conservative transforms, class-weighted
  loss, checkpoint save/resume, and closed-set evaluation reports;
- open-set decision policy returning base class, registered class or unknown,
  with `conservative`, `classifier_first` and `prototype_priority` arbitration,
  an optional runner-up margin, and reported L2 distance;
- few-shot registration with reference-count bounds, measured reference
  coherence, base-class label-collision refusal and protected-root enforcement;
- runtime frozen-model fingerprinting and checkpoint SHA-256 provenance, so
  registration provably cannot alter a model weight;
- `OpenSetRecognizer` rebuilt from a checkpoint, sharing one forward pass between
  closed-set probabilities and the embedding;
- unseen-class acquisition tooling: source discovery, Mapillary metadata
  planning, human-review packaging, and an intake audit that blocks performance
  claims when no defensible unseen class exists;
- 252 tests, mypy clean across 47 source files.

Not yet done:

- calibrated thresholds validated on genuinely unseen-class data; the current
  values come from a leave-one-class-out proxy, not from signs absent from every
  dataset here (see `docs/new_sign_photo_shoot_plan.md`);
- a measured conflict-resolution policy for when base-classifier and prototype
  evidence both qualify; `conservative` currently rejects such cases as ambiguous
  rather than arbitrating between two uncalibrated scores;
- a few-shot/open-set experiment on independently reviewed unseen-class
  photographs, which remains the project's main open question;
- any improvement over the 60.32% baseline; four attempts are documented above;
- FastAPI endpoints and Streamlit UI.

Measured open-set results exist for the leave-one-class-out proxy only
(`outputs/loco_results/`). No latency result, and no result on a sign genuinely
absent from both datasets, is claimed.

General adapter limitations: JSON manifests support only a top-level sample list or
an object containing `samples`; CSV/JSON bounding-box or detection annotations
are discovered but not interpreted; flat directories are inspectable but cannot
provide supervised labels; archives are not extracted; and image validation
depends on the codecs available in OpenCV. Class imbalance is flagged when an
empty class exists or the largest non-empty class has more than 1.5 times the
samples of the smallest.

The Dataset B dHash threshold is a screening heuristic and can produce false
positives or miss transformed duplicates. Candidate labels and the two ambiguous
orientation mappings to Dataset A require visual review before group-aware split
generation or training.
