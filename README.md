# Adaptive Indian Road Sign Recognition Using Open-Set and Few-Shot Learning

This internship project builds a computer-vision system that recognizes Indian
road signs while allowing new sign classes to be registered from a few examples.
The base dataset remains unchanged, and registering a sign never retrains the
full base classifier.

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
├── scripts/                 # Future entry-point scripts
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

Python 3.11 is required. Select the PyTorch wheel appropriate for the target CPU
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

Two arbitration strategies are configurable. `classifier_first` (the default)
accepts a base class whose confidence clears `base_confidence_threshold`, then
consults the registry, then returns `unknown`. `prototype_priority` reverses the
first two steps, which is appropriate when an operator has deliberately
registered a sign the base model is known to misread. A registered class is
accepted only when its similarity clears `prototype_similarity_threshold` and,
when a runner-up exists, beats it by at least `prototype_margin`.

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
and every parameter is explicitly frozen after loading, so registration is
structurally incapable of updating a weight.

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

Implemented:

- timm-backed MobileNetV3-Small feature extractor;
- reusable supervised classification head;
- normalized multi-shot prototypes;
- add, overwrite, remove, lookup, and cosine search operations;
- atomic compressed NPZ persistence with Unicode labels and JSON metadata;
- pickle-disabled loading with schema and shape validation;
- comprehensive prototype-registry unit tests;
- read-only directory, split-directory, flat-directory, and manifest inspection;
- configurable CSV/JSON and directory dataset adapter;
- deterministic class mappings and optional per-sample metadata;
- path-containment, missing-file, extension, and corrupt-image validation;
- reproducible stratified splitting and external generated manifests;
- temporary-fixture dataset tests and configuration foundation.
- deterministic CPU/CUDA training infrastructure and conservative transforms;
- AdamW/SGD, cross-entropy, cosine/plateau/disabled scheduler factories;
- epoch training/validation, best/last checkpoints, resume, and measured history;
- closed-set sklearn metrics and CSV/JSON prediction reports;
- dependency-aware unit and synthetic infrastructure smoke tests.
- reproducible VQA schema/image audit, conservative one-photo/one-label derivation,
  duplicate-group screening, and Dataset A/B semantic alignment.
- deterministic six-class manual-review manifest and class contact sheets;
- strict review application with protected-field and decision validation;
- reference-only six-class Dataset A pool with hash/template/augmentation groups;
- fixed-epoch, no-validation training mode that saves `last.pt` without
  manufacturing a validation-selected `best.pt`;
- locked six-class baseline configuration and pre-training protocol.
- relocked five-class manifests and contiguous model mapping;
- first fixed 30-epoch measured baseline and one-shot external evaluation;
- per-image external predictions retaining stable manual-review IDs.

- open-set decision policy returning base class, registered class, or unknown,
  with the arbitrating rule recorded in every decision;
- `classifier_first` and `prototype_priority` arbitration strategies and an
  optional runner-up margin test;
- few-shot registration with reference-count bounds, measured reference
  coherence, and protected-root enforcement;
- `OpenSetRecognizer` rebuilt from a checkpoint, sharing one forward pass
  between closed-set probabilities and the embedding;
- unit tests for the decision policy and registration, plus an end-to-end
  synthetic pipeline test asserting that registration changes no model weight.

Not yet implemented:

- threshold calibration on held-out validation and out-of-distribution data;
- any measured open-set result;
- follow-up experiment addressing the measured domain gap;
- FastAPI endpoints and Streamlit UI.

Thresholds in the YAML file are placeholders, not measured values. The measured
accuracy above is closed-set five-class performance only; no latency or open-set
performance result is claimed.

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
