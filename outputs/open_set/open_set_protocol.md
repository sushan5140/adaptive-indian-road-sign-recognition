# Open-set and few-shot experimental protocol status

## Frozen backbone

The closed-set backbone is the Baseline V2 `best.pt` checkpoint from epoch 21
(one-based), SHA-256
`0f990f21c7f844f5611e91f867740b7f980e851426681c69deb2fefadbea8ff4`.
Every feature-extractor and classifier parameter is frozen. Evaluation uses the
checkpoint's recorded 224-pixel resize and ImageNet normalization; training
augmentation is never applied. Embeddings are L2-normalized and must have 1024
dimensions.

## Registration protocol

One or more independently chosen reference images are embedded by the frozen
pipeline. Each embedding is normalized, their mean is calculated, and the mean
is normalized again before registration in the existing pickle-free NPZ
registry. Registration must not collide with a base-class label or overwrite an
existing incremental class unless an explicit future operation authorizes it.

## Required experimental partitions

For each unseen class, complete perceptual/source groups must be assigned to
exactly one of reference, calibration, or query evaluation. Prototype references
must never appear in query evaluation. Threshold calibration must use neither
query images nor the locked V2 test split. Multiple reference selections and
seeds should be used for 1-shot, 3-shot, and 5-shot conditions when group counts
permit.

## Scoring and calibration

The implemented raw scores are maximum base softmax probability, maximum cosine
similarity to registered prototypes, and normalized nearest-prototype L2
distance. Softmax is not described as an unknown score. Thresholds have no code
defaults and must be selected on future validation/calibration data using known
accuracy, unknown precision/recall/F1, false acceptance/rejection rates, and
AUROC only when the sampling protocol makes it scientifically meaningful.

If both base and incremental thresholds pass, the current uncalibrated policy
conservatively rejects the input as ambiguous. Any other conflict rule must be
separately calibrated and documented.

## Current stop condition

No scientifically defensible unseen-class experiment can be run with the current
data. Dataset B's only non-base candidates are `one_way`, `stop`, and
`y_junction`; all were manually rejected. Dataset A contains only one original
template per populated class and its remaining files are dependent augmentations,
with exact duplicates and cross-label duplicate groups. It cannot provide
independent reference and query photographs.

Accordingly, no thresholds were selected and no few-shot, unknown-detection,
base-retention, latency, or open-set performance metrics were generated. The
locked V2 test split was not used.
