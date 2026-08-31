# Five-Class Controlled Baseline Protocol

Status: relocked before training and before external Dataset B evaluation.

## Scientific basis

Dataset A contains 32×32 augmentation-derived images. Every selected class
traces to exactly one original template, and its remaining samples are dependent
augmentations or copies. Exact duplicates also exist. Consequently, Dataset A
cannot provide a statistically independent validation set, and an ordinary
image-level random split would create source leakage.

Dataset B consists of independent real-world photographs and is reserved as the
external test set. It is prohibited from training, validation, epoch selection,
hyperparameter selection, preprocessing selection, checkpoint selection, or any
other model-selection decision.

The earlier six-class alignment was manually reviewed. All 16 Dataset B
`y_junction` candidates depicted side-road junction signs or lacked a reliably
visible sign. Dataset A class 41 was separately verified as a correctly labelled
true Y-junction. Therefore, class 41 is excluded only from this baseline; it
remains unchanged in Dataset A and its global metadata.

## Fixed classes and indices

| Model index | Class | Dataset A ID | Dataset A samples | Dataset B external samples |
|---:|---|---:|---:|---:|
| 0 | give_way | 0 | 201 | 18 |
| 1 | no_entry | 1 | 201 | 14 |
| 2 | no_right_turn | 16 | 201 | 11 |
| 3 | road_hump | 30 | 201 | 53 |
| 4 | filling_station | 55 | 616 | 21 |
| | **Total** | | **1,420** | **117** |

Model indices are contiguous classifier outputs. They are not Dataset A IDs;
the original IDs remain columns in the manifests.

## Locked training schedule

- Python 3.11 and PyTorch/timm
- `mobilenetv3_small_100` with ImageNet-pretrained initialization
- five output classes and dropout 0.2
- RGB input resized to 224×224
- ImageNet mean `[0.485, 0.456, 0.406]`
- ImageNet standard deviation `[0.229, 0.224, 0.225]`
- horizontal flip probability 0.0
- affine rotation limited to 5 degrees
- brightness and contrast jitter 0.15
- cross-entropy with label smoothing 0.0
- AdamW, learning rate 0.001, weight decay 0.0001
- cosine scheduler with minimum learning rate 0.000001
- batch size 32, workers 0, pin memory disabled
- CPU device
- seed 42 and deterministic algorithms enabled
- exactly 30 epochs on all 1,420 Dataset A samples

Training loss and accuracy are optimization diagnostics only. No validation
metric exists. Training ends after epoch 30 unless a non-finite loss or runtime
failure aborts it. The predetermined artifact is `last.pt`, which represents the
completed epoch-30 state. No validation-selected `best.pt` may be created.

## External evaluation

Only after the 30 epochs complete and `last.pt` is locked may that checkpoint be
evaluated once on the 117-row five-class Dataset B manifest. The evaluation must
report top-1 accuracy; macro and weighted precision, recall, and F1; per-class
precision, recall, F1, support, and correct counts; a confusion matrix; common
confusion pairs; and per-image predictions.

Maximum softmax probability is closed-set classifier confidence only. It is not
unknown-sign or open-set confidence. No result from Dataset B may cause this run
to be retrained, extended, or reconfigured. Any later improvement is a new
experiment with Dataset B already considered observed.

## Limitations

- Dataset A has one independent template per class and cannot measure internal
  generalization.
- Dataset A is low resolution and augmentation-derived.
- Dataset B labels originate from VQA annotations, although the included rows
  have been manually reviewed.
- The external set is imbalanced and contains only five sign classes.
- One external test set cannot cover all geography, devices, weather, distances,
  occlusions, or illumination conditions.
- ImageNet pretrained weights may require a one-time network download.

Open-set rejection, prototype registration, calibration, incremental inference,
FastAPI, and Streamlit are outside this experiment.
