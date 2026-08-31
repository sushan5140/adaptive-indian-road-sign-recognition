# Six-Class Controlled Baseline Protocol

Status: locked before training and before any Dataset B external-test evaluation.

## Scientific rationale

Dataset A contains 13,971 32×32 files, but each populated class traces to one
original numeric template. The other files are augmentation generations, and
exact duplicates also exist. A random image-level split would place derivatives
of the same template in multiple partitions and cannot measure independent
generalization.

Dataset B contains real-world photographs. Its six exact semantic overlaps are
reserved for one external evaluation after manual review. Dataset B must not be
used to choose epochs, checkpoints, hyperparameters, preprocessing, or class
definitions.

## Classes and review

| Dataset B class | Dataset A ID | Dataset A name |
|---|---:|---|
| filling_station | 55 | Filling station |
| give_way | 0 | Give way |
| no_entry | 1 | No entry |
| no_right_turn | 16 | No right turn |
| road_hump | 30 | Road hump |
| y_junction | 41 | Y-junction |

Every proposed Dataset B photograph receives a stable review ID. A human must
mark it approved, rejected, or relabelled to another one of these six classes.
Pending, malformed, missing, duplicated, or path-altered rows block external
manifest generation. Rejected images are excluded. Approved and valid relabelled
images form the external-test manifest. Perceptual groups remain identified.

## Model selection and stopping

Option 2 is required: fixed training configuration with no held-out Dataset A
validation. All usable six-class Dataset A files form the training pool. Training
loss and training accuracy are optimization diagnostics only. They are not model
selection or generalization estimates.

Training stops after exactly 30 completed epochs unless a non-finite loss or
runtime error aborts the run. The final artifact is `last.pt`. No `best.pt` is
created because no legitimate validation metric exists. Dataset B remains unseen
until this fixed training run is complete.

## Locked configuration

- Python: 3.11
- Framework: PyTorch and timm
- Architecture: `mobilenetv3_small_100`
- Initialization: ImageNet pretrained weights
- Output classes: 6
- Dropout: 0.2
- Input: RGB, resized to 224×224
- Normalization mean: `[0.485, 0.456, 0.406]`
- Normalization standard deviation: `[0.229, 0.224, 0.225]`
- Horizontal flip probability: 0.0
- Maximum rotation: 5 degrees
- Brightness jitter: 0.15
- Contrast jitter: 0.15
- Batch size: 32
- Workers: 0
- Pin memory: false
- Epochs: 30 fixed
- Loss: cross-entropy, label smoothing 0.0
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Scheduler: cosine annealing
- Minimum learning rate: 0.000001
- Seed: 42
- Deterministic algorithms: enabled
- Device for the first baseline: CPU
- Checkpoint: `last.pt` only

The machine-readable configuration is `configs/six_class_baseline.yaml`. These
values must not be silently tuned after viewing Dataset B results.

## External evaluation

After review completion and fixed training, evaluate exactly once on the approved
Dataset B external manifest. Report top-1 accuracy; macro and weighted precision,
recall, and F1; per-class precision, recall, F1, and support; confusion matrix;
and per-image predictions. Dataset B performance is the first independent
real-world measurement and is not a model-selection signal.

## Known limitations

- Dataset A has one source template per class, so internal independent validation
  and uncertainty estimates across source photographs are impossible.
- Dataset A is very low resolution and augmentation-derived.
- Dataset B labels are derived from VQA annotations and require human confirmation.
- The six-class benchmark does not cover the full Indian road-sign vocabulary.
- A single external dataset cannot characterize every geography, device, weather,
  distance, occlusion, or illumination condition.
- Pretrained weights require network availability on first use; no weights are
  downloaded during preparation or verification.

No training or Dataset B evaluation is authorized by this protocol-preparation
phase. Training begins only after the completed review CSV is supplied.
