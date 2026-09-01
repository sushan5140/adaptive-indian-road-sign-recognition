# Baseline V1 vs Baseline V2

| Item | Baseline V1 | Baseline V2 |
|---|---:|---:|
| Data design | Dataset A template/augmentation-derived training; external real-world evaluation | Real-world, manually reviewed, group-safe splits |
| Classes | 5 | 17 |
| Test accuracy | 14.53% (external Dataset B) | 60.32% |
| Test macro F1 | 6.54% | 67.56% |

V1 showed a severe synthetic/template-to-real-world domain gap. V2 uses 17 real-world classes and a validation-selected checkpoint, with the locked test split evaluated once after training. These are not directly equivalent benchmark scores because their datasets, class sets, and experimental designs differ.
