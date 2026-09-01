# Baseline V1 vs Baseline V2

| Item | V1 | V2 |
|---|---:|---:|
| Classes | 5 | 17 |
| Test samples | 117 | 63 |
| Top-1 accuracy | 0.145299 | 0.603175 |
| Macro F1 | 0.065393 | 0.675581 |

The datasets and class sets differ, so this is descriptive rather than a controlled head-to-head comparison. V2 used validation macro F1 for checkpoint selection and evaluated the locked test split once afterward.
