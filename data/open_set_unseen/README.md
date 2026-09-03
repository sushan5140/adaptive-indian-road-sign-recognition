# Unseen-class data intake

This directory is a quarantined intake area for real photographs of road signs that
are outside the frozen 17-class Baseline V2 taxonomy. It is not part of the base
training dataset. Nothing here may be used in an experiment until a human has
reviewed it and the reference, calibration, and query groups have been locked.

## Folder contract

- `raw/<class>/`: newly acquired, unmodified real photographs awaiting audit.
- `review/`: source metadata and generated human-review manifests.
- `approved/<class>/`: reserved for a later, explicit post-review export. The audit
  command never copies, moves, or approves images.

Accepted image extensions are configured in `configs/open_set_unseen.yaml`. Put each
photograph under its proposed class directory and add one row to
`review/source_metadata.csv`. `relative_path` is relative to `raw/`. Record a stable
source identifier, original URL, licence, and attribution. Images with unclear reuse
rights must not be added.

Run:

```powershell
python scripts/audit_unseen_dataset.py
```

The command detects corrupt files, exact duplicates, difference-hash near-duplicate
candidates, cross-label group conflicts, and repeated source identifiers. Every new
row remains `pending`; there is no automatic approval. Reviewers may edit only
`review_status`, `review_label`, `rejection_reason`, and `review_notes` in the CSV.

An operational class minimum is 15 independent source/perceptual groups: up to five
prototype references, at least five threshold-calibration groups, and at least five
query groups. The acquisition target is 30–50 real photographs per class. Do not
partition a class merely because it reaches 15 files: human approval and group
independence are both required.
