# Baseline V2 Dataset B Review Protocol

## Scientific status

Baseline V1 remains immutable. It trained MobileNetV3-Small on augmentation-derived
Dataset A template imagery and measured a severe domain gap on a manually reviewed
real-world Dataset B subset. Baseline V2 is a separate development experiment that
will train on independent real-world Dataset B photographs.

Dataset B was observed during V1 development and evaluation. It is therefore not a
pristine unseen external benchmark for V2, and eventual V1 and V2 scores will not be
presented as directly equivalent benchmark results.

## Candidate construction

The existing audit selected 530 unique photographs in 20 candidate classes. Its
conservative semantic filters require a direct sign-identity question, an explicitly
supported answer alias, minimum candidate viability, and no conflicting labels within
a perceptual duplicate group. These filters make a label plausible but do not replace
visual human verification.

The V2 review manifest carries forward a previous decision only when both
`source_image_id` and `proposed_class` match exactly. Only `approved` and `rejected`
decisions are reused. Every other candidate remains `pending`; no new VQA-derived
label is automatically approved.

## Human review instructions

Review only rows whose `review_status` is `pending`, using the pending-only contact
sheets in `outputs/v2_review/contact_sheets/` and the original full-resolution image
when a thumbnail is ambiguous.

- Set `review_status` to `approved` when the visible sign matches `proposed_class`.
- Set it to `rejected` when it does not provide a reliable classification example.
- Use `relabel` and `review_label` only when the visible class is unambiguous and the
  intended review-application code explicitly supports that target.
- Record concise reasoning in `review_notes` for rejected or relabeled samples.
- Do not edit protected/source columns or add duplicate source images.

## Retention and leakage rules

After review, a class may be retained only with at least 10 approved unique photos
and at least 8 independent approved `perceptual_group_id` values; 15 photos are
preferred where available. No replacement images may be invented.

If review becomes complete, splitting will use leakage groups—not individual images—
with deterministic seed 42. Any later exact-hash, filename-family, or visual evidence
of dependency must merge the affected groups and be recorded before splitting.

## Current status

The V2 manual review is complete: 412 candidates are approved, 118 are rejected,
and none remain pending or relabeled. Seventeen classes satisfy the fixed viability
rule; `one_way`, `stop`, and `y_junction` are excluded because they have no approved
photos or approved perceptual groups.

Deterministic seed-42 group-safe train, validation, and test manifests now exist.
`final_review_summary.json` and `v2_split_summary.json` are the authoritative measured
records for this phase. Baseline V2 training and evaluation have not started.
