# New-sign photo shoot plan

Goal: collect real, independent photographs of signs the model has never
seen in any form — not held out from training like the LOCO experiment, but
genuinely absent from Dataset A and Dataset B. This is the one input the
open-set claim still needs.

## Pick two sign types, not one

The leave-one-class-out results found a real pattern worth testing directly:
the open-set mechanism works well when a new sign looks nothing like the 17
known classes, and struggles when it resembles an existing visual family.
Photographing one sign of each kind repeats that comparison with real
evidence instead of a proxy.

- **Visually distinct sign** — recommended: **Stop** (unique octagon, red,
  single word) or **No Parking** (circular, blue/red ring). Nothing in the
  current 17 classes shares this shape or color pattern, so this is the
  "should be easy" case — expect something closer to the `filling_station`
  result (93.75% recognition, near-zero false accepts).
- **Visually similar sign** — recommended: a triangular warning sign such as
  **Men at Work**, **Cattle**, or **Falling Rocks**. These share the same
  shape, color, and "figure inside a red-bordered triangle" pattern as
  `school_ahead`, `pedestrian_crossing_ahead`, and `road_hump` — the cluster
  that already confuses the base classifier. Expect something closer to the
  `school_ahead` result (weak separation), which is itself a useful,
  reportable finding rather than a failure.

## How many photos, and how to split them

**15–20 different physical locations per sign** (not 15 photos of the same
sign, and not crops/zooms of one photo — each must be a distinct real-world
instance, the way Dataset B's `perceptual_group_id` audit already enforces
for the existing 17 classes). More than 15 gives margin: the manual reviews
already run on this project (`v2_class_viability.csv`) show 10–40% of
candidate photos typically get rejected on second look (blur, ambiguous
framing, wrong sign entirely).

Split each sign's photos the same way `leave_one_class_out_eval.py` already
splits registration from evaluation:

| Purpose | Count | Notes |
|---|---|---|
| Registration (teach) | 5 | These become the prototype. Pick the 5 clearest, most "typical" shots. |
| Threshold tuning | 3–5 | Held out, used only if thresholds need re-calibrating against a real negative/positive pair instead of the LOCO proxy. |
| Test (final, locked) | 5–10 | Touched once, at the end, to report the real recognition rate. Never used to pick anything. |

## Shooting checklist, per photo

- One physical sign instance per photo — never two different sign types in
  frame (Dataset B's audit specifically flagged and excluded ambiguous
  multi-sign photos; don't reintroduce that problem).
- Vary distance: some close (sign fills half the frame), some at a normal
  walking/driving distance (sign is a smaller part of a street scene).
- Vary angle: mostly front-on/slightly oblique, since the training data
  disables horizontal flip augmentation deliberately (mirroring changes
  sign meaning) — extreme side angles won't match how the model was trained
  to see signs.
- Vary lighting/time of day if practical (overcast, bright sun, dusk) —
  even 2–3 different lighting conditions beats all-identical light.
- Shoot at native phone resolution, JPEG, no heavy digital zoom (the
  pipeline resizes to 224×224 regardless, but a zoomed/cropped source loses
  real detail a genuine photo would have).
- Different physical locations across a city/area, not the same sign
  photographed from 15 slightly different spots.

## File naming and manifest, to match the existing pipeline

Name files so they're easy to turn into a manifest CSV like the ones
already in `outputs/manifests/` (`image_path,class_name,split`):

```
stop_001.jpg, stop_002.jpg, ... stop_018.jpg
men_at_work_001.jpg, men_at_work_002.jpg, ... men_at_work_018.jpg
```

Keep a simple log while shooting — even a plain text line per photo is
enough — recording roughly where/when each was taken. That's what let this
project's existing audits (`dataset_b_audit`) catch near-duplicates and
build the group-aware split; the same discipline here means the eventual
manifest can assign locations to train/tune/test without any single
physical sign leaking across the split.

## What happens after the photos exist

1. Run the same kind of manual review this project already does for
   Dataset B (`scripts/apply_manual_review.py` pattern) — reject blurry,
   ambiguous, or multi-sign photos before anything else touches them.
2. Register each sign from its 5 teaching photos via
   `OpenSetRecognizer.register_sign_from_paths`.
3. Evaluate on the locked test set — this is the real version of what
   `leave_one_class_out_eval.py` approximated. No retraining needed; the
   existing V2 checkpoint's backbone is used as-is.
4. Compare the two signs' recognition/false-accept rates against each
   other and against the LOCO proxy numbers (`school_ahead` 42.4%/22.8%,
   `filling_station` 31.3%/0%) to see whether the proxy under- or
   over-estimated real-world performance.
