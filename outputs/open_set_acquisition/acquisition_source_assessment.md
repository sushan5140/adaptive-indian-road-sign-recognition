# Unseen-class acquisition source assessment

No data was downloaded during this phase. Any photograph entering the intake must
have a recorded source identifier, URL, licence, and attribution, and must remain
pending until visual review.

## Potential sources

- **Locally captured photographs (preferred):** obtain permission from the
  photographer and record an explicit project-compatible licence or consent. Keep
  bursts, video frames, and multiple crops from one scene under one source identifier.
- **Mapillary:** the official platform documents traffic-sign map-data downloads and
  API access, but use is subject to Mapillary terms and commercial terms. Confirm the
  intended image use, redistribution, attribution, and research rights before
  downloading anything. Do not assume map detections grant unrestricted rights to
  underlying imagery.
  <https://help.mapillary.com/hc/en-us/articles/4407521157138-Downloading-map-data-via-the-Mapillary-web-app>
- **Mapillary Traffic Sign Dataset:** Meta describes a diverse global traffic-sign
  research dataset. Confirm the current dataset-specific licence and access terms,
  and verify Indian sign semantics, before acquisition.
  <https://ai.meta.com/ai-for-good/datasets/mapillary-training-datasets/>
- **GTSRB:** the official benchmark contains real German traffic-sign photographs.
  Its different geography and sign design make it a possible cross-domain diagnostic,
  not the preferred primary Indian unseen-class source. Licence/terms and attribution
  still require review before use.
  <https://benchmark.ini.rub.de/gtsrb_dataset.html>
- **NIT Rourkela Indian traffic-sign dataset write-up:** potentially relevant to the
  target geography, but no image was acquired because the precise download and reuse
  licence must first be confirmed with the publisher.
  <https://www.nitrkl.ac.in/docs/CS/Database/dataset_writeup.pdf>

## Rejected source types

Synthetic renders, isolated sign templates, icons, screenshots of sign catalogues,
generated images, Dataset A augmentations, and the previously rejected Dataset B
`stop` examples are not acceptable independent unseen-class photographs. The
WIRIndic resource is an Indian road-scene text-recognition corpus rather than a
fine-grained road-sign-class dataset, so it is not suitable for this evaluation.

## Dataset A overlap with unseen-class targets (verified 2026-09-04)

The archive held at `Downloads/archive.zip` was inspected read-only and confirmed
to be Dataset A, the same manually supplied Kaggle archive already extracted to
`data/raw/indian_traffic_sign_dataset/`. It is not a new source. Its structure
matches the earlier audit exactly: 13,971 images across 58 numbered class folders
(0-58, with 39 declared in `traffic_sign.csv` but having no image folder), and 44
of those 58 folders hold exactly 201 images each, consistent with one original
template plus 200 generated variants. `unseen_class_data_audit.json` already
records Dataset A as not defensible for independent reference/query evaluation,
and nothing in this inspection changes that.

Two of the current unseen-class targets have a counterpart there:

| Target class | Dataset A class | Note |
| --- | --- | --- |
| `no_left_turn` | 15, "No left turn" | exact semantic match |
| `no_parking` | 21, "No parking" | exact semantic match |
| `maximum_speed_limit_50_km_h` | 18/19 are 90 and 110 km/h | same sign family, different numeral |
| `stop` | none | class 52 is "Bus stop", a different sign |

Baseline V2 was trained on Dataset B only and has never seen Dataset A, so these
overlaps do not contaminate the trained model. They do affect how a result may be
described. If `no_left_turn` or `no_parking` is later registered from independent
photographs and evaluated, the write-up must disclose that a class of the same
identity exists in Dataset A, rather than presenting either as a sign absent from
every dataset in the project. `stop` has no counterpart at all and therefore
remains the cleanest genuinely unseen candidate.

Dataset A carries no licence information of any kind. The archive contains no
README, LICENSE, NOTICE, attribution, or citation file; its only non-image member
is the 1,280-byte `traffic_sign.csv` class map. The provenance table in
`data/README.md` accordingly still records dataset name, source, licence and date
obtained as TBD. This is unresolved, and is a stricter problem than the pending
Mapillary terms question: Mapillary candidates at least carry a documented CC
BY-SA position and per-image attribution strings, whereas Dataset A has neither.
Resolve the original listing's licence before publishing anything derived from it.

## Unseen-class contamination after the Dataset C import (2026-09-04)

Importing the HuggingFace supplement (Dataset C, see README.md) for training
removes most of the scaffolded unseen-class candidates from consideration: a class
the base model trains on is no longer unseen. Two kinds of contamination must be
kept apart.

- **Training contamination.** The class is in Dataset C, now in the training pool.
  Any model trained on it has seen the class. Disqualifying for unseen evidence.
- **Descriptive contamination.** The class exists in Dataset A, which has never
  been trained on. A result stays valid, but the write-up must disclose that a
  class of the same identity is held by the project.

| Scaffolded class | Dataset A | Dataset C (kept) | Status |
| --- | --- | --- | --- |
| `stop` | absent (52 is "Bus stop") | STOP, 34 | training-contaminated |
| `no_left_turn` | class 15 | LEFT_TURN_PROHIBITED, 33 | training-contaminated + disclosure |
| `no_parking` | class 21 | NO_PARKING, 4 | training-contaminated + disclosure |
| `maximum_speed_limit_50_km_h` | absent (A has 90, 110) | SPEED_LIMIT_50, 136 | training-contaminated |
| `roundabout_ahead` | class 44 "Roundabout" | ROUNDABOUT, 14 | training-contaminated + disclosure |
| `bus_stop` | class 52 "Bus stop" | absent | clean for training; disclosure only |

`bus_stop` is the only one of the six free of training contamination and is
therefore the primary target for the photo shoot in
`docs/new_sign_photo_shoot_plan.md`. It is not clean in the absolute sense:
Dataset A carries a "Bus stop" class, so an eventual result must state that while
noting Dataset A was never trained on.

`stop` was previously recorded here as the cleanest candidate, on the grounds that
it has no Dataset A counterpart. Dataset C contains 34 kept `STOP` crops, so that
assessment is superseded.

### Consequence for the frozen Mapillary plans

`stop_pixel_acquisition_plan_20260904_r01.csv` and
`no_left_turn_pixel_acquisition_plan_20260904_r01.csv` no longer serve the unseen
purpose they were frozen for. They are left unchanged, all rows still
`pixel_download_authorized: no` and
`terms_status: blocked_pending_manual_logged_in_terms_confirmation`. If the
logged-in Mapillary terms are ever resolved they could supply extra real training
images for those classes, but that is optional work, not a blocker on the
open-set claim.

## Independence protocol

Treat photographs as dependent when they share an original frame, burst, video,
photographer scene, source post, or perceptual duplicate group. Each class needs at
least 15 approved independent groups before partitioning: up to 5 reference groups,
at least 5 separate calibration groups, and at least 5 separate query groups. The
target is 30–50 approved real photographs per class to reduce instability.
