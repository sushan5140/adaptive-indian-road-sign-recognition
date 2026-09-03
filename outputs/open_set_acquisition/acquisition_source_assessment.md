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

## Independence protocol

Treat photographs as dependent when they share an original frame, burst, video,
photographer scene, source post, or perceptual duplicate group. Each class needs at
least 15 approved independent groups before partitioning: up to 5 reference groups,
at least 5 separate calibration groups, and at least 5 separate query groups. The
target is 30–50 approved real photographs per class to reduce instability.
