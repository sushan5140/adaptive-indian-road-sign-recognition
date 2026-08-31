# Repository Instructions

## Scope and invariants

- Preserve the separation between the immutable base training dataset and the incremental prototype registry.
- Never copy incremental examples into the base dataset and never retrain the full base classifier during sign registration.
- Keep dataset-dependent behavior configurable and documented. Do not download datasets or pretrained weights implicitly.
- Do not claim model accuracy or other experimental results without recorded experiments.
- Mark unfinished, dataset-dependent, or calibration-dependent behavior explicitly.

## Engineering standards

- Support Python 3.11, CPU, and CUDA without hard-coding a device.
- Add type hints and docstrings to public classes and functions.
- Use structured logging rather than `print` in application code.
- Validate external inputs and raise specific, informative exceptions.
- Keep random seeds configurable and use the shared reproducibility utilities once implemented.
- Store prototype metadata without pickle or executable serialization formats.

## Repository workflow

- Read the exact source before editing; repository-memory or graph summaries are navigation aids only.
- Preserve user changes and avoid destructive Git or filesystem operations.
- Keep changes within the requested phase. Training, API, and Streamlit work are intentionally deferred.
- Run `black --check .`, `isort --check-only .`, `mypy models`, and `pytest` before completion.
- Add focused tests for behavior, validation, persistence, and failure cases whenever implementation changes.
