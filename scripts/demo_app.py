"""Local Gradio demo for base, unknown, and few-shot sign recognition.

The interface is intentionally thin: model construction, preprocessing,
classification, prototype search, open-set arbitration, and registration all
remain in :mod:`inference.pipeline`, :mod:`inference.decision`, and
:mod:`inference.registration`.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import gradio as gr
import numpy as np
import numpy.typing as npt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.decision import OpenSetDecision, OpenSetThresholds, Verdict  # noqa: E402
from inference.pipeline import InferenceError, OpenSetRecognizer  # noqa: E402
from inference.registration import (  # noqa: E402
    RegistrationError,
    RegistrationPolicy,
)
from utils.config import (  # noqa: E402
    ConfigurationError,
    load_yaml_config,
    require_mapping,
)

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "v2_work" / "checkpoints" / "best.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT_URL = (
    "https://github.com/sushan5140/adaptive-indian-road-sign-recognition/"
    "releases/download/v2-checkpoint/best.pt"
)
UNKNOWN_MESSAGE = (
    "This doesn't match any sign I've learned yet — flagged as unseen/new."
)

READABLE_NAMES: dict[str, str] = {
    "filling_station": "Filling Station",
    "gap_in_median": "Gap in Median",
    "give_way": "Give Way",
    "hairpin_bend_ahead": "Hairpin Bend Ahead",
    "left_curve_ahead": "Left Curve Ahead",
    "major_road_ahead": "Major Road Ahead",
    "maximum_speed_limit_30_km_h": "Maximum Speed Limit 30 km/h",
    "maximum_speed_limit_40_km_h": "Maximum Speed Limit 40 km/h",
    "maximum_speed_limit_80_km_h": "Maximum Speed Limit 80 km/h",
    "no_entry": "No Entry",
    "no_right_turn": "No Right Turn",
    "pass_either_side": "Pass Either Side",
    "pedestrian_crossing_ahead": "Pedestrian Crossing Ahead",
    "right_curve_ahead": "Right Curve Ahead",
    "road_hump": "Road Hump",
    "school_ahead": "School Ahead",
    "side_road_right": "Side Road Right",
}


class DemoError(RuntimeError):
    """Raised when the local demo cannot be configured or started safely."""


def readable_name(label: str) -> str:
    """Return a presentation-friendly class name for a model label."""
    return READABLE_NAMES.get(label, label.replace("_", " ").strip().title())


def build_recognizer(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    config_path: str | Path = CONFIG_PATH,
    *,
    device: str = "auto",
) -> OpenSetRecognizer:
    """Build the project's recognizer with the configured open-set policy.

    Args:
        checkpoint_path: V2 checkpoint containing weights and training metadata.
        config_path: Project YAML containing measured thresholds and registration
            policy.
        device: ``auto``, ``cpu``, or ``cuda``.

    Returns:
        A frozen recognizer with an empty, in-memory prototype registry.

    Raises:
        DemoError: If required files or configuration are unavailable.
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise DemoError(
            f"Checkpoint not found at {checkpoint}. Download {CHECKPOINT_URL} "
            "and place it at outputs/v2_work/checkpoints/best.pt."
        )
    try:
        config = load_yaml_config(config_path)
        thresholds = OpenSetThresholds.from_config(require_mapping(config, "open_set"))
        registration_policy = RegistrationPolicy.from_config(
            require_mapping(config, "registration"), project_root=PROJECT_ROOT
        )
        return OpenSetRecognizer.from_checkpoint(
            checkpoint,
            thresholds=thresholds,
            registration_policy=registration_policy,
            device=device,
        )
    except (ConfigurationError, InferenceError, RegistrationError) as error:
        raise DemoError(f"Could not initialize the sign recognizer: {error}") from error


class DemoController:
    """Thread-safe UI adapter around one in-memory open-set recognizer."""

    def __init__(self, recognizer: OpenSetRecognizer) -> None:
        self.recognizer = recognizer
        self._lock = threading.RLock()

    def classify(self, image: np.ndarray | None) -> tuple[dict[str, float], str]:
        """Classify one Gradio RGB image and format the decision for people."""
        if image is None:
            return {}, "Upload or capture an image first."
        try:
            rgb = _as_rgb_uint8(image)
            with self._lock:
                decision = self.recognizer.predict_images([rgb], top_k=3)[0]
        except (InferenceError, ValueError, TypeError) as error:
            return {}, f"Could not classify this image: {error}"
        return _format_decision(decision)

    def register(self, label: str, reference_files: Sequence[Any] | None) -> str:
        """Register one incremental class from 3–5 uploaded image files."""
        cleaned_label = " ".join(str(label).strip().split())
        if not cleaned_label:
            return "Enter a name for the new sign."
        try:
            paths = _file_paths(reference_files)
            if not 3 <= len(paths) <= 5:
                return f"Upload 3–5 reference photos; received {len(paths)}."
            with self._lock:
                result = self.recognizer.register_sign_from_paths(
                    cleaned_label,
                    paths,
                    metadata={"source": "gradio_demo_session"},
                    persist=False,
                )
        except (InferenceError, RegistrationError, ValueError, TypeError) as error:
            return f"Registration failed: {error}"
        return (
            f"Registered {readable_name(result.label)} from "
            f"{result.reference_count} photos in memory for this demo session. "
            "Now use a different photo below to test it."
        )


def _as_rgb_uint8(image: np.ndarray) -> npt.NDArray[np.uint8]:
    """Validate the RGB array supplied by Gradio without changing semantics."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("expected an RGB image")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return cast(npt.NDArray[np.uint8], np.ascontiguousarray(array))


def _file_paths(files: Sequence[Any] | None) -> list[Path]:
    """Normalize Gradio's filepath values while rejecting missing files."""
    paths: list[Path] = []
    for value in files or ():
        raw = value if isinstance(value, (str, Path)) else getattr(value, "name", value)
        path = Path(str(raw)).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"reference image does not exist: {path.name}")
        paths.append(path)
    return paths


def _base_ranking(decision: OpenSetDecision) -> dict[str, float]:
    """Convert ranked base evidence to Gradio's label-confidence mapping."""
    return {
        readable_name(label): float(probability)
        for label, probability in decision.base.ranking[:3]
    }


def _format_decision(decision: OpenSetDecision) -> tuple[dict[str, float], str]:
    """Render a project decision without presenting rejected evidence as truth."""
    if decision.verdict is Verdict.BASE_CLASS:
        label = readable_name(decision.label)
        return _base_ranking(decision), (
            f"Predicted: {label} sign ({decision.score:.0%} confidence)."
        )
    if decision.verdict is Verdict.REGISTERED_CLASS:
        label = readable_name(decision.label)
        return {label: float(decision.score)}, (
            f"Recognized registered sign: {label} "
            f"({decision.score:.0%} prototype similarity)."
        )

    rejected = {
        f"Rejected candidate: {readable_name(label)}": float(probability)
        for label, probability in decision.base.ranking[:3]
    }
    return {"Unseen / new sign": 1.0, **rejected}, UNKNOWN_MESSAGE


def create_demo(recognizer: OpenSetRecognizer | None = None) -> gr.Blocks:
    """Create the two-tab Gradio application."""
    controller = DemoController(recognizer or build_recognizer())
    with gr.Blocks(title="Adaptive Indian Road Sign Recognition") as demo:
        gr.Markdown(
            "# Adaptive Indian Road Sign Recognition\n"
            "Classify one of the 17 trained signs, reject an unseen sign, or "
            "teach the frozen model a new sign from a few photos."
        )
        with gr.Tab("Classify a sign"):
            classify_image = gr.Image(
                sources=["upload", "webcam"],
                type="numpy",
                label="Road-sign image",
            )
            classify_button = gr.Button("Classify", variant="primary")
            classify_labels = gr.Label(
                label="Decision and top candidates", num_top_classes=4
            )
            classify_message = gr.Textbox(label="Result", interactive=False)
            classify_button.click(
                controller.classify,
                inputs=classify_image,
                outputs=[classify_labels, classify_message],
            )

        with gr.Tab("Register a new sign (few-shot)"):
            gr.Markdown(
                "Upload **3–5 photos of the same new sign class**. Registration "
                "creates only an in-memory prototype; it does not retrain the "
                "base model or change the training dataset."
            )
            new_label = gr.Textbox(label="New sign name", placeholder="e.g. Bus Stop")
            reference_files = gr.File(
                label="Reference photos",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            register_button = gr.Button("Register sign", variant="primary")
            registration_message = gr.Textbox(
                label="Registration status", interactive=False
            )
            register_button.click(
                controller.register,
                inputs=[new_label, reference_files],
                outputs=registration_message,
            )

            gr.Markdown("### Test with a different photo")
            test_image = gr.Image(
                sources=["upload", "webcam"],
                type="numpy",
                label="Fresh test photo",
            )
            test_button = gr.Button("Recognize registered sign")
            test_labels = gr.Label(
                label="Decision and top candidates", num_top_classes=4
            )
            test_message = gr.Textbox(label="Result", interactive=False)
            test_button.click(
                controller.classify,
                inputs=test_image,
                outputs=[test_labels, test_message],
            )
    return demo


def main() -> int:
    """Load the frozen V2 recognizer and launch the local Gradio server."""
    demo = create_demo()
    demo.launch(
        inbrowser=True,
        # Set share=True only when the presenter intentionally needs a public,
        # temporary Gradio link. Uploaded images then pass through that link.
        # share=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
