# Team demo

This demo runs the frozen 17-class MobileNetV3-Small classifier, the calibrated
open-set decision policy, and the existing few-shot prototype registrar through
a local two-tab Gradio interface. Registration stays in memory for the running
process: it does not change the base dataset, retrain the model, or write a
prototype registry to disk.

## Download the two required pieces

1. Download or clone the repository. A GitHub source ZIP does **not** contain
   the trained checkpoint because model weights are intentionally excluded from
   Git history.
2. Download [`best.pt` from the `v2-checkpoint` release](https://github.com/sushan5140/adaptive-indian-road-sign-recognition/releases/download/v2-checkpoint/best.pt).
3. Create `outputs/v2_work/checkpoints/` and place the file at:

   ```text
   outputs/v2_work/checkpoints/best.pt
   ```

The expected file size is **18,689,098 bytes** and its SHA-256 is:

```text
0f990f21c7f844f5611e91f867740b7f980e851426681c69deb2fefadbea8ff4
```

## Install and run

Use Python 3.11 from the repository root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/demo_app.py
```

The browser opens the local Gradio interface. To present remotely, deliberately
uncomment `share=True` near the bottom of `scripts/demo_app.py`; this creates a
temporary public Gradio link, so use it only when sharing uploaded images that
are safe to send through that service.

## One-paragraph live script

Start in **Classify a sign** and upload a known V2 sign, pointing out the plain
English label and confidence. Next upload a sign outside the 17 trained classes:
instead of forcing a familiar label, the calibrated open-set policy says it is
unseen/new. Move to **Register a new sign (few-shot)**, name that new class and
upload three to five reference photos of it. Registration creates an embedding
prototype without retraining. Finally, upload a different photo of the same sign
in the test area and show that the existing open-set pipeline now recognizes the
registered name when its calibrated prototype rule accepts the match.

This is a qualitative team demonstration, not a replacement for the locked
benchmark or an additional experimental result. Few-shot success depends on
coherent reference photos and a query sufficiently similar to their prototype;
the application reports an honest unknown result when configured thresholds are
not met.
