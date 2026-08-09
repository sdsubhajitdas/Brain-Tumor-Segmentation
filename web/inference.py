"""Loads the trained model once (at FastAPI startup) and exposes a simple
predict wrapper reused across requests -- unlike api.py's CLI, which reloads
the model on every invocation.

Uploaded images are processed entirely in memory (bytes -> PIL -> tensor ->
inference -> PNG bytes) and never written to disk. bts/classifier.py itself
is untouched; the two adaptations needed for web use (a dummy mask for
uploads, since predict() requires one; torch.no_grad() around the forward
pass) are applied here at the call site only.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bts.classifier import BrainTumorClassifier
from bts.model import DynamicUNet

# Single-request-at-a-time CPU inference doesn't benefit from intra-op
# parallelism; capping avoids torch defaulting to the host machine's full
# core count (which can exceed this container's cgroup CPU quota) and
# wasting memory on unused thread-local MKL/oneDNN buffers.
torch.set_num_threads(1)

FILTER_LIST = [16, 32, 64, 128, 256]
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "saved_models"
    / "UNet-[16, 32, 64, 128, 256].pt"
)


@dataclass
class PredictionResult:
    mask: np.ndarray  # (512, 512) uint8, values 0 or 255
    overlay: np.ndarray  # (512, 512, 3) uint8 RGB
    dice_score: float | None  # None when there's no ground-truth mask (uploads)
    inference_ms: float


class InferenceEngine:
    """Load once via InferenceEngine(); call .predict_sample()/.predict_from_bytes() many times."""

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        model = DynamicUNet(FILTER_LIST).to(self.device)
        self.classifier = BrainTumorClassifier(model, self.device)
        self.classifier.restore_model(str(MODEL_PATH))
        self.classifier.model.eval()

    def predict_sample(self, image_path: Path, mask_path: Path, threshold: float = 0.5) -> PredictionResult:
        """Curated sample: real ground-truth mask, dice score included."""
        image_tensor = self._tensor_from_pil(Image.open(image_path))
        mask_tensor = self._tensor_from_pil(Image.open(mask_path))
        return self._run(image_tensor, mask_tensor, threshold, has_ground_truth=True)

    def predict_from_bytes(self, raw_bytes: bytes, threshold: float = 0.5) -> PredictionResult:
        """Visitor upload: no ground-truth mask available."""
        import io

        image_tensor = self._tensor_from_pil(Image.open(io.BytesIO(raw_bytes)))
        dummy_mask_tensor = torch.zeros_like(image_tensor)
        return self._run(image_tensor, dummy_mask_tensor, threshold, has_ground_truth=False)

    def _tensor_from_pil(self, image: Image.Image) -> torch.Tensor:
        # Matches torchvision's Grayscale + Resize((512, 512)) + to_tensor
        # pipeline this used to run (torchvision's PIL grayscale codepath is
        # just convert("L"); Resize on a PIL image is just Image.resize()),
        # without depending on torchvision itself for three lines of PIL/numpy.
        image = image.convert("L").resize((512, 512), Image.BILINEAR)
        arr = np.array(image, dtype=np.uint8)
        return torch.from_numpy(arr).float().div(255.0).unsqueeze(0)

    def _run(
        self,
        image_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        threshold: float,
        has_ground_truth: bool,
    ) -> PredictionResult:
        start = time.perf_counter()
        data = {"image": image_tensor, "mask": mask_tensor}
        with torch.no_grad():
            image_arr, _mask_arr, output_arr, score = self.classifier.predict(data, threshold=threshold)
        inference_ms = (time.perf_counter() - start) * 1000

        mask_uint8 = (output_arr * 255).astype(np.uint8)
        overlay = _make_overlay(image_arr, output_arr)

        return PredictionResult(
            mask=mask_uint8,
            overlay=overlay,
            dice_score=float(score) if has_ground_truth else None,
            inference_ms=inference_ms,
        )


def _make_overlay(image_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    """Red-tinted predicted mask over the grayscale original."""
    base = (np.clip(image_arr, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    red_tint = np.zeros_like(rgb)
    red_tint[..., 0] = 255
    alpha = 0.4
    mask_bool = mask_arr.astype(bool)
    rgb[mask_bool] = rgb[mask_bool] * (1 - alpha) + red_tint[mask_bool] * alpha
    return rgb.astype(np.uint8)
