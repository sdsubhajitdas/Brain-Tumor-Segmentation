from pydantic import BaseModel


class PredictResponse(BaseModel):
    ok: bool = True
    source: str  # "sample" | "upload"
    sample_id: str | None = None
    dice_score: float | None = None
    mask_png_base64: str
    overlay_png_base64: str
    inference_ms: float
