import base64
import io

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from web import samples
from web.rate_limit import is_rate_limited
from web.schemas import PredictResponse

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.post("/api/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile | None = None,
    sample_id: str | None = Form(None),
):
    client_ip = request.client.host if request.client else "unknown"
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests, please slow down.")

    if file is None and sample_id is None:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'sample_id'.")
    if file is not None and sample_id is not None:
        raise HTTPException(status_code=400, detail="Provide only one of 'file' or 'sample_id'.")

    engine = request.app.state.inference_engine

    # Blocking torch inference is offloaded to Starlette's threadpool so it
    # doesn't stall the event loop (e.g. /healthz) for the duration of a
    # multi-second CPU forward pass.
    if sample_id is not None:
        if sample_id not in samples.SAMPLE_IDS:
            raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id}")
        image_path, mask_path = samples.get_sample_paths(sample_id)
        result = await run_in_threadpool(engine.predict_sample, image_path, mask_path)
        source = "sample"
    else:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 10MB).")
        try:
            Image.open(io.BytesIO(raw)).verify()
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Not a valid image file.")
        result = await run_in_threadpool(engine.predict_from_bytes, raw)
        source = "upload"

    return PredictResponse(
        source=source,
        sample_id=sample_id,
        dice_score=result.dice_score,
        mask_png_base64=_encode_png(result.mask, mode="L"),
        overlay_png_base64=_encode_png(result.overlay, mode="RGB"),
        inference_ms=result.inference_ms,
    )


def _encode_png(array, mode: str) -> str:
    image = Image.fromarray(array, mode)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
