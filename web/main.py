import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.inference import InferenceEngine
from web.routes import api as api_routes
from web.routes import pages as page_routes

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

IDLE_CHECK_INTERVAL_SECONDS = 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = InferenceEngine()
    engine.bind_event_loop(asyncio.get_running_loop())
    app.state.inference_engine = engine

    async def idle_unload_loop():
        while True:
            # Nothing to check while the model isn't loaded -- block here
            # instead of polling on a fixed interval, and wake up the
            # instant a request loads it.
            await engine.loaded_event.wait()
            while True:
                await asyncio.sleep(IDLE_CHECK_INTERVAL_SECONDS)
                if engine.unload_if_idle():
                    break  # went idle -- back to waiting for the next load

    task = asyncio.create_task(idle_unload_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Brain Tumor Segmentation", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
app.include_router(page_routes.router)
app.include_router(api_routes.router)


@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return templates.TemplateResponse(request, "404.html", {}, status_code=404)
