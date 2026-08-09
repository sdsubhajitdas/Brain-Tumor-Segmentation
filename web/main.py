from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.inference import InferenceEngine
from web.routes import api as api_routes
from web.routes import pages as page_routes

WEB_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_engine = InferenceEngine()
    yield


app = FastAPI(title="Brain Tumor Segmentation", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
app.include_router(page_routes.router)
app.include_router(api_routes.router)
