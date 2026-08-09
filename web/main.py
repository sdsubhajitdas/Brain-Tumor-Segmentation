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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_engine = InferenceEngine()
    yield


app = FastAPI(title="Brain Tumor Segmentation", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
app.include_router(page_routes.router)
app.include_router(api_routes.router)


@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return templates.TemplateResponse(request, "404.html", {}, status_code=404)
