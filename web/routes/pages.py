from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from web import gallery, samples

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@router.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"sample_ids": samples.SAMPLE_IDS})


@router.get("/about")
def about(request: Request):
    return templates.TemplateResponse(request, "about.html", {})


@router.get("/gallery")
def gallery_page(request: Request):
    return templates.TemplateResponse(request, "gallery.html", {"gallery_items": gallery.list_gallery_items()})
