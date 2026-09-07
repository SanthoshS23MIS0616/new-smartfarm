from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.config import REPORTS_DIR, STATIC_DIR

# Ensure dirs exist so StaticFiles mount never crashes
STATIC_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Crop Intelligence Platform",
    version="1.0.0",
    description="Crop recommendation, yield prediction, and ranking service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h2>SmartFarm: index.html not found in static/</h2>", status_code=404)
    return FileResponse(str(index_file), media_type="text/html")
