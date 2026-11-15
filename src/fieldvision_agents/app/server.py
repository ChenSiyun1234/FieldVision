"""The web console: a thin JSON API over service.py and one static page.

    fieldvision serve            then open http://127.0.0.1:8765

Runs are synchronous: an offline run takes seconds, a Nemotron run up to a minute.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import service

STATIC = Path(__file__).with_name("static")
app = FastAPI(title="FieldVision analyst console")


class RunRequest(BaseModel):
    fips: str
    year: int
    offline: bool = False


class ModeRequest(BaseModel):
    offline: bool = False


class AskRequest(BaseModel):
    question: str
    offline: bool = False


def _guard(fn, *args, **kw):
    """Turn pipeline errors into readable 4xx/5xx answers for the page."""
    try:
        return fn(*args, **kw)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # show the problem on the page instead of a bare 500
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return service.status()


@app.get("/api/counties")
def counties():
    return service.counties()


@app.get("/api/runs")
def runs():
    return service.runs()


@app.post("/api/runs")
def run(req: RunRequest):
    result = _guard(service.run, req.fips, req.year, offline=req.offline or not service.has_key())
    detail = _guard(service.detail, result["run_id"])
    detail["result"]["seconds"] = result["seconds"]
    return detail


@app.get("/api/runs/{run_id}")
def detail(run_id: str):
    return _guard(service.detail, run_id)


@app.post("/api/runs/{run_id}/report")
def report(run_id: str, req: ModeRequest):
    return _guard(service.report, run_id, offline=req.offline or not service.has_key())


@app.post("/api/runs/{run_id}/ask")
def ask(run_id: str, req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Type a question.")
    return _guard(service.ask, run_id, req.question, offline=req.offline or not service.has_key())


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def serve(host="127.0.0.1", port=8765):
    import uvicorn
    print(f"FieldVision console on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
