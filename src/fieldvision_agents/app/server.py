"""The FieldVision app: a JSON API over the pipeline and one static page.

    fieldvision serve            then open http://127.0.0.1:8765

Two roles share the page. A dairy owner chats with the silage finder and sees matched fields on a map.
A USDA analyst chats with the full pipeline, opens runs in the console, and reads the dashboard.
Calls are synchronous: offline answers take seconds, Nemotron runs up to a minute.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..settings import REPO
from . import chat as chat_agent
from . import dairy, dashboard, geo, service

STATIC = Path(__file__).with_name("static")
VENDOR = REPO / "app" / "vendor"                     # Leaflet, shared with the field map demo
app = FastAPI(title="FieldVision")


class RunRequest(BaseModel):
    fips: str
    year: int
    offline: bool = False


class ModeRequest(BaseModel):
    offline: bool = False


class AskRequest(BaseModel):
    question: str
    offline: bool = False


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    role: str
    messages: list[Message]
    context: dict = {}


class SilageRequest(BaseModel):
    fips: str
    year: int
    as_of: str | None = None
    dairy_id: str | None = None
    lat: float | None = None
    lon: float | None = None
    head: float | None = None
    tons: float | None = None
    max_miles: float | None = None


def _guard(fn, *args, **kw):
    """Turn pipeline errors into readable 4xx/5xx answers for the page."""
    try:
        return fn(*args, **kw)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except (RuntimeError, ValueError) as e:
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


# --- US map: states, their counties, one county's data -------------------------------

@app.get("/api/geo/state/{state}")
def geo_state(state: str):
    return _guard(geo.state_summary, state)


@app.get("/api/geo/county/{fips}")
def geo_county(fips: str):
    return _guard(geo.county_detail, fips)


# --- chat (both roles) -----------------------------------------------------------

@app.get("/api/suggestions/{role}")
def suggestions(role: str):
    return chat_agent.SUGGESTIONS.get(role, [])


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.messages or not req.messages[-1].content.strip():
        raise HTTPException(400, "Type a message.")
    return _guard(chat_agent.chat, req.role, [m.model_dump() for m in req.messages], req.context)


# --- dairy owner ---------------------------------------------------------------

@app.get("/api/dairies")
def dairies():
    return _guard(dairy.dairies)


@app.post("/api/silage")
def silage(req: SilageRequest):
    kw = req.model_dump()
    return _guard(dairy.find_silage, kw.pop("fips"), kw.pop("year"), **kw)


# --- USDA analyst ----------------------------------------------------------------

@app.get("/api/dashboard")
def dashboard_data():
    return _guard(dashboard.build)


@app.get("/api/whatif")
def what_if(points: float = 1.0):
    return _guard(dashboard.what_if, points)


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
if VENDOR.is_dir():
    app.mount("/vendor", StaticFiles(directory=VENDOR), name="vendor")


def serve(host="127.0.0.1", port=8765):
    import uvicorn
    print(f"FieldVision on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
