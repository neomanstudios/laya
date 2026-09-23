"""HTTP API for Laya, with Swagger UI at ``/docs`` and the OpenAPI spec at ``/openapi.json``.

    pip install "laya[serve]"
    uvicorn laya.server:app --host 0.0.0.0 --port 8000

One process holds one :class:`~laya.router.Router`, so checkpoints are loaded once and shared
by every request. Endpoints are plain ``def``, which FastAPI runs in a worker thread: a forward
pass never blocks the event loop, and the router's inference path is thread-safe.

Configuration is environment only (see ``docs/API.md``):

    LAYA_PRELOAD              checkpoints to build at startup, e.g. "english,multilingual" or
                              "all"; empty (default) loads on first use
    LAYA_DEVICE               "cuda" / "cpu" / "mps"; empty picks the best available
    LAYA_MAX_LOADED           how many checkpoints stay resident (default 1, LRU eviction)
    LAYA_DEFAULT_MODEL        checkpoint for states with no letters to detect (default english)
    LAYA_AUTO_TASK_DETECTION  "1" routes the four typed-decisions workflows automatically
    LAYA_API_KEY              when set, /v1/* requires this value in the X-API-Key header
    HF_TOKEN                  Hugging Face token, for gated or private checkpoints
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__
from .email import email_state
from .lang import analyse
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import Router, _repo_str

PRESETS = {
    "triage": triage_questions,
    "email": email_questions,
    "guard": guard_questions,
    "moderation": moderation_questions,
    "router": router_questions,
}
PresetName = Literal["triage", "email", "guard", "moderation", "router"]

# Descriptive only; the repo each name resolves to comes from the live Router.
MODEL_INFO = {
    "english": {"encoder": "ModernBERT-large", "params": "421M", "context": 512,
                "use_for": "English"},
    "multilingual": {"encoder": "mmBERT-base", "params": "322M", "context": 1024,
                     "use_for": "100+ languages, 2x faster"},
    "typed-decisions": {"encoder": "ModernBERT-large", "params": "421M", "context": 1024,
                        "use_for": "the typed-decisions workflows"},
}

DEVICE = os.environ.get("LAYA_DEVICE", "").strip()
PRELOAD = os.environ.get("LAYA_PRELOAD", "").strip()
MAX_LOADED = int(os.environ.get("LAYA_MAX_LOADED", "1"))
DEFAULT_MODEL = os.environ.get("LAYA_DEFAULT_MODEL", "english").strip() or "english"
AUTO_TASK_DETECTION = os.environ.get("LAYA_AUTO_TASK_DETECTION", "").strip().lower() in ("1", "true", "yes", "on")
API_KEY = os.environ.get("LAYA_API_KEY", "").strip()

_router: Optional[Router] = None


def get_router() -> Router:
    if _router is None:                                  # only before startup / after shutdown
        raise HTTPException(status_code=503, detail="the model router is not ready yet")
    return _router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _router
    _router = Router(device=DEVICE or None, max_loaded=MAX_LOADED, default=DEFAULT_MODEL,
                     auto_task_detection=AUTO_TASK_DETECTION)
    names = [n.strip() for n in PRELOAD.split(",") if n.strip()]
    if names:
        try:
            _router.preload(None if names == ["all"] else names)
            print("[laya] preloaded: %s" % ", ".join(_router.loaded), flush=True)
        except Exception as e:                           # a blip here must not kill the server:
            print("[laya] preload failed (%s); checkpoints will load on first request" % e,
                  flush=True)                            # requests still load on demand
    yield
    _router = None


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                               description="Required only when the server sets LAYA_API_KEY.")


def require_api_key(key: Optional[str] = Security(_api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


# --------------------------------------------------------------------------- schemas
State = Union[str, Dict[str, Any], List[Any]]
STATE_FIELD = Field(
    description="What to decide about: plain text, a JSON object (email, ticket, record) or a "
                "list of conversation turns.",
    examples=[{"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
               "body": "We were billed twice for March. Refund the duplicate today or we cancel."}],
)


class Question(BaseModel):
    """One typed question. Every question in a request is answered in the same forward pass."""

    type: Literal["choice", "score", "noul"] = Field(
        description="choice: pick one label. score: an ordered scale, lowest level first. "
                    "noul: a yes/no probability.")
    instructions: str = Field(min_length=1, description="What to decide. Backtick a key of the "
                                                        "state (`body`) to point at one field.")
    criteria: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        default=None,
        description="choice: {label: description} or [labels]. score: ordered [level, ...]. "
                    "noul: optional {\"true\": ..., \"false\": ...}.")

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"type": "choice", "instructions": "Which department should handle this?",
         "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                      "sales": "pricing, new contracts", "other": "everything else"}},
    ]})

    @model_validator(mode="after")
    def _check_criteria(self):
        if self.type == "choice" and not self.criteria:
            raise ValueError("a choice question needs criteria: {label: description} or [labels]")
        if self.type == "score" and (not isinstance(self.criteria, list) or len(self.criteria) < 2):
            raise ValueError("a score question needs criteria as a list of at least two levels, "
                             "lowest first")
        if self.type == "noul" and self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError('a noul question takes criteria as {"true": ..., "false": ...}, or none')
        return self


class RoutingOptions(BaseModel):
    """Overrides for checkpoint selection. Leave empty to let Laya detect the language."""

    model: Optional[str] = Field(default=None, description="Force a checkpoint: english, "
                                                           "multilingual or typed-decisions.")
    task: Optional[str] = Field(default=None, description="Force a task, e.g. typed_decisions.")
    lang: Optional[str] = Field(default=None, description="Language hint (BCP-47, e.g. de) when "
                                                          "you already know it.")


class RouteRequest(RoutingOptions):
    state: State = STATE_FIELD
    questions: Optional[Dict[str, Question]] = Field(
        default=None, description="Optional; only the question ids matter here, and only when the "
                                  "server runs with LAYA_AUTO_TASK_DETECTION=1.")


class PredictRequest(RouteRequest):
    preset: Optional[PresetName] = Field(
        default=None, description="Use a built-in question set instead of `questions`. "
                                  "See GET /v1/presets.")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "state": {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
                  "body": "We were billed twice for March. Refund the duplicate today or we cancel."},
        "questions": {
            "department": {"type": "choice",
                           "instructions": "Which department should handle this request?",
                           "criteria": {"billing": "invoices, payments, refunds",
                                        "technical": "bugs, outages, system errors",
                                        "sales": "pricing, new contracts",
                                        "other": "everything else"}},
            "urgency": {"type": "score", "instructions": "How urgent is this request?",
                        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
            "churn_risk": {"type": "noul",
                           "instructions": "Does the user threaten to cancel or leave?"},
        },
    }]})

    @model_validator(mode="after")
    def _one_question_source(self):
        if bool(self.questions) == bool(self.preset):
            raise ValueError("send either `questions` or `preset`, not both and not neither")
        return self

    def resolved_questions(self) -> Dict[str, Any]:
        if self.preset:
            return PRESETS[self.preset]()
        return {qid: q.model_dump(exclude_none=True) for qid, q in self.questions.items()}


class EmailRequest(RoutingOptions):
    """Clean an email into a state and answer the email preset about it."""

    subject: str = Field(default="", examples=["Duplicate charge on invoice #4411"])
    body: str = Field(min_length=1, examples=[
        "Hi, we were billed twice for March. Please refund the duplicate today.\n\n"
        "Best regards,\nSam\n\n> On Mon, support wrote:\n> Thanks for reaching out."])
    sender: Optional[str] = Field(default=None, examples=["user@acme.com"])
    categories: Optional[Dict[str, str]] = Field(
        default=None, description="Override the preset's routing categories: {team: description}.")
    clean: bool = Field(default=True, description="Strip quoted replies, signatures and "
                                                  "disclaimers before deciding.")


class DetectRequest(BaseModel):
    state: State = STATE_FIELD


class Answer(BaseModel):
    """One answer. Which value field is set follows `type`."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice", "score", "noul"]
    choice: Optional[str] = Field(default=None, description="choice: the winning label.")
    score: Optional[float] = Field(default=None, description="score: expected level over the "
                                                             "scale, so 1.7 sits between 1 and 2.")
    noul: Optional[float] = Field(default=None, description="noul: probability the statement holds.")
    probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="Calibrated distribution over labels or levels.")
    legend: Optional[Dict[str, str]] = Field(default=None, description="score: level -> criterion.")
    confidence: float = Field(description="0-1. Gate on this to hand borderline cases to a "
                                          "slower model or a human.")
    action: Dict[str, float] = Field(description="Head outputs, e.g. act_probability.")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class RouteResponse(BaseModel):
    """Which checkpoint was chosen, and why."""

    model: str = Field(description="english, multilingual or typed-decisions.")
    repo: str
    reason: str
    detection: Optional[Dict[str, Any]] = Field(
        default=None, description="Script/language evidence, or null when the route was forced.")
    workflow: Optional[str] = Field(default=None, description="Matched typed-decisions workflow.")


class PredictResponse(BaseModel):
    model: str = Field(description="Runtime name, always laya-rl-agent.")
    answers: Dict[str, Answer] = Field(description="Keyed by your question ids.")
    usage: Usage
    routing: RouteResponse
    latency_ms: float = Field(description="Server-side time for routing plus the forward pass.")


class EmailResponse(PredictResponse):
    state: Dict[str, Any] = Field(description="The cleaned state the answers were computed over.")


class LanguageReport(BaseModel):
    """What routing sees. Cheap enough to call on its own (microseconds, no model)."""

    model_config = ConfigDict(extra="allow")

    script: str
    language: Optional[str]
    is_english: bool
    language_undecided: bool
    diacritic_rate: float
    non_latin_fraction: float
    script_profile: Dict[str, float]


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    repo: str
    loaded: bool = Field(description="Resident in memory, so no download or build on next use.")


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    device: str
    loaded: List[str]


# --------------------------------------------------------------------------- app
DESCRIPTION = """
Typed decisions over any state, in 100+ languages, in a single forward pass. No text is
generated, so there is nothing to parse and nothing to hallucinate.

Ask [`POST /v1/predict`](#/decisions/predict_v1_predict_post) for `choice`, `score` and `noul`
(yes/no) answers about one state. Every question is answered in the same pass, and each answer
carries a calibrated probability, so you can route low-confidence cases elsewhere.

A checkpoint is chosen per request from the script and language of the state; `POST /v1/route`
shows that decision without running a model. Pass `model` to force one.

The first call for a checkpoint downloads it from the Hugging Face hub and can take minutes;
set `LAYA_PRELOAD` to pay that at startup instead. Full guide: `docs/API.md`.
"""

app = FastAPI(
    title="Laya API",
    version=__version__,
    summary="Multilingual, non-autoregressive System 1 decision engine.",
    description=DESCRIPTION,
    license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    lifespan=lifespan,
    openapi_tags=[
        {"name": "decisions", "description": "Ask typed questions about a state."},
        {"name": "routing", "description": "Which checkpoint answers, and why."},
        {"name": "catalogue", "description": "Checkpoints and built-in question sets."},
        {"name": "service", "description": "Liveness."},
    ],
)


@app.exception_handler(ValueError)
async def _value_error(_request, exc: ValueError):
    """A bad question set is the caller's error, not a 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(OSError)
async def _os_error(_request, exc: OSError):
    """Missing weights, no disk, hub unreachable: the request cannot be served right now."""
    return JSONResponse(status_code=503, content={"detail": "could not load the checkpoint: %s" % exc})


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=Health, tags=["service"], summary="Liveness and loaded models")
def health():
    return {"status": "ok", "version": __version__, "device": DEVICE or "auto",
            "loaded": _router.loaded if _router else []}


v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@v1.post("/predict", response_model=PredictResponse, tags=["decisions"],
         response_model_exclude_none=True,
         summary="Answer typed questions about a state")
def predict(req: PredictRequest, router: Router = Depends(get_router)) -> Any:
    """Route the state to a checkpoint and answer every question in one forward pass."""
    t0 = time.perf_counter()
    result = router.predict(req.state, req.resolved_questions(),
                            model=req.model, task=req.task, lang=req.lang)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@v1.post("/email", response_model=EmailResponse, tags=["decisions"],
         response_model_exclude_none=True,
         summary="Triage an email (cleans quoted replies and signatures first)")
def email(req: EmailRequest, router: Router = Depends(get_router)) -> Any:
    state = email_state(req.subject, req.body, sender=req.sender, clean=req.clean)
    t0 = time.perf_counter()
    result = router.predict(state, email_questions(req.categories),
                            model=req.model, task=req.task, lang=req.lang)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    result["state"] = state
    return result


@v1.post("/route", response_model=RouteResponse, tags=["routing"],
         summary="Show the checkpoint choice without running a model")
def route(req: RouteRequest, router: Router = Depends(get_router)) -> Any:
    return router.route(req.state, dict(req.questions or {}),
                        model=req.model, task=req.task, lang=req.lang)


@v1.post("/detect", response_model=LanguageReport, tags=["routing"],
         summary="Script and language detection only")
def detect(req: DetectRequest) -> Any:
    return analyse(req.state)


@v1.get("/models", response_model=List[ModelInfo], tags=["catalogue"],
        summary="Checkpoints this server can serve")
def models(router: Router = Depends(get_router)) -> Any:
    loaded = router.loaded
    return [dict(MODEL_INFO.get(name, {}), name=name, repo=_repo_str(spec), loaded=name in loaded)
            for name, spec in router.models.items()]


@v1.get("/presets", response_model=Dict[str, Dict[str, Question]], tags=["catalogue"],
        summary="Every built-in question set")
def presets() -> Any:
    """Copy one into `questions`, or pass its name as `preset`."""
    return {name: build() for name, build in PRESETS.items()}


@v1.get("/presets/{name}", response_model=Dict[str, Question], tags=["catalogue"],
        summary="One built-in question set")
def preset(name: PresetName) -> Any:
    return PRESETS[name]()


app.include_router(v1)
