"""HTTP API tests: the real app, real routing, no weights.

Everything here runs against `laya.server.app` itself. The endpoints exercised (route, detect,
models, presets, validation) never touch a checkpoint, so the suite stays offline and fast.

Set LAYA_TEST_MODEL to also run one real forward pass through /v1/predict:

    LAYA_TEST_MODEL=multilingual python tests/test_server.py
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
except ImportError:                                  # the serve extra is optional
    print('skipped: install the server extra first -- pip install "laya[serve]"')
    sys.exit(0)

from laya import __version__  # noqa: E402
from laya import server  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


QUESTIONS = {
    "department": {"type": "choice", "instructions": "Who should handle this?",
                   "criteria": {"billing": "refunds", "technical": "bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["not urgent", "soon", "critical"]},
    "churn_risk": {"type": "noul", "instructions": "Do they threaten to leave?"},
}

client = TestClient(server.app)
with client:                                         # `with` runs startup, so the Router exists

    # ----------------------------------------------------------------- service
    r = client.get("/health")
    check("health/status", r.status_code, 200)
    check("health/reports the package version", r.json()["version"], __version__)
    check("health/nothing loaded before a prediction", r.json()["loaded"], [])
    check("root/redirects to the swagger page", client.get("/").url.path, "/docs")

    # ----------------------------------------------------------------- openapi
    spec = client.get("/openapi.json").json()
    check("openapi/paths", sorted(spec["paths"]),
          ["/health", "/v1/detect", "/v1/email", "/v1/models", "/v1/predict", "/v1/presets",
           "/v1/presets/{name}", "/v1/route"])
    check("openapi/version matches the package", spec["info"]["version"], __version__)
    check_true("openapi/documents the answer shape", "Answer" in spec["components"]["schemas"])
    check_true("openapi/predict carries a request example",
               bool(spec["components"]["schemas"]["PredictRequest"].get("examples")))
    check("swagger ui", client.get("/docs").status_code, 200)
    check("redoc", client.get("/redoc").status_code, 200)

    # ----------------------------------------------------------------- routing
    r = client.post("/v1/route", json={"state": {"body": "मुझसे दो बार शुल्क लिया गया"}})
    check("route/non-latin goes multilingual", r.json()["model"], "multilingual")
    check("route/repo is a string, not a tuple",
          r.json()["repo"], "convaiinnovations/laya/multilingual")
    check("route/english stays english",
          client.post("/v1/route", json={"state": "I was charged twice"}).json()["model"], "english")
    check("route/explicit model wins",
          client.post("/v1/route", json={"state": "I was charged twice",
                                         "model": "typed-decisions"}).json()["model"],
          "typed-decisions")
    check("route/lang hint is honoured",
          client.post("/v1/route", json={"state": "short", "lang": "de"}).json()["model"],
          "multilingual")
    r = client.post("/v1/route", json={"state": "x", "model": "nope"})
    check("route/unknown model is a 400", r.status_code, 400)
    check_true("route/unknown model says what is valid", "english" in r.json()["detail"])

    r = client.post("/v1/detect", json={"state": "Le client a été facturé deux fois"})
    check("detect/script", r.json()["script"], "latin")
    check("detect/not english", r.json()["is_english"], False)
    check("detect/needs no model", client.get("/health").json()["loaded"], [])

    # ----------------------------------------------------------------- catalogue
    models = client.get("/v1/models").json()
    check("models/all three are listed", sorted(m["name"] for m in models),
          ["english", "multilingual", "typed-decisions"])
    check("models/none resident yet", [m["name"] for m in models if m["loaded"]], [])

    presets = client.get("/v1/presets").json()
    check("presets/names", sorted(presets), ["email", "guard", "moderation", "router", "triage"])
    check("presets/one set", sorted(client.get("/v1/presets/guard").json()), sorted(presets["guard"]))
    check("presets/unknown name is rejected", client.get("/v1/presets/nope").status_code, 422)
    check_true("presets/are valid question definitions",
               all(q.get("type") in ("choice", "score", "noul")
                   for qs in presets.values() for q in qs.values()))

    # ----------------------------------------------------------------- validation
    def bad(body):
        return client.post("/v1/predict", json=body).status_code

    check("validation/questions or preset, not both",
          bad({"state": "x", "questions": QUESTIONS, "preset": "triage"}), 422)
    check("validation/questions or preset, not neither", bad({"state": "x"}), 422)
    check("validation/state is required", bad({"preset": "triage"}), 422)
    check("validation/choice needs criteria",
          bad({"state": "x", "questions": {"q": {"type": "choice", "instructions": "i"}}}), 422)
    check("validation/score needs two levels or more",
          bad({"state": "x", "questions": {"q": {"type": "score", "instructions": "i",
                                                 "criteria": ["only one"]}}}), 422)
    check("validation/noul criteria must be a mapping",
          bad({"state": "x", "questions": {"q": {"type": "noul", "instructions": "i",
                                                 "criteria": ["true", "false"]}}}), 422)
    check("validation/unknown question type",
          bad({"state": "x", "questions": {"q": {"type": "ranking", "instructions": "i"}}}), 422)
    check("validation/empty instructions",
          bad({"state": "x", "questions": {"q": {"type": "noul", "instructions": ""}}}), 422)
    check("validation/email needs a body", client.post("/v1/email", json={"subject": "hi"}).status_code, 422)

    # ----------------------------------------------------------------- api key
    server.API_KEY = "secret"
    try:
        check("apikey/rejects a request without the header",
              client.post("/v1/route", json={"state": "x"}).status_code, 401)
        check("apikey/rejects the wrong key",
              client.post("/v1/route", json={"state": "x"}, headers={"X-API-Key": "nope"}).status_code, 401)
        check("apikey/accepts the right key",
              client.post("/v1/route", json={"state": "x"}, headers={"X-API-Key": "secret"}).status_code, 200)
        check("apikey/never guards /health", client.get("/health").status_code, 200)
    finally:
        server.API_KEY = ""

    # ----------------------------------------------------------------- real weights (opt-in)
    weights = os.environ.get("LAYA_TEST_MODEL", "").strip()
    if weights:
        body = {"state": {"body": "We were billed twice and want a refund today."},
                "questions": QUESTIONS, "model": weights}
        r = client.post("/v1/predict", json=body)
        check("predict/ok", r.status_code, 200)
        payload = r.json()
        answers = payload.get("answers", {})
        check("predict/answers every question", sorted(answers), sorted(QUESTIONS))
        check_true("predict/choice is one of the labels",
                   answers["department"]["choice"] in QUESTIONS["department"]["criteria"])
        check_true("predict/score sits on the scale", 0.0 <= answers["urgency"]["score"] <= 2.0)
        check_true("predict/noul is a probability", 0.0 <= answers["churn_risk"]["noul"] <= 1.0)
        check_true("predict/null fields are dropped", "choice" not in answers["churn_risk"])
        check("predict/routing is reported", payload["routing"]["model"], weights)
        check_true("predict/counts input tokens", payload["usage"]["input_tokens"] > 0)
        check_true("predict/times the call", payload["latency_ms"] > 0)
        check("predict/model is now resident", client.get("/health").json()["loaded"], [weights])
    else:
        print("note: set LAYA_TEST_MODEL=multilingual to also run a real forward pass")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all server tests passed")
sys.exit(1 if FAIL else 0)
