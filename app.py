"""Student Result Prediction API with request logging and Prometheus metrics."""
from __future__ import annotations
import logging, os, time, uuid
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_result_model.pkl"
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("student-result-api")
HTTP_REQUESTS = Counter("student_api_http_requests_total", "Total HTTP requests", ["method", "path", "status"])
HTTP_DURATION = Histogram("student_api_http_request_duration_seconds", "HTTP request duration", ["method", "path"])
PREDICTIONS = Counter("student_api_predictions_total", "Prediction attempts", ["outcome"])
MODEL_LOADED = Gauge("student_api_model_loaded", "Model loaded: 1=yes, 0=no")
try:
    model: Any | None = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
    MODEL_LOADED.set(1 if model is not None else 0)
    logger.info("model_loaded=%s", model is not None)
except Exception:
    model = None
    MODEL_LOADED.set(0)
    logger.exception("model_load_failed")

app = FastAPI(title="Student Result Prediction API", version="1.0.0")

@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start, status = time.perf_counter(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception("request_failed request_id=%s path=%s", request_id, request.url.path)
        raise
    finally:
        duration = time.perf_counter() - start
        HTTP_REQUESTS.labels(request.method, request.url.path, str(status)).inc()
        HTTP_DURATION.labels(request.method, request.url.path).observe(duration)
        logger.log(logging.WARNING if status >= 400 else logging.INFO,
                   "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
                   request_id, request.method, request.url.path, status, duration * 1000)

@app.get("/")
def root():
    return {"service": "student-result-prediction", "status": "ok", "health": "/health", "docs": "/docs"}

@app.get("/health")
def health():
    ready = model is not None
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "ok" if ready else "not_ready", "model_loaded": ready})

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(payload: dict[str, Any]):
    missing = [key for key in FEATURES if key not in payload]
    if missing:
        PREDICTIONS.labels("invalid").inc()
        logger.warning("prediction_validation_failed missing=%s", ",".join(missing))
        return JSONResponse(status_code=400, content={"error": "Missing required fields", "missing_fields": missing})
    values = {}
    for key in FEATURES:
        try:
            values[key] = float(payload[key])
        except (TypeError, ValueError):
            PREDICTIONS.labels("invalid").inc()
            return JSONResponse(status_code=400, content={"error": "All features must be numeric", "field": key})
        if not 0 <= values[key] <= 100:
            PREDICTIONS.labels("invalid").inc()
            return JSONResponse(status_code=400, content={"error": "Values must be between 0 and 100", "field": key})
    if model is None:
        PREDICTIONS.labels("error").inc()
        return JSONResponse(status_code=503, content={"error": "Model is not loaded"})
    try:
        code = int(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
        result = "PASS" if code == 1 else "FAIL"
        PREDICTIONS.labels("success").inc()
        logger.info("prediction_completed result=%s", result)
        return {"prediction": result, "prediction_code": code}
    except Exception:
        PREDICTIONS.labels("error").inc()
        logger.exception("prediction_failed")
        return JSONResponse(status_code=500, content={"error": "Prediction failed"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
