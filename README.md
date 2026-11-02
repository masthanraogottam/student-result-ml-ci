# Student Result Prediction — MLOps Practicals 2–12

End-to-end educational MLOps project: reproducible model training, CI quality gate, FastAPI model serving, Docker, GHCR publication, Render deployment, structured logging, Prometheus metrics, and Grafana visualization.

## Model
The training script generates 300 synthetic records and fits a StandardScaler + LogisticRegression pipeline. The label uses attendance 25%, internal marks 35%, assignments 20%, and previous score 20%; weighted score >=60 maps to PASS. Synthetic data is for practical use only, not real student assessment.

## API
POST /predict accepts numeric values 0–100 for attendance, internal_marks, assignment_marks, previous_score.
- GET / — service information
- GET /health — model readiness
- POST /predict — PASS/FAIL
- GET /metrics — Prometheus metrics
- GET /docs — Swagger UI

## Run locally
```bash
python -m pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest discover -v
uvicorn app:app --host 0.0.0.0 --port 8000
```
Example:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"attendance":90,"internal_marks":85,"assignment_marks":88,"previous_score":80}'
```

## Docker (Practicals 6–7)
```bash
docker build -t student-result-prediction:latest .
docker run --rm -p 8000:8000 student-result-prediction:latest
```

## GHCR (Practical 8 manual activity)
Authenticate with a GitHub token having read:packages and write:packages. Never commit the token.
```bash
docker tag student-result-prediction:latest ghcr.io/masthanraogottam/student-result-prediction:v1
docker push ghcr.io/masthanraogottam/student-result-prediction:v1
docker pull ghcr.io/masthanraogottam/student-result-prediction:v1
```
Practical 9 automates publishing using repository GITHUB_TOKEN.

## Render (Practical 10)
Create an image-backed Web Service from `ghcr.io/masthanraogottam/student-result-prediction:latest`. Set PORT=8000 and health-check path /health. For private GHCR packages configure registry credentials in Render, not the repository. Trigger redeploy after a new image is published.

## Logging (Practical 11)
The API emits request ID, method, path, status, duration, model readiness and prediction outcomes to standard logs. Raw student identifiers, payloads and secrets are not logged. Inspect Render runtime logs.

## Prometheus and Grafana (Practical 12)
Update `prometheus.yml` with the actual Render hostname. Configure Grafana with Prometheus as its data source. Suggested PromQL:
```promql
sum(rate(student_api_http_requests_total[5m]))
sum(rate(student_api_http_requests_total{status=~"4..|5.."}[5m]))
sum(rate(student_api_predictions_total[5m]))
histogram_quantile(0.95, sum(rate(student_api_http_request_duration_seconds_bucket[5m])) by (le))
student_api_model_loaded
up{job="student-result-api"}
```
The included alert rules monitor target availability and repeated client errors.

## Workflows
- `.github/workflows/ml-ci.yml`: train, quality gate, ML/API tests and release artifact.
- `.github/workflows/docker-publish.yml`: build, container health/prediction tests, then GHCR publish with latest and commit-SHA tags.
Enable GitHub Actions and grant this repository write access to an existing GHCR package if necessary.
