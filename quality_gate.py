"""Fail the workflow when model accuracy is below the configured threshold."""
import json
import os
from pathlib import Path

metrics = json.loads(Path("metrics.json").read_text(encoding="utf-8"))
accuracy = float(metrics["accuracy"])
threshold = float(os.getenv("MIN_MODEL_ACCURACY", "0.70"))
print(f"Model accuracy: {accuracy:.4f}; required minimum: {threshold:.4f}")
if accuracy < threshold:
    raise SystemExit("QUALITY GATE FAILED: model accuracy is below the minimum.")
print("QUALITY GATE PASSED")
