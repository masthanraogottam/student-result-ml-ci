"""API tests; train_model.py must run before this module."""
import unittest

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


class TestPredictionAPI(unittest.TestCase):
    def test_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])

    def test_predict_pass(self):
        response = client.post("/predict", json={
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "PASS")

    def test_missing_fields(self):
        response = client.post("/predict", json={"attendance": 90})
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.json())

    def test_non_numeric_input(self):
        response = client.post("/predict", json={
            "attendance": "high",
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80,
        })
        self.assertEqual(response.status_code, 400)

    def test_metrics(self):
        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("student_api_http_requests_total", response.text)


if __name__ == "__main__":
    unittest.main()
