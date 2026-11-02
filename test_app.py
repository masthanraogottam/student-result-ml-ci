import unittest
from fastapi.testclient import TestClient
from app import app
client = TestClient(app)

class TestPredictionAPI(unittest.TestCase):
    def test_root(self):
        self.assertEqual(client.get("/").status_code, 200)
    def test_health(self):
        self.assertEqual(client.get("/health").status_code, 200)
    def test_pass_prediction(self):
        r = client.post("/predict", json={"attendance":90,"internal_marks":85,"assignment_marks":88,"previous_score":80})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["prediction"], "PASS")
    def test_missing_fields(self):
        self.assertEqual(client.post("/predict", json={"attendance":90}).status_code, 400)
    def test_bad_numeric_input(self):
        r = client.post("/predict", json={"attendance":"high","internal_marks":85,"assignment_marks":88,"previous_score":80})
        self.assertEqual(r.status_code, 400)
    def test_metrics(self):
        self.assertIn("student_api_http_requests_total", client.get("/metrics").text)

if __name__ == "__main__":
    unittest.main()
