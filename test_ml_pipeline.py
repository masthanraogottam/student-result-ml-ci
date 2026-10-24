"""Automated tests for the student-result ML pipeline."""
import json
from pathlib import Path
import unittest

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


class TestMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_path = ROOT / "student_results.csv"
        cls.model_path = ROOT / "student_result_model.pkl"
        cls.metrics_path = ROOT / "metrics.json"

    def test_dataset_exists(self):
        self.assertTrue(self.dataset_path.is_file())

    def test_model_exists(self):
        self.assertTrue(self.model_path.is_file())

    def test_metrics_exist(self):
        self.assertTrue(self.metrics_path.is_file())

    def test_accuracy_is_between_zero_and_one(self):
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_prediction_is_valid_class(self):
        model = joblib.load(self.model_path)
        sample = pd.DataFrame([{
            "attendance": 85,
            "internal_marks": 75,
            "assignment_marks": 80,
            "previous_score": 78,
        }])
        prediction = int(model.predict(sample)[0])
        self.assertIn(prediction, (0, 1))

    def test_high_performance_sample_passes(self):
        model = joblib.load(self.model_path)
        sample = pd.DataFrame([{
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80,
        }])
        self.assertEqual(int(model.predict(sample)[0]), 1)

    def test_low_performance_sample_fails(self):
        model = joblib.load(self.model_path)
        sample = pd.DataFrame([{
            "attendance": 55,
            "internal_marks": 30,
            "assignment_marks": 40,
            "previous_score": 35,
        }])
        self.assertEqual(int(model.predict(sample)[0]), 0)


if __name__ == "__main__":
    unittest.main()
