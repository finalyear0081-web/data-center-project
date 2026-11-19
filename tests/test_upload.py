"""
Integration Tests for Dataset Upload API and Background Tracking
"""
import io
import os
import unittest
import pandas as pd
from unittest.mock import patch
from backend.app import create_app
from backend.config import Config

class TestUploadPipeline(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.testing = True
        self.hist_path = os.path.join(Config.STORAGE_DIR, "dataset_history.json")
        self.orig_hist = None
        if os.path.exists(self.hist_path):
            try:
                with open(self.hist_path, "r", encoding="utf-8") as f:
                    self.orig_hist = f.read()
            except Exception:
                pass

    def tearDown(self):
        if self.orig_hist is not None:
            try:
                with open(self.hist_path, "w", encoding="utf-8") as f:
                    f.write(self.orig_hist)
            except Exception:
                pass

    @patch("backend.services.prediction_service.PredictionService.start_pipeline_job")
    def test_upload_valid_csv(self, mock_job):
        """Test uploading a valid 10-column dataset CSV file without overwriting production model."""
        mock_job.return_value = {"accepted": True, "message": "Pipeline queued in background"}
        df = pd.read_csv(Config.DEFAULT_DATASET_PATH, nrows=350)
        csv_bytes = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        
        data = {
            "file": (csv_bytes, "test_upload_valid.csv")
        }
        resp = self.client.post("/api/dataset/upload", data=data, content_type="multipart/form-data")
        self.assertIn(resp.status_code, [200, 202])
        json_data = resp.get_json()
        self.assertTrue(json_data["success"])
        self.assertIn("pipeline_job", json_data["data"])
        self.assertEqual(json_data["data"]["total_records"], 350)

    def test_reject_invalid_extension(self):
        """Test non-csv file extension rejection."""
        fake_txt = io.BytesIO(b"sensor,data\n1,2")
        data = {"file": (fake_txt, "telemetry.txt")}
        resp = self.client.post("/api/dataset/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["success"])

    def test_reject_missing_required_columns(self):
        """Test rejection when required columns are absent."""
        invalid_csv = io.BytesIO(b"timestamp,temp\n2024-01-01 00:00:00,24.5")
        data = {"file": (invalid_csv, "incomplete.csv")}
        resp = self.client.post("/api/dataset/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 422)
        json_data = resp.get_json()
        self.assertFalse(json_data["success"])
        self.assertIn("inlet_temperature", json_data["validation_report"]["missing_columns"])

    def test_dataset_status_polling(self):
        """Test GET /api/dataset/status endpoint returns valid pipeline state."""
        resp = self.client.get("/api/dataset/status")
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertTrue(json_data["success"])
        self.assertIn("state", json_data["data"])
        self.assertIn("progress", json_data["data"])

if __name__ == "__main__":
    unittest.main()
