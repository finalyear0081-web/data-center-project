"""
Comprehensive End-to-End REST API Integration Tests
"""
import unittest
from backend.app import create_app

class TestRestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.client.testing = True

    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertEqual(json_data["status"], "healthy")
        self.assertTrue(json_data["model_ready"])

    def test_html_views(self):
        routes = ["/dashboard", "/dataset", "/predictions", "/cooling", "/model-performance", "/login", "/register"]
        for r in routes:
            resp = self.client.get(r)
            self.assertEqual(resp.status_code, 200, f"Route {r} failed with status {resp.status_code}")

    def test_dataset_endpoints(self):
        resp_prev = self.client.get("/api/dataset/preview?limit=5")
        self.assertEqual(resp_prev.status_code, 200)
        self.assertTrue(resp_prev.get_json()["success"])

        resp_stats = self.client.get("/api/dataset/statistics")
        self.assertEqual(resp_stats.status_code, 200)
        self.assertIn("inlet_temperature", resp_stats.get_json()["data"])

        resp_hist = self.client.get("/api/dataset/history")
        self.assertEqual(resp_hist.status_code, 200)

    def test_prediction_endpoints(self):
        resp_fc = self.client.post("/api/prediction/forecast")
        self.assertEqual(resp_fc.status_code, 200)
        data = resp_fc.get_json()["data"]
        self.assertIn("prediction", data)
        self.assertIn(len(data["prediction"]["forecast_temperatures"]), [30, 144, 288, 720])

        resp_res = self.client.get("/api/prediction/results")
        self.assertEqual(resp_res.status_code, 200)

        resp_exp = self.client.get("/api/prediction/export")
        self.assertEqual(resp_exp.status_code, 200)
        self.assertIn("text/csv", resp_exp.content_type)

    def test_cooling_endpoints(self):
        resp_status = self.client.get("/api/cooling/status")
        self.assertEqual(resp_status.status_code, 200)
        status_data = resp_status.get_json()["data"]
        self.assertIn(status_data["chiller_state"], ["RUNNING", "STOPPED"])

        # Mode switch
        resp_mode = self.client.post("/api/cooling/mode", json={"mode": "MANUAL"})
        self.assertEqual(resp_mode.status_code, 200)
        self.assertEqual(resp_mode.get_json()["data"]["operating_mode"], "MANUAL")

        # Manual control
        resp_man = self.client.post("/api/cooling/manual", json={"action": "ON"})
        self.assertEqual(resp_man.status_code, 200)
        self.assertEqual(resp_man.get_json()["data"]["chiller_state"], "RUNNING")

        # Return to automatic
        resp_auto = self.client.post("/api/cooling/mode", json={"mode": "AUTOMATIC"})
        self.assertEqual(resp_auto.status_code, 200)

        # Thresholds
        resp_th = self.client.put("/api/cooling/thresholds", json={"on_threshold": 28.5, "off_threshold": 26.5})
        self.assertEqual(resp_th.status_code, 200)

    def test_dashboard_endpoints(self):
        resp_sum = self.client.get("/api/dashboard/summary")
        self.assertEqual(resp_sum.status_code, 200)
        sum_data = resp_sum.get_json()["data"]
        self.assertIn("current_inlet_temperature", sum_data)
        self.assertIn("predicted_max_temperature", sum_data)

        resp_telem = self.client.get("/api/dashboard/temperature?hours=12")
        self.assertEqual(resp_telem.status_code, 200)
        self.assertIn(resp_telem.get_json()["data"]["total_points"], [72, 144, 360, 720])

        resp_alerts = self.client.get("/api/dashboard/alerts")
        self.assertEqual(resp_alerts.status_code, 200)

    def test_model_endpoints(self):
        resp_status = self.client.get("/api/model/status")
        self.assertEqual(resp_status.status_code, 200)

        resp_metrics = self.client.get("/api/model/metrics")
        self.assertEqual(resp_metrics.status_code, 200)
        eval_data = resp_metrics.get_json()["data"]["evaluation"]
        self.assertIn("LSTM", eval_data["models"])

if __name__ == "__main__":
    unittest.main()
