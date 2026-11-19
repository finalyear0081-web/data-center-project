"""
External REST API Connection & Data Source Integration Service
Securely connects to authorized external telemetry REST endpoints,
handles authentication, validates responses, and ingests telemetry data.
"""
import os
import json
import time
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import requests
import pandas as pd

from backend.config import Config
from backend.preprocessing.suitability_engine import DatasetSuitabilityEngine
from backend.services.dataset_service import get_dataset_service

CONFIG_FILE = os.path.join(Config.STORAGE_DIR, "datasource_config.json")

def _obfuscate_key(key: str) -> str:
    """Masks secret API key for safe frontend display and logging."""
    if not key or len(key) <= 6:
        return "******"
    return f"{key[:3]}****{key[-3:]}"

class DataSourceService:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "endpoint_url": "",
            "auth_type": "NONE",  # NONE, BEARER_TOKEN, API_KEY_HEADER, API_KEY_QUERY, BASIC_AUTH
            "api_key_encrypted": "",
            "header_name": "Authorization",
            "custom_headers": {},
            "json_path": "",
            "polling_interval_minutes": 15,
            "sync_mode": "HISTORICAL",
            "last_connected_at": None,
            "last_sync_status": "NOT_CONFIGURED",
            "last_record_count": 0,
            "error": None
        }

    def _save_config(self):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Returns safe status dictionary with masked credentials."""
        c = self._config.copy()
        raw_key = self._get_secret_key()
        c["api_key_masked"] = _obfuscate_key(raw_key) if raw_key else "None"
        c.pop("api_key_encrypted", None)
        return c

    def _get_secret_key(self) -> str:
        # Check environment variable first
        env_key = os.getenv("EXTERNAL_TELEMETRY_API_KEY")
        if env_key:
            return env_key
        # Check encrypted store
        enc = self._config.get("api_key_encrypted", "")
        if enc:
            try:
                return base64.b64decode(enc.encode("utf-8")).decode("utf-8")
            except Exception:
                pass
        return ""

    def _set_secret_key(self, raw_key: str):
        if raw_key:
            enc = base64.b64encode(raw_key.encode("utf-8")).decode("utf-8")
            self._config["api_key_encrypted"] = enc
        else:
            self._config["api_key_encrypted"] = ""

    def save_connection(self, endpoint_url: str, auth_type: str = "NONE",
                        api_key: Optional[str] = None, header_name: Optional[str] = None,
                        custom_headers: Optional[Dict[str, str]] = None,
                        json_path: Optional[str] = None,
                        polling_interval_minutes: int = 15,
                        sync_mode: str = "HISTORICAL") -> Dict[str, Any]:
        """Saves and updates external API connection configuration securely."""
        self._config["endpoint_url"] = endpoint_url.strip()
        self._config["auth_type"] = auth_type.upper()
        if api_key is not None:
            self._set_secret_key(api_key.strip())
        self._config["header_name"] = header_name or "Authorization"
        self._config["custom_headers"] = custom_headers or {}
        self._config["json_path"] = json_path or ""
        self._config["polling_interval_minutes"] = max(1, int(polling_interval_minutes))
        self._config["sync_mode"] = sync_mode.upper()
        self._config["error"] = None
        self._save_config()
        return self.get_status()

    def test_connection(self, endpoint_url: Optional[str] = None,
                        auth_type: Optional[str] = None,
                        api_key: Optional[str] = None,
                        header_name: Optional[str] = None,
                        custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Tests endpoint connectivity and parses response payload without saving."""
        url = endpoint_url or self._config.get("endpoint_url")
        if not url:
            return {"success": False, "error": "No endpoint URL provided."}

        a_type = (auth_type or self._config.get("auth_type", "NONE")).upper()
        key = api_key if api_key is not None else self._get_secret_key()
        h_name = header_name or self._config.get("header_name", "Authorization")
        req_headers = dict(custom_headers or self._config.get("custom_headers", {}))

        # Build auth headers
        params = {}
        if a_type == "BEARER_TOKEN" and key:
            req_headers["Authorization"] = f"Bearer {key}"
        elif a_type == "API_KEY_HEADER" and key:
            req_headers[h_name] = key
        elif a_type == "API_KEY_QUERY" and key:
            params["api_key"] = key
        elif a_type == "BASIC_AUTH" and key:
            req_headers["Authorization"] = f"Basic {base64.b64encode(key.encode()).decode()}"

        start_time = time.time()
        try:
            resp = requests.get(url, headers=req_headers, params=params, timeout=10)
            latency_ms = round((time.time() - start_time) * 1000, 1)

            if not (200 <= resp.status_code < 300):
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "latency_ms": latency_ms,
                    "error": f"API returned non-success HTTP status code: {resp.status_code} ({resp.reason})"
                }

            json_resp = resp.json()
            records, detected_keys = self._extract_records(json_resp)

            return {
                "success": True,
                "status_code": resp.status_code,
                "latency_ms": latency_ms,
                "records_detected": len(records),
                "sample_keys": detected_keys[:15],
                "message": f"Successfully connected to API ({len(records)} telemetry records detected in {latency_ms}ms)."
            }
        except Exception as e:
            return {
                "success": False,
                "latency_ms": round((time.time() - start_time) * 1000, 1),
                "error": f"Connection failed: {str(e)}"
            }

    def _extract_records(self, json_data: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Extracts tabular records array from variable JSON hierarchies."""
        records = []
        path = self._config.get("json_path", "")

        if path and isinstance(json_data, dict):
            parts = path.split(".")
            curr = json_data
            for p in parts:
                if isinstance(curr, dict) and p in curr:
                    curr = curr[p]
                else:
                    break
            if isinstance(curr, list):
                records = curr

        if not records:
            if isinstance(json_data, list):
                records = json_data
            elif isinstance(json_data, dict):
                # Search for any key containing list of dicts (e.g. data, readings, telemetry, items)
                for k in ["data", "readings", "telemetry", "items", "results", "measurements"]:
                    if k in json_data and isinstance(json_data[k], list) and len(json_data[k]) > 0:
                        records = json_data[k]
                        break
                if not records:
                    records = [json_data]

        sample_keys = list(records[0].keys()) if (records and isinstance(records[0], dict)) else []
        return records, sample_keys

    def sync_data(self) -> Dict[str, Any]:
        """
        Pulls telemetry from configured endpoint, converts to canonical tabular dataset,
        saves snapshot, and triggers automatic suitability analysis.
        """
        test_res = self.test_connection()
        if not test_res.get("success"):
            self._config["last_sync_status"] = "FAILED"
            self._config["error"] = test_res.get("error")
            self._save_config()
            return {"success": False, "error": test_res.get("error")}

        url = self._config["endpoint_url"]
        a_type = self._config["auth_type"]
        key = self._get_secret_key()
        h_name = self._config["header_name"]
        req_headers = dict(self._config.get("custom_headers", {}))
        params = {}

        if a_type == "BEARER_TOKEN" and key:
            req_headers["Authorization"] = f"Bearer {key}"
        elif a_type == "API_KEY_HEADER" and key:
            req_headers[h_name] = key
        elif a_type == "API_KEY_QUERY" and key:
            params["api_key"] = key

        resp = requests.get(url, headers=req_headers, params=params, timeout=20)
        json_resp = resp.json()
        records, _ = self._extract_records(json_resp)

        if not records or not isinstance(records[0], dict):
            return {"success": False, "error": "No valid tabular telemetry records found in API response."}

        df = pd.DataFrame(records)
        now_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"api_telemetry_sync_{now_ts}.csv"
        saved_path = os.path.join(Config.UPLOADS_DIR, filename)
        os.makedirs(Config.UPLOADS_DIR, exist_ok=True)
        df.to_csv(saved_path, index=False)

        # Run suitability analysis on synced dataset
        suitability_report = DatasetSuitabilityEngine.evaluate_dataset(df)

        # Update dataset service active dataset
        dataset_svc = get_dataset_service()
        dataset_svc.active_dataset_path = saved_path
        dataset_svc.storage.record_dataset_history({
            "filename": filename,
            "file_path": saved_path,
            "source_type": "EXTERNAL_REST_API",
            "endpoint_url": url,
            "total_records": len(df),
            "suitability_verdict": suitability_report["verdict"],
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        })

        self._config["last_connected_at"] = datetime.now(timezone.utc).isoformat()
        self._config["last_sync_status"] = "SUCCESS"
        self._config["last_record_count"] = len(df)
        self._config["error"] = None
        self._save_config()

        return {
            "success": True,
            "filename": filename,
            "file_path": saved_path,
            "total_records": len(df),
            "suitability": suitability_report,
            "message": f"Successfully ingested {len(df)} records from API into active pipeline."
        }

_datasource_service = None

def get_datasource_service() -> DataSourceService:
    global _datasource_service
    if _datasource_service is None:
        _datasource_service = DataSourceService()
    return _datasource_service
