"""
Application Configuration Module
Defines paths, environment settings, security keys, and machine learning constraints.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("KERAS_HOME", os.path.join(BASE_DIR, ".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
MODELS_DIR = os.path.join(STORAGE_DIR, "models")
PREDICTIONS_DIR = os.path.join(STORAGE_DIR, "predictions")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "hazardnet_cleaned_aggregate_data.csv")

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "datacenter-predictive-cooling-secret-key-2026")
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    
    # Upload limits: 100 MB max file size
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", "csv", "xlsx", "xls", "json"}
    
    # Cloudinary Configuration
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
    CLOUDINARY_URL = os.getenv("CLOUDINARY_URL", "")
    CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "data_center_datasets")
    
    # Paths
    BASE_DIR = BASE_DIR
    FRONTEND_DIR = FRONTEND_DIR
    TEMPLATES_DIR = TEMPLATES_DIR
    STATIC_DIR = STATIC_DIR
    STORAGE_DIR = STORAGE_DIR
    UPLOADS_DIR = UPLOADS_DIR
    MODELS_DIR = MODELS_DIR
    PREDICTIONS_DIR = PREDICTIONS_DIR
    DATASET_DIR = DATASET_DIR
    DEFAULT_DATASET_PATH = DEFAULT_DATASET_PATH
    
    # Saved Model & Scaler paths
    MODEL_PATH = os.path.join(MODELS_DIR, "lstm_temperature_model.keras")
    SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
    METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    TRAINING_STATUS_PATH = os.path.join(MODELS_DIR, "training_status.json")
    JOB_STATUS_PATH = os.path.join(STORAGE_DIR, "job_status.json")
    
    # Cooling Hysteresis Setpoints (°C)
    DEFAULT_ON_THRESHOLD = float(os.getenv("DEFAULT_ON_THRESHOLD", 28.5))
    DEFAULT_OFF_THRESHOLD = float(os.getenv("DEFAULT_OFF_THRESHOLD", 26.5))
    
    # Keras & Sandbox defaults
    KERAS_HOME = os.getenv("KERAS_HOME", os.path.join(BASE_DIR, ".keras"))
    TF_ENABLE_ONEDNN_OPTS = "0"
