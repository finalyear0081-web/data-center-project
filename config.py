"""
Configuration Module for Data Center Predictive Cooling System
Handles environment variables, security keys, storage paths, and defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "datacenter-predictive-cooling-secret-key-2026")
    ENVIRONMENT = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))
    HOST = os.getenv("HOST", "0.0.0.0")

    # File paths
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "data_center_temperature_dataset.csv")
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "dataset", "uploads")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
    MODEL_PATH = os.path.join(MODEL_DIR, "lstm_temperature_model.keras")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

    # Firebase Admin SDK Configuration
    FIREBASE_SERVICE_ACCOUNT_KEY = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
    FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "datacenter-cooling-system")

    # Cloudinary Configuration
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

    # Cooling defaults
    DEFAULT_ON_THRESHOLD = 28.0
    DEFAULT_OFF_THRESHOLD = 26.0

    # Keras & TF sandbox defaults
    KERAS_HOME = os.getenv("KERAS_HOME", os.path.join(PROJECT_ROOT, ".keras"))
    TF_ENABLE_ONEDNN_OPTS = "0"
