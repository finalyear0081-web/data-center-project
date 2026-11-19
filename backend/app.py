"""
Data Center Temperature Prediction & Predictive Cooling Optimization System
Production Flask Application Factory
"""
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("KERAS_HOME", os.path.join(PROJECT_ROOT, ".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from flask import Flask, render_template, redirect, url_for, jsonify
from backend.config import Config
from backend.extensions import cors
from backend.routes.upload_routes import upload_bp
from backend.routes.dataset_routes import dataset_bp
from backend.routes.prediction_routes import prediction_bp
from backend.routes.cooling_routes import cooling_bp
from backend.routes.dashboard_routes import dashboard_bp
from backend.routes.model_routes import model_bp
from backend.routes.datasource_routes import datasource_bp

def create_app(config_class=Config) -> Flask:
    app = Flask(
        __name__,
        template_folder=config_class.TEMPLATES_DIR,
        static_folder=config_class.STATIC_DIR
    )
    app.config.from_object(config_class)

    # Initialize extensions
    cors.init_app(app)

    # Register API Blueprints
    app.register_blueprint(upload_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(cooling_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(datasource_bp)

    # Frontend Navigation Web Routes (8 Primary Sections)
    @app.route("/")
    def index():
        return redirect(url_for("dashboard_view"))

    # 1. Overview
    @app.route("/dashboard")
    def dashboard_view():
        return render_template("dashboard.html", active_page="dashboard")

    # 2. Data Source Management
    @app.route("/datasources")
    def datasources_view():
        return render_template("datasources.html", active_page="datasources")

    # Legacy /dataset route aliases to datasources
    @app.route("/dataset")
    def dataset_view():
        return render_template("datasources.html", active_page="datasources")

    # 3. Dataset Analysis
    @app.route("/dataset-analysis")
    def dataset_analysis_view():
        return render_template("dataset_analysis.html", active_page="dataset_analysis")

    # 4. Model Training
    @app.route("/model-training")
    def model_training_view():
        return render_template("model_training.html", active_page="model_training")

    # 5. Model Performance
    @app.route("/model-performance")
    def model_performance_view():
        return render_template("model_performance.html", active_page="model_performance")

    # 6. 24-Hour Predictions
    @app.route("/predictions")
    def predictions_view():
        return render_template("predictions.html", active_page="predictions")

    # 7. Chiller Control
    @app.route("/cooling")
    def cooling_view():
        return render_template("cooling.html", active_page="cooling")

    # 8. Alerts and Reports
    @app.route("/alerts-reports")
    def alerts_reports_view():
        return render_template("alerts_reports.html", active_page="alerts_reports")

    @app.route("/login")
    def login_view():
        return render_template("login.html")

    @app.route("/register")
    def register_view():
        return render_template("register.html")

    # Health Check for Docker & Render deployment
    @app.route("/health")
    def health_check():
        model_ready = os.path.exists(config_class.MODEL_PATH) and os.path.exists(config_class.SCALER_PATH)
        return jsonify({
            "status": "healthy",
            "model_ready": model_ready,
            "system": "Data Center Predictive Cooling System",
            "version": "2.0.0"
        }), 200

    @app.errorhandler(404)
    def handle_404(e):
        return jsonify({"error": "Resource not found", "status": 404}), 404

    @app.errorhandler(500)
    def handle_500(e):
        return jsonify({"error": "Internal server error", "status": 500}), 500

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
