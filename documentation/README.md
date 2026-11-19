# LSTM-Based Data Center Temperature Prediction and Predictive Cooling Optimization System

An intelligent, end-to-end industrial IoT predictive cooling platform combining time-series deep learning (stacked LSTM), physics-aware thermodynamic simulation, hysteresis control algorithms, and a real-time web monitoring dashboard.

> **Academic & Experimental Disclaimer**:
> This project operates using high-fidelity synthetic simulated sensor telemetry and simulated chiller control logic for research and evaluation purposes. It does not interface with physical actuators or high-voltage cooling machinery.

---

## 1. Project Highlights & Key Features

- **1-Year Synthetic Sensor Dataset**: Exactly 105,120 records at 5-minute sampling intervals across 10 physical data center columns.
- **Strict Data-Leakage Prevention**: Chronological 70% Train, 15% Validation, 15% Test split. Min-Max scalers fit solely on training data.
- **Stacked LSTM Multi-Step Predictor**: 64-unit LSTM $\to$ Dropout(0.2) $\to$ 32-unit LSTM $\to$ Dropout(0.2) $\to$ Dense(32) $\to$ Dense(288) outputs. Generates a full 24-hour future temperature forecast in 5-minute intervals.
- **Predictive Hysteresis Controller**: Implements dual-threshold switching ($T_{\text{ON}} = 28.0^\circ\text{C}$, $T_{\text{OFF}} = 26.0^\circ\text{C}$) to prevent rapid chiller short-cycling.
- **Dynamic Alerts & Early Warnings**: Projects threshold crossings hours in advance, pinpointing the expected activation timestamp.
- **Interactive Monitoring Dashboard**: Built with Bootstrap 5 and Chart.js, featuring responsive telemetry inspection, 24-hour forecast overlays, dataset inspection, and model benchmarking.
- **Role-Based Security**: Dual roles (Admin and Operator) supporting Firebase Admin SDK token verification with seamless offline fallback.

---

## 2. Experimental Benchmark Results (IEEE Research Evaluation)

Evaluated on the held-out 15% chronological test split (634 sequences of 288-step forecasts) using the same input horizon and features:

| Model Architecture | MAE (°C) | RMSE (°C) | $R^2$ Score | Forecast Behavior |
|---|---|---|---|---|
| **LSTM Neural Network (Proposed)** | **0.5434** | **0.6833** | **0.7817** | Smooth, diurnal-tracking multi-step trajectory |
| **Random Forest Regressor** | 0.5486 | 0.6901 | 0.7773 | Multi-output ensemble with tree-based split |
| **Persistence Baseline (Last Known)** | 1.6305 | 2.0068 | -0.8830 | Propagates current temperature across 288 steps |

*All metrics are calculated using actual model predictions against ground truth without synthetic fabrication.*

---

## 3. Technology Stack

- **Backend**: Python 3.11 / Flask / Gunicorn / Flask-CORS
- **Machine Learning**: TensorFlow 2.x / Keras / NumPy / Pandas / Scikit-learn / Joblib
- **Frontend**: HTML5 / CSS3 / JavaScript (ES6) / Bootstrap 5.3 / Chart.js 4.4
- **Database & Storage**: Firestore (Firebase Admin SDK) with local fallback / Cloudinary
- **Containerization & Cloud**: Docker / Render / VS Code

---

## 4. Project Directory Structure

```text
data-center-predictive-cooling/
├── app.py                      # Main Flask application entry point
├── config.py                   # Configuration & environment variable settings
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition for production
├── render.yaml                 # Render Blueprint deployment specification
├── .env.example                # Sample environment variables
├── .gitignore                  # Git ignore rules
│
├── dataset/
│   ├── data_center_temperature_dataset.csv  # 105,120 sensor records
│   ├── generate_dataset.py     # Reproducible synthetic physics generator
│   └── uploads/                # Directory for uploaded user CSV datasets
│
├── models/
│   ├── lstm_model.py           # Keras stacked LSTM architecture definition
│   ├── train.py                # Training workflow with EarlyStopping & Checkpoints
│   ├── predict.py              # 288-step inference service & inverse scaling
│   ├── evaluate.py             # Benchmark comparison (LSTM, RF, Persistence)
│   └── saved/                  # Trained model, scaler, and evaluation metrics
│
├── preprocessing/
│   ├── data_loader.py          # CSV loading and 10-column validation
│   ├── cleaning.py             # Sensor physical boundary validation & imputation
│   ├── normalization.py        # Train-only MinMax scaling
│   └── sequence_generator.py   # Chronological split & sliding window sequences
│
├── cooling/
│   ├── controller.py           # Simulated chiller state machine & hysteresis
│   ├── threshold.py            # Safety bounds & threshold manager
│   └── alerts.py               # Early warning & critical thermal alert engine
│
├── routes/
│   ├── auth_middleware.py      # Role-based access control decorator
│   ├── auth_routes.py          # Token verification, login, register, logout
│   ├── dataset_routes.py       # Metadata, preview, statistics, and upload
│   ├── model_routes.py         # Retraining, status, metrics, and inference
│   ├── cooling_routes.py       # Chiller status, event logs, and thresholds
│   └── dashboard_routes.py     # Summary KPIs, telemetry, and 24h forecasts
│
├── services/
│   ├── firebase_service.py     # Firebase Admin SDK & Firestore with local fallback
│   ├── cloudinary_service.py   # Cloudinary CSV upload with local fallback
│   └── prediction_service.py   # System coordinator and caching layer
│
├── templates/
│   ├── login.html              # Sign-in page with demo credentials
│   ├── register.html           # User registration page
│   └── dashboard.html          # Professional real-time monitoring dashboard
│
├── static/
│   ├── css/style.css           # Clean, slate-gray industrial theme
│   └── js/
│       ├── dashboard.js        # Dashboard state & API synchronization
│       ├── charts.js           # Chart.js visualization engine
│       └── prediction.js       # Live inference & retrain polling
│
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing & splits
│   ├── test_model.py           # Unit tests for model dimensions & inference
│   ├── test_cooling.py         # Unit tests for hysteresis & thresholds
│   └── test_api.py             # Integration tests for REST endpoints
│
└── documentation/
    ├── README.md               # User manual and project documentation
    └── system_architecture.md  # Architectural diagrams and mathematical specs
```

---

## 5. Local Setup and Execution (VS Code / CLI)

### Prerequisites
- Python 3.10 or 3.11 installed
- Git

### Step 1: Clone or Navigate to Directory
```bash
cd "c:\Users\Eswar\OneDrive\Desktop\data center"
```

### Step 2: Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Automated Test Suites
Execute the comprehensive test suites verifying preprocessing, modeling, cooling logic, and API endpoints:
```bash
python -m unittest discover tests
```
*Expected result: All tests pass with code 0.*

### Step 5: Start the Local Development Server
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000/dashboard
```

### Demo Credentials
- **Administrator**: `admin@datacenter.internal` (Full control, threshold adjustment, retrain trigger)
- **Operator**: `operator@datacenter.internal` (View and monitor telemetry)

---

## 6. Docker Container Deployment

### Build Docker Image
```bash
docker build -t datacenter-predictive-cooling:latest .
```

### Run Docker Container Locally
```bash
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY="datacenter-production-secret" \
  datacenter-predictive-cooling:latest
```
Access the application at `http://localhost:5000`.

---

## 7. Render Cloud Deployment Instructions

1. Push your repository to GitHub.
2. Sign in to your [Render Dashboard](https://dashboard.render.com).
3. Click **New +** $\to$ **Blueprint**.
4. Connect your GitHub repository. Render will automatically detect `render.yaml`.
5. Configure Environment Variables in Render:
   - `SECRET_KEY`: (Auto-generated by Render)
   - `FIREBASE_PROJECT_ID`: (Optional, your Firebase project ID)
   - `FIREBASE_SERVICE_ACCOUNT_KEY`: (Optional, path or raw JSON)
   - `CLOUDINARY_CLOUD_NAME`: (Optional, Cloudinary account)
   - `CLOUDINARY_API_KEY`: (Optional)
   - `CLOUDINARY_API_SECRET`: (Optional)
6. Click **Apply**. Render will automatically build the Dockerfile and deploy the application with live health-check probes at `/health`.
