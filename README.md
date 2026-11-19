# LSTM-Based Data Center Temperature Prediction and Predictive Cooling Optimization System

A complete, production-structured industrial IoT platform combining multivariate time-series deep learning (stacked LSTM), automated CSV ingestion and validation, dual-mode predictive chiller control (Automatic Hysteresis & Manual Override), and an interactive data-center monitoring web interface.

> **Academic & Experimental Notice**:
> This platform operates using simulated sensor readings and a software-simulated chiller controller for research, forecasting, and optimization evaluation. There is no direct connection to high-voltage physical cooling machinery.

---

## 1. Project Architecture & End-to-End Workflow

The platform operates through an automated pipeline:

```text
User CSV Upload (Web UI)
        ↓
Strict Schema Validation (10 Sensor Columns, Monotonic Timestamps, Boundary Checks)
        ↓
Automated Sampling Detection (5m -> 288 steps, 15m -> 96 steps, 60m -> 24 steps)
        ↓
Data Preprocessing (Deduplication, Time-Series Interpolation, Train-Only MinMax Normalization)
        ↓
LSTM Model Training / Loading (Stacked LSTM with 24-Hour Horizon Dense Projection)
        ↓
Next 24-Hour Temperature Prediction (Unscaled Celsius Forecast Array)
        ↓
Predictive Cooling Evaluation (Dual Setpoint Hysteresis: ON >= 28.0°C, OFF <= 26.0°C)
        ↓
Real-Time Dashboard & Chiller Component (RUNNING vs STOPPED with Animated Fan & Manual Overrides)
```

---

## 2. Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6), Bootstrap 5.3, Chart.js 4.4, Lucide Icons
- **Backend**: Python 3.11, Flask 3.x, Gunicorn, REST APIs, Flask-CORS
- **Machine Learning**: TensorFlow 2.x, Keras, NumPy, Pandas, Scikit-learn, Joblib
- **Deployment & DevOps**: Docker, Render, VS Code

---

## 3. Production Project Folder Structure

```text
data-center-predictive-cooling/
│
├── frontend/
│   ├── templates/
│   │   ├── base.html                  # Core layout with sidebar navigation & topbar
│   │   ├── dashboard.html             # Main telemetry overview & KPI metrics
│   │   ├── dataset.html               # Drag & drop upload, validation & preview
│   │   ├── predictions.html           # 24h forecast chart & paginated 288-step table
│   │   ├── cooling.html               # Visual chiller component & Auto/Manual controls
│   │   ├── model_performance.html     # Loss curves, actual vs predicted, benchmarks
│   │   ├── login.html                 # Operator sign-in page
│   │   └── register.html              # Operator enrollment page
│   │
│   └── static/
│       ├── css/
│       │   ├── style.css              # Industrial clean theme & color tokens
│       │   ├── dashboard.css          # KPI cards, chiller fan animations, tables
│       │   └── responsive.css         # Mobile and tablet viewport styles
│       └── js/
│           ├── app.js                 # Global sync & icon initialization
│           ├── upload.js              # Drag-and-drop & pipeline progress polling
│           ├── dashboard.js           # Overview KPIs and live polling
│           ├── charts.js              # Chart.js visualization configurations
│           ├── predictions.js         # Forecast rendering, search, sort, pagination
│           └── cooling.js             # Chiller visual state, Auto/Manual mode actions
│
├── backend/
│   ├── app.py                         # Production Flask factory and UI routes
│   ├── config.py                      # Centralized configuration & file paths
│   ├── extensions.py                  # Flask extensions (CORS)
│   │
│   ├── routes/
│   │   ├── upload_routes.py           # POST /api/dataset/upload, GET status
│   │   ├── dataset_routes.py          # GET /api/dataset/preview, statistics, history
│   │   ├── prediction_routes.py       # POST forecast, GET results, GET export
│   │   ├── cooling_routes.py          # GET status, POST mode, POST manual, PUT thresholds
│   │   ├── dashboard_routes.py        # GET summary, temperature, forecast, alerts
│   │   └── model_routes.py            # POST train, GET status, GET metrics, GET summary
│   │
│   ├── preprocessing/
│   │   ├── validation.py              # CSV schema, 10 columns, intervals, duplicates
│   │   ├── data_loader.py             # CSV parsing and statistical summaries
│   │   ├── cleaning.py                # Sensor bounds repair & linear interpolation
│   │   ├── normalization.py           # Train-only MinMax scaling (leakage prevention)
│   │   └── sequence_generator.py      # Chronological 70/15/15 splits & sliding windows
│   │
│   ├── models/
│   │   ├── lstm_model.py              # Stacked LSTM multi-step neural network builder
│   │   ├── train.py                   # Asynchronous background training pipeline
│   │   ├── predict.py                 # Multi-step 24h temperature forecasting engine
│   │   ├── evaluation.py              # Benchmark comparison (LSTM, RF, Persistence)
│   │   └── saved/                     # Default pre-trained model & scaler weights
│   │
│   ├── cooling/
│   │   ├── controller.py              # Dual-mode state machine (Automatic / Manual)
│   │   ├── threshold.py               # Hysteresis safety limits & setpoint manager
│   │   └── alerts.py                  # Thermal excursion early warnings & logs
│   │
│   ├── services/
│   │   ├── dataset_service.py         # File ingestion, active dataset manager
│   │   ├── prediction_service.py      # Background pipeline orchestration & caching
│   │   └── storage_service.py         # Local persistent file & metadata manager
│   │
│   └── utils/
│       ├── response.py                # Standardized success & error JSON helpers
│       └── logging_config.py          # Structured application logger
│
├── storage/
│   ├── uploads/                       # Persisted user-uploaded CSV datasets
│   ├── models/                        # Persisted trained model weights (.keras) & scalers
│   └── predictions/                   # Cached forecasts and job status JSON
│
├── tests/
│   ├── test_upload.py                 # Upload API, format rejection, job status
│   ├── test_validation.py             # 10 columns check, missing column reporting
│   ├── test_preprocessing.py          # Interpolation, chronological split, leakage test
│   ├── test_prediction.py             # Exactly 288 records verification & bounds
│   ├── test_cooling.py                # Hysteresis deadband, manual ON/OFF override
│   └── test_api.py                    # Complete REST API & HTML routes integration
│
├── app.py                             # Root execution entry point
├── requirements.txt                   # Production dependencies
├── Dockerfile                         # Container definition with healthcheck
├── render.yaml                        # Render cloud deployment blueprint
├── .env.example                       # Sample environment configuration
├── .gitignore                         # Git exclusion rules
└── README.md                          # Documentation and user manual
```

---

## 4. Empirical Evaluation Results (IEEE Benchmark)

Evaluated on the held-out 15% chronological test split (634 sequences of 288-step forecasts) using the same input horizon and features:

| Model Architecture | MAE (°C) | RMSE (°C) | $R^2$ Score | Forecast Trajectory |
|---|---|---|---|---|
| **Stacked LSTM (Proposed)** | **0.5434** | **0.6833** | **0.7817** | Smooth multi-step curve capturing diurnal trends |
| **Random Forest Regressor** | 0.5486 | 0.6901 | 0.7773 | Multi-output ensemble with tree-based split |
| **Persistence Baseline** | 1.6305 | 2.0068 | -0.8830 | Propagates current temperature across 288 steps |

*All metrics were calculated using actual model predictions against test ground truth without synthetic fabrication.*

---

## 5. CSV Dataset Upload Requirements

The web application allows users to upload custom data center telemetry CSV files via drag-and-drop or file browsing.

### Required 10 Sensor Columns:
1. `timestamp` (e.g. `2024-01-01 00:00:00`)
2. `inlet_temperature` (**Primary Prediction Target**, °C)
3. `outlet_temperature` (°C)
4. `hot_aisle_temperature` (°C)
5. `cold_aisle_temperature` (°C)
6. `humidity` (%)
7. `cpu_usage` (%)
8. `it_power_consumption` (kW)
9. `outdoor_temperature` (°C)
10. `cooling_power_consumption` (kW)

### Sampling Intervals and Forecast Horizons:
The system automatically detects the sampling interval from consecutive timestamps:
- **5-Minute Interval**: Yields **288 predicted values** for 24 hours.
- **10-Minute Interval**: Yields **144 predicted values** for 24 hours.
- **15-Minute Interval**: Yields **96 predicted values** for 24 hours.
- **30-Minute Interval**: Yields **48 predicted values** for 24 hours.
- **60-Minute Interval**: Yields **24 predicted values** for 24 hours.

---

## 6. Dual-Mode Chiller Control & Visual Component

### Visual Chiller Indicator:
- **RUNNING (ON)**: Visual cabinet glows green, status pill displays `RUNNING`, CSS fan animation spins actively, power consumption reflects active cooling (135.0 kW), and the activation reason is logged.
- **STOPPED (OFF)**: Status displays `STOPPED` with neutral gray indicator, fan animation stops completely, power consumption drops to standby (48.0 kW), and deactivation reason is logged.

### Operating Modes:
1. **Automatic Mode (LSTM Forecast Driven)**:
   - Uses dual-threshold hysteresis:
     $$\text{State}_{t} = \begin{cases} \text{RUNNING}, & \text{if } T \ge 28.0^\circ\text{C} \\ \text{STOPPED}, & \text{if } T \le 26.0^\circ\text{C} \\ \text{State}_{t-1}, & \text{if } 26.0^\circ\text{C} < T < 28.0^\circ\text{C} \end{cases}$$
   - Evaluates the entire 24h forecast, logs prospective threshold crossings, and displays expected activation timestamps.
2. **Manual Mode (Operator Override)**:
   - Buttons: **TURN CHILLER ON** and **TURN CHILLER OFF**.
   - Overrides automatic logic immediately, updating backend state and frontend without page reload.
   - Returning to Automatic Mode recalculates state based on current forecast and hysteresis setpoints.

---

## 7. Local Installation & Execution (VS Code / Terminal)

### Step 1: Navigate to Workspace
```bash
cd "c:\Users\Eswar\OneDrive\Desktop\data center"
```

### Step 2: Set Environment Variables & Run Automated Test Suites
Run the comprehensive test suite verifying upload handling, validation, preprocessing, prediction dimensions, hysteresis, and REST APIs:
```powershell
$env:KERAS_HOME="$pwd\.keras"
$env:TF_ENABLE_ONEDNN_OPTS="0"
python -m unittest discover tests
```
*Expected output: `Ran 30 tests ... OK`.*

### Step 3: Start the Web Application
```powershell
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/dashboard
```

### Navigation Pages:
- **Overview**: `http://localhost:5000/dashboard`
- **Dataset Upload**: `http://localhost:5000/dataset`
- **24h Predictions**: `http://localhost:5000/predictions`
- **Chiller Control**: `http://localhost:5000/cooling`
- **Model Performance**: `http://localhost:5000/model-performance`

---

## 8. Docker & Render Cloud Deployment

### Run with Docker Locally:
```bash
docker build -t datacenter-predictive-cooling:latest .
docker run -p 5000:5000 datacenter-predictive-cooling:latest
```
Access at `http://localhost:5000`.

### Deploy on Render:
1. Push this project to GitHub.
2. In Render, select **New +** $\to$ **Blueprint**.
3. Link your repository. Render automatically reads `render.yaml`, builds the production Docker container, binds to `$PORT`, and probes `/health` for deployment readiness.
