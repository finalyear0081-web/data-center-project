# End-to-End Enterprise Reference Process (ERP) & System Architecture Specification
## LSTM-Based Data Center Temperature Prediction & Predictive Cooling Optimization System

---

## Executive Overview

The **LSTM-Based Data Center Temperature Prediction and Predictive Cooling Optimization System** operates as an enterprise-grade cyber-physical platform that bridges multi-sensor telemetry ingestion, deep learning time-series forecasting, and automated cooling actuation. 

This document defines the complete **End-to-End Reference Process (ERP)**, outlining:
1. **The 8-Phase Core Operational Pipeline** (Ingestion $\rightarrow$ Validation $\rightarrow$ Isolated Preprocessing $\rightarrow$ Residual LSTM Training $\rightarrow$ Multi-Horizon Forecasting $\rightarrow$ Hysteresis Control $\rightarrow$ Alert Dispatch $\rightarrow$ Facility Visualization).
2. **The Enterprise Resource Planning & DCIM Integration Model** (Asset tracking, energy accounting, PUE/CoP calculation, and maintenance automation).
3. **The Entity-Relationship (ER) Data Model** (Entities, relationships, attributes, and foreign keys).
4. **Predictive Closed-Loop Hysteresis State Machine** (Anti-chatter chiller control logic).

---

## 1. End-to-End Process Flowchart

```mermaid
flowchart TD
    subgraph PHASE1 ["Phase 1: Ingestion & Schema Recognition"]
        UPLOAD["CSV Upload / Telemetry Ingestion"]
        DETECT{"Schema Detection Engine"}
        UPLOAD --> DETECT
        DETECT -->|"experiment_id present"| AAU["AAU Experimental Schema<br/>(16 Experiments, 2-min sampling)"]
        DETECT -->|"timestamp present"| HZ["HazardNet Operational Schema<br/>(Continuous stream, 10-min sampling)"]
    end

    subgraph PHASE2 ["Phase 2: Validation & Governance"]
        VAL["Multi-Schema Validation Engine"]
        AAU --> VAL
        HZ --> VAL
        VAL --> CHECK{"Pass Physical & Schema Rules?"}
        CHECK -->|"No"| ERR["HTTP 422 Rejection + Diagnostic Report"]
        CHECK -->|"Yes"| CLEAN["Deduplication & Boundary-Aware Imputation"]
    end

    subgraph PHASE3 ["Phase 3: Preprocessing & Boundary Isolation"]
        ISO["Experiment Boundary Isolation Engine"]
        CLEAN --> ISO
        ISO --> SPLIT["Chronological Splitting (70% Train, 15% Val, 15% Test)"]
        SPLIT --> SCALER["Train-Only MinMax Normalization (Zero Data Leakage)"]
        SCALER --> SEQ_GEN["Sliding Sequence Generator (Input: W steps, Horizon: H steps)"]
    end

    subgraph PHASE4 ["Phase 4: Residual LSTM Training & Evaluation"]
        LSTM["Residual Stacked LSTM Network<br/>(Persistence Anchor Skip Connection)"]
        SEQ_GEN --> LSTM
        LSTM --> BENCH["Chronological Test Evaluation vs Baselines<br/>(Residual LSTM vs Persistence vs Random Forest)"]
        BENCH --> METRICS["Metrics Registry (MAE, RMSE, R²)"]
    end

    subgraph PHASE5 ["Phase 5: Multi-Horizon Forecasting"]
        RECENT["Historical Input Window (Last W steps of target experiment/stream)"]
        RECENT --> INFER["Multi-Step Inference Engine"]
        LSTM -.->|Model Weights| INFER
        SCALER -.->|Inverse Scaling| INFER
        INFER --> PROD_FC["Temperature Forecast Vector (H steps in °C)"]
    end

    subgraph PHASE6 ["Phase 6: Closed-Loop Cooling Optimization"]
        CONTROLLER["Predictive Cooling Controller with Hysteresis"]
        THRESH["Threshold Settings (ON >= 28°C, OFF <= 26°C)"]
        PROD_FC --> CONTROLLER
        THRESH --> CONTROLLER
        CONTROLLER --> DECISION["Actuation State: RUNNING (ON) / STOPPED (OFF)"]
        CONTROLLER --> PROJ["Projected Chiller Schedule & Running Hours"]
    end

    subgraph PHASE7 ["Phase 7: Enterprise Dispatch & Alerting"]
        ALERT_ENG["Thermal Alert Engine"]
        CONTROLLER --> ALERT_ENG
        ALERT_ENG -->|"Temp >= 29.5°C"| CRIT["CRITICAL Alarm -> Dispatch HVAC Tech"]
        ALERT_ENG -->|"Temp >= 28.0°C"| WARN["WARNING -> Pre-Cooling Activation"]
        ALERT_ENG -->|"Temp < 28.0°C"| NORM["NORMAL -> Standard Economizer Mode"]
    end

    subgraph PHASE8 ["Phase 8: Facility DCIM & UI Dashboard"]
        DASH["Real-Time Monitoring Dashboard"]
        API["Flask REST APIs (/api/dashboard, /api/prediction, /api/cooling)"]
        DECISION --> API
        PROD_FC --> API
        METRICS --> API
        ALERT_ENG --> API
        API --> DASH
    end
```

---

## 2. Functional Enterprise Modules (ERP Decomposition)

```
┌────────────────────────────────────────────────────────────────────────┐
│             DATA CENTER PREDICTIVE COOLING ERP PLATFORM                 │
├─────────────────┬─────────────────┬──────────────────┬─────────────────┤
│ 1. DATA         │ 2. MACHINE      │ 3. DECISION      │ 4. FACILITY     │
│ GOVERNANCE      │ LEARNING & AI   │ & HVAC CONTROL   │ & ENERGY ERP    │
├─────────────────┼─────────────────┼──────────────────┼─────────────────┤
│ • Multi-Schema  │ • Residual      │ • Dual-Threshold │ • PUE & CoP     │
│   Ingestion     │   Stacked LSTM  │   Hysteresis     │   Accounting    │
│ • Experiment    │ • Persistence   │ • Anti-Chatter   │ • Work Order    │
│   Isolation     │   Anchoring     │   Hold Logic     │   Generation    │
│ • Domain Bounds │ • Multi-Horizon │ • Manual/Auto    │ • Chiller Run-  │
│   Verification  │   Projection    │   Overrides      │   Time Tracking │
│ • Train-Only    │ • Baseline      │ • Thermal Alarm  │ • Historical    │
│   MinMax Scale  │   Benchmarking  │   Escalation     │   Auditing      │
└─────────────────┴─────────────────┴──────────────────┴─────────────────┘
```

---

## 3. Entity-Relationship (ER) Data Model

```mermaid
erDiagram
    DATASET ||--o{ EXPERIMENT : "contains"
    DATASET ||--o{ TELEMETRY_READING : "stores"
    EXPERIMENT ||--o{ TELEMETRY_READING : "groups"
    DATASET ||--o{ MODEL_VERSION : "trains"
    MODEL_VERSION ||--o{ EVALUATION_METRIC : "produces"
    MODEL_VERSION ||--o{ FORECAST_RUN : "generates"
    FORECAST_RUN ||--o{ FORECAST_STEP : "contains"
    FORECAST_RUN ||--o{ COOLING_DECISION : "drives"
    COOLING_DECISION ||--o{ ALERT_LOG : "triggers"

    DATASET {
        string dataset_id PK
        string filename
        string schema_type
        string schema_name
        int total_records
        int total_experiments
        float sampling_interval_minutes
        datetime uploaded_at
        boolean is_active
    }

    EXPERIMENT {
        string experiment_id PK
        string dataset_id FK
        string configuration_name
        float it_load_percentage
        int record_count
        float duration_hours
    }

    TELEMETRY_READING {
        int reading_id PK
        string dataset_id FK
        string experiment_id FK
        datetime timestamp
        float elapsed_time_minutes
        float inlet_temperature_c
        float outlet_temperature_c
        float cooling_supply_temp_c
        float cooling_return_temp_c
        float it_power_kw
        float facility_power_kw
        float pue
        float cop
    }

    MODEL_VERSION {
        string model_id PK
        string dataset_id FK
        string architecture_type
        int input_window_steps
        int forecast_horizon_steps
        int epochs_trained
        float final_val_loss
        string weights_filepath
        string scaler_filepath
        datetime trained_at
    }

    EVALUATION_METRIC {
        int metric_id PK
        string model_id FK
        string model_name
        float mae_celsius
        float rmse_celsius
        float r2_score
        float high_temp_mae
        int test_sample_count
    }

    FORECAST_RUN {
        string forecast_id PK
        string model_id FK
        string experiment_id FK
        datetime generated_at
        float predicted_min_celsius
        float predicted_max_celsius
        float predicted_avg_celsius
        int total_steps
    }

    FORECAST_STEP {
        int step_id PK
        string forecast_id FK
        int step_number
        string timestamp_label
        float predicted_temperature_c
        string projected_chiller_state
    }

    COOLING_DECISION {
        string decision_id PK
        string forecast_id FK
        string operating_mode
        string current_state
        float trigger_temperature
        float on_threshold
        float off_threshold
        datetime expected_activation_time
        float projected_running_hours
        string activation_reason
    }

    ALERT_LOG {
        string alert_id PK
        string decision_id FK
        string severity
        string title
        string message
        float metric_value
        datetime created_at
        boolean acknowledged
    }
```

---

## 4. Closed-Loop Predictive Cooling State Machine

```mermaid
stateDiagram-v2
    [*] --> STOPPED : Initial State (Standby)

    STOPPED --> RUNNING : Inlet Temp >= 28.0°C (Trigger Activation)
    STOPPED --> STOPPED : Inlet Temp < 28.0°C (Stay in Standby)

    RUNNING --> STOPPED : Inlet Temp <= 26.0°C (Trigger Deactivation)
    RUNNING --> RUNNING : Inlet Temp > 26.0°C (Continue Cooling)

    note right of RUNNING
        Deadband (26.0°C - 28.0°C):
        If currently RUNNING, chiller stays RUNNING.
        Prevents equipment short-cycling.
    end note

    note left of STOPPED
        Deadband (26.0°C - 28.0°C):
        If currently STOPPED, chiller stays STOPPED.
        Prevents premature compressor starts.
    end note

    state MANUAL_OVERRIDE {
        [*] --> MANUAL_RUNNING : Operator sets ON
        [*] --> MANUAL_STOPPED : Operator sets OFF
    }

    STOPPED --> MANUAL_OVERRIDE : Operator manual switch
    RUNNING --> MANUAL_OVERRIDE : Operator manual switch
    MANUAL_OVERRIDE --> STOPPED : Reset to AUTOMATIC (T <= 26°C)
    MANUAL_OVERRIDE --> RUNNING : Reset to AUTOMATIC (T >= 28°C)
```

---

## 5. End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Facility Operator / Engineer
    participant UI as Web Dashboard
    participant API as Flask REST Gateway
    participant Ingestion as Dataset Service
    participant Engine as LSTM Training Engine
    participant Infer as Predictor Service
    participant Controller as Cooling Controller
    participant Storage as File & Cache Store

    User->>UI: Uploads CSV Dataset
    UI->>API: POST /api/dataset/upload (multipart/form-data)
    API->>Ingestion: handle_upload(file)
    Ingestion->>Storage: Save raw file to storage/uploads/
    Ingestion->>Ingestion: validate_csv_dataset(saved_path)
    
    alt Validation Failed
        Ingestion-->>API: Return Validation Errors (Missing cols, bad bounds)
        API-->>UI: 422 Unprocessable Entity + Diagnostic Report
        UI-->>User: Display Error Modal with Required Schema Columns
    else Validation Succeeded
        Ingestion-->>API: 202 Accepted (Pipeline Job Initiated)
        API-->>UI: Return Job Token & Dataset Metadata Profile
        UI-->>User: Display Active Dataset Profile & Progress Bar
    end

    Note over Ingestion,Engine: Background Asynchronous Processing Pipeline

    Ingestion->>Engine: start_pipeline_job(file_path, force_retrain=True)
    Engine->>Engine: clean_and_repair_dataset()
    Engine->>Engine: split_chronologically() [Per-experiment isolation]
    Engine->>Engine: DataCenterScaler.fit(train_df strictly)
    Engine->>Engine: create_experiment_isolated_sequences()
    Engine->>Engine: Train Residual LSTM (10 Epochs, EarlyStopping)
    Engine->>Storage: Save Model (.keras) & Scaler (.pkl)
    Engine->>Engine: run_model_evaluation() [Benchmark vs Baselines]
    Engine->>Storage: Save evaluation_metrics.json & loss_history.json

    Engine->>Infer: predict_24h(clean_df, experiment_id)
    Infer->>Infer: Load model + Scaler & Predict multi-step sequence
    Infer-->>Controller: Pass Forecast Temperatures & Timestamps
    Controller->>Controller: evaluate_forecast(thresholds: 28°C ON, 26°C OFF)
    Controller->>Storage: Update latest_forecast.json & cooling_events.json

    loop Progress Polling
        UI->>API: GET /api/dataset/status
        API-->>UI: Return State: TRAINING (40%) -> PREDICTING (75%) -> COMPLETED (100%)
    end

    UI->>API: GET /api/dashboard/summary & /api/dashboard/forecast
    API->>Storage: Read latest_forecast.json & telemetry
    API-->>UI: Return Current State, Forecast Curve, Chiller Status
    UI-->>User: Refresh Dynamic Charts, Thermal Alerts, and Chiller State
```

---

## 6. Enterprise Integration & Maintenance Roadmap

| Integration Layer | Protocol / Interface | Enterprise Function |
| :--- | :--- | :--- |
| **BMS / SCADA** | BACnet/IP, Modbus TCP | Dispatches physical start/stop commands to variable-speed chillers and chilled-water pumps. |
| **IT Telemetry** | Redfish API, IPMI, SNMPv3 | Ingests real-time server rack power consumption, inlet airflow, and internal compute load. |
| **Enterprise ERP** | REST API / Webhooks | Logs daily cooling energy consumption ($kWh$), calculates operational expenditures ($\$$), and tracks PUE trends. |
| **ITSM (ServiceNow)** | Incident Management API | Automatically triggers high-priority incident tickets when forecasted inlet temperatures exceed $29.5^\circ\text{C}$ for $> 30$ continuous minutes. |
| **Asset CMMS** | Maintenance Work Orders | Monitors compressor start/stop cycle frequency and flags maintenance intervals when runtime exceeds 2,500 hours. |
