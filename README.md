# LoanLens 🚀

**LoanLens** is a full-stack, state-of-the-art Loan Default Prediction system. It combines high-performance machine learning (XGBoost) with modern web technologies (React & FastAPI) and Explainable AI (SHAP) to provide transparent and actionable credit risk insights.

![LoanLens Banner](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react)
![XGBoost](https://img.shields.io/badge/XGBoost-1e3a8a?style=for-the-badge)

## ✨ Core Features

- **Advanced Modeling**: XGBoost classifier with SMOTETomek resampling to handle extreme class imbalance (93% default rate).
- **Embedded Pipeline**: Model artifacts now include a full `StandardScaler` pipeline, allowing seamless prediction without manual feature scaling.
- **Explainable AI (XAI)**: SHAP-based feature importance scores integrated directly into every prediction.
- **Full-Stack Architecture**:
  - **Backend**: FastAPI with async support for real-time predictions, model management, and health monitoring.
  - **Frontend**: Responsive React dashboard with interactive metrics and model management UI.
- **Batch Processing**: Support for CSV-based batch predictions with downloadable results.
- **Dockerized Ready**: Fully containerized setup for easy deployment.

## 📂 Project Structure

```text
LoanLens/
├── api/                    # FastAPI Backend
│   ├── main_extended.py    # Main API with all endpoints
├── frontend/               # React Dashboard (Material UI)
├── src/                    # Core ML Logic
│   ├── train_model.py      # Fixed Pipeline training (SMOTE + Scaler)
│   ├── models.py           # Model Manager for inference
│   ├── data_handler.py     # CSV validation and cleaning
│   └── monitoring.py       # Health and drift detection logic
├── data/                   # Dataset Storage
├── notebooks/              # EDA & Research Notebooks
├── test_low_risk.csv       # Verification data for healthy profiles
├── test_high_risk.csv      # Verification data for default profiles
├── run_api.py              # Easy launcher for backend
├── config.yaml             # Global project configuration
└── requirements.txt        # Python dependencies
```

## 🛠️ Quick Start

### 1. Requirements

Ensure you have **Python 3.10+** and **Node.js** installed.

### 2. Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
python api/main_extended.py
```

> The API will be available at: [http://localhost:8000](http://localhost:8000)
> Interactive Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Frontend Setup

```bash
cd frontend
npm install  # First time only
npm start
```

> The Dashboard will be available at: [http://localhost:3000](http://localhost:3000)

## 📊 Model Training

We resolved the common "Always Predicts High Risk" bias by implementing an `ImbPipeline` that correctly isolates SMOTE logic to training folds and embeds scaling within the model artifact.

To retrain the model from scratch:

```bash
python src/train_model.py
```

*Current Metrics: ROC AUC ~0.94, F1-Score ~0.88*

## 🧪 Testing

You can use the provided testing files in the root directory:

- **`test_low_risk.csv`**: Five records of users with high income and low utilization (Expected: LOW RISK).
- **`test_high_risk.csv`**: Five records of users with past-due history and high utilization (Expected: HIGH RISK).
