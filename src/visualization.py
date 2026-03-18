"""Visualization and UI components module."""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def render_header():
    """Render application header."""
    st.set_page_config(page_title="LoanLens", layout="wide")
    st.title("📊 LoanLens - Loan Default Prediction")


def render_sidebar():
    """Render sidebar with model upload."""
    st.sidebar.header("🔍 Model Upload")
    return st.sidebar.file_uploader(
        "Upload one or more trained models", 
        type=["pkl"], 
        accept_multiple_files=True
    )


def render_data_upload():
    """Render data upload section."""
    st.header("📄 Upload CSV Data")
    return st.file_uploader("Upload your input CSV file", type=["csv"])


def render_data_preview(df: pd.DataFrame):
    """Render data preview."""
    st.write("📌 Sample Data Preview:")
    st.dataframe(df)


def render_error(message: str):
    """Render error message."""
    st.error(f"❌ {message}")


def render_warning(message: str):
    """Render warning message."""
    st.warning(f"⚠️ {message}")


def render_success(message: str):
    """Render success message."""
    st.success(f"✅ {message}")


def render_predictions(predictions_df: pd.DataFrame):
    """Render predictions section."""
    st.subheader("📈 Predictions")
    st.dataframe(predictions_df)


def render_threshold_tuning(
    model_names: list, 
    default_threshold: float = 0.5
) -> tuple:
    """Render threshold tuning section.
    
    Returns:
        Tuple of (selected_model_idx, threshold)
    """
    st.subheader("🎯 Threshold Tuning & Default Classification")
    
    selected_idx = st.selectbox(
        "Select a model to evaluate",
        range(len(model_names)),
        format_func=lambda i: model_names[i],
        key="eval_model"
    )
    
    threshold = st.slider(
        "Select classification threshold for 'default'", 
        0.0, 1.0, default_threshold, 0.01
    )
    
    return selected_idx, threshold


def render_true_labels_upload():
    """Render true labels upload section."""
    return st.file_uploader(
        "📂 Upload true labels CSV (must include column `SeriousDlqin2yrs`)", 
        type=["csv"]
    )


def render_confusion_matrix(cm: np.ndarray):
    """Render confusion matrix."""
    st.markdown("### 🧮 Confusion Matrix")
    cm_df = pd.DataFrame(
        cm, 
        index=["Actual: No Default", "Actual: Default"],
        columns=["Predicted: No Default", "Predicted: Default"]
    )
    st.dataframe(cm_df)


def render_classification_report(report: Dict):
    """Render classification report."""
    st.markdown("### 📋 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())


def render_roc_auc(score: float):
    """Render ROC AUC score."""
    st.markdown(f"### 📈 ROC AUC Score: `{score:.4f}`")


def render_shap_section():
    """Render SHAP explainability section header."""
    st.subheader("📉 SHAP Explainability")


def render_model_selector(model_names: list, key: str = "model_select") -> int:
    """Render model selector for SHAP analysis."""
    return st.selectbox(
        "Select model for SHAP analysis", 
        range(len(model_names)), 
        format_func=lambda i: model_names[i],
        key=key
    )


def render_record_selector(max_index: int, key: str = "record_index") -> int:
    """Render record index selector."""
    st.markdown("### 🔎 SHAP Bar Plot Per Record")
    return st.number_input(
        "Select record index for Bar Plot", 
        min_value=0, 
        max_value=max_index, 
        step=1, 
        key=key
    )


def render_shap_summary():
    """Render SHAP summary plot section."""
    st.markdown("### SHAP Summary Plot (Global Feature Impact)")


def render_shap_record():
    """Render SHAP record section."""
    st.markdown("#### 🔹 Feature Impact for Selected Record")


def render_explanation_guide():
    """Render explanation guide."""
    st.markdown("---")
    st.markdown("### ℹ️ What Do These Predictions Mean?")
    st.markdown("""
    - **Prediction Output**: The `Default_Prob_*` columns show the model's confidence that a person will default.
    - **SHAP Summary Plot**: Highlights which features most influenced the model across all predictions.
    - **SHAP Bar Plot**: Shows per-record feature impact. Positive values push toward default; negative away.
    """)


def render_footer():
    """Render application footer."""
    st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; font-size: 0.9em; color: gray;'>"
        "Developed by Group 21 for the LoanLens project."
        "</div>",
        unsafe_allow_html=True
    )
