import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from streamlit_shap import st_shap

from src import visualization as viz

# Import modular components
from src.config import (
    APP_NAME,
    DEFAULT_THRESHOLD,
    DROP_COLUMNS,
    LAYOUT,
    PAGE_TITLE,
    REQUIRED_FEATURES,
)
from src.data_handler import DataHandler
from src.explainability import ExplainabilityAnalyzer
from src.models import ModelManager


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)
    st.title(f"📊 {APP_NAME} - Loan Default Prediction")

    # Initialize components
    model_manager = ModelManager()
    data_handler = DataHandler(REQUIRED_FEATURES, DROP_COLUMNS)

    # Sidebar - Model Upload
    st.sidebar.header("🔍 Model Upload")
    model_files = st.sidebar.file_uploader(
        "Upload one or more trained models", type=["pkl"], accept_multiple_files=True
    )

    # Load models
    if model_files:
        for model_file in model_files:
            try:
                model, name = model_manager.load_model(model_file)
                model_manager.add_model(model, name)
                st.sidebar.success(f"✅ Loaded: {name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load {model_file.name}: {e}")

    # Main - Data Upload
    st.header("📄 Upload CSV Data")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    df = None
    if uploaded_file:
        df = data_handler.prepare_data(uploaded_file)

        if df is not None:
            viz.render_data_preview(df)
        else:
            viz.render_error("Data validation failed. Check required features.")

    # Predictions
    if df is not None and len(model_manager) > 0:
        predictions_df = model_manager.predict_all(df)
        viz.render_predictions(predictions_df)

        for i, name in enumerate(model_manager.model_names):
            df[f"Default_Prob_{name}"] = predictions_df[name]

        # Threshold Tuning
        selected_idx, threshold = viz.render_threshold_tuning(model_manager.model_names, DEFAULT_THRESHOLD)

        selected_proba = predictions_df.iloc[:, selected_idx]
        predicted_labels = (selected_proba >= threshold).astype(int)

        # True labels evaluation
        y_true_file = viz.render_true_labels_upload()

        if y_true_file:
            try:
                y_true_df = pd.read_csv(y_true_file)
                if "SeriousDlqin2yrs" in y_true_df.columns:
                    y_true = y_true_df["SeriousDlqin2yrs"]

                    cm = confusion_matrix(y_true, predicted_labels)
                    viz.render_confusion_matrix(cm)

                    report = classification_report(y_true, predicted_labels, output_dict=True)
                    viz.render_classification_report(report)

                    roc_auc = roc_auc_score(y_true, selected_proba)
                    viz.render_roc_auc(roc_auc)
                else:
                    viz.render_warning("CSV must contain 'SeriousDlqin2yrs' column")
            except Exception as e:
                viz.render_error(f"Error loading true labels: {e}")

        # SHAP Explainability
        viz.render_shap_section()

        shap_model_idx = viz.render_model_selector(model_manager.model_names, "shap_model")
        selected_model = model_manager.get_model(shap_model_idx)

        try:
            analyzer = ExplainabilityAnalyzer(selected_model)
            analyzer.compute_shap_values(df)

            viz.render_shap_summary()
            fig_summary = analyzer.get_summary_plot(df)
            st_shap(fig_summary, height=400)

            record_idx = viz.render_record_selector(len(df) - 1, "bar_plot")
            explanation = analyzer.get_explanation(record_idx, df)

            viz.render_shap_record()
            fig_bar = analyzer.get_bar_plot(explanation)
            st_shap(fig_bar, height=400)

        except Exception as e:
            viz.render_error(f"SHAP analysis error: {e}")

    viz.render_explanation_guide()
    viz.render_footer()


if __name__ == "__main__":
    main()
