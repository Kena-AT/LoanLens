import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import numpy as np
from shap import Explanation
import streamlit as st

st.set_page_config(page_title="LoanLens", layout="wide")
st.title("📊 LoanLens - Loan Default Prediction")

st.sidebar.header("🔍 Model Upload")
model_files = st.sidebar.file_uploader("Upload one or more trained models", type=["pkl"], accept_multiple_files=True)

st.header("📄 Upload CSV Data")
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

df = None
models = []
model_names = []

# Load uploaded models
for model_file in model_files or []:
    try:
        model = joblib.load(model_file)
        models.append(model)
        model_names.append(model_file.name)
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model {model_file.name}: {e}")

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📌 Sample Data Preview:")
        st.dataframe(df)

        # Drop unnecessary columns
        df.drop(columns=[col for col in ['Unnamed: 0', 'SeriousDlqin2yrs'] if col in df.columns], inplace=True, errors='ignore')

        # Required features
        required_features = [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents"
        ]
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            st.error(f"❗ Missing required features: {missing}")
            df = None

    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

# Prediction and SHAP logic
if df is not None and models:
    st.subheader("📈 Predictions")

    comparison_df = pd.DataFrame()
    for i, model in enumerate(models):
        try:
            proba = model.predict_proba(df)[:, 1]
            col_name = f"Default_Prob_{model_names[i]}"
            df[col_name] = proba
            comparison_df[model_names[i]] = proba
        except Exception as e:
            st.error(f"❌ Error using model {model_names[i]}: {e}")

    st.dataframe(df[[col for col in df.columns if col.startswith("Default_Prob_")]])


    # 🔍 Threshold Tuning & Evaluation Section
    st.subheader("🎯 Threshold Tuning & Default Classification")

    selected_model_idx_eval = st.selectbox(
        "Select a model to evaluate",
        range(len(models)),
        format_func=lambda i: model_names[i],
        key="eval_model"
    )

    selected_model_eval = models[selected_model_idx_eval]
    selected_model_name_eval = model_names[selected_model_idx_eval]
    selected_proba = df[f"Default_Prob_{selected_model_name_eval}"]

    threshold = st.slider("Select classification threshold for 'default'", 0.0, 1.0, 0.5, 0.01)
    predicted_labels = (selected_proba >= threshold).astype(int)

    # True labels uploader
    y_true_file = st.file_uploader("📂 Upload true labels CSV (must include column `SeriousDlqin2yrs`)", type=["csv"])

    if y_true_file is not None:
        try:
            y_true_df = pd.read_csv(y_true_file)
            if 'SeriousDlqin2yrs' in y_true_df.columns:
                from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

                y_true = y_true_df['SeriousDlqin2yrs']

                st.markdown("### 🧮 Confusion Matrix")
                cm = confusion_matrix(y_true, predicted_labels)
                cm_df = pd.DataFrame(cm, index=["Actual: No Default", "Actual: Default"],
                                        columns=["Predicted: No Default", "Predicted: Default"])
                st.dataframe(cm_df)

                st.markdown("### 📋 Classification Report")
                report = classification_report(y_true, predicted_labels, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                roc_auc = roc_auc_score(y_true, selected_proba)
                st.markdown(f"### 📈 ROC AUC Score: `{roc_auc:.4f}`")
            else:
                st.warning("Uploaded CSV must contain a column named 'SeriousDlqin2yrs'.")
        except Exception as e:
            st.error(f"Error loading true labels: {e}")


    # SHAP Explainability
    
    st.subheader("📉 SHAP Explainability")

    selected_model_idx = st.selectbox("Select model for SHAP analysis", range(len(models)), format_func=lambda i: model_names[i])
    selected_model = models[selected_model_idx]

    try:
        explainer = shap.TreeExplainer(selected_model)
        shap_values_raw = explainer.shap_values(df)
        expected_value = explainer.expected_value

        # Handle binary/multiclass model
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]  # Probabilities for class 1 (default)
            expected_value = expected_value[1]
        else:
            shap_values = shap_values_raw

        # SHAP Summary Plot
        st.markdown("### SHAP Summary Plot (Global Feature Impact)")
        fig_summary = shap.summary_plot(shap_values, df, show=False)
        st_shap(fig_summary, height=400)

        # SHAP Bar Plot Per Record
        st.markdown("### 🔎 SHAP Bar Plot Per Record")
        record_index = st.number_input("Select record index for Bar Plot", min_value=0, max_value=df.shape[0]-1, step=1, key="bar_plot_index")

        # Create SHAP Explanation object
        record_values = shap_values[record_index]                   # shape: (n_features,)
        record_base_value = expected_value                         # scalar
        record_data = df.iloc[record_index].values                 # shape: (n_features,)
        record_feature_names = df.columns.tolist()

        explanation = shap.Explanation(
            values=np.array(record_values),
            base_values=np.array([record_base_value]),   # Wrap scalar
            data=np.array([record_data]),                # Make 2D
            feature_names=record_feature_names
        )

        st.markdown("#### 🔹 Feature Impact for Selected Record")
        st_shap(shap.plots.bar(explanation), height=400)

        # Explanation Section
        st.markdown("---")
        st.markdown("### ℹ️ What Do These Predictions Mean?")
        st.markdown("""
        - **Prediction Output**: The `Default_Prob_*` columns show the model's confidence that a person will default.
        - **SHAP Summary Plot**: Highlights which features most influenced the model across all predictions.
        - **SHAP Bar Plot**: Shows per-record feature impact. Positive values push toward default; negative away.
        """)

    except Exception as e:
        st.error(f"❌ SHAP explainability error: {e}")

# Add spacer to push footer to bottom
st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)

# Footer
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; font-size: 0.9em; color: gray;'>"
    "Developed by Group 21 for the LoanLens project."
    "</div>",
    unsafe_allow_html=True
)