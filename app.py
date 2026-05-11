import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Project paths and constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "credit_risk_model.pkl"
FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "emp_length",
    "term",
]

LOW_RISK_CUTOFF = 0.20
HIGH_RISK_CUTOFF = 0.50


# -----------------------------
# Page setup and styling
# -----------------------------
st.set_page_config(
    page_title="AI-Powered Credit Risk Assistant",
    page_icon="credit-risk",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f6f8fb;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    h1 {
        color: #0f172a;
        letter-spacing: 0;
    }
    h2, h3 {
        color: #1e293b;
        letter-spacing: 0;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] p {
        font-size: 0.92rem;
        line-height: 1.45;
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.35rem 0.75rem;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
    }
    .section-header {
        border-top: 1px solid #e5e7eb;
        margin-top: 1.7rem;
        padding-top: 1.25rem;
        margin-bottom: 0.75rem;
    }
    .section-title {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.25;
    }
    .section-caption {
        color: #64748b;
        font-size: 0.94rem;
        margin-top: 0.25rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.05rem;
        min-height: 112px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.75rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.15;
        margin-top: 0.25rem;
    }
    .metric-help {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.35;
        margin-top: 0.35rem;
    }
    .about-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.85rem 0.9rem;
        margin-bottom: 1rem;
    }
    .about-box p {
        margin: 0.25rem 0;
    }
    .explanation-card {
        min-height: auto;
        margin-bottom: 0.65rem;
    }
    .risk-low {
        color: #166534;
        font-weight: 700;
    }
    .risk-medium {
        color: #92400e;
        font-weight: 700;
    }
    .risk-high {
        color: #991b1b;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data loading and preparation
# -----------------------------
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load the included Lending Club sample dataset."""
    return pd.read_csv(DATA_PATH, low_memory=False)


def clean_term(value):
    """Convert values like '36 months' into 36."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace("months", "").replace("month", "").strip()
    return pd.to_numeric(text, errors="coerce")


def clean_emp_length(value):
    """Convert employment length text into a numeric year value."""
    if pd.isna(value):
        return np.nan

    text = str(value).lower().strip()
    if "10+" in text:
        return 10
    if "< 1" in text:
        return 0

    text = (
        text.replace("+", "")
        .replace("years", "")
        .replace("year", "")
        .strip()
    )
    return pd.to_numeric(text, errors="coerce")


def create_target(df: pd.DataFrame) -> pd.Series | None:
    """Create the binary default target when the dataset contains labels."""
    if "default" in df.columns:
        return pd.to_numeric(df["default"], errors="coerce")

    if "loan_status" not in df.columns:
        return None

    final_statuses = ["Fully Paid", "Charged Off", "Default"]
    labeled = df["loan_status"].isin(final_statuses)
    target = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)
    return target.where(labeled)


def prepare_model_data(df: pd.DataFrame, require_target: bool = False) -> tuple[pd.DataFrame, pd.Series | None]:
    """Reuse the notebook's selected features and cleaning logic."""
    prepared = df.copy()

    if "term" in prepared.columns:
        prepared["term"] = prepared["term"].apply(clean_term)
    if "emp_length" in prepared.columns:
        prepared["emp_length"] = prepared["emp_length"].apply(clean_emp_length)

    missing_features = [feature for feature in FEATURES if feature not in prepared.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_features)}")

    X = prepared[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = create_target(prepared)

    if require_target and y is not None:
        labeled_rows = y.notna()
        X = X.loc[labeled_rows]
        y = y.loc[labeled_rows].astype(int)

    return X, y


def fill_missing_values(X: pd.DataFrame, fill_values: pd.Series | None = None):
    """Fill missing values with the median, a simple beginner-friendly strategy."""
    if fill_values is None:
        fill_values = X.median(numeric_only=True)

    cleaned = X.fillna(fill_values)
    cleaned = cleaned.fillna(0)
    return cleaned, fill_values


def categorize_risk(probability: float) -> str:
    """Translate probability into business-friendly risk groups."""
    if probability < LOW_RISK_CUTOFF:
        return "Low Risk"
    if probability < HIGH_RISK_CUTOFF:
        return "Medium Risk"
    return "High Risk"


def format_reasons(reasons: list[str]) -> str:
    """Join explanation reasons in a readable sentence."""
    if not reasons:
        return ""
    if len(reasons) == 1:
        return reasons[0]
    if len(reasons) == 2:
        return f"{reasons[0]} and {reasons[1]}"
    return ", ".join(reasons[:-1]) + f", and {reasons[-1]}"


def explain_prediction(row: pd.Series, probability: float, fill_values: pd.Series) -> str:
    """Generate a simple plain-English explanation for one applicant."""
    protective_factors = []
    risk_factors = []

    dti_reference = fill_values.get("dti", 20)
    rate_reference = fill_values.get("int_rate", 13)
    fico_reference = fill_values.get("fico_range_low", 700)
    income_reference = fill_values.get("annual_inc", 65000)
    employment_reference = fill_values.get("emp_length", 5)

    if row["int_rate"] <= rate_reference:
        protective_factors.append("a lower interest rate")
    else:
        risk_factors.append("a higher interest rate")

    if row["dti"] <= dti_reference:
        protective_factors.append("a manageable debt-to-income ratio")
    else:
        risk_factors.append("a higher debt-to-income ratio")

    if row["fico_range_low"] >= fico_reference:
        protective_factors.append("a stronger FICO score")
    else:
        risk_factors.append("a lower FICO score")

    if row["term"] < 60:
        protective_factors.append("a shorter loan term")
    else:
        risk_factors.append("a longer loan term")

    if row["annual_inc"] >= income_reference:
        protective_factors.append("income above the training median")
    else:
        risk_factors.append("income below the training median")

    if row["emp_length"] >= employment_reference:
        protective_factors.append("steady employment history")

    risk_label = categorize_risk(probability)
    probability_text = f"{probability:.1%}"
    protective_text = format_reasons(protective_factors[:3])
    risk_text = format_reasons(risk_factors[:3])

    if risk_label == "Low Risk":
        if protective_text and risk_text:
            return (
                f"This applicant is classified as low risk with an estimated default probability of "
                f"{probability_text}. The strongest protective signs are {protective_text}. "
                f"There are still a few items to review, such as {risk_text}, but the overall model score remains low."
            )
        if protective_text:
            return (
                f"This applicant is classified as low risk with an estimated default probability of "
                f"{probability_text}. The profile looks stronger because of {protective_text}."
            )
        return (
            f"This applicant is classified as low risk with an estimated default probability of "
            f"{probability_text}. The main factors are close to the typical lower-risk profile in the training data."
        )

    if risk_label == "Medium Risk":
        if protective_text and risk_text:
            return (
                f"This applicant is classified as medium risk with an estimated default probability of "
                f"{probability_text}. The profile is mixed: {protective_text} help, while {risk_text} increase risk."
            )
        if risk_text:
            return (
                f"This applicant is classified as medium risk with an estimated default probability of "
                f"{probability_text}. The model is mainly watching {risk_text}."
            )
        return (
            f"This applicant is classified as medium risk with an estimated default probability of "
            f"{probability_text}. The model does not see extreme risk signals, but the profile is not clearly low risk."
        )

    if protective_text and risk_text:
        return (
            f"This applicant is classified as high risk with an estimated default probability of "
            f"{probability_text}. The main concerns are {risk_text}. Some positives exist, such as "
            f"{protective_text}, but they do not fully offset the risk signals."
        )
    if risk_text:
        return (
            f"This applicant is classified as high risk with an estimated default probability of "
            f"{probability_text}. The main concerns are {risk_text}."
        )
    return (
        f"This applicant is classified as high risk with an estimated default probability of "
        f"{probability_text}. The model score is elevated based on the combined feature pattern."
    )


# -----------------------------
# Model training, loading, saving
# -----------------------------
def train_model(df: pd.DataFrame) -> dict:
    """Train logistic regression using the same foundation as the notebook."""
    X, y = prepare_model_data(df, require_target=True)

    if y is None:
        raise ValueError("Training requires a 'loan_status' or 'default' column.")

    X, fill_values = fill_missing_values(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    auc_score = roc_auc_score(y_test, probabilities)
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    coefficients = pd.DataFrame(
        {
            "feature": FEATURES,
            "coefficient": model.coef_[0],
            "importance": np.abs(model.coef_[0]),
        }
    ).sort_values("importance", ascending=False)

    return {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "fill_values": fill_values,
        "auc": auc_score,
        "fpr": fpr,
        "tpr": tpr,
        "coefficients": coefficients,
    }


def save_model(bundle: dict) -> None:
    """Save the trained model bundle for reuse."""
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(bundle, file)


def load_model() -> dict | None:
    """Load a saved model if one already exists."""
    if not MODEL_PATH.exists():
        return None

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def predict_risk(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Create probabilities, categories, and explanations for uploaded rows."""
    X, _ = prepare_model_data(df, require_target=False)
    X, _ = fill_missing_values(X, bundle["fill_values"])

    scaled = bundle["scaler"].transform(X)
    probabilities = bundle["model"].predict_proba(scaled)[:, 1]

    results = df.loc[X.index].copy()
    results["default_probability"] = probabilities
    results["risk_category"] = results["default_probability"].apply(categorize_risk)
    results["plain_english_explanation"] = [
        explain_prediction(X.iloc[i], probabilities[i], bundle["fill_values"])
        for i in range(len(X))
    ]
    return results


# -----------------------------
# Visualization helpers
# -----------------------------
def plot_roc_curve(bundle: dict):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(bundle["fpr"], bundle["tpr"], color="#2563eb", linewidth=2.5, label=f"AUC = {bundle['auc']:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#f97316", linewidth=2)
    ax.set_title("ROC Curve - Credit Risk Model")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_risk_distribution(results: pd.DataFrame):
    order = ["Low Risk", "Medium Risk", "High Risk"]
    counts = results["risk_category"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#16a34a", "#f59e0b", "#dc2626"]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Applicant Risk Distribution")
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Number of Applicants")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_probability_histogram(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(results["default_probability"], bins=20, color="#2563eb", edgecolor="white")
    ax.axvline(LOW_RISK_CUTOFF, color="#16a34a", linestyle="--", label="Low / Medium cutoff")
    ax.axvline(HIGH_RISK_CUTOFF, color="#dc2626", linestyle="--", label="Medium / High cutoff")
    ax.set_title("Predicted Default Probability")
    ax.set_xlabel("Default Probability")
    ax.set_ylabel("Applicant Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_feature_importance(bundle: dict):
    coefficients = bundle["coefficients"].sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = np.where(coefficients["coefficient"] >= 0, "#dc2626", "#16a34a")
    ax.barh(coefficients["feature"], coefficients["coefficient"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def display_metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = ""):
    """Display consistent section headings across the dashboard."""
    caption_html = f'<div class="section-caption">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-title">{title}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Streamlit application
# -----------------------------
st.title("AI-Powered Credit Risk Assistant")
st.caption("Developed by Fikreyohannes Getnet Anlay | Georgia Tech MS Analytics")
st.write("Upload loan data, train or load a logistic regression model, and explain applicant default risk in plain English.")
st.info(
    "This model is a simplified educational portfolio project and should not be used as the sole basis for real lending decisions."
)

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        <div class="about-box">
            <p><strong>Goal:</strong> Predict loan default risk</p>
            <p><strong>Dataset:</strong> Lending Club loan data sample</p>
            <p><strong>Model:</strong> Logistic Regression</p>
            <p><strong>Tools:</strong> Python, Streamlit, Pandas, Scikit-learn, Matplotlib</p>
            <p><strong>Note:</strong> This is a portfolio project, not financial advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.header("Workflow")
    st.write("1. Upload or use the sample CSV")
    st.write("2. Train or load the logistic regression model")
    st.write("3. Review predictions, charts, and explanations")

    st.divider()
    model_choice = st.radio(
        "Model option",
        ["Train model from selected data", "Load saved model if available"],
        help="Training needs either a loan_status column or a default column.",
    )

    save_trained_model = st.checkbox(
        "Save trained model to model folder",
        value=True,
        help="This creates model/credit_risk_model.pkl for future app sessions.",
    )

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file, low_memory=False)
    data_source = "Uploaded CSV"
else:
    raw_df = load_sample_data()
    data_source = "Included sample data"

section_header(
    "Data Preview",
    "Review the selected dataset before training or scoring applicants.",
)
left, right = st.columns([2, 1])

with left:
    st.dataframe(raw_df.head(25), width="stretch", height=360, hide_index=True)

with right:
    display_metric_card("Data source", data_source)
    display_metric_card("Rows", f"{raw_df.shape[0]:,}")
    display_metric_card("Columns", f"{raw_df.shape[1]:,}")

missing_summary = raw_df.isna().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

with st.expander("Missing values summary"):
    if missing_summary.empty:
        st.success("No missing values found.")
    else:
        st.write("The app fills missing model features with the training median.")
        st.dataframe(
            missing_summary.rename("missing_values").to_frame(),
            width="stretch",
            height=260,
        )

section_header(
    "Model Training and Evaluation",
    "Train a simple, interpretable logistic regression model using the notebook feature set.",
)

try:
    if model_choice == "Load saved model if available":
        bundle = load_model()
        if bundle is None:
            st.info("No saved model found yet. Training a model from the selected data instead.")
            bundle = train_model(raw_df)
            if save_trained_model:
                save_model(bundle)
        else:
            st.success(f"Loaded saved model from {MODEL_PATH.name}.")
    else:
        bundle = train_model(raw_df)
        if save_trained_model:
            save_model(bundle)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        display_metric_card("Model", "Logistic Regression", "Interpretable baseline classifier")
    with metric_cols[1]:
        display_metric_card("AUC", f"{bundle['auc']:.2f}", "ROC-AUC on a 30% test split")
    with metric_cols[2]:
        display_metric_card("Features", str(len(FEATURES)), "Notebook feature set reused")
    with metric_cols[3]:
        display_metric_card("Risk bands", "3", "Low, Medium, High")

    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.pyplot(plot_roc_curve(bundle), width="stretch")
    with chart_cols[1]:
        st.pyplot(plot_feature_importance(bundle), width="stretch")

    section_header(
        "Risk Predictions",
        "Score each applicant and group predicted default risk into practical review categories.",
    )
    results = predict_risk(raw_df, bundle)

    risk_cols = st.columns(3)
    with risk_cols[0]:
        low_count = (results["risk_category"] == "Low Risk").sum()
        display_metric_card("Low Risk", f"{low_count:,}", "Predicted probability below 20%")
    with risk_cols[1]:
        medium_count = (results["risk_category"] == "Medium Risk").sum()
        display_metric_card("Medium Risk", f"{medium_count:,}", "Predicted probability from 20% to 50%")
    with risk_cols[2]:
        high_count = (results["risk_category"] == "High Risk").sum()
        display_metric_card("High Risk", f"{high_count:,}", "Predicted probability of 50% or higher")

    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    prediction_cols = st.columns(2)
    with prediction_cols[0]:
        st.pyplot(plot_risk_distribution(results), width="stretch")
    with prediction_cols[1]:
        st.pyplot(plot_probability_histogram(results), width="stretch")

    section_header(
        "Prediction Table",
        "The first 100 scored applicants are shown below. Download the full results if needed.",
    )
    display_columns = [
        column
        for column in FEATURES
        if column in results.columns
    ] + [
        "default_probability",
        "risk_category",
        "plain_english_explanation",
    ]

    formatted_results = results[display_columns].copy()
    formatted_results["default_probability"] = formatted_results["default_probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(formatted_results.head(100), width="stretch", height=430, hide_index=True)

    csv_download = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_download,
        file_name="credit_risk_predictions.csv",
        mime="text/csv",
    )

    section_header(
        "Plain-English Explanation Examples",
        "A few sample explanations showing how the model describes risk in business-friendly language.",
    )
    for _, row in formatted_results.head(5).iterrows():
        risk_class = row["risk_category"].lower().replace(" ", "-")
        st.markdown(
            f"""
            <div class="metric-card explanation-card">
                <span class="risk-{risk_class.split('-')[0]}">{row['risk_category']}</span>
                <p style="margin: 0.5rem 0 0 0;">{row['plain_english_explanation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

except ValueError as error:
    st.error(str(error))
    st.info(
        "To train the model, upload a CSV with the notebook's feature columns plus either "
        "'loan_status' or 'default'. To score unlabeled applicants, first load or train a saved model."
    )
except Exception as error:
    st.error("Something went wrong while running the credit risk assistant.")
    st.exception(error)
