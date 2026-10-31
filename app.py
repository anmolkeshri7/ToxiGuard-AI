# .\venv\Scripts\python.exe train_model.py
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass    
# >> .\venv\Scripts\Activate.ps1
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ToxiGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #080d1c 0%, #10172c 55%, #17132d 100%);
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: #0d1426;
    border-right: 1px solid #29334d;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #ffffff;
    margin-bottom: 4px;
}

.gradient-text {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 28px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 20px;
    margin-bottom: 12px;
}

.metric-card {
    background: linear-gradient(145deg, #18233b, #111b30);
    border: 1px solid #33415e;
    border-radius: 16px;
    padding: 20px;
    min-height: 120px;
}

.metric-label {
    color: #a5b4cc;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.result-card {
    background: #18233b;
    border: 1px solid #465575;
    border-radius: 16px;
    padding: 22px;
    margin-top: 18px;
}

.stTextArea textarea,
.stTextInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #64748b !important;
    border-radius: 12px !important;
    caret-color: #ffffff !important;
}

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border: 1px solid #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2) !important;
}

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6d28d9, #4338ca);
    transform: translateY(-1px);
}

div[data-baseweb="select"] > div {
    background-color: #1e293b;
    border-color: #475569;
}

[data-testid="stMetric"] {
    background: #18233b;
    border: 1px solid #33415e;
    padding: 16px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px 0 5px 0;
    font-size: 12px;
}

hr {
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    required_files = [
        "models/naive_bayes.pkl",
        "models/logistic_regression.pkl",
        "models/vectorizer.pkl",
        "models/metrics.json"
    ]

    missing = [file for file in required_files if not os.path.exists(file)]

    if missing:
        return None, None, None, None, missing

    nb_model = joblib.load("models/naive_bayes.pkl")
    lr_model = joblib.load("models/logistic_regression.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)

    return nb_model, lr_model, vectorizer, metrics, []


try:
    nb_model, lr_model, vectorizer, metrics, missing_files = load_models()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

if missing_files:
    st.error("Some model files are missing:")
    for file in missing_files:
        st.write(f"- {file}")
    st.info("Run python train_model.py to generate the required files.")
    st.stop()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 0 25px 0;">
        <div style="font-size:45px;">🛡️</div>
        <div style="font-size:25px;font-weight:800;color:white;">
            ToxiGuard <span style="color:#a78bfa;">AI</span>
        </div>
        <div style="font-size:11px;color:#94a3b8;">
            INTELLIGENT TOXICITY DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "📊 Analytics", "ℹ️ About"],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("### Model Information")
    st.caption("Machine Learning Based")
    st.caption("TF-IDF Text Vectorization")
    st.caption("Naive Bayes + Logistic Regression")

    st.markdown("---")
    st.caption("ToxiGuard AI | College ML Project")


# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.markdown("""
    <div class="main-title">
        ToxiGuard <span class="gradient-text">AI</span>
    </div>
    <div class="subtitle">
        Detect toxic language using Machine Learning and Natural Language Processing.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Analyze Your Comment</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_model = st.selectbox(
            "Select Machine Learning Algorithm",
            ["Naive Bayes", "Logistic Regression"]
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Choose a model to analyze your text.")

    user_comment = st.text_area(
        "Enter your comment",
        height=160,
        placeholder="Type or paste your comment here...",
        key="comment_input"
    )

    analyze = st.button("🔍 Analyze Comment", use_container_width=True)

    if analyze:

        if not user_comment.strip():
            st.warning("Please enter a comment before analyzing.")

        else:

            if selected_model == "Naive Bayes":
                model = nb_model
            else:
                model = lr_model

            input_vector = vectorizer.transform([user_comment.strip()])

            if input_vector.nnz == 0:

                st.warning(
                    "The model does not recognize any words in this comment. "
                    "Try using different wording."
                )

            else:

                prediction = model.predict(input_vector)[0]
                probabilities = model.predict_proba(input_vector)[0]

                classes = list(model.classes_)

                non_toxic_index = classes.index(0)
                toxic_index = classes.index(1)
                predicted_index = classes.index(prediction)

                non_toxic_probability = probabilities[non_toxic_index] * 100
                toxic_probability = probabilities[toxic_index] * 100
                confidence = probabilities[predicted_index] * 100

                if prediction == 1:
                    result = "TOXIC"
                    result_color = "#fb7185"
                    result_icon = "⚠️"
                    description = "This comment contains language classified as toxic."
                else:
                    result = "NON-TOXIC"
                    result_color = "#34d399"
                    result_icon = "✅"
                    description = "This comment is classified as non-toxic."

                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:14px;color:#94a3b8;">
                        ANALYSIS RESULT
                    </div>
                    <div style="font-size:30px;font-weight:800;color:{result_color};margin-top:8px;">
                        {result_icon} {result}
                    </div>
                    <div style="font-size:14px;color:#cbd5e1;margin-top:8px;">
                        {description}
                    </div>
                    <div style="font-size:15px;color:white;margin-top:16px;">
                        Prediction confidence:
                        <b>{confidence:.2f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-title">Class Probabilities</div>',
                            unsafe_allow_html=True)

                p1, p2 = st.columns(2)

                with p1:
                    st.metric(
                        "Non-Toxic Probability",
                        f"{non_toxic_probability:.2f}%"
                    )
                    st.progress(float(non_toxic_probability / 100))

                with p2:
                    st.metric(
                        "Toxic Probability",
                        f"{toxic_probability:.2f}%"
                    )
                    st.progress(float(toxic_probability / 100))

                # Probability chart
                probability_df = pd.DataFrame({
                    "Category": ["Non-Toxic", "Toxic"],
                    "Probability": [
                        non_toxic_probability,
                        toxic_probability
                    ]
                }).set_index("Category")

                st.markdown('<div class="section-title">Probability Comparison</div>',
                            unsafe_allow_html=True)

                st.bar_chart(
                    probability_df,
                    horizontal=True
                )

                st.caption(
                    "Probabilities are model estimates, not guaranteed "
                    "measures of correctness. Review comments with care."
                )


# ==========================================
# ANALYTICS PAGE
# ==========================================

elif page == "📊 Analytics":

    st.markdown("""
    <div class="main-title">
        Model <span class="gradient-text">Analytics</span>
    </div>
    <div class="subtitle">
        Evaluate model performance using unseen test data.
    </div>
    """, unsafe_allow_html=True)

    model_names = {
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Logistic Regression"
    }

    if not all(name in metrics for name in model_names):
        st.error(
            "Metrics file does not contain both expected models. "
            "Please retrain using the updated train_model.py."
        )
        st.stop()

    selected_metric_model = st.selectbox(
        "Select model for detailed analytics",
        ["Naive Bayes", "Logistic Regression"]
    )

    key = (
        "naive_bayes"
        if selected_metric_model == "Naive Bayes"
        else "logistic_regression"
    )

    selected_metrics = metrics[key]

    # Metric cards
    st.markdown('<div class="section-title">Performance Metrics</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    metric_items = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1")
    ]

    for column, (label, metric_key) in zip(
        [c1, c2, c3, c4], metric_items
    ):

        value = selected_metrics[metric_key] * 100

        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Model comparison
    st.markdown('<div class="section-title">Model Comparison</div>',
                unsafe_allow_html=True)

    comparison_data = []

    for model_key, model_name in model_names.items():

        model_metric = metrics[model_key]

        comparison_data.append({
            "Model": model_name,
            "Accuracy": model_metric["accuracy"] * 100,
            "Precision": model_metric["precision"] * 100,
            "Recall": model_metric["recall"] * 100,
            "F1 Score": model_metric["f1"] * 100
        })

    comparison_df = pd.DataFrame(comparison_data).set_index("Model")

    st.bar_chart(
        comparison_df,
        horizontal=True
    )

    # Confusion matrix
    st.markdown('<div class="section-title">Confusion Matrix</div>',
                unsafe_allow_html=True)

    cm = np.array(selected_metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 4))

    fig.patch.set_facecolor("#111b30")
    ax.set_facecolor("#18233b")

    image = ax.imshow(cm, interpolation="nearest", cmap="Purples")

    ax.set_title(
        f"{selected_metric_model} Confusion Matrix",
        color="white",
        pad=15
    )

    ax.set_xlabel("Predicted Label", color="white")
    ax.set_ylabel("Actual Label", color="white")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Non-Toxic", "Toxic"], color="white")
    ax.set_yticklabels(["Non-Toxic", "Toxic"], color="white")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold"
            )

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#475569")

    fig.colorbar(image, ax=ax)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Rows represent actual classes; columns represent predicted classes. "
        "The matrix uses label order 0 = Non-Toxic and 1 = Toxic."
    )

    # Dataset information
    st.markdown('<div class="section-title">Dataset Information</div>',
                unsafe_allow_html=True)

    if os.path.exists("dataset.csv"):

        dataset = pd.read_csv("dataset.csv")

        total = len(dataset)
        non_toxic_count = int((dataset["label"] == 0).sum())
        toxic_count = int((dataset["label"] == 1).sum())

        d1, d2, d3 = st.columns(3)

        d1.metric("Total Comments", total)
        d2.metric("Non-Toxic Comments", non_toxic_count)
        d3.metric("Toxic Comments", toxic_count)

    else:
        st.info("Dataset file was not found.")


# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "ℹ️ About":

    st.markdown("""
    <div class="main-title">
        About <span class="gradient-text">ToxiGuard AI</span>
    </div>
    <div class="subtitle">
        An AI-powered toxic language detection project.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🛡️ Project Overview

    ToxiGuard AI is a machine learning-based application designed
    to classify text comments into two categories:

    - **Toxic:** Comments containing insulting, abusive or harmful language.
    - **Non-Toxic:** Comments that do not fall into the toxic category.

    ### ⚙️ Technologies Used

    | Technology | Purpose |
    |---|---|
    | Python | Programming language |
    | Streamlit | Interactive web application |
    | Pandas | Dataset handling |
    | Scikit-learn | Machine learning |
    | TF-IDF | Text feature extraction |
    | Naive Bayes | Text classification |
    | Logistic Regression | Text classification |
    | Matplotlib | Data visualization |
    | Joblib | Model serialization |

    ### 🧠 How It Works

    1. User enters a text comment.
    2. TF-IDF converts the comment into numerical features.
    3. The selected machine learning model analyzes the features.
    4. The application displays the predicted category and class probabilities.
    5. Analytics displays evaluation results from the test dataset.

    ### 🎯 Project Objective

    The objective is to demonstrate how Natural Language Processing
    and Machine Learning can help identify potentially toxic language
    in online discussions.

    ### ⚠️ Disclaimer

    This is an educational prototype. Its predictions can be incorrect,
    and it should not be used as the sole basis for moderation decisions.
    """, unsafe_allow_html=False)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    🛡️ ToxiGuard AI | Machine Learning Project
    <br>
    Built with Python, Streamlit and Scikit-learn
</div>
""", unsafe_allow_html=True)
