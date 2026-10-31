# streamlit run app.py
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\venv\Scripts\Activate.ps1
import streamlit as st
import joblib
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ToxiGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
/* Comment input box */
.stTextArea textarea,
.stTextInput input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #64748b !important;
    border-radius: 10px !important;
}

/* Placeholder text */
.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
}

/* Text area focus */
.stTextArea textarea:focus,
.stTextInput input:focus {
    border: 1px solid #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2) !important;
}
.stApp {
    background-color: #0b1020;
    color: #f1f5f9;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(120deg, #24245a, #172554);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #394278;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    color: #c7d2fe;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: #171f35;
    border: 1px solid #293653;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}

div.stButton > button {
    background: #6366f1;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #4f46e5;
    color: white;
}

div[data-baseweb="textarea"] textarea {
    background-color: #171f35;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #11182b;
}

.result-card {
    padding: 22px;
    border-radius: 15px;
    margin: 12px 0;
    font-size: 20px;
    font-weight: bold;
}

.safe {
    background: #123c32;
    border: 1px solid #16a34a;
    color: #86efac;
}

.danger {
    background: #451e2a;
    border: 1px solid #ef4444;
    color: #fca5a5;
}

.small-note {
    color: #94a3b8;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_models():
    nb = joblib.load("models/naive_bayes.pkl")
    lr = joblib.load("models/logistic_regression.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)

    return nb, lr, vectorizer, metrics


try:
    nb_model, lr_model, vectorizer, metrics = load_models()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.warning(
        "Check that all three model files and metrics.json "
        "exist inside the models folder."
    )
    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🛡️ ToxiGuard AI")
st.sidebar.caption("Intelligent Content Moderation")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Analytics", "About"]
)

st.sidebar.divider()
st.sidebar.success("Machine Learning Models Loaded")

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ ToxiGuard AI</h1>
        <p>Intelligent Toxic Comment Detection System</p>
        <p>Making online conversations safer with AI.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Analyze Your Comment")

    st.write(
        "Enter any sentence below to check whether "
        "it is classified as toxic or non-toxic."
    )

    comment = st.text_area(
        "Your comment",
        placeholder="Example: You are a wonderful person...",
        height=130
    )

    algorithm = st.selectbox(
        "Select Machine Learning Algorithm",
        ["Naive Bayes", "Logistic Regression"]
    )

    if st.button("🔍 Analyze Comment", use_container_width=True):

        if not comment.strip():
            st.warning("Please enter a comment first.")

        else:

            model = (
                nb_model
                if algorithm == "Naive Bayes"
                else lr_model
            )

            text_vector = vectorizer.transform([comment])

            prediction = int(model.predict(text_vector)[0])

            probabilities = model.predict_proba(text_vector)[0]

            class_index = list(model.classes_).index(prediction)

            confidence = float(probabilities[class_index]) * 100

            st.divider()
            st.subheader("Analysis Results")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:

                    st.markdown("""
                    <div class="result-card danger">
                        ⚠️ TOXIC COMMENT DETECTED
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="result-card safe">
                        ✅ NON-TOXIC COMMENT
                    </div>
                    """, unsafe_allow_html=True)

            with col2:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            st.caption(
                "Confidence is the model's estimated probability, "
                "not its measured accuracy."
            )

            # Probability chart

            st.subheader("Prediction Breakdown")

            class_names = [
                "Non-Toxic" if int(c) == 0 else "Toxic"
                for c in model.classes_
            ]

            chart_data = pd.DataFrame({
                "Category": class_names,
                "Probability (%)": probabilities * 100
            })

            chart_data = chart_data.set_index("Category")

            st.bar_chart(
                chart_data,
                horizontal=True
            )

            st.caption(
                "Predictions are based on the trained dataset "
                "and may not identify every form of harmful language."
            )

# ---------------- ANALYTICS PAGE ----------------

elif page == "Analytics":

    st.markdown("""
    <div class="hero">
        <h1>📊 Model Analytics</h1>
        <p>Performance comparison of machine learning algorithms</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Performance Overview")

    names = list(metrics.keys())

    selected = st.selectbox(
        "Select model for detailed analysis",
        names
    )

    result = metrics[selected]

    # Metrics cards

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{result['accuracy']}%")
    c2.metric("Precision", f"{result['precision']}%")
    c3.metric("Recall", f"{result['recall']}%")
    c4.metric("F1 Score", f"{result['f1_score']}%")

    st.divider()

    # Model comparison chart

    st.subheader("Algorithm Comparison")

    comparison = pd.DataFrame({
        "Naive Bayes": [
            metrics["Naive Bayes"]["accuracy"],
            metrics["Naive Bayes"]["precision"],
            metrics["Naive Bayes"]["recall"],
            metrics["Naive Bayes"]["f1_score"]
        ],
        "Logistic Regression": [
            metrics["Logistic Regression"]["accuracy"],
            metrics["Logistic Regression"]["precision"],
            metrics["Logistic Regression"]["recall"],
            metrics["Logistic Regression"]["f1_score"]
        ]
    }, index=["Accuracy", "Precision", "Recall", "F1 Score"])

    st.bar_chart(
        comparison,
        horizontal=True,
        stack=False
    )

    # Confusion matrix

    st.divider()
    st.subheader(f"{selected} - Confusion Matrix")

    cm = np.array(result["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 4))

    fig.patch.set_facecolor("#171f35")
    ax.set_facecolor("#171f35")

    image = ax.imshow(
        cm,
        cmap="Purples"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Non-Toxic", "Toxic"],
        color="white"
    )

    ax.set_yticklabels(
        ["Non-Toxic", "Toxic"],
        color="white"
    )

    ax.set_xlabel("Predicted Label", color="white")
    ax.set_ylabel("Actual Label", color="white")

    ax.set_title(
        "Confusion Matrix",
        color="white",
        fontsize=14
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white",
                fontsize=16,
                fontweight="bold"
            )

    ax.tick_params(colors="white")

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Metrics are calculated on the held-out test dataset. "
        "A small dataset can produce unstable performance scores."
    )

# ---------------- ABOUT PAGE ----------------

elif page == "About":

    st.markdown("""
    <div class="hero">
        <h1>About ToxiGuard AI</h1>
        <p>AI-powered text classification and content moderation</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Objective")

    st.write("""
    ToxiGuard AI is a machine learning application designed
    to classify written comments into toxic and non-toxic
    categories using Natural Language Processing.
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Matplotlib
    """)

    st.subheader("Algorithms")

    st.markdown("""
    **Naive Bayes:** A probabilistic algorithm used for text classification.

    **Logistic Regression:** A classification algorithm used to estimate
    the probability of a text belonging to a category.
    """)

    st.subheader("Working Process")

    st.write("""
    1. User enters a comment.
    2. Text is converted into numerical features using TF-IDF.
    3. The selected machine learning model analyses the features.
    4. The application displays the predicted category and probability.
    """)

# ---------------- FOOTER ----------------

st.sidebar.divider()
st.sidebar.caption("ToxiGuard AI | College ML Project")
