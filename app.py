
import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="ToxiGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- LOAD MACHINE LEARNING MODELS ----------------

@st.cache_resource
def load_models():
    nb_model = joblib.load("models/naive_bayes.pkl")
    lr_model = joblib.load("models/logistic_regression.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    return nb_model, lr_model, vectorizer

try:
    nb_model, lr_model, vectorizer = load_models()
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.info("Please make sure all three model files exist in the models folder.")
    st.stop()

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #4338ca;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
}

div.stButton > button {
    background-color: #4f46e5;
    color: white;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #3730a3;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🛡️ ToxiGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Toxic Comment Detection System</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# ---------------- INTRODUCTION ----------------

st.markdown("### Welcome to ToxiGuard AI")

st.write(
    "ToxiGuard AI is a Machine Learning based application "
    "that analyses text and identifies potentially toxic "
    "or non-toxic comments."
)

st.info(
    "Enter a comment below and select a machine learning "
    "algorithm to analyse the text."
)

# ---------------- USER INPUT ----------------

st.markdown("### Enter Your Comment")

comment = st.text_area(
    "Write or paste your comment here:",
    placeholder="Example: You are a wonderful person...",
    height=130
)

model_choice = st.selectbox(
    "Choose Machine Learning Algorithm",
    ["Naive Bayes", "Logistic Regression"]
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze Comment", use_container_width=True):

    if not comment.strip():
        st.warning("Please enter a comment before analysing.")

    else:
        text_vector = vectorizer.transform([comment])

        if model_choice == "Naive Bayes":
            model = nb_model
        else:
            model = lr_model

        prediction = int(model.predict(text_vector)[0])

        probabilities = model.predict_proba(text_vector)[0]

        confidence = float(probabilities[prediction]) * 100

        st.divider()
        st.markdown("### Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠️ TOXIC COMMENT")
                st.write("The model classified this comment as toxic.")
            else:
                st.success("✅ NON-TOXIC COMMENT")
                st.write("The model classified this comment as non-toxic.")

        with col2:
            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        # Probability chart
        result_data = pd.DataFrame(
            {
                "Category": ["Non-Toxic", "Toxic"],
                "Probability (%)": probabilities * 100
            }
        )

        st.markdown("### Prediction Probability")

        st.bar_chart(
            result_data.set_index("Category")
        )

# ---------------- MODEL INFORMATION ----------------

st.divider()

st.markdown("### Machine Learning Algorithms")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Naive Bayes")
    st.write(
        "A probabilistic machine learning algorithm "
        "used for text classification."
    )

with col2:
    st.markdown("#### 2. Logistic Regression")
    st.write(
        "A classification algorithm that estimates "
        "the probability of different text categories."
    )

st.divider()

st.markdown("### Technologies Used")

st.write(
    "Python | Natural Language Processing | "
    "TF-IDF | Scikit-learn | Streamlit"
)

st.caption(
    "ToxiGuard AI | College Machine Learning Project"
)

st.caption(
    "Note: Predictions are educational model outputs "
    "and may not correctly identify every toxic comment."
)
