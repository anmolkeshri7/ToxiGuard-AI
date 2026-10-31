# 🛡️ ToxiGuard AI
### AI-Powered Toxic Comment Detection System

ToxiGuard AI is a machine learning-based web application that detects potentially toxic language in text comments. It uses Natural Language Processing (NLP) and machine learning algorithms to classify comments as **Toxic** or **Non-Toxic**.

The project demonstrates how machine learning can help identify abusive, threatening, hateful, harassing, and aggressive language in online conversations.

<p align="center">
  <a href="https://toxiguard-ai.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Try%20Now-7C3AED?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Scikit--learn-orange?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

---

## 🌐 Live Demo

**Try ToxiGuard AI:** https://toxiguard-ai.streamlit.app/

Enter a comment, select a machine learning algorithm, and view its predicted category and class probabilities.

---

## ✨ Features

- **Toxic Comment Detection:** Classifies text as Toxic or Non-Toxic.
- **Multiple ML Algorithms:** Supports Naive Bayes and Logistic Regression.
- **Text Processing:** Uses TF-IDF vectorization to convert text into numerical features.
- **Prediction Confidence:** Displays model-estimated class probabilities.
- **Analytics Dashboard:** Shows accuracy, precision, recall, and F1 score.
- **Confusion Matrix:** Visualizes classification performance.
- **Interactive Interface:** Built with Streamlit and custom CSS.
- **Dataset Analytics:** Displays the number of Toxic and Non-Toxic examples.

---

## 🧠 Machine Learning Workflow

The application follows this workflow:

1. Collect and prepare labelled text comments.
2. Clean and preprocess the dataset.
3. Split the dataset into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train Naive Bayes and Logistic Regression models.
6. Evaluate both models using classification metrics.
7. Save trained models using Joblib.
8. Use the trained models to classify user-entered comments.

### Algorithms Used

| Algorithm | Purpose |
|---|---|
| Multinomial Naive Bayes | Text classification |
| Logistic Regression | Binary classification |
| TF-IDF | Text feature extraction |

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Core programming |
| Streamlit | Web application |
| Pandas | Dataset processing |
| Scikit-learn | Machine learning |
| TF-IDF | Text vectorization |
| Multinomial Naive Bayes | Classification |
| Logistic Regression | Classification |
| Joblib | Model saving and loading |
| Matplotlib | Data visualization |

---

## 📂 Project Structure

```text
ToxiGuard-AI/
│
├── app.py
├── train_model.py
├── dataset.csv
├── requirements.txt
├── README.md
│
└── models/
    ├── naive_bayes.pkl
    ├── logistic_regression.pkl
    ├── vectorizer.pkl
    └── metrics.json
```

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/anmolkeshri7/ToxiGuard-AI.git
```

### 2. Navigate to the project directory

```bash
cd ToxiGuard-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the machine learning models

```bash
python train_model.py
```

This generates the trained models and evaluation metrics inside the `models/` directory.

### 7. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

---

## 📊 Dataset

The project currently uses a custom-labelled demonstration dataset containing two classes:

| Label | Category |
|---|---|
| 0 | Non-Toxic |
| 1 | Toxic |

The dataset includes examples covering:

- Insults and personal attacks
- Threats and intimidation
- Harassment and bullying
- Hate and discriminatory language
- Aggressive language
- Positive and neutral conversations

**Note:** The current dataset is a small, manually constructed demonstration dataset. It is not a professionally annotated benchmark and should not be treated as representative of real-world language. Integration of larger public datasets, such as Jigsaw Toxic Comment Classification, is a planned improvement.

---

## 🚀 Future Improvements

- Integration of the Jigsaw Toxic Comment Classification dataset.
- Multi-label toxicity classification.
- Improved text preprocessing and feature engineering.
- Evaluation on larger, independently labelled datasets.
- Improved handling of sarcasm and context-dependent language.
- More advanced NLP models.
- Better generalization to real-world online conversations.

---

## ⚠️ Disclaimer

ToxiGuard AI is an educational machine learning prototype. Predictions may be incorrect, particularly for sarcasm, context-dependent language, and unfamiliar expressions. It should not be used as the sole basis for moderation or other consequential decisions.

---

## 👨‍💻 Author

**Anmol Keshri**

B.Tech – Computer Science and Engineering

GitHub: [@anmolkeshri7](https://github.com/anmolkeshri7)

Project: [ToxiGuard AI](https://github.com/anmolkeshri7/ToxiGuard-AI)

---

<p align="center">
  ⭐ If you find this project interesting, consider giving it a star!
</p>