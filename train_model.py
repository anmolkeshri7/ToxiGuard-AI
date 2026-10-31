
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("Starting ToxiGuard AI model training...")

# 1. Load dataset
data = pd.read_csv("dataset.csv")
data = data.dropna(subset=["comment", "label"])

X = data["comment"].astype(str)
y = data["label"].astype(int)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 3. Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = {}

os.makedirs("models", exist_ok=True)

# 5. Train and evaluate
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    results[name] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)) * 100, 2),
        "precision": round(float(precision_score(
            y_test, predictions, zero_division=0
        )) * 100, 2),
        "recall": round(float(recall_score(
            y_test, predictions, zero_division=0
        )) * 100, 2),
        "f1_score": round(float(f1_score(
            y_test, predictions, zero_division=0
        )) * 100, 2),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist()
    }

    filename = (
        "naive_bayes.pkl"
        if name == "Naive Bayes"
        else "logistic_regression.pkl"
    )

    joblib.dump(model, f"models/{filename}")

    print(f"{name} Accuracy: {results[name]['accuracy']}%")

# 6. Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

# 7. Save analytics
with open("models/metrics.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nAll models trained successfully!")
print("Model files and analytics saved in models folder.")
