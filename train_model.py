
import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
df = pd.read_csv("dataset.csv")

df = df.dropna(subset=["comment", "label"])
df["comment"] = df["comment"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)
df = df[df["label"].isin([0, 1])]
df = df.drop_duplicates(subset=["comment"])

X = df["comment"]
y = df["label"]

print("Dataset size:", len(df))
print("\nClass distribution:")
print(y.value_counts())

if y.nunique() != 2:
    raise ValueError("Dataset must contain both labels: 0 and 1.")

# 2. Split data before fitting vectorizer
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Define models
models = {
    "naive_bayes": MultinomialNB(alpha=0.5),
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=42
    )
}

os.makedirs("models", exist_ok=True)

# 5. Train and evaluate
all_metrics = {}

for name, model in models.items():

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist()
    }

    all_metrics[name] = metrics

    joblib.dump(model, f"models/{name}.pkl")

    print(f"\n{name.upper()}")
    print("Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
    print("Precision:", round(metrics["precision"] * 100, 2), "%")
    print("Recall:", round(metrics["recall"] * 100, 2), "%")
    print("F1 Score:", round(metrics["f1"] * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Non-Toxic", "Toxic"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

# 6. Save vectorizer and metrics
joblib.dump(vectorizer, "models/vectorizer.pkl")

with open("models/metrics.json", "w") as file:
    json.dump(all_metrics, file, indent=4)

print("\nTraining completed successfully!")
print("Updated models saved in the models folder.")
