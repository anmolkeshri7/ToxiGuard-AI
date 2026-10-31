
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("dataset.csv")

X = data["comment"].astype(str)
y = data["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate models
nb_predictions = nb_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

print("Naive Bayes Accuracy:",
      accuracy_score(y_test, nb_predictions))

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_predictions))

print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_predictions))

# Save models
os.makedirs("models", exist_ok=True)

joblib.dump(nb_model, "models/naive_bayes.pkl")
joblib.dump(lr_model, "models/logistic_regression.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModels trained and saved successfully!")
