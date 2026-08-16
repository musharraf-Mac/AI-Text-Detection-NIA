import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle, os

df = pd.read_csv("../data_p/processed/dataset_comb_train.csv")

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

print("Train:", train_df.shape, "Test:", test_df.shape)
train_df["author"].value_counts()

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["author"]
y_test = test_df["author"]

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["human", "ai"])
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["human","ai"], yticklabels=["human","ai"], cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.savefig("results/figures/confusion_matrix.png")
plt.show()

test_df = test_df.copy()
test_df["predicted"] = y_pred

test_df.to_csv("results/test_predictions.csv", index=False)
print("Saved predictions for Objective 3 team")

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/baseline_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved")

with open("results/metrics.md", "w") as f:
    f.write("# Baseline Model Results\n\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
    f.write("```\n" + classification_report(y_test, y_pred) + "\n```\n")