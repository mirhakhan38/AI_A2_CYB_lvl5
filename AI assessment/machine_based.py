#Phishing Website Detection: Logistic Regression vs Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#Load Dataset
data = pd.read_csv("dataset.csv")

data = data.drop(columns=["id"])

#Separate features and target
X = data.drop(columns=["Result"])
y = data["Result"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, pos_label=1))
print("Recall:", recall_score(y_test, y_pred_lr, pos_label=1))
print("F1-score:", f1_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

#Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, pos_label=1))
print("Recall:", recall_score(y_test, y_pred_rf, pos_label=1))
print("F1-score:", f1_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

#Comparison Table
results_table = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
    "Precision": [precision_score(y_test, y_pred_lr, pos_label=1), precision_score(y_test, y_pred_rf, pos_label=1)],
    "Recall": [recall_score(y_test, y_pred_lr, pos_label=1), recall_score(y_test, y_pred_rf, pos_label=1)],
    "F1-score": [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)]
})

print("\n=== Results Comparison Table ===")
print(results_table)
