import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


print("======================================")
print("AI Flood Predictor - Model Comparison")
print("======================================")


# Step 1: Load dataset
data = pd.read_csv("data/flood_data.csv")

print("\nDataset loaded successfully")
print("Total rows:", len(data))
print("Columns:", list(data.columns))


# Step 2: Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# Step 3: Clean data
data = data.drop_duplicates()

print("\nAfter removing duplicates")
print("Total rows:", len(data))


# Step 4: Separate input features and output label
X = data.drop("risk_level", axis=1)
y = data["risk_level"]


# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nData split completed")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# Step 6: Create multiple ML models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=3))
    ])
}


# Step 7: Train and compare all models
results = []
best_model_name = None
best_model = None
best_f1_score = -1

for model_name, model in models.items():
    print(f"\nTraining model: {model_name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    })

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    # Select best model using F1 score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = model_name
        best_model = model


# Step 8: Save comparison results
os.makedirs("models", exist_ok=True)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1_score", ascending=False)

results_df.to_csv("models/model_comparison_results.csv", index=False)

print("\n======================================")
print("Model Comparison Results")
print("======================================")
print(results_df)


# Step 9: Save best model
joblib.dump(best_model, "models/flood_model.pkl")

best_model_info = {
    "best_model_name": best_model_name,
    "best_f1_score": round(best_f1_score, 4)
}

with open("models/best_model_info.json", "w") as file:
    json.dump(best_model_info, file, indent=4)

print("\nBest Model Selected:", best_model_name)
print("Best F1 Score:", round(best_f1_score, 4))
print("Best model saved at: models/flood_model.pkl")
print("Model comparison saved at: models/model_comparison_results.csv")
print("Best model info saved at: models/best_model_info.json")


# Step 10: Final report for best model
best_prediction = best_model.predict(X_test)

print("\nClassification Report for Best Model:")
print(classification_report(y_test, best_prediction, zero_division=0))

print("\nConfusion Matrix for Best Model:")
print(confusion_matrix(y_test, best_prediction))


# Step 11: Sample prediction test
sample_data = pd.DataFrame({
    "rainfall_24h": [220],
    "rainfall_3d": [480],
    "river_level": [7.5],
    "humidity": [96],
    "temperature": [25],
    "elevation": [10],
    "distance_to_river": [0.5],
    "past_flood_count": [5]
})

sample_prediction = best_model.predict(sample_data)[0]

if hasattr(best_model, "predict_proba"):
    sample_confidence = best_model.predict_proba(sample_data).max() * 100
else:
    sample_confidence = 0

print("\nSample Prediction Test:")
print(sample_data)
print("Predicted Flood Risk:", sample_prediction)
print(f"Prediction Confidence: {sample_confidence:.2f}%")
