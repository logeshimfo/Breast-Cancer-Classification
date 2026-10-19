import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully.")
print("Features:", X.shape)
print("Target:", y.shape)


# --------------------------------------------------
# 2. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# --------------------------------------------------
# 3. Logistic Regression Pipeline
# --------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]
)


# --------------------------------------------------
# 4. SVC Pipeline
# --------------------------------------------------

svc_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]
)


# --------------------------------------------------
# 5. Hyperparameter Tuning - Logistic Regression
# --------------------------------------------------

logistic_param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"]
}

logistic_grid = GridSearchCV(
    estimator=logistic_model,
    param_grid=logistic_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

logistic_grid.fit(X_train, y_train)

print("\nLogistic Regression")
print("Best Parameters:", logistic_grid.best_params_)
print("Best CV F1:", logistic_grid.best_score_)

best_logistic = logistic_grid.best_estimator_


# --------------------------------------------------
# 6. Hyperparameter Tuning - SVC
# --------------------------------------------------

svc_param_grid = {
    "model__C": [0.1, 1, 10, 100],
    "model__kernel": ["linear", "rbf"],
    "model__gamma": ["scale", "auto"]
}

svc_grid = GridSearchCV(
    estimator=svc_model,
    param_grid=svc_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

svc_grid.fit(X_train, y_train)

print("\nSupport Vector Classifier")
print("Best Parameters:", svc_grid.best_params_)
print("Best CV F1:", svc_grid.best_score_)

best_svc = svc_grid.best_estimator_


# --------------------------------------------------
# 7. Evaluate Logistic Regression
# --------------------------------------------------

logistic_pred = best_logistic.predict(X_test)
logistic_prob = best_logistic.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print("Accuracy :", accuracy_score(y_test, logistic_pred))
print("Precision:", precision_score(y_test, logistic_pred))
print("Recall   :", recall_score(y_test, logistic_pred))
print("F1       :", f1_score(y_test, logistic_pred))
print("ROC-AUC  :", roc_auc_score(y_test, logistic_prob))


# --------------------------------------------------
# 8. Evaluate SVC
# --------------------------------------------------

svc_pred = best_svc.predict(X_test)
svc_prob = best_svc.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("SVC RESULTS")
print("=" * 50)

print("Accuracy :", accuracy_score(y_test, svc_pred))
print("Precision:", precision_score(y_test, svc_pred))
print("Recall   :", recall_score(y_test, svc_pred))
print("F1       :", f1_score(y_test, svc_pred))
print("ROC-AUC  :", roc_auc_score(y_test, svc_prob))


# --------------------------------------------------
# 9. Final Model Evaluation
# --------------------------------------------------

final_model = best_svc

final_pred = final_model.predict(X_test)
final_prob = final_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("FINAL MODEL RESULTS")
print("=" * 50)

print("Accuracy :", accuracy_score(y_test, final_pred))
print("Precision:", precision_score(y_test, final_pred))
print("Recall   :", recall_score(y_test, final_pred))
print("F1       :", f1_score(y_test, final_pred))
print("ROC-AUC  :", roc_auc_score(y_test, final_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        final_pred,
        target_names=data.target_names
    )
)


# --------------------------------------------------
# 10. Save Final Model
# --------------------------------------------------

model_path = "models/breast_cancer_model.pkl"

joblib.dump(final_model, model_path)

print("\nFinal model saved to:", model_path)


# --------------------------------------------------
# 11. Test Model Loading
# --------------------------------------------------

loaded_model = joblib.load(model_path)

print("Model loaded successfully.")

# Test one prediction
sample_prediction = loaded_model.predict(X_test[:1])

print("Sample prediction:", sample_prediction[0])
print("Actual value      :", y_test[0])