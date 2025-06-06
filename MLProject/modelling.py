import os
import argparse
import time
import pickle
import pandas as pd
import numpy as np
import mlflow
import optuna

from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, roc_auc_score, confusion_matrix, classification_report
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline

DAGSHUB_REPO_OWNER = 'mpfordreamer'
DAGSHUB_REPO_NAME  = 'Cryptonews-analysis'
EXPERIMENT_NAME    = "crypto_sentiment_modeling_docker_v5"

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()
if DAGSHUB_TOKEN:
    from dagshub import init as dagshub_init
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    dagshub_init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True
    )

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[INFO] MLflow Experiment: {EXPERIMENT_NAME}")
print(f"[INFO] MLflow Tracking URI: {mlflow.get_tracking_uri()}")

def load_and_preprocess_data(data_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, data_path)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found at {csv_file}")
    df = pd.read_csv(csv_file)

    X_text = df['text_clean']
    y      = df['sentiment_encoded']

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    global X_train_original, y_train_original
    X_train_original, y_train_original = X_train, y_train

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, X_test, y_resampled, y_test


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 10, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('lgbm', lgb.LGBMClassifier(**params))
    ])
    scores = cross_val_score(
        pipeline,
        X_train_original,
        y_train_original,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1
    )
    return float(np.mean(scores))


def train_best_model(study, X_train, X_test, y_train, y_test, output_model_name):
    os.makedirs("models", exist_ok=True)

    # Log the chosen hyperparameters to the active run
    mlflow.log_params(study.best_params)

    start_time = time.time()
    best_model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
    best_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    mlflow.log_metric("training_time", training_time)

    y_pred   = best_model.predict(X_test)
    test_f1  = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    precision= precision_score(y_test, y_pred, average='weighted')
    roc_auc  = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')

    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.log_param("model_type", "LGBMClassifier")
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_samples", X_train.shape[0])

    # Save .pkl
    pkl_path = os.path.join("models", output_model_name)
    with open(pkl_path, "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact(pkl_path, artifact_path="best_lgbm_pkl")

    # Save native LightGBM model (.txt)
    native_path = os.path.join("models", "best_lgbm_native.txt")
    best_model.booster_.save_model(native_path)
    mlflow.log_artifact(native_path, artifact_path="best_lgbm_native")

    # Save confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join("models", "confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

    # Save classification report
    report = classification_report(y_test, y_pred)
    report_path = os.path.join("models", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path, artifact_path="classification_report")

    return best_model, test_f1, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="CSV filename (e.g. preprocessed_cryptonews.csv)"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Output .pkl filename (e.g. best_lgbm_model.pkl)"
    )
    args = parser.parse_args()

    # Load + preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(args.data_path)

    # Run the Optuna study (no mlflow.start_run() here—just return mean_f1)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Everything from this point on is logged to the active MLflow run
    mlflow.log_params({
        "best_trial_number": study.best_trial.number,
        "best_value":         study.best_value,
        "n_trials":           50,
        "optimization_direction": "maximize"
    })

    print("Label before SMOTE:", pd.Series(y_train_original).value_counts())
    print("Label after SMOTE:", pd.Series(y_train).value_counts())
    print("\n[INFO] Training best model...")

    best_model, test_f1, accuracy = train_best_model(
        study, X_train, X_test, y_train, y_test, args.output_model
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("accuracy", accuracy)

    print("\n[RESULTS] Best Parameters:")
    print(study.best_params)
    print(f"\n[RESULTS] Best CV F1 Score: {study.best_value:.4f}")
    print(f"[Test] F1 Score: {test_f1:.4f}")
    print(f"[Test] Accuracy: {accuracy:.4f}")
    print(f"\n[INFO] Best model saved as '{args.output_model}'")
