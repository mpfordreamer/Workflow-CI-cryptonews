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
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline

# — DagsHub & MLflow configuration
DAGSHUB_REPO_OWNER = 'mpfordreamer'
DAGSHUB_REPO_NAME  = 'Cryptonews-analysis'
EXPERIMENT_NAME    = "crypto_sentiment_modeling_docker_v5"

# Get token from environment. If missing, skip DagsHub.
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "").strip()
if DAGSHUB_TOKEN:
    from dagshub import init as dagshub_init
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    dagshub_init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    print("[INFO] DagsHub initialized with token.")
else:
    print("[INFO] DAGSHUB_TOKEN not found. Using local MLflow.")

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[INFO] MLflow Experiment: {EXPERIMENT_NAME}")
print(f"[INFO] MLflow Tracking URI: {mlflow.get_tracking_uri()}")


def load_and_preprocess_data(data_path):
    """
    Load preprocessed_cryptonews.csv from data_path folder,
    perform TF-IDF, split and apply SMOTE, then return
    (X_train_resampled, X_test, y_train_resampled, y_test).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, data_path, 'preprocessed_cryptonews.csv')
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
    """
    Optuna objective function for LightGBM hyperparameter optimization,
    logging params and mean CV F1 to MLflow.
    """
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

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
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
        mean_f1 = float(np.mean(scores))
        mlflow.log_metric("mean_cv_f1", mean_f1)
        return mean_f1


def train_best_model(study, X_train, X_test, y_train, y_test, output_model_name):
    """
    Train model using best parameters from study, save artifacts:
    - Model pickle (.pkl)
    - Native LightGBM model (.txt)
    - Confusion matrix image (.png)
    - Classification report (.txt)
    All logged to MLflow.
    """
    os.makedirs("models", exist_ok=True)

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(study.best_params)

        start_time = time.time()
        best_model = lgb.LGBMClassifier(
            **study.best_params,
            random_state=42,
            verbose=-1
        )
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

        # Save model as a pickle file
        pkl_path = os.path.join("models", output_model_name)
        with open(pkl_path, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(pkl_path, artifact_path="best_lgbm_pkl")

        # Save native LightGBM model
        native_path = os.path.join("models", "best_lgbm_native.txt")
        best_model.booster_.save_model(native_path)
        mlflow.log_artifact(native_path, artifact_path="best_lgbm_native")

        # Save confusion matrix image
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
        "--data_dir",
        dest="data_path",
        type=str,
        required=True,
        help="Folder containing preprocessed_cryptonews.csv"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Name of the output .pkl file (e.g., best_lgbm_model.pkl)"
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_and_preprocess_data(args.data_path)

    study = optuna.create_study(direction='maximize')
    with mlflow.start_run(run_name="optimization") as _:
        study.optimize(objective, n_trials=50)
        mlflow.log_params({
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "n_trials": 50,
            "optimization_direction": "maximize"
        })

    train_best_model(
        study, X_train, X_test, y_train, y_test, args.output_model
    )

    print(f"[INFO] Model saved at: models/{args.output_model}")
