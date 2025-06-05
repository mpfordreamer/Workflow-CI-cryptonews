import os
import time
import pandas as pd
import numpy as np
import mlflow
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_score,
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
from imblearn.pipeline import Pipeline
from mlflow.utils.file_utils import TempDir 

# MLflow setup
DAGSHUB_REPO_OWNER = 'mpfordreamer'
DAGSHUB_REPO_NAME = 'Cryptonews-analysis'
EXPERIMENT_NAME = "crypto_sentiment_modeling_docker_v5" 

DAGSHUB_USER_TOKEN_FROM_ENV = os.getenv("DAGSHUB_TOKEN")

print(f"--- DEBUG: DAGSHUB_TOKEN from env: '{DAGSHUB_USER_TOKEN_FROM_ENV}' ---")

if DAGSHUB_USER_TOKEN_FROM_ENV and DAGSHUB_USER_TOKEN_FROM_ENV.strip() != "":
    print(f"[INFO] DAGSHUB_TOKEN found. Setting MLflow environment variables for DagsHub.")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER 
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_USER_TOKEN_FROM_ENV
    
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    print("[INFO] DagsHub initialized. MLflow should use provided credentials.")
else:
    print(f"[WARNING] DAGSHUB_TOKEN environment variable not set or is empty. Attempting DagsHub initialization without explicit token (will likely attempt OAuth).")
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[INFO] MLflow tracking URI set to: {mlflow.get_tracking_uri()}") 
print(f"[INFO] MLflow experiment set to: {EXPERIMENT_NAME}")

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'cryptonews_preprocessing', 'preprocessed_cryptonews.csv')
    
    df = pd.read_csv(file_path)
    X_text = df['text_clean']
    y = df['sentiment_encoded']
    
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(X_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Cross-validation
    global X_train_original, y_train_original
    X_train_original, y_train_original = X_train, y_train

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, X_test, y_resampled, y_test

def objective(trial):
    """Optuna objective function with MLflow tracking"""
    # Define parameters for this trial
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
        # Log parameters
        mlflow.log_params(params)
        
        # Train and evaluate model
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
        mean_f1 = np.mean(scores)
        
        # Log metrics
        mlflow.log_metric("mean_cv_f1", mean_f1)
        
        return mean_f1

def train_best_model(study, X_train, X_test, y_train, y_test):
    """Train and evaluate the best model"""
    
    # Create models directory if not exists
    os.makedirs("models", exist_ok=True)
    
    with mlflow.start_run(run_name="best_model"):
        # Log best parameters
        mlflow.log_params(study.best_params)
        
        # Start timing for training duration
        start_time = time.time()
        
        # Train model with best parameters
        best_model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1, probability=True)
        best_model.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')
        
        # Log metrics
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model parameters
        mlflow.log_param("model_type", "LGBMClassifier")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", X_train.shape[0])
        
        # Log to MLflow tracking server
        mlflow.lightgbm.log_model(best_model, "model")

        # Also save locally
        mlflow.lightgbm.save_model(best_model, "models/mlflow_model")
        
        # Save and log model in native LightGBM format
        model_txt_path = 'models/best_lgbm_native.txt'  # Native format file
        best_model.booster_.save_model(model_txt_path)
        mlflow.log_artifact(model_txt_path, artifact_path="best_lgbm_native")  # Log as artifact
        
        # Save confusion matrix as an image
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        cm_image_path = 'models/confusion_matrix.png'
        plt.savefig(cm_image_path)
        mlflow.log_artifact(cm_image_path, artifact_path="confusion_matrix")
        
        # Save classification report as text file
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = 'models/classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(report_path, artifact_path="classification_report")
        
        return best_model, test_f1, accuracy

if __name__ == "__main__":
    # Load and prepare data
    print("[INFO] Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Optimization
    print("[INFO] Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    
    # Run optimization with MLflow tracking
    with mlflow.start_run(run_name="optimization") as run:
        study.optimize(objective, n_trials=50)
        
        # Log best trial info and parameters
        mlflow.log_params({
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "n_trials": 50,
            "optimization_direction": "maximize"
        })
    
    # Train and evaluate best model
    print("Label before SMOTE:", pd.Series(y_train_original).value_counts())
    print("Label after SMOTE:", pd.Series(y_train).value_counts())
    print("\n[INFO] Training best model...")
    best_model, test_f1, accuracy = train_best_model(study, X_train, X_test, y_train, y_test)
    
    # Print results
    print("\n[RESULTS] Best Parameters:")
    print(study.best_params)
    print(f"\n[RESULTS] Best CV F1 Score: {study.best_value:.4f}")
    print(f"[Test] F1 Score: {test_f1:.4f}")
    print(f"[Test] Accuracy: {accuracy:.4f}")
    print("\n[INFO] Best model saved as 'best_lgbm_model.txt'")