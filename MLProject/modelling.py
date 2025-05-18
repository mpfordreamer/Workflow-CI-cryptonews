import argparse
import sys
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

def main(data_path, output_model):
    """Main function to train and evaluate models"""
    # Get correct path to data file
    if data_path == "default":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "preprocessed_cryptonews.csv")
    else:
        file_path = data_path

    print(f"[INFO] Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print("[SUCCESS] Data loaded successfully!")
    except FileNotFoundError:
        print(f"[ERROR] File not found at: {file_path}")
        print("\nMake sure preprocessed_cryptonews.csv exists in:")
        print(f"{os.path.dirname(os.path.abspath(__file__))}")
        raise SystemExit("❌ Data file not found")


    X_text = df['text_clean']
    y = df['sentiment_encoded']

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(X_text).toarray()

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train models
    print("\n[INFO] Training models with SMOTE...\n")

    models = {
        "LGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(max_iter=1000, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"[TRAIN] {name}")
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1 Score': f1
        })

        # Log to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, f"model_{name}")

    # Save best model
    best_model = models["LGBM"]
    best_model.fit(X_resampled, y_resampled)
    best_model_file = output_model
    pd.to_pickle(best_model, best_model_file)

    # Simpan juga confusion matrix sebagai artefak tambahan
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Best Model')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment models and log with MLflow")
    
    # Set default path to local directory
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "preprocessed_cryptonews.csv"
    )
    
    parser.add_argument("--data-path", type=str, default=default_path,
                       help="Path to preprocessed_cryptonews.csv file")
    parser.add_argument("--output-model", type=str, default="best_lgbm_model.pkl",
                       help="Path to save the trained model")
    
    args = parser.parse_args()
    main(args.data_path, args.output_model)