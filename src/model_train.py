import re
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix, 
    classification_report, PrecisionRecallDisplay, accuracy_score, 
    precision_score, recall_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_column_names(df):
    df_copy = df.copy()
    df_copy.columns = [re.sub(r'[\[\]\{\}\:\,\s"]', '_', str(col)) for col in df_copy.columns]
    return df_copy

class ModelTrain:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = clean_column_names(x_train)
        self.y_train = y_train
        self.x_test = clean_column_names(x_test)
        self.y_test = y_test

    def train_log_reg(self, n_iter=1000):
        logging.info("Training Baseline Logistic Regression...")
        
        lg = LogisticRegression(max_iter=n_iter, class_weight='balanced', random_state=42)
        lg.fit(self.x_train, self.y_train)

        pred = lg.predict(self.x_test)
        pred_prob = lg.predict_proba(self.x_test)[:, 1]
        cm = confusion_matrix(self.y_test, pred)

        print('The AUC ROC score: ', roc_auc_score(self.y_test, pred_prob))
        print('The Average Precision Score: ', average_precision_score(self.y_test, pred_prob))
        print('The F1 Score is ', f1_score(self.y_test, pred))
        print(f'THe True Negative: {cm[0][0]} \n The False Postives: {cm[0][1]} \n The False Negatives: {cm[1][0]} \n The true Positives: {cm[1][1]}')
        print(classification_report(self.y_test, pred, target_names=['Not Fraud', 'Fraud']))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        labels = [
            [f"True Negative\n{cm[0][0]}\n(Clean & Allowed)", f"False Positive\n{cm[0][1]}\n(Blocked Innocent User)"],
            [f"False Negative\n{cm[1][0]}\n(Fraud Slipped By)", f"True Positive\n{cm[1][1]}\n(Caught Fraud)"]
        ]
        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=axes[0],
                    xticklabels=['Predicted Not Fraud', 'Predicted Fraud'],
                    yticklabels=['Actual Not Fraud', 'Actual Fraud'])
        axes[0].set_title("Logistic Regression Confusion Matrix")
        
        PrecisionRecallDisplay.from_predictions(self.y_test, pred_prob, name="Logistic Regression", ax=axes[1])
        axes[1].set_title("Precision-Recall Curve")
        plt.tight_layout()
        plt.show()

        return lg
    
    def train_and_evaluate_models(self):
        pos_count = np.sum(self.y_train == 1)
        neg_count = np.sum(self.y_train == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        model_dict = {
            "Random_Forest": (
                RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1),
                {
                    "n_estimators": [200, 300],
                    "max_depth": [6, 10, 15]
                }
            ),
            "XGBoost": (
                XGBClassifier(random_state=42, eval_metric='aucpr', scale_pos_weight=scale_pos_weight, n_jobs=1),
                {
                    "n_estimators": [200, 300],
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1]
                }
            ),
            "LightGBM": (
                LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, num_threads=1),
                {
                    "n_estimators": [200, 300],
                    "max_depth": [4, 6, 10],
                    "learning_rate": [0.01, 0.1]
                }
            )
        }
        
        best_models = {}

        for name, (model, params) in model_dict.items():
            logging.info(f"Starting Hyperparameter Tuning for {name}...")
            
            grid = GridSearchCV(model, params, cv=3, scoring='average_precision', n_jobs=-1)
            grid.fit(self.x_train, self.y_train)
            
            best_model = grid.best_estimator_
            best_models[name] = best_model
            
            logging.info(f"Best parameters for {name}: {grid.best_params_}")

            preds = best_model.predict(self.x_test)
            preds_prob = best_model.predict_proba(self.x_test)[:, 1]  
            cm = confusion_matrix(self.y_test, preds)
            
            print("\n" + "="*50)
            print(f"PERFORMANCE METRICS FOR: {name.upper()}")
            print("="*50)
            print(f"The Average Precision Score:   {average_precision_score(self.y_test, preds_prob):.4f}")
            print(f"The F1 Score:                  {f1_score(self.y_test, preds):.4f}")
            print(f"Recall (Fraud Class):          {recall_score(self.y_test, preds):.4f}")
            print(f'THe True Negative (True Non-Frauds): {cm[0][0]} \n The False Postives (Non-Frauds being passed as Frauds): {cm[0][1]} \n The False Negatives (Frauds being passed as Non-Fruads): {cm[1][0]} \n The true Positives (True Fruads caught): {cm[1][1]}')
            print(classification_report(self.y_test, preds, target_names=['Not Fraud', 'Fraud']))
            print("-"*50)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            labels = [
                [f"True Negative\n{cm[0][0]}\n(Clean)", f"False Positive\n{cm[0][1]}\n(Blocked)"],
                [f"False Negative\n{cm[1][0]}\n(Missed)", f"True Positive\n{cm[1][1]}\n(Caught)"]
            ]
            sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=axes[0],
                        xticklabels=['Predicted Not Fraud', 'Predicted Fraud'],
                        yticklabels=['Actual Not Fraud', 'Actual Fraud'])
            axes[0].set_title(f"{name}: Real-World Business Impact")

            PrecisionRecallDisplay.from_predictions(self.y_test, preds_prob, name=name, ax=axes[1])
            axes[1].set_title(f"{name}: Precision-Recall Curve (AUC-PR)")
            axes[1].grid(True, linestyle='--', alpha=0.5)

            plt.suptitle(f"Visual Evaluation for {name}", fontsize=16, weight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        return best_models
    
    def cross_validate_single(self, X_train=None, y_train=None, model=None, scoring=None):
        """
        Perform Stratified K-Fold Cross-Validation.
        Reports mean ± std and per-fold results for reliable performance estimation.
        """
        
        import sklearn.model_selection as sms
        
        if X_train is None:
            X_train = self.x_train
        if y_train is None:
            y_train = self.y_train
    
        if X_train is self.x_train:
            X_clean = self.x_train
        else:
            X_clean = clean_column_names(X_train.copy())
        
        if model is None:
            logging.warning("No model provided. Using Logistic Regression with default settings.")
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        
        if scoring is None:
            scoring = ['average_precision', 'roc_auc', 'f1', 'precision', 'recall']
        
        stk = sms.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        logging.info(f"Starting {len(scoring)} metric Stratified K-Fold CV (k=5) for {model.__class__.__name__}...")
        
        if hasattr(model, 'n_jobs'):
            model.n_jobs = -1
        if hasattr(model, 'num_threads'):
            model.num_threads = -1

        cv_results = sms.cross_validate(
            model, X_clean, y_train,
            cv=stk,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION RESULTS for {model.__class__.__name__}")
        print("="*60)
        
        for metric in scoring:
            scores = cv_results[f'test_{metric}']
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"{metric.replace('_', ' ').title():<25}: {mean_score:.4f} ± {std_score:.4f}")
        
        print("\nPer-fold scores:")
        for i in range(5):
            print(f"Fold {i+1}:")
            for metric in scoring:
                score = cv_results[f'test_{metric}'][i]
                print(f"  {metric.replace('_', ' ').title():<20}: {score:.4f}")
        
        return cv_results

    def compare_all_models(self, best_models):
        """Compares all pre-tuned best models side-by-side using Stratified K-Fold CV."""
        results_summary = {}
        
        print("\n" + "="*70)
        print("MODEL COMPARISON: SIDE-BY-SIDE SUMMARY")
        print("="*70)
        
        for name, model in best_models.items():
            logging.info(f"Running full CV comparison for {name}...")
            cv_results = self.cross_validate_single(model=model)
            
            results_summary[name] = {
                metric: cv_results[f'test_{metric}'].mean() 
                for metric in ['average_precision', 'f1', 'recall']
            }
            
        print("\n" + "="*50)
        print(f"{'Model':<20} | {'Avg Prec':<10} | {'F1':<10} | {'Recall':<10}")
        print("-"*50)
        for name, metrics in results_summary.items():
            print(f"{name:<20} | {metrics['average_precision']:.4f}     | {metrics['f1']:.4f}     | {metrics['recall']:.4f}")
        print("="*50)
        
        return results_summary
    
    
    
