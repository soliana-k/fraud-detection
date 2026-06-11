import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, confusion_matrix, classification_report, PrecisionRecallDisplay, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import re
def clean_column_names(df):
    df.columns = [re.sub(r'[\[\]\{\}\:\,\s"]', '_', str(col)) for col in df.columns]
    return df
class ModelTrain:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        

    def train_log_reg(self, n_iter):
        lg=LogisticRegression(max_iter=n_iter, class_weight='balanced')
        lg.fit(self.x_train, self.y_train)

        pred=lg.predict(self.x_test)
        pred_prob=lg.predict_proba(self.x_test)[:, 1]

        cm=confusion_matrix(self.y_test, pred)
       

        print('The AUC ROC score: ',roc_auc_score(self.y_test, pred_prob))
        print('The Average Precision Score: ',average_precision_score(self.y_test, pred_prob))
        print('The F1 Score is ',f1_score(self.y_test, pred))
        print('The confusion Matrix ',cm)
        print(f'THe True Negative: {cm[0][0]} \n The False Postives: {cm[0][1]} \n The False Negatives: {cm[1][0]} \n The true Positives: {cm[1][1]}')

        print(classification_report(self.y_test, pred, target_names=['Not Fraud', 'Fraud']))

        plt.figure(figsize=(7, 5))
        labels = [
            [f"True Negative\n{cm[0][0]}\n(Clean & Allowed)", f"False Positive\n{cm[0][1]}\n(Blocked Innocent User)"],
            [f"False Negative\n{cm[1][0]}\n(Fraud Slipped By)", f"True Positive\n{cm[1][1]}\n(Caught Fraud)"]
        ]
        
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=['Predicted Not Fraud', 'Predicted Fraud'],
            yticklabels=['Actual Not Fraud', 'Actual Fraud']
        )
        plt.title("Fraud Model", fontsize=14, pad=15)
        plt.ylabel("Ground Truth (Actuals)", fontsize=12)
        plt.xlabel("Model Predictions", fontsize=12)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5))
        display = PrecisionRecallDisplay.from_predictions(
            self.y_test, pred_prob, name="Logistic Regression", ax=plt.gca()
        )
        plt.title("Precision-Recall Curve (AUC-PR)", fontsize=14, pad=15)
        plt.tight_layout()
        plt.show()


    def train_and_evaluate_models(self):

        X_train = clean_column_names(self.x_train)
        X_test = clean_column_names(self.x_test)

        model_dict = {
            "Random_Forest": (RandomForestClassifier(random_state=42, class_weight='balanced'),
                {
                    "n_estimators":[100, 200],
                    "max_depth": [6, 10]
                }
            ),
            "XGBoost": (
                XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=10),
                {
                    "n_estimators":[100, 200],
                    "max_depth":[4, 6],
                    "learning_rate": [0.01, 0.1]
                }
            ),
            "LightGBM": (
                LGBMClassifier(random_state=42, verbose=-1),
                {
                    "n_estimators":[100, 200],
                    "max_depth":[4, 6],
                    "learning_rate": [0.01, 0.1]
                }
            )
        }
        
        best_models = {}

        for name, (model, params) in model_dict.items():
            logging.info(f"Starting Hyperparameter Tuning for {name}...")
            
           
            grid = GridSearchCV(model, params, cv=3, scoring='average_precision', n_jobs=-1)
            grid.fit(X_train, self.y_train)
            
            best_model = grid.best_estimator_
            best_models[name] = best_model
            
            logging.info(f"Best parameters for {name}: {grid.best_params_}")

            preds = best_model.predict(X_test)
            preds_prob = best_model.predict_proba(X_test)[:, 1]  

            cm = confusion_matrix(self.y_test, preds)
            
            print("\n" + "="*50)
            print(f"PERFORMANCE METRICS FOR: {name.upper()}")
            print("="*50)
            print(f"The AUC ROC Score:             {roc_auc_score(self.y_test, preds_prob):.4f}")
            print(f"The Average Precision Score:   {average_precision_score(self.y_test, preds_prob):.4f}")
            print(f"The F1 Score:                  {f1_score(self.y_test, preds):.4f}")
            print(f"Accuracy:                      {accuracy_score(self.y_test, preds):.4f}")
            print(f"Precision (Fraud Class):       {precision_score(self.y_test, preds):.4f}")
            print(f"Recall (Fraud Class):          {recall_score(self.y_test, preds):.4f}")
            print("-"*50)
            print(f"True Negatives (Clean Allowed):   {cm[0][0]}")
            print(f"False Positives (Innocent Blocked): {cm[0][1]}")
            print(f"False Negatives (Fraud Missed):    {cm[1][0]}")
            print(f"True Positives (Fraud Caught):     {cm[1][1]}")
            print("-"*50)
            print("Detailed Classification Report:")
            print(classification_report(self.y_test, preds, target_names=['Not Fraud', 'Fraud']))

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            labels = [
                [f"True Negative\n{cm[0][0]}\n(Clean & Allowed)", f"False Positive\n{cm[0][1]}\n(Blocked Innocent)"],
                [f"False Negative\n{cm[1][0]}\n(Fraud Slipped By)", f"True Positive\n{cm[1][1]}\n(Caught Fraud)"]
            ]
            sns.heatmap(
                cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=['Predicted Not Fraud', 'Predicted Fraud'],
                yticklabels=['Actual Not Fraud', 'Actual Fraud']
            )
            axes[0].set_title(f"{name}: Real-World Business Impact", fontsize=12, pad=10)
            axes[0].set_ylabel("Ground Truth (Actuals)")
            axes[0].set_xlabel("Model Predictions")

            PrecisionRecallDisplay.from_predictions(
                self.y_test, preds_prob, name=name, ax=axes[1]
            )
            axes[1].set_title(f"{name}: Precision-Recall Curve (AUC-PR)", fontsize=12, pad=10)
            axes[1].grid(True, linestyle='--', alpha=0.5)

            plt.suptitle(f"Visual Evaluation for {name}", fontsize=16, weight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        return best_models
    

    def cross_validate(self, X_train, y_train, X_test, y_test):
        stk=StratifiedKFold(n_splits=5)
