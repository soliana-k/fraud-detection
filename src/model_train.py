import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, confusion_matrix, classification_report


class ModelTrain:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test

    def train_log_reg(self):
        lg=LogisticRegression()
        lg.fit(self.x_train, self.y_train)
        pred=lg.predict(self.x_test)

        print(roc_auc_score(self.y_test, pred))
        print(average_precision_score(self.y_test, pred))
        print(f1_score(self.y_test, pred))
        print(confusion_matrix(self.y_test, pred))
        print(classification_report(self.y_test, pred))