import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import logging
import os
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates new features specifically for the Fraud dataset.

    """

    def __init__(self, processed_data_path: str):
        self.path = processed_data_path

    def load_data(self) -> pd.DataFrame:
        """
        Load the processed CSV produced by EdaPipeline.
        parse_dates ensures signup_time/purchase_time arrive as datetime64
        so .dt accessors work without an AttributeError.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Data file not found: {self.path}")

        df = pd.read_csv(
            self.path,
            parse_dates=['signup_time', 'purchase_time'] if 'signup_time' in pd.read_csv(self.path, nrows=0).columns else None,
        )
        logger.info(f"Processed data loaded: {df.shape}")
        return df

    def engineer_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature engineering on the complete dataframe BEFORE splitting

        Args:
            df: Full dataframe straight from load_data(), before train/test split.

        Returns:
            DataFrame ready for train_test_split → preprocess → SMOTE.
        """
        required = ['purchase_time', 'signup_time', 'user_id']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"engineer_full() requires {required} but these are missing: {missing}.\n"
            )

        df = df.copy()

        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds()

        
        df['transaction_frequency'] = df.groupby('user_id')['user_id'].transform('count')
        logger.info(
            f"transaction_frequency — "
            f"min={df['transaction_frequency'].min()}, "
            f"max={df['transaction_frequency'].max()}, "
            f"mean={df['transaction_frequency'].mean():.2f}, "
            f"nunique={df['transaction_frequency'].nunique()}"
        )
        if 'country' in df.columns:
            logger.info("Applying Target Encoding to 'country'...")
            te = TargetEncoder(cols=['country'], smoothing=10, min_samples_leaf=5)
            df['country_target_enc'] = te.fit_transform(df['country'], df['class'])['country']
            df = df.drop(columns=['country'])

        
        df = df.drop(['signup_time', 'purchase_time', 'user_id', 'device_id'],
                     axis=1, errors='ignore')
        
        logger.info('Ran Feature Engineering.......')
        return df


class DataPreprocessor:
    """
    Handles scaling and one-hot encoding.
    Scaling is skipped for the CreditCard dataset (already PCA-transformed).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        is_fraud: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Args:
            X_train: Training features (post-engineering, pre-SMOTE).
            X_test:  Test features (post-engineering).
            is_fraud: True for Fraud dataset, False for CreditCard.

        Returns:
            Tuple of (X_train_processed, X_test_processed).
        """
        X_train = X_train.copy().reset_index(drop=True)
        X_test = X_test.copy().reset_index(drop=True)

        for df_ in (X_train, X_test):
            if 'user_id' in df_.columns:
                df_.drop(columns=['user_id'], inplace=True)

        known_cat_cols = ['source', 'browser', 'sex']
        inferred_cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = list(dict.fromkeys(known_cat_cols + inferred_cat_cols))
        cat_cols = [c for c in cat_cols if c in X_train.columns]

        if is_fraud:
            for col in X_train.select_dtypes(include=['int32']).columns:
                X_train[col] = X_train[col].astype('int64')
                X_test[col] = X_test[col].astype('int64')

            candidate_cols = [
                c for c in X_train.select_dtypes(include=['number']).columns
                if c not in cat_cols
            ]

            
            num_cols = [c for c in candidate_cols if X_train[c].std() > 0]
            skipped = [c for c in candidate_cols if c not in num_cols]
            if skipped:
                logger.warning(
                    f"Skipping scaling for zero-variance columns (will keep raw values): {skipped}"
                )

            X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = self.scaler.transform(X_test[num_cols])
            logger.info('Standard Scaling done....')

        # One got encoding
        if cat_cols:
            encoded_train = self.encoder.fit_transform(X_train[cat_cols])
            encoded_test = self.encoder.transform(X_test[cat_cols])

            feat_names = self.encoder.get_feature_names_out(cat_cols)
            encoded_train_df = pd.DataFrame(encoded_train, columns=feat_names, index=X_train.index)
            encoded_test_df  = pd.DataFrame(encoded_test,  columns=feat_names, index=X_test.index)

            X_train = pd.concat([X_train.drop(columns=cat_cols), encoded_train_df], axis=1)
            X_test  = pd.concat([X_test.drop(columns=cat_cols),  encoded_test_df],  axis=1)
            logger.info('One Hot Encoging done....')
        return X_train, X_test
    

class ImbalanceHandler:
    """
    Handles class imbalance using SMOTE with before/after visualization.
    """

    def handle(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to the training set. Test set is passed through unchanged.
        """
        has_categorical = any(
            col.startswith(('source_', 'browser_', 'sex_')) 
            for col in X_train.columns
        )

        if has_categorical:
            logger.info("Categorical features detected → Using SMOTENC")
            cat_mask = [col.startswith(('source_', 'browser_', 'sex_')) 
                       for col in X_train.columns]
            
            smote = SMOTENC(
                sampling_strategy=0.5,
                categorical_features=cat_mask,
                random_state=42,
                k_neighbors=5
            )
        else:
            logger.info("No categorical features → Using regular SMOTE")
            smote = SMOTE(
                sampling_strategy=0.5,
                random_state=42,
                k_neighbors=5
            )
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        print(f"Training shape before SMOTE: {X_train.shape}")
        print(f"Training shape after  SMOTE: {X_train_res.shape}")

        self._plot_imbalance(y_train, y_train_res)
        return X_train_res, y_train_res, X_test, y_test

    def _plot_imbalance(self, y_before: pd.Series, y_after: pd.Series):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        pd.Series(y_before).value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title('Before SMOTE')
        pd.Series(y_after).value_counts().plot(kind='bar', ax=ax2, color=['skyblue', 'salmon'])
        ax2.set_title('After SMOTE')
        plt.tight_layout()
        plt.show()


class FraudDetectionPipeline:
    """
    Orchestrates the full preprocessing pipeline for Fraud and CreditCard datasets.
    """

    def __init__(self, processed_data_path: str, is_fraud: bool):
        """
        Args:
            processed_data_path: Path to the CSV saved by EdaPipeline.
            is_fraud: True for Fraud dataset, False for CreditCard.
        """
        self.feature_eng = FeatureEngineer(processed_data_path)
        self.preprocessor = DataPreprocessor()
        self.balancer = ImbalanceHandler()
        self.is_fraud = is_fraud

    def run(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Pipeline: load → engineer (full df) → split → preprocess → balance

        engineer_full runs on the complete dataframe before splitting so that
        transaction_frequency has real variance and temporal features are
        derived from timestamps that would otherwise be lost after the split.

        Returns:
            Tuple: (X_train_resampled, y_train_resampled, X_test, y_test)
        """
        df = self.feature_eng.load_data()

        if self.is_fraud:
            df = self.feature_eng.engineer_full(df)

        target_col = next((c for c in ['class', 'Class'] if c in df.columns), None)
        if not target_col:
            raise ValueError("Target column ('class' or 'Class') not found in data.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        X_train, X_test = self.preprocessor.preprocess(X_train, X_test, self.is_fraud)

        return self.balancer.handle(X_train, y_train, X_test, y_test)