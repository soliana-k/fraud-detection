import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging
import os
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Provides static validation methods to ensure required columns are present 
    for both Fraud and CreditCard datasets.
    
    This class enforces data contract expectations early, preventing cryptic 
    downstream errors.
    """

    @staticmethod
    def validate_fraud_data(df: pd.DataFrame):
        """
        Validate that the DataFrame contains all required columns for the Fraud dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame to validate.
        
        Raises:
            ValueError: If any required column is missing.
        """

        required = ['ip_address', 'signup_time', 'purchase_time', 'user_id', 'class']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for Fraud: {missing}\nAvailable: {list(df.columns)}")

    @staticmethod
    def validate_creditcard_data(df: pd.DataFrame):
        """
        Validate that the DataFrame contains all required columns for the CreditCard dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame to validate.
        
        Raises:
            ValueError: If any required column is missing.
        """

        required = ['Time', 'Amount', 'Class']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for CreditCard: {missing}\nAvailable: {list(df.columns)}")


class DataLoader:
    """
    Handles loading of Fraud and CreditCard datasets with automatic detection 
    of dataset type based on presence of IP file.
    """
    
    def __init__(self, data_filepath: str, ip_filepath: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_filepath (str): Path to the main transaction dataset.
            ip_filepath (Optional[str]): Path to IP-to-Country mapping (only for Fraud dataset).
        """

        self.data_filepath = os.path.normpath(data_filepath)
        self.ip_filepath = os.path.normpath(ip_filepath) if ip_filepath else None
        self.is_fraud = self.ip_filepath is not None

    def load_data(self) -> pd.DataFrame:
        """
        Load the main dataset and validate its structure.
        
        Returns:
            pd.DataFrame: Loaded and validated DataFrame.
        
        Raises:
            FileNotFoundError: If the data file does not exist.
            ValueError: If required columns are missing.
        """
        
        if not os.path.exists(self.data_filepath):
            raise FileNotFoundError(f"Data file not found: {self.data_filepath}")

        df = pd.read_csv(self.data_filepath)
        
        if self.is_fraud:
            DataValidator.validate_fraud_data(df)
            logger.info(f"Fraud data loaded: {df.shape}")
        else:
            DataValidator.validate_creditcard_data(df)
            logger.info(f"CreditCard data loaded: {df.shape}")
        
        return df

    def load_ip_data(self) -> Optional[pd.DataFrame]:
        """
        Load the IP-to-Country mapping file if provided.
        
        Returns:
            Optional[pd.DataFrame]: IP mapping DataFrame or None if not applicable.
        """

        if not self.ip_filepath or not os.path.exists(self.ip_filepath):
            return None
        ip_df = pd.read_csv(self.ip_filepath)
        logger.info(f"IP-to-Country data loaded: {ip_df.shape}")
        return ip_df


class DataCleaner:
    """
    Performs basic data cleaning operations including duplicate removal,
    missing value reporting, datetime conversion, and type optimization.
    """

    def __init__(self, max_categories: int = 7):
        """
        Args:
            max_categories (int): Threshold for converting object columns to categorical.
        """
        self.max_categories = max_categories

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame with comprehensive reporting.
        
        Args:
            df (pd.DataFrame): Raw input DataFrame.
        
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logger.info("Starting data cleaning...")

        missing_pct = df.isna().mean() * 100
        print('=' * 80)
        print("Missing Values Percentage:\n", missing_pct[missing_pct > 0])
        print('=' * 80)

        
        dup_pct = df.duplicated().mean() * 100
        print('=' * 80)
        print(f"Duplicate Rows: {dup_pct:.2f}%")
        print('=' * 80)
        if dup_pct > 0:
            df = df.drop_duplicates()
            logger.info(f"Duplicates removed ({dup_pct:.2f}%)")

       
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='raise')
                logger.info(f"Converted {col} to datetime")
            except Exception:
                pass

        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() <= self.max_categories:
                df[col] = df[col].astype('category')

        print('=' * 80)
        print("Data Info after cleaning:")
        df.info()
        print('=' * 80)

        return df


class EDAVisualizer:
    """
    Comprehensive Exploratory Data Analysis with rich visualizations and 
    business-oriented insights for both Fraud and CreditCard datasets.
    """
    def full_eda(self, df: pd.DataFrame, dataset_name: str):
        """
        Run complete EDA pipeline including univariate, bivariate, multivariate,
        and insightful visualizations.
        
        Args:
            df (pd.DataFrame): Data to analyze.
            dataset_name (str): Name of the dataset ('Fraud' or 'CreditCard').
        """

        print('=' * 90)
        print(f" FULL EDA for {dataset_name} Dataset")
        print('=' * 90)

        print(f"Dataset Shape: {df.shape}")
        print("\nData Types:")
        print(df.dtypes)

        print("\nMissing Values Summary:")
        print(df.isna().sum()[df.isna().sum() > 0])

        print("\n" + "="*80)
        print("Numerical Features Summary")
        print(df.describe(exclude=['object', 'category']).T)
        
        print("\nCategorical / Object Features Summary")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            cat_summary = df[cat_cols].describe()
            print(cat_summary.T)
        else:
            print("No categorical/object columns found in the dataset.")

        self.bivariate_with_target(df, dataset_name)
        self.multivariate_analysis(df, dataset_name)
        self.insight_visuals(df, dataset_name)

    def bivariate_with_target(self, df: pd.DataFrame, dataset_name: str):
        """
        Analyze relationship between features and the target variable.
        """

        target = next((c for c in ['class', 'Class'] if c in df.columns), None)
        if not target:
            return

        print("\nBivariate Analysis vs Target:")
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            if col == target:
                continue
            print(f"\n{col} vs {target}:")
            print(df.groupby(target)[col].describe())
            print('-' * 70)

        imbalance = df[target].value_counts(normalize=True) * 100
        print(f"\nTarget Imbalance:\n{imbalance}")
        print("[Business Risk] High imbalance → risk of missing fraud events (revenue loss).")

    def multivariate_analysis(self, df: pd.DataFrame, dataset_name: str):
        """
        Compute and display correlation insights, especially with the target.
        """

        print("\n" + "="*80)
        print("Multivariate / Correlation Analysis")
        
        num_df = df.select_dtypes(include=['number'])
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            print("Top Correlations with Target:")
            target = next((c for c in ['class', 'Class'] if c in corr.columns), None)
            if target:
                print(corr[target].sort_values(key=abs, ascending=False).head(10))
        print("="*80)

    def insight_visuals(self, df: pd.DataFrame, dataset_name: str):
        """
        Generate key insight visualizations including heatmaps, boxplots, 
        and country-specific fraud analysis.
        """

        target = next((c for c in ['class', 'Class'] if c in df.columns), None)

        num_df = df.select_dtypes(include=['number'])
        if len(num_df.columns) > 1:
            plt.figure(figsize=(14, 10))
            sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title(f'Correlation Heatmap - {dataset_name}')
            plt.tight_layout()
            plt.show()

        
        num_cols = [c for c in df.select_dtypes(include=['number']).columns if c != target]
        for col in num_cols[:6]:  
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target, y=col, data=df, palette='Set2')
            plt.title(f'{col} Distribution by Target (Fraud vs Non-Fraud)')
            plt.ylabel(col)
            plt.xlabel('Target (0 = Legit, 1 = Fraud)')
            plt.tight_layout()
            plt.show()

        
        if dataset_name.lower() == "fraud" and 'purchase_value' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='country', y='purchase_value', data=df, order=df['country'].value_counts().iloc[:10].index)
            plt.xticks(rotation=90)
            plt.title('Purchase Value by Top Countries')
            plt.tight_layout()
            plt.show()

        if dataset_name.lower() == "fraud" and 'country' in df.columns and target:
            self.plot_fraud_by_country(df, target)

    def plot_fraud_by_country(self, df: pd.DataFrame, target: str):
        """
        Generate detailed fraud analysis visualizations by country.
        
        """
        print("\n" + "="*80)
        print("Fraud Analysis by Country")
        
        country_stats = df.groupby('country')[target].agg(
            total_transactions='count',
            fraud_cases='sum',
            fraud_rate='mean'
        ).sort_values(by='fraud_cases', ascending=False)

        print("Top 10 Countries by Fraud Cases:")
        print(country_stats.head(10))

        # at least 50 transactions
        significant = country_stats[country_stats['total_transactions'] >= 50].sort_values(by='fraud_rate', ascending=False)

        # pot 1: Fraud Rate by Country
        plt.figure(figsize=(14, 8))
        sns.barplot(x=significant.index[:15], y=significant['fraud_rate'][:15], palette='Reds_r')
        plt.title('Fraud Rate by Country (Countries with ≥ 50 transactions)')
        plt.ylabel('Fraud Rate')
        plt.xlabel('Country')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        # Plto 2: Absolute Fraud Cases by Country
        plt.figure(figsize=(14, 8))
        top_fraud_countries = significant.sort_values(by='fraud_cases', ascending=False).head(15)
        sns.barplot(x=top_fraud_countries.index, y=top_fraud_countries['fraud_cases'], palette='Oranges_r')
        plt.title('Number of Fraud Cases by Country (Top 15)')
        plt.ylabel('Number of Fraud Cases')
        plt.xlabel('Country')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_numerical(self, df: pd.DataFrame):

        """Plot distributions for all numerical features."""

        for col in df.select_dtypes(include=['number']).columns:
            plt.figure(figsize=(14, 8))
            if df[col].nunique() <= 10:
                sns.countplot(data=df, x=col)
            else:
                sns.histplot(data=df, x=col, kde=True, bins=40)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.show()

    def plot_categorical(self, df: pd.DataFrame):

        """Plot distributions for all categorical/object features."""

        for col in df.select_dtypes(include=['object', 'category']).columns:
            plt.figure(figsize=(14, 8))
            if df[col].nunique() > 25:
                order = df[col].value_counts().iloc[:10].index
                sns.countplot(data=df, y=col, order=order)
                plt.title(f'Top 10 Values - {col}')
            else:
                sns.countplot(data=df, x=col, order=df[col].value_counts().index)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.show()


class GeolocationIntegrator:
    """
    Handles IP address to country mapping using merge_asof for efficient range-based joining.
    """
    def integrate(self, df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich transaction data with country information using IP ranges.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame with 'ip_address'.
            ip_df (pd.DataFrame): IP-to-Country mapping DataFrame.
        
        Returns:
            pd.DataFrame: Enriched DataFrame with 'country' column.
        """
        df = df.copy()
        ip_df = ip_df.copy()

        df['ip_address'] = df['ip_address'].astype('int64')
        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('int64')
        ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('int64')

        ip_df = ip_df.sort_values('lower_bound_ip_address')
        df = df.sort_values('ip_address')

        merged = pd.merge_asof(df, ip_df, left_on='ip_address', right_on='lower_bound_ip_address', direction='backward')

        invalid = (merged['ip_address'] > merged['upper_bound_ip_address']) | merged['country'].isna()
        merged.loc[invalid, 'country'] = 'Unknown'
        merged = merged.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, errors='ignore')
        return merged


class FeatureEngineer:
    """
    Creates new features specifically for the Fraud dataset.
    """
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on fraud transaction data.
        
        Args:
            df (pd.DataFrame): Input DataFrame with timestamp columns.
        
        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df = df.copy()
        if all(col in df.columns for col in ['purchase_time', 'signup_time']):
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.dayofweek
            df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
            df['transaction_frequency'] = df.groupby('user_id')['user_id'].transform('count')
            df = df.drop(['signup_time', 'purchase_time', 'device_id'], axis=1, errors='ignore')
        return df


class DataPreprocessor:
    """
    Handles scaling and encoding. Skips scaling for CreditCard dataset 
    (already PCA-transformed).
    """
    def preprocess(self, df: pd.DataFrame, is_fraud: bool) -> pd.DataFrame:
        """
        Preprocess data: scale numerical features (if fraud) and one-hot encode categoricals.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            is_fraud (bool): Whether this is the Fraud dataset.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame ready for modeling.
        """
        df = df.copy().reset_index(drop=True)

        if is_fraud:
            scaler = StandardScaler()
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns
            cols_to_scale = [c for c in num_cols if c not in ['class', 'Class', 'user_id']]
            if cols_to_scale:
                df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        else:
            logger.info("CreditCard: Skipping scaling (already PCA transformed)")

        # One got encoding
        cat_cols = ['source', 'browser', 'sex', 'country']
        cat_cols = [c for c in cat_cols if c in df.columns]
        if cat_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
            df = df.drop(columns=cat_cols)
            df = pd.concat([df, encoded_df], axis=1)

        return df


class ImbalanceHandler:
    """
    Handles class imbalance using SMOTE and provides before/after visualization.
    """
    def handle(self, df: pd.DataFrame):
        """
        Apply SMOTE oversampling to balance the training set.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame with target column.
        
        Returns:
            Tuple containing: X_train_res, y_train_res, X_test, y_test
        """

        target_col = next((c for c in ['class', 'Class'] if c in df.columns), None)
        if not target_col:
            raise ValueError("Target column not found.")

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        print(f"\nOriginal Training Shape before SMOTE: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        print(f"Resampled Training Shape after SMOTE: {X_train_res.shape}")

        self._plot_imbalance(y_train, y_train_res)
        return X_train_res, y_train_res, X_test, y_test

    def _plot_imbalance(self, y_before, y_after):

        """Private method to visualize class distribution before and after SMOTE."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        pd.Series(y_before).value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title('Before SMOTE')
        pd.Series(y_after).value_counts().plot(kind='bar', ax=ax2, color=['skyblue', 'salmon'])
        ax2.set_title('After SMOTE')
        plt.tight_layout()
        plt.show()


class FraudDetectionPipeline:
    """
    Main orchestrator class that coordinates the entire fraud detection pipeline.
    Supports both Fraud (with IP mapping) and CreditCard datasets.
    """
    def __init__(self, data_filepath: str, ip_filepath: Optional[str] = None):
        """
        Initialize the complete fraud detection pipeline.
        
        Args:
            data_filepath (str): Path to main dataset.
            ip_filepath (Optional[str]): Path to IP-to-Country file (for Fraud dataset only).
        """
        self.loader = DataLoader(data_filepath, ip_filepath)
        self.cleaner = DataCleaner()
        self.geo = GeolocationIntegrator()
        self.eda = EDAVisualizer()
        self.feature_eng = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        self.balancer = ImbalanceHandler()
        self.is_fraud = self.loader.is_fraud

    def run(self):
        """
        Execute the full pipeline: load → geo (if fraud) → EDA → clean → engineer → preprocess → balance.
        
        Returns:
            Tuple: (X_train_resampled, y_train_resampled, X_test, y_test)
        """

        df = self.loader.load_data()
        ip_df = self.loader.load_ip_data() if self.is_fraud else None

        dataset_name = "Fraud" if self.is_fraud else "CreditCard"

        if self.is_fraud and ip_df is not None:
            df = self.geo.integrate(df, ip_df)
            logger.info("Geolocation integration completed.")

        self.eda.full_eda(df, dataset_name)
        self.eda.plot_numerical(df)
        self.eda.plot_categorical(df)

        df = self.cleaner.clean(df)

        if self.is_fraud:
            df = self.feature_eng.engineer(df)
        df = self.preprocessor.preprocess(df, self.is_fraud)

        return self.balancer.handle(df)


