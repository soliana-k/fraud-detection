import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Eda:

    


    def __init__(self, filepath):
        self.filepath=filepath
        self.df=None

        pass

    def load_data(self):
        try:
            file_path = os.path.normpath(self.filepath)
    
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Could not find the file at: {file_path}")
            
            self.df=pd.read_csv(file_path)
            logger.info('Data is loaded successfully')
        
        except Exception as e:
            logger.error(f'errror: {e}')
            raise
            
    def cleanup_data(self):
        logger.info(f'cleaning up the data for {self.filepath}\n')
        MAX_CATEGORIES = 7

        print('=' * 70)
        print('Data missing value percentage\n')
        print(self.df.isna().mean() *100)
        print('=' * 70)

        print('=' * 70)
        print('Data Duplicate percentage\n')
        print(self.df.duplicated().mean() *100)
        print('=' * 70)

        print('=' * 70)
        print('Data Types Check\n')
        print(self.df.info())
        print('=' * 70)

        for col in self.df.select_dtypes(include=['object', 'string']).columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col], format='%Y-%m-%d %H:%M:%S', errors='raise')
                print(f"Strict match success: Converted '{col}'")
            except (ValueError, TypeError):
                print(f"Strict match failed: Kept '{col}' in original format")

        for col in self.df.select_dtypes(include=['object', 'string']).columns:
                try:
                    unique_count = self.df[col].nunique()
                        
                    if unique_count <= MAX_CATEGORIES:
                        self.df[col] = self.df[col].astype('category')
                        print(f"Converted '{col}' to category ({unique_count} categories).")
                    else:
                        print(f"Skipped '{col}' - Too many unique values ({unique_count}).")
                except Exception as e:
                    print(f"Error processing category for '{col}': {e}")
        
        print('=' * 70)
        print('Data Types Check\n')
        print(self.df.info())
        print('=' * 70)
        return self.df
    
    # def univariate_analysis(self):
    #     plt.Figure((16, 9))
    #     sns.pairplot(data=self.df, kind='kde')
    #     plt.show()


    def plot_numerical(self):
            for col in self.df.select_dtypes(include=['number']).columns:
                plt.figure(figsize=(14, 9))
                if self.df[col].nunique() <= 10:
                    sns.countplot(data=self.df, x=col)
                else:
                    sns.histplot(data=self.df, x=col, kde=True, bins=40)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.show()

    def plot_categorical(self):
            for col in self.df.select_dtypes(include=['object']).columns:
                plt.figure(figsize=(14, 9))
                if self.df[col].nunique() > 25:
                    order = self.df[col].value_counts().iloc[:10].index
                    sns.countplot(data=self.df, y=col, order=order)
                    plt.title(f'Top 10 Values - {col}')
                else:
                    sns.countplot(data=self.df, x=col, order=self.df[col].value_counts().index)
                    plt.xticks(rotation=45, ha='right')
                    plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.show()
    


    def run(self):
        self.load_data()
        self.cleanup_data()
        self.plot_numerical()
        self.plot_categorical()