import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Eda:

    


    def __init__(self, fraud_filepath, ip_filepath=None):
        self.fraud_filepath=fraud_filepath
        self.ip_filepath=ip_filepath
        self.df=None
        self.ip_df=None


    def load_frauddata(self):
        try:
            file_path = os.path.normpath(self.fraud_filepath)
    
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Could not find the file at: {file_path}")
            
            self.df=pd.read_csv(file_path)
            logger.info('Data is loaded successfully')
        
        except Exception as e:
            logger.error(f'errror: {e}')
            raise

    def load_ipdata(self):
        try:
            
            ip_data_path = os.path.normpath(self.ip_to_country)
    
            if not os.path.exists(ip_data_path):
                raise FileNotFoundError(f"Could not find the file at: {ip_data_path}")
            
            self.ip_df = pd.read_csv(ip_data_path)
            logger.info('IpAddress_to_country Data is loaded successfully')
        
        except Exception as e:
            logger.error(f'errror: {e}')
            raise
            
    def cleanup_data(self):
        logger.info(f'cleaning up the data for {self.fraud_filepath}\n')
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

        if self.df.duplicated().mean() != 0:
            self.df=self.df.drop_duplicates()
            print('=' * 70)
            print('Data Duplicate percentage after dropping\n')
            print(self.df.duplicated().mean() *100)
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
    
    def univariate_analysis(self):
        print('=' * 70)
        logger.info('Doing Univariate describtion on numerical columns')
        print(self.df.describe(exclude=['object', 'category', 'string']).T)
        print('=' * 70)
        logger.info('Doing Univariate describtion on categoricals columns')
        print(self.df.describe(include=['category']).T)
        




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
            for col in self.df.select_dtypes(include=['object', 'category']).columns:
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


    def bivariate_analysis(self):
        print('=' * 70)
        for col in self.df.select_dtypes(include=['category']).columns:
            for num_cols in self.df.select_dtypes(include=['number', 'float']).columns:
                agg_dict = {f'Amount of {num_cols} per {col}': 'count',
                            f'Average {num_cols} per {col}': 'mean'}
                print(self.df.groupby(col)[num_cols].agg(**agg_dict))
                print('-' * 70)
        print('=' * 70) 
        target = next((c for c in ['class', 'Class'] if c in self.df.columns), None)
        if target:
            quantified = self.df[target].value_counts(normalize=True) * 100
        print(f'The Class IMbalance percentage in the Target variable is {quantified} ')
        print('=' * 70) 
        
    
    def bivariate_with_feature(self):
        X=self.df.copy()
        X=X.drop('class', axis=1)
        y='class'
        df_melted = pd.melt(
                self.df, 
                id_vars=['class'],                
                value_vars=X.columns     
            )

        g = sns.FacetGrid(df_melted, col="variable", sharex=False, col_wrap=4)
        g.map(sns.scatterplot, "value", "class")

        plt.show()

    def integrate_geolocation(self):
        try:
        
            self.df['ip_address'] = self.df['ip_address'].astype('int64')
            
            self.ip_df['lower_bound_ip_address'] = self.ip_df['lower_bound_ip_address'].astype('int64')
            self.ip_df['upper_bound_ip_address'] = self.ip_df['upper_bound_ip_address'].astype('int64')
            self.ip_df = self.ip_df.sort_values('lower_bound_ip_address')
            self.df = self.df.sort_values('ip_address')
            self.df = pd.merge_asof(
                self.df,
                self.ip_df,
                left_on='ip_address',
                right_on='lower_bound_ip_address',
                direction='backward'
            )
            
            # self.df = self.df[
            #     (self.df['ip_address'] <= self.df['upper_bound_ip_address']) | 
            #     (self.df['country'].isna())
            # ]
            invalid_mask = (self.df['ip_address'] > self.df['upper_bound_ip_address'])
            self.df.loc[invalid_mask | self.df['country'].isna(), 'country'] = 'Unknown'
            self.df = self.df.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1)
            
            
            logger.info("Geolocation integrated successfully.")
        except Exception as e:
            print(f'error {e}')
            raise
        
    def analyze_fraud_by_country(self):
        logger.info("Analyzing fraud patterns by country...")
        country_stats = self.df.groupby('country')['class'].agg(['count', 'mean']).sort_values(by='mean', ascending=False)
        
        print("Fraud Statistics by Country (Top 10):")
        print(country_stats.head(10))
        
        plt.figure(figsize=(12, 6))
        significant_countries = country_stats[country_stats['count'] > 100]
        
        sns.barplot(x=significant_countries.index, y=significant_countries['mean'])
        plt.xticks(rotation=90)
        plt.title('Fraud Rate by Country (Countries with > 100 transactions)')
        plt.ylabel('Fraud Rate (Mean of Class)')
        plt.tight_layout()
        plt.show()

    def feature_engineer(self):
        if 'purchase_time' in self.df.columns:
            try:
                
                self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
                self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
                self.df['time_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds()
                user_counts = self.df.groupby('user_id')['user_id'].transform('count')
                self.df['transaction_frequency'] = user_counts
                self.df = self.df.drop(['signup_time', 'purchase_time', 'device_id'], axis=1)
                
                logger.info("Feature engineering completed.")
            except Exception as e:
                print(f'error {e}')
                raise
        else:
            pass

    def transform_data(self):
        if 'country' in self.df.columns:

            try:
                self.df.reset_index(drop=True, inplace=True)

                scaler = StandardScaler()
                num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
                cols_to_scale = [c for c in num_cols if c not in ['class', 'user_id']]
                self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])

                # one got encoding 
                self.df['country'] = self.df['country'].astype('category')
                cat_cols = ['source', 'browser', 'sex', 'country']
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(self.df[cat_cols])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
                self.df = self.df.drop(columns=cat_cols)
                self.df = pd.concat([self.df, encoded_df], axis=1)
                
                logger.info("Transformation complete using OneHotEncoder.")

            except Exception as e:
                print(f'error {e}')
                raise
        else:
            pass

    def handle_imbalance(self):
        target_col = None
        if 'class' in self.df.columns:
            target_col = 'class'
        elif 'Class' in self.df.columns:
            target_col = 'Class'
            
        if not target_col:
            logger.warning("No target 'class' or 'Class' column found. Skipping balancing.")
            return None, None, None, None
        try:
                
            X = self.df.drop(target_col, axis=1)
            y = self.df[target_col]

            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"Original training shape: {X_train.shape}")
            print(f"Resampled training shape: {X_train_res.shape}")

            before = pd.Series(y_train).value_counts()
            after = pd.Series(y_train_res).value_counts()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
            before.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
            ax1.set_title('Before SMOTE (Imbalanced)')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
                
            after.plot(kind='bar', ax=ax2, color=['skyblue', 'salmon'])
            ax2.set_title('After SMOTE (Balanced)')
            ax2.set_xlabel('Class')
                
            plt.tight_layout()
            plt.show()
            return X_train_res, y_train_res, X_test, y_test
        
        except Exception as e:
            print(f'error {e}')
            raise
        

    def run(self):
        self.load_frauddata()
        if self.ip_df is not None:
            self.load_ipdata()
            self.univariate_analysis()
            self.plot_numerical()
            self.plot_categorical()
            self.bivariate_analysis()
            self.integrate_geolocation()
            self.analyze_fraud_by_country()
            
        self.cleanup_data()
        self.plot_numerical()
        self.feature_engineer()
        self.transform_data()
        return self.handle_imbalance()