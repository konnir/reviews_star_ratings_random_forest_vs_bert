from typing import List
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

class TrainUtils(object):
    """ A class to help with the training process. """
    def __init__(self, ds_path: str):
        """
        The constructor for the TrainUtils class loads the ds and lean duplicates.
        :param ds_path: String, path for the dataset.
        """
        self.df = pd.read_csv(ds_path)
        df_len = len(self.df)
        self.df = self.df.drop_duplicates(keep='first')
        print(f"There are {(df_len - len(self.df)) / df_len * 100}% duplicates, removed")

    def handle_prices(self) -> pd.DataFrame:
        """ Convert prices to floats and replace missing values with the median price. """
        # First, ensure price is a string to avoid errors with replace method
        self.df['price'] = self.df['price'].astype(str)
        # Use regex to extract numeric values correctly and handle cases where conversion might fail
        self.df['price'] = self.df['price'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
        # Now, fill missing values with the median price
        self.df['price'].fillna(self.df['price'].median(), inplace=True)
        return self.df

    def transform_to_categorical(self, column_name: str, categories_dict: dict) -> pd.DataFrame:
        """ Convert categorical values to integers. """
        self.df[column_name] = self.df[column_name].apply(lambda x: self.map_values(categories_dict, x))
        return self.df

    def map_values(self, categories_dict: dict, value: str):
        return categories_dict.get(value, -1)

    def move_to_categorical(self, guided_df: pd.DataFrame) -> pd.DataFrame:
        """ Convert categorical values to integers. """
        encoder = LabelEncoder()
        # Encode categorical variables
        self.df['verified'] = encoder.fit_transform(self.df['verified'])
        self.df['brand_encoded'] = encoder.fit_transform(self.df['brand'].fillna('unknown'))
        self.df['category_encoded'] = encoder.fit_transform(self.df['category'].fillna('unknown'))

        # Display the first few rows to verify changes
        self.df[['verified', 'brand_encoded', 'category_encoded', 'price']].head()
        return self.df

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Splits the dataset into train and test sets.
        :param test_size: Float between 0.0 and 1.0, the proportion of the dataset to include in the test split.
        :param random_state: Integer to use as random state.
        :return:
        """
        # Preprocessing for categorical features
        categorical_features = ['verified', 'category', 'brand']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Vectorizer for 'reviewText'
        text_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

        # Applying transformations separately to allow for different preprocessing
        # Encode categorical features
        cat_data = self.df[categorical_features].fillna('Unknown')
        cat_transformed = categorical_transformer.fit_transform(cat_data)

        # Vectorize 'reviewText'
        text_data = self.df['reviewText'].fillna('Unknown')
        text_transformed = text_vectorizer.fit_transform(text_data)

        # Combine the features: categorical and text
        X_combined = hstack([cat_transformed, text_transformed])

        # Split the combined features into training and testing sets
        y = self.df['rating'].values  # Target variable
        X_train_comb, X_test_comb, y_train_comb, y_test_comb \
            = train_test_split(X_combined, y, test_size=0.2, random_state=42)

        return X_train_comb, X_test_comb, y_train_comb, y_test_comb

    def check_df(self, df: pd.DataFrame, outliers_degree=2) -> pd.DataFrame:
        """ Show for each column - how many unique values nad data """
        fields = []
        for column in df.columns:
            non_na_values = df[column].dropna()
            d_type = df[column].dtypes
            c_type = 'other'
            if pd.api.types.is_numeric_dtype(df[column]):   # Type of field
                c_type = 'numeric'
            elif pd.api.types.is_string_dtype(df[column]) :
                c_type = 'string'
            unique = df[column].nunique()   # how many unique values
            nulls = df[column].isnull().sum() / len(df) # % of nulls for column
            mean_freq = df[column].dropna().mean() if c_type == 'numeric' else df[column].dropna().mode()[0] # mean (numeric), frequent for (string)
            median = df[column].dropna().median() if c_type == 'numeric' else None # median (numeric)
            std = df[column].dropna().std() if c_type == 'numeric' else None    # std (numeric)
            values, outliers, n_entropy = None, None, None
            if c_type == 'numeric':
                values = [non_na_values.min(), non_na_values.max()]    # min_max_or_categories (numeric)
                try:
                    z_scores = np.abs(stats.zscore(non_na_values))
                    # outliers = len(df[np.abs(z_scores) > outliers_degree]) / len(df)   # outliers (numeric)
                    outliers = (z_scores > outliers_degree).sum() / len(df)   # outliers (numeric)
                except TypeError as e:
                    print(f"Error calculating z-scores for column {column}: {e}")

            elif c_type == 'string':     # normalized entropy for categorical (string)
                values = {category: i for i, category in enumerate(df[column].dropna().unique())} # min_max_or_categories (categorical)
                probabilities = df[column].dropna().value_counts() / len(df)
                n_entropy = -np.sum(probabilities * np.log2(probabilities)) / np.log2(len(df[column].value_counts()))
            fields.append([column, d_type, c_type, unique, nulls, mean_freq, median, std, n_entropy, outliers, values])

        c_df = pd.DataFrame(fields, columns=['name', 'd_type', 'c_type', 'unique', '%_nulls', 'mean_freq',
                                             'median', 'std', 'n_entropy', 'outliers_%', 'min_max_or_categories'])
        return c_df
