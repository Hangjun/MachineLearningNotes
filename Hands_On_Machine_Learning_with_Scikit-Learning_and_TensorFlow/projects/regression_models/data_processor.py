import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data_loader

RAND_STATE = 42


class DefaultDataProcessor:
    def __init__(self, path, num_features, cat_features, strat_feature, label):
        self.path = path
        self.num_features = num_features
        self.cat_features = cat_features
        self.strat_feature = strat_feature
        self.label = label

    def load_and_process(self):
        df_raw = data_loader.load_data(self.path)
        df_train, df_test = self.training_test_split(df_raw)
        return df_train, df_test

    # Get train, test split
    def training_test_split(self, df, test_size=0.2):
        if self.strat_feature is None:
            return train_test_split(df, test_size=test_size, random_state=RAND_STATE)

        # first discretize the feature
        df['strat_cat'] = pd.cut(df[self.strat_feature], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1, 2, 3, 4, 5])

        strat_train_set = pd.DataFrame()
        strat_test_set = pd.DataFrame()
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RAND_STATE)
        for train_index, test_index in split.split(df, df['strat_cat']):
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop('strat_cat', axis=1, inplace=True)
        return strat_train_set, strat_test_set

    # Apply fit_transform() to training data, and only fit() to test data
    def create_data_process_pipeline(self):
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ('one_hot_encoder', OneHotEncoder()),
        ])

        full_pipeline = ColumnTransformer([
            ("numerical", num_pipeline, self.num_features),
            ("categorical", cat_pipeline, self.cat_features),
        ])

        return full_pipeline

    def extract_label(self, df):
        df_feature = df.drop(self.label, axis=1)
        df_label = df[self.label].copy()

        return df_feature, df_label
