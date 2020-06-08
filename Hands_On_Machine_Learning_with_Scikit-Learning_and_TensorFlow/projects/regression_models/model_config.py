from pandas._libs.properties import CachedProperty
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from . import data_processor


class ModelConfig:
    def __init__(self, config):
        self.config = config

    @CachedProperty
    def get_model(self):
        config = self.config['model_loader']

        model_type = config['type']
        if model_type == 'linear_regression':
            return LinearRegression()

        if model_type == 'decision_tree':
            return DecisionTreeRegressor()

        if model_type == 'random_forest':
            return RandomForestRegressor()

        raise RuntimeError('Unknown model!')

    @CachedProperty
    def get_data_processor(self):
        config = self.config['data_processor']

        processor_type = config['type']
        data_path = config['data_path']
        num_features = config['num_features']
        cat_features = config['cat_features']
        strat_feature = config['strat_feature']
        label = config['label']

        if processor_type == 'default':
            return data_processor.DefaultDataProcessor(data_path, num_features, cat_features, strat_feature, label)

        raise RuntimeError('Unknown data processor!')

    @CachedProperty
    def get_hp_tuner(self):
        config = self.config['hp_tuner']
        tuner_type = config['type']
        if tuner_type == 'grid':
            return GridSearchCV(self.get_model, config['param_grid'], cv=10, scoring='neg_mean_squared_error',
                                return_train_score=True)
        if tuner_type == 'none':
            return None

        raise RuntimeError('Unknown hyperparameter tuner!')
