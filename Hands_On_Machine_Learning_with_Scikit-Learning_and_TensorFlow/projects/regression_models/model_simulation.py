from . import model


def simulate_linear_regression():
    config = {
        'data_processor': {
            'type': 'default',
            'data_path': 'datasets/housing/housing.csv',
            'num_features': ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                             'population', 'households', 'median_income'],
            'cat_features': ['ocean_proximity'],
            'strat_feature': 'median_income',
            'label': 'median_house_value',
        },

        'model_loader': {
            'type': 'linear_regression',
        },

        'hp_tuner': {
            'type': 'none',
        },
    }

    model.simulate(config)


def simulate_decision_tree_regression():
    config = {
        'data_processor': {
            'type': 'default',
            'data_path': 'datasets/housing/housing.csv',
            'num_features': ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                             'population', 'households', 'median_income'],
            'cat_features': ['ocean_proximity'],
            'strat_feature': 'median_income',
            'label': 'median_house_value',
        },

        'model_loader': {
            'type': 'decision_tree',
        },

        'hp_tuner': {
            'type': 'none',
        },
    }

    model.simulate(config)


def simulate_random_forest_regression():
    config = {
        'data_processor': {
            'type': 'default',
            'data_path': 'datasets/housing/housing.csv',
            'num_features': ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                             'population', 'households', 'median_income'],
            'cat_features': ['ocean_proximity'],
            'strat_feature': 'median_income',
            'label': 'median_house_value',
        },

        'model_loader': {
            'type': 'random_forest',
        },

        'hp_tuner': {
            'type': 'grid',
            'param_grid': [
                {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
            ],
        },
    }

    model.simulate(config)
