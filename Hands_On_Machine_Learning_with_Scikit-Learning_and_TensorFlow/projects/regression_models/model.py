import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from projects.regression_models.model_config import ModelConfig


def compute_cv_rmse(model, training_set, label, cv=10):
    scores = cross_val_score(model, training_set, label, scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    print("RMSE:", rmse_scores)
    print("RMSE mean:", rmse_scores.mean())
    print("RMSE std:", rmse_scores.std())


def compute_rmse(model, df, label):
    predictions = model.predict(df)
    mse = mean_squared_error(label, predictions)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)
    return rmse


def grid_search_cv_error(grid_search):
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


def simulate(config):
    # Parse config
    model_config = ModelConfig(config)

    model = model_config.get_model
    data_processor_ = model_config.get_data_processor
    hp_tuner = model_config.get_hp_tuner
    df_train, df_test = data_processor_.load_and_process()
    full_pipeline = data_processor_.create_data_process_pipeline()

    df_train_feature, df_train_label = data_processor_.extract_label(df_train)
    df_test_feature, df_test_label = data_processor_.extract_label(df_test)

    df_train_feature_prepared = full_pipeline.fit_transform(df_train_feature)
    model.fit(df_train_feature_prepared, df_train_label)

    # Cross validation error on training data
    print('RMSE on training set:')
    compute_rmse(model, df_train_feature_prepared, df_train_label)

    print('Cross validation error on training set:')
    compute_cv_rmse(model, df_train_feature_prepared, df_train_label)

    # Hyper-parameter tuning via grid search
    if hp_tuner is not None:
        hp_tuner.fit(df_train_feature_prepared, df_train_label)
        grid_search_cv_error(hp_tuner)
        print('Grid search best parameters:', hp_tuner.best_params_)

    if hp_tuner is not None:
        final_model = hp_tuner.best_estimator_
    else:
        final_model = model

    # Evaluate final model performance on test set
    df_test_feature_prepared = full_pipeline.transform(df_test_feature)
    final_model.fit(df_test_feature_prepared, df_test_label)
    print('RMSE on test set:')
    compute_rmse(final_model, df_test_feature_prepared, df_test_label)

    # Persist model
    joblib.dump(final_model, "housing_prediction_final_model.pkl")
