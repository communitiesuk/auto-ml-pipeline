# fixing directory to root of project
import git
import os
import sys
repo = git.Repo('.', search_parent_directories=True)
os.chdir(repo.working_tree_dir)
sys.path.append(repo.working_tree_dir)

import pandas as pd
from scipy.stats import uniform, loguniform, randint
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from src.utils.utils import int_loguniform
from src.modelling.pipeline.ml_pipeline import (
    preprocess_features,
    preprocess_target,
    FilterFeatures,
    model_pipeline,
)


# demonstration model run using LDC data to predict IMD score at MSOA level
regression_data = pd.read_csv("Q:/SDU/LDC/modelling/data/processed/msoa_ldc_imd_sample.csv")
# target variables
target_var_list = ["imd_avg_score"]
# drop any unecessary variables from the model. In this case, we are dropping the geographical identifier.
drop_variables = ["msoa11cd"]
# model dictionary and hyperparameter search space
model_param_dict = {
        LinearRegression(): {},
        Lasso(): {
            "model__fit_intercept": [True, False],
            "model__alpha": loguniform(1e-4, 1), 
        },
        RandomForestRegressor(): {
            "model__max_depth": randint(1, 100),
            "model__max_features": [1, 0.5, "sqrt"],
            "model__min_samples_leaf": randint(1, 20),
            "model__min_samples_split": randint(2, 20),
            "model__n_estimators": randint(5, 300),
        },
        XGBRegressor(): {
            "model__max_depth": randint(2, 20),
            "model__learning_rate": loguniform(1e-4, 0.1),
            "model__subsample": uniform(0.3, 0.7),
            "model__n_estimators": int_loguniform(5, 5000),
        },
    }
# optional controls:
# select features list - use to subset specific features of interest, if blank it will use all features.
# change feature_filter__filter_features hyperparam when using this
select_features_list = []
# optional - user specified model for evaluation plots. e.g. user_model = "Lasso"
# if left blank out the best performing model will be used for the evaluation plots
user_model = ""
# shortened feature name label for evaluation plots
col_labels = {}
# custom pre-processing pipeline - remove to use default pre-processing pipeline: FilterFeatures(), StandardScaler()
pre_processing_pipeline_steps = [
    ("feature_filter", FilterFeatures()),
    ("knn_imputer", KNNImputer()),
    ("scaler", MinMaxScaler()),
]

# run pipeline for all models
for target_var in target_var_list:
    # pre-processing
    # drop cols, convert to set to drop unique cols only
    cols_to_drop = list(set([target_var] + drop_variables))
    features = preprocess_features(df=regression_data, cols_to_drop=cols_to_drop)
    target_df = preprocess_target(df=regression_data, target_col=target_var)
    # test set of 20%
    x_train, x_test, y_train, y_test = train_test_split(
        features, target_df, test_size=0.20, random_state=36
    )
    # run model pipeline
    model_pipeline(
        model_param_dict=model_param_dict,
        target_var=target_var,
        target_df=target_df,
        id_col="msoa11cd",
        original_df=regression_data,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        output_path="Q:/SDU/LDC/modelling/outputs",
        output_label="msoa_demo",
        col_label_map=col_labels,
        pd_y_label="IMD Average Score",
        user_evaluation_model=user_model,
        shap_plots=True,
        shap_id_keys=["E02000266", "E02000503"],
        custom_pre_processing_steps=pre_processing_pipeline_steps,
    )
