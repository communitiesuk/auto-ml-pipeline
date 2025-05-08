# fixing directory to root of project
import git
import os
import sys

repo = git.Repo(".", search_parent_directories=True)
os.chdir(repo.working_tree_dir)
sys.path.append(repo.working_tree_dir)

import pandas as pd
from scipy.stats import uniform, loguniform, randint
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from src.utils.utils import int_loguniform
from src.modelling.pipeline.ml_pipeline import (
    preprocess_features,
    preprocess_target,
    FilterFeatures,
    model_pipeline,
)



# demonstration model run using ONS Subnational indicator data to predict life satisfaction score at LAD level
# The ONS have collated subnational metrics for local authorities which we can read in and use as an example test case.
# data available from "https://www.ons.gov.uk/visualisations/dvc1786/machine_readable.csv"
ons_ess_data = pd.read_csv("https://www.ons.gov.uk/visualisations/dvc1786/machine_readable.csv", skiprows=1, storage_options = {'User-Agent': 'Mozilla/5.0'})
# filtering and tranforming data for use in modelling pipeline
indicators = [
    "Gross disposable household income per head",
    "Aged 16 to 64 years employment rate (Great Britain)",
    "Gigabit capable broadband",
    "Male healthy life expectancy",
    "Female healthy life expectancy"
]
# filter for local authority disctricts only and the indicators of interest
ons_ess_data = ons_ess_data[(ons_ess_data["Geography"] == "Local Authority District") & (ons_ess_data["Indicator"].isin(indicators))]
# pivot into tabular format
regression_data = pd.pivot_table(ons_ess_data, values="Value", columns="Indicator", index=["AREACD", "AREANM"]).reset_index()
# remove rows with missing data
regression_data.dropna(inplace=True) 

# target variables
target_var_list = ["Gross disposable household income per head"]
# drop any unecessary variables from the model. In this case, we are dropping the geographical identifier.
drop_variables = ["AREACD", "AREANM"]
# model dictionary and hyperparameter search space
model_param_dict = {
    LinearRegression(): {},

}
# optional controls:
# select features list - use to subset specific features of interest, if blank it will use all features.
# change feature_filter__filter_features hyperparam when using this
select_features_list = []
# optional - user specified model for evaluation plots. e.g. user_model = "Lasso"
# if left blank out the best performing model will be used for the evaluation plots
user_model = ""
# shortened feature name label for evaluation plots
col_labels = {"Life satisfaction": "Life satisfaction score"}
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

    # run model pipeline
    model_pipeline(
        model_param_dict=model_param_dict,
        target_var=target_var,
        target_df=target_df,
        feature_df=features,
        id_col="AREANM",
        original_df=regression_data,
        scoring_metrics=["r2", "neg_root_mean_squared_error"],
        output_path="outputs",
        output_label="demo",
        col_label_map=col_labels,
        user_evaluation_model=user_model,
        shap_id_keys=["Merthyr Tydfil", "Monmouthshire"],
        custom_pre_processing_steps=pre_processing_pipeline_steps,
    )
