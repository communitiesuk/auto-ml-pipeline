from operator import truediv
import pandas as pd 
# fixing directory to root of project
import git
import os
import sys

repo = git.Repo(".", search_parent_directories=True)
os.chdir(repo.working_tree_dir)
sys.path.append(repo.working_tree_dir)


import pandas as pd
import numpy as np
from scipy.stats import uniform, loguniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from src.utils.utils import int_loguniform
from src.modelling.pipeline.ml_pipeline import (
    preprocess_features,
    preprocess_target,
    FilterFeatures,
    model_pipeline,
    )


def get_survey_timeseries_data(survey_files: list, survey_data_path: str, survey_vars: list) -> pd.DataFrame:
    """
    Get a combined timeseries DataFrame for all user provided waves variables. If variables are not present in a wave, the cells are fill with with N/A.

    Args:
        survey_files (list): List of file names for each wave.
        survey_data_path (str): Path to the survey data files.
        survey_vars (list): List of variables to create timeseries.

    Returns:
        pd.DataFrame: Timeseries DataFrame for the survey data combined across waves.
    """
    df_cols = ["wave"] + survey_vars
    combined_wave_data = pd.DataFrame(columns=df_cols)

    # Iterate through waves
    for file in survey_files:
        wave_letter = file[0]
        # read data
        wave_data = pd.read_table(f"{survey_data_path}/{file}")
        # rename cols
        rename_dict = {wave_letter + "_" + var: var for var in survey_vars}
        wave_data.rename(columns=rename_dict, inplace=True)
        # subset for cols 
        vars_present = [var for var in survey_vars if var in wave_data.columns]
        wave_data = wave_data[vars_present]
         # Add wave marker
        wave_data["wave"] = wave_letter
        # concat
        combined_wave_data = pd.concat([combined_wave_data, wave_data], axis=0)
        
    return combined_wave_data

# get timeseries data
us_survey_data_path = "Q:/SDU/simulation-modelling/data/understanding_society_survey_data_timeseries/UKDA-6614-tab/tab/ukhls"

ind_vars = ["pidp", "istrtdaty", "helphours1", "dvage", "hospc1", "health", "fruitamt", "vegeamt", "vwhrs", "mwhrs"]

ind_files = ["m_indresp.tab"] 

timeseries_df = get_survey_timeseries_data(ind_files, us_survey_data_path, ind_vars)

# create features 

# create social care binary
timeseries_df["has_social_care"] = 0
timeseries_df.loc[timeseries_df["helphours1"] > 0, ["has_social_care"]] = 1


timeseries_df["has_disability"] = 0
timeseries_df.loc[timeseries_df["health"] == 1, ["has_disability"]] = 1

# has been to hospital due to condition
timeseries_df["hospital_condition"] = 0
timeseries_df.loc[timeseries_df["hospc1"] == 1, ["hospital_condition"]] = 1

# healthy eating
timeseries_df["healthy_eater"] = 0
timeseries_df.loc[timeseries_df["fruitamt"] + timeseries_df["vegeamt"] > 5, ["healthy_eater"]] = 1

# at least 2.5 hours exercise 
timeseries_df["physically_active"] = 0
timeseries_df.loc[timeseries_df["vwhrs"] + timeseries_df["mwhrs"] > 2.5, ["physically_active"]] = 1

# convert to numeric 
timeseries_df["dvage"] =  pd.to_numeric(timeseries_df["dvage"], errors="coerce")

timeseries_df = timeseries_df[["wave","pidp", "has_social_care", "dvage", "physically_active", "healthy_eater", "has_disability", "hospital_condition"]]

# drop nas
timeseries_df.dropna(inplace=True)

# train models
target_var_list = ["has_social_care"]
# drop any unecessary variables from the model. In this case, we are dropping the geographical identifier.
drop_variables = ["pidp", "wave"]
# model dictionary and hyperparameter search space
model_param_dict = {
    LogisticRegression():{}
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

# run pipeline for all models
for target_var in target_var_list:
    # pre-processing
    # drop cols, convert to set to drop unique cols only
    cols_to_drop = list(set([target_var] + drop_variables))
    features = preprocess_features(df=timeseries_df, cols_to_drop=cols_to_drop)
    target_df = preprocess_target(df=timeseries_df, target_col=target_var)
    print(features)
    # run model pipeline
    model_pipeline(
        model_param_dict=model_param_dict,
        target_var=target_var,
        target_df=target_df,
        feature_df=features,
        id_col="pidp",
        original_df=timeseries_df,
        scoring_metrics = ["f1", "accuracy"],
        output_label="classification_sipher_timeseries",
        output_path="outputs",
        col_label_map=col_labels,
        user_evaluation_model=user_model,
    )

