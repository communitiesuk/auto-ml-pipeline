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
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from src.utils.utils import int_loguniform
from src.modelling.pipeline.ml_pipeline import (
    preprocess_features,
    preprocess_target,
    FilterFeatures,
    model_pipeline,
    )


from sklearn.datasets import load_iris

# read in demo data
iris = load_iris()

iris_data = pd.DataFrame(np.concatenate((iris.data, np.array([iris.target]).T), axis=1), columns=iris.feature_names + ['target'])
iris_data = iris_data[iris_data["target"].isin([0, 1.0])]

# target variables
target_var_list = ["target"]
# drop any unecessary variables from the model. In this case, we are dropping the geographical identifier.
drop_variables = []
# model dictionary and hyperparameter search space
model_param_dict = {
    LogisticRegression(): {},
    RandomForestClassifier(): {}
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
    features = preprocess_features(df=iris_data, cols_to_drop=cols_to_drop)
    target_df = preprocess_target(df=iris_data, target_col=target_var)

    # run model pipeline
    model_pipeline(
        model_param_dict=model_param_dict,
        target_var=target_var,
        target_df=target_df,
        feature_df=features,
        id_col="",
        original_df=iris_data,
        scoring_metrics=["f1"],
        output_path="outputs",
        output_label="classification_demo",
        col_label_map=col_labels,
        user_evaluation_model=user_model,
    )
