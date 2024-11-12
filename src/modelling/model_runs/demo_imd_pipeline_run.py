import git
import os
repo = git.Repo('.', search_parent_directories=True)
os.chdir(repo.working_tree_dir)
import sys
sys.path.append(repo.working_tree_dir)

import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from src.modelling.pipeline.ml_pipeline import preprocess_features, preprocess_target, model_grid_cv_pipeline


# demonstration model run using LDC data to predict IMD score at LSOA level

# read in feature data
ldc_density_metrics = pd.read_csv("Q:/SDU/LDC/modelling/data/processed/broad_category_density_lsoa11cd.csv")
# target imd data
imd_data = pd.read_excel("Q:/SDU/LDC/modelling/data/raw/File_5_-_IoD2019_Scores.xlsx", sheet_name="IoD2019 Scores")[["LSOA code (2011)", "Index of Multiple Deprivation (IMD) Score"]]

# merge with imd data
ldc_density_metrics = pd.merge(left=ldc_density_metrics, right=imd_data, left_on="lsoa11cd", right_on="LSOA code (2011)")
# drop remaining nans
ldc_density_metrics = ldc_density_metrics.dropna()
# take sample to speed up run time
ldc_density_metrics = ldc_density_metrics.head(1000)

# drop any unecessary variables to  from model
drop_variables = []
# geography variables
geography_variables=['lsoa11cd', 'LSOA code (2011)']
# target variables
target_var_list = ["Index of Multiple Deprivation (IMD) Score"]

# select features list - use to subset specific features of interest, if blank it will use all features. Change feature_filter__filter_features hyperparm when usings
select_features_list = []

# model dictionary and hyperparameter search space
model_param_dict = { 
        # LinearRegression(): {
        #     'feature_filter__filter_features':  [False],
        #     'feature_filter__feature_filter_list': [select_features_list]
        #     },
        # Lasso(): {
        #     'model__fit_intercept': [True, False],
        #     'model__alpha': [0.001, 0.01, 0.1, 0.5, 1],
        #     'feature_filter__filter_features': [False],
        #     'feature_filter__feature_filter_list': [select_features_list]
        #     },
        RandomForestRegressor(): {
            'model__max_depth': [None, 25, 50],
            'model__max_features': [1, 0.5, 'sqrt'],
            'model__min_samples_leaf':  [1, 4, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__n_estimators': [10, 50, 200]
            },
        # XGBRegressor():{
        #     'model__max_depth': [2, 3, 5, 10],
        #     'model__learning_rate': [0.1, 0.01, 0.001],
        #     'model__subsample': [0.5, 0.7, 1],
        #     'model__n_estimators':[10, 50, 100, 500, 2000],
        #     'feature_filter__filter_features': [False],
        #     'feature_filter__feature_filter_list': [select_features_list]
        #     }
}

# optional - user specified model for evaluation plots. e.g. user_model = "Lasso"
# if left blank out the best performing model will be used for the evaluation plots
user_model = ""

# shortened feature name label for evaluation plots
col_labels = {
    }

# run pipeline for all models   
for target_var in target_var_list:
    # pre-processing
    # drop cols, convert to set to drop unique cols only
    cols_to_drop = list(set([target_var] + drop_variables + geography_variables))
    features = preprocess_features(df=ldc_density_metrics, cols_to_drop=cols_to_drop)
    target_df = preprocess_target(df=ldc_density_metrics, target_col=target_var)
    # test set of 20% 
    x_train, x_test, y_train, y_test = train_test_split(features, target_df, test_size=0.20, random_state=36)
    # run model pipeline
    model_grid_cv_pipeline(model_param_dict=model_param_dict, 
                           target_var=target_var, 
                           target_df=target_df,
                           id_col="lsoa11cd",
                           original_df=ldc_density_metrics,
                           x_train=x_train, 
                           y_train=y_train, 
                           x_test=x_test, 
                           y_test=y_test, 
                           output_path="Q:/SDU/LDC/modelling/outputs",
                           output_label="autolog_test", 
                           col_label_map=col_labels, 
                           pd_y_label="IMD Average Score", 
                           user_evaluation_model=user_model,
                           shap_plots=True, 
                           shap_id_keys=["E01000001", "E01001328"])
