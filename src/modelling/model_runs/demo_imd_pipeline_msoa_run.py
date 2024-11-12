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

# run model for specificed cutoff value of premises
def cut_off_model_run(cut_off):
    # read in data
    ldc_density_metrics = pd.read_csv("Q:/SDU/LDC/modelling/data/processed/granular_category_density_" + str(cut_off) + "_msoa11cd.csv")
    #ldc_prop_density_metrics = pd.read_csv("Q:/SDU/LDC/modelling/data/processed/broad_category_density_proportion_msoa11cd.csv")
    #ldc_density_metrics = pd.merge(left=ldc_density_metrics, right=ldc_prop_density_metrics, on="msoa11cd", suffixes=['_count','_prop'])

    imd_data = pd.read_csv("Q:/SDU/LDC/modelling/data/raw/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv")[["LSOA code (2011)", "Index of Multiple Deprivation (IMD) Score", "Total population: mid 2015 (excluding prisoners)"]]

    # group imd into msoas
    imd_data["pop_weighted_score"] = imd_data["Index of Multiple Deprivation (IMD) Score"] * imd_data["Total population: mid 2015 (excluding prisoners)"]
    # merge with msoa lookup
    lsoa_to_msoa_lookup = pd.read_csv("Q:/SDU/LDC/modelling/data/lookups/Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2011)_Lookup_in_England_and_Wales.csv")[["LSOA11CD", "MSOA11CD"]]
    imd_data = pd.merge(left=imd_data, right=lsoa_to_msoa_lookup, left_on="LSOA code (2011)", right_on="LSOA11CD")
    imd_data = imd_data.groupby(by="MSOA11CD").sum(numeric_only=True).reset_index()
    #nw divide by summed pop
    imd_data["imd_avg_score"] = imd_data["pop_weighted_score"] / imd_data["Total population: mid 2015 (excluding prisoners)"]
    imd_data = imd_data[["imd_avg_score", "MSOA11CD"]]
    # merge with rural urban lookup and filter for predominantly urban msoas
    high_st_msoas = pd.read_csv("Q:/SDU/LDC/modelling/data/processed/high_st_msoas.csv")[["MSOA11CD"]]
    ldc_density_metrics = pd.merge(left=high_st_msoas, right=ldc_density_metrics, how="left", left_on="MSOA11CD", right_on="msoa11cd")
    ldc_density_metrics = ldc_density_metrics.fillna(0)
    # merge with imd data
    ldc_density_metrics = pd.merge(left=ldc_density_metrics, right=imd_data, how="left", on="MSOA11CD")

    # drop remaining nans
    ldc_density_metrics = ldc_density_metrics.dropna()


    # drop gdhi variables to remove from model
    drop_variables = []
    # geography variables
    geography_variables=['msoa11cd', 'MSOA11CD']
    # target variables
    target_var_list = ["imd_avg_score"]

    # select features list
    select_features_list = []

    # model dictionary and hyperparameter search space
    model_param_dict = { 
        LinearRegression(): {
            },
        Lasso(): {
            'model__fit_intercept': [True, False],
            'model__alpha': [0.001, 0.01, 0.1, 0.5, 1],
            },
        RandomForestRegressor(): {
            'model__max_depth': [None, 5, 25, 50],
            'model__max_features': [1, 0.5, 'sqrt', 'log2'],
            'model__min_samples_leaf': [1, 2, 4, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__n_estimators': [5, 30, 100, 200],
            },
        XGBRegressor():{
            'model__max_depth': [2, 3, 5, 10],
            'model__learning_rate': [0.1, 0.01, 0.001],
            'model__subsample': [0.5, 0.7, 1],
            'model__n_estimators':[10, 50, 100, 500, 2000],
            }
    }

    # optional - user specified model for evaluation plots. e.g. user_model = "Lasso"
    # if left blank out the best performing model will be used for the evaluation plots
    user_model = ""

    # shortened feature name label for evaluation plots
    col_labels = {
        }
    # areas to create shap plots for
    shap_plot_areas = [
        'E02002680', 'E02002707', 'E02002720', 'E02002710', 'E02002696', 'E02002719', 
        'E02002725', 'E02002702', 'E02002709', 'E02002701', 'E02002712', 
        'E02002704'
        ]

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
        model_grid_cv_pipeline(model_param_dict, target_var, target_df, "msoa11cd", ldc_density_metrics,
                            x_train, y_train, x_test, y_test, output_path="Q:/SDU/LDC/modelling/outputs",
                            output_label="imd_highsts_msoa", col_label_map=col_labels, 
                            pd_y_label="IMD Average Score", user_evaluation_model=user_model, shap_plots=True,
                            shap_id_keys=shap_plot_areas)
    return


# look through cut off vals 
for cut_off in [100]:
    cut_off_model_run(cut_off)
