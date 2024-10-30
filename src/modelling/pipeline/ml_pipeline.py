
from typing import Any
import git
import os
import pickle
import datetime
repo = git.Repo('.', search_parent_directories=True)
os.chdir(repo.working_tree_dir)
import sys
sys.path.append(repo.working_tree_dir)


import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from src.visualise.regression_evaluation_plots import create_model_evaluation_plots


from sklearn import set_config
set_config(transform_output="pandas")


# Define preprocessing functions
def preprocess_target(df: pd.DataFrame, target_col: str) -> np.ndarray:
    """
    Extract the target variable from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_col (str): The name of the target column.

    Returns:
    np.ndarray: The target variable as a NumPy array.
    """
    target = df[target_col].values
    return target


def preprocess_features(df: pd.DataFrame, cols_to_drop: list=None, encode_categoricals: bool=True) -> pd.DataFrame:
    """
    Preprocess features by dropping specified columns and performing dummy encoding.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - cols_to_drop (List[str], optional): List of columns to be dropped. Defaults to None.
    - encode_categoricals (bool, optional): Whether to perform dummy encoding. Defaults to True.

    Returns:
    - pd.DataFrame: The preprocessed features DataFrame.
    """
    # Extract features series
    features = df.drop(cols_to_drop, axis=1)
    # Dummy encoding of any remaining categorical data (if specified)
    if encode_categoricals:
        features = pd.get_dummies(features, drop_first=True)
    return features


class FilterFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, filter_features=True, feature_filter_list=None): #
        self.filter_features = filter_features
        self.feature_filter_list = feature_filter_list
    def fit(self, x, y=None):
        return self # nothing else to do
    def transform(self, x, y=None):
        if self.filter_features:
            x_filtered = x[self.feature_filter_list]
            return x_filtered
        else:
            return x


def display_scores(scores: np.ndarray) -> tuple:
    """
    Display and return the scores, mean, and standard deviation.

    Parameters:
    - scores (np.ndarray): An array of scores.

    Returns:
    Tuple[np.ndarray, float, float]: The input scores, mean, and standard deviation.
    """
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    return (scores, scores.mean(), scores.std())


def evaluate_model(full_pipeline: object, best_model: object, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray) -> tuple:
    """
    Evaluate the performance of the given model on training and test data.

    Parameters:
    - full_pipeline (object): Trained regression model pipeline.
    - best_model (object): Trained regression model.
    - x_train (pd.DataFrame): Training input data.
    - y_train (np.ndarray): Training target data.
    - x_test (pd.DataFrame): Test input data.
    - y_test (np.ndarray): Test target data.

    Returns: tuple: Tuple containing train_rmse, train_rmse_sd, test_rmse, train_r2, train_r2_sd, test_r2, train_predictions, and test_predictions.
    """
    # Access best model index
    best_model_idx = full_pipeline.best_index_
    # Access cv_results_ dictionary
    cv_results = full_pipeline.cv_results_
    # avg training RMSE from CV test sets
    train_rmse = -cv_results["mean_test_neg_root_mean_squared_error"][best_model_idx]
    train_rmse_sd = cv_results["std_test_neg_root_mean_squared_error"][best_model_idx]
    # training predictions for eval plots
    train_predictions = best_model.predict(x_train)
    # avg training R2 from CV test sets
    train_r2 = cv_results["mean_test_r2"][best_model_idx]
    train_r2_sd = cv_results["std_test_r2"][best_model_idx]
    # test on test data - RMSE
    test_predictions = best_model.predict(x_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    # test R^2
    test_r2 = r2_score(y_test, test_predictions)
    return  train_rmse, train_rmse_sd, test_rmse, train_r2, train_r2_sd, test_r2, train_predictions, test_predictions


def output_evaluation_metrics_and_plots(user_evaluation_model: str, best_evaluation_model: str, final_model: str, full_pipeline: object, 
                                        model_name: str, best_params: dict, target_var: str, target_df: pd.DataFrame, id_col: str, original_df: pd.DataFrame,
                                        x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray, 
                                        train_predictions: np.ndarray, test_predictions: np.ndarray, train_rmse: float, train_rmse_sd: float, 
                                        test_rmse: float, train_r2: float, train_r2_sd: float, test_r2: float, 
                                        output_label: str = "", output_path: str = "",col_label_map: dict={}, pd_y_label: str = "", 
                                        shap_plots: bool=False, shap_id_keys: list=[], index_mapping: dict={}) -> None:
    """
    Output evaluation metrics and create plots for the regression model.

    Parameters:

    - user_evaluation_model (str): User defined model to use when creating evaluation plots. 
      If not defined, evaluation plots will be created for the best performing model.
    - best_evaluation_model (str): best performing model so far.
    - final_model (str): final model in model dictionary.
    - full_pipeline (object): Trained regression model pipeline.
    - model_name (str): Name of the regression model.
    - best_params (dict): Best hyperparameters found during GridSearchCV.
    - target_var (str): Name of the target variable.
    - target_df (pd.DataFrame): DataFrame containing the target variable.
    - id_col (str): Name of the unique id variable for each row in the dataset.
    - original_df (str): Original full feature and target df with id col.
    - x_train (pd.DataFrame): Training input data.
    - y_train (pd.DataFrame): Training target data.
    - x_test (np.ndarray): Test input data.
    - y_test (np.ndarray): Test target data.
    - train_predictions (np.ndarray): Predictions on the training set.
    - test_predictions (np.ndarray): Predictions on the test set.
    - train_rmse (float): Training RMSE.
    - train_rmse_sd (float): Standard deviation of training RMSE.
    - test_rmse (float): Test RMSE.
    - train_r2 (float): Training R^2.
    - train_r2_sd (float): Training R^2 standard deviation from CV.
    - test_r2 (float): Test R^2.
    - output_label (str): A label to add to the output files saved.
    - output_path (str): A path to the directory where the output files will be saved.
    - col_label_map (dict): A map of shortened feature names for the evaluation plots.
    - pd_y_label (str, optional): A label the y axis of the PD plots.
    - shap_plots (bool, optional): Toggle to create shap plots for rows specified by shap_id_keys list.
    - shap_id_keys (list, optional): List for rows to create shap plots for.
    - index_mapping (dict, optional): a mapping dictionary: original_df index -> (x_train or x_test, index)

    Returns: None
    """
   #create an empty csv for the results
    filename = f"{output_path}/{output_label}_regression_model_summary.csv"
    if os.path.isfile(filename):
        all_models_evaluation_df = pd.read_csv(filename)
    else:
        all_models_evaluation_df = pd.DataFrame()
    # remove feature list hyperparameter from output 
    del best_params['feature_filter__feature_filter_list']
        
    model_evaluation_dict = {"time": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                            "model": model_name,
                            "params": str(best_params),
                            "target_variable": target_var,
                            "target_mean": target_df.mean(),
                            "train_RMSE": train_rmse.round(2),  
                            "train_RMSE_sd": train_rmse_sd.round(3),     
                            "test_RMSE": test_rmse.round(2),
                            "test_RMSE_perc_mean": (test_rmse / target_df.mean()).round(2),
                            "train_R^2": train_r2.round(2),
                            "train_R^2_sd": train_r2_sd.round(2),
                            "test_R^2": round(test_r2, 2)
                            }
    model_evaluation_df = pd.DataFrame([model_evaluation_dict])
    output = pd.concat([all_models_evaluation_df, model_evaluation_df])
    output.drop_duplicates().to_csv(filename, index=False)
    # create evaluation plots
    feature_importance_model = full_pipeline.best_estimator_.named_steps["model"]
    if model_name == user_evaluation_model:
        print("Creating evaluation plots")
        create_model_evaluation_plots(full_pipeline, feature_importance_model, target_var, x_train, y_train, x_test, y_test, train_predictions, test_predictions, output_label, output_path, col_label_map, pd_y_label, shap_plots, shap_id_keys, index_mapping)
    # if no user defined model then create plots for best performing model
    elif user_evaluation_model == "":
        if model_name == final_model:
            best_model = best_evaluation_model.best_estimator_.named_steps["model"]
            train_predictions = best_evaluation_model.predict(x_train)
            test_predictions = best_evaluation_model.predict(x_test)
            print("The best performing model is: " + str(best_model))
            print("Creating evaluation plots")
            create_model_evaluation_plots(best_evaluation_model, best_model, target_var, id_col, original_df, x_train, y_train, x_test, y_test, train_predictions, test_predictions, output_label, output_path, col_label_map, pd_y_label, shap_plots, shap_id_keys, index_mapping)
    return

def log_gridsearch_results_to_mlflow(grid_search: object, model_name: str, output_label: str="") -> None:
    """
    Log all GridSearchCV results and hyperparameters to MLflow.
    
    Parameters:
    - grid_search (object): The GridSearchCV object after it has been fitted.
    - model_name (str): A name to identify the model in the MLflow logs.
    - output_label (str): A label to add to the output files saved.
    """
    # set experiment id as output label if present, default experiment if not
    if output_label:
        mlflow.set_experiment(output_label)
    # Loop through the results for each parameter combination
    # End any existing runs
    mlflow.end_run()
    run_name = output_label + model_name
    with mlflow.start_run(run_name=run_name) as parent_run:
        # log the model name
        mlflow.set_tag("model_name", model_name)
        for i in range(len(grid_search.cv_results_['params'])):
            with mlflow.start_run(run_name="hyper_parmam_" + run_name, nested=True):
                # Log the hyperparameters
                params = grid_search.cv_results_['params'][i]
                mlflow.log_params(params)

                # Log the metrics for this parameter combination
                mean_train_rmse = -grid_search.cv_results_['mean_train_neg_root_mean_squared_error'][i]
                mean_test_rmse = -grid_search.cv_results_['mean_test_neg_root_mean_squared_error'][i]
                mean_train_r2 = grid_search.cv_results_['mean_train_r2'][i]
                mean_test_r2 = grid_search.cv_results_['mean_test_r2'][i]

                mlflow.log_metric("train_rmse", mean_train_rmse)
                mlflow.log_metric("test_rmse", mean_test_rmse)
                mlflow.log_metric("train_r2", mean_train_r2)
                mlflow.log_metric("test_r2", mean_test_r2)
    return


def model_grid_cv_pipeline(model_param_dict: dict, target_var: str, target_df: pd.DataFrame, 
                           id_col: str, original_df: pd.DataFrame, x_train: pd.DataFrame, y_train: np.ndarray, 
                           x_test: pd.DataFrame, y_test: np.ndarray, output_label: str = "", output_path: str = "",
                           col_label_map: dict={}, pd_y_label: str = "", user_evaluation_model: str="", 
                           shap_plots: bool=False, shap_id_keys: list=[]) -> None:
    """
    Perform a grid search cross-validation for multiple regression models.

    Parameters:
    - model_param_dict (dict): Dictionary containing regression models and their hyperparameter grids.
    - target_var (str): Name of the target variable.
    - target_df (str): Original target df.
    - id_col (str): Name of the unique id variable for each row in the dataset.
    - original_df (str): Original full feature and target df with id col.
    - x_train (pd.DataFrame): Training input data.
    - y_train (np.ndarray): Training target data.
    - x_test (pd.DataFrame): Test input data.
    - y_test (np.ndarray): Test target data.
    - output_label (str): A label to add to the output files saved.
    - output_path (str): A path to the directory where the output files will be saved.
    - col_label_map (dict): A map of shortened feature names for the evaluation plots
    - pd_y_label (str, optional): A label the y axis of the PD plots.
    - user_evaluation_model (str, optional): User defined model to use when creating evaluation plots. 
      If not defined, evaluation plots will be created for the best performing model.
    - shap_plots (bool, optional): Toggle to create shap plots for rows specified by shap_id_keys list.
    - shap_id_keys (list, optional): List for rows to create shap plots for.

    Returns: None
    """

    # initialise evaluation metric checker to track best performing model
    best_r2 = -100
    # final model name to trigger evaluation chart plotting
    final_model = str(list(model_param_dict.keys())[-1]).split("(")[0]

    # Create a mapping dictionary for Shap plots: original_df index -> (x_train or x_test, index)
    # Get the indices for the train and test sets
    train_indices = x_train.index
    test_indices = x_test.index
    index_mapping = {}
    # Fill the mapping for train indices
    for idx in train_indices:
        index_mapping[idx] = ('train', train_indices.get_loc(idx))

    # Fill the mapping for test indices
    for idx in test_indices:
        index_mapping[idx] = ('test', test_indices.get_loc(idx))

    for model in model_param_dict.keys():
        model_name = str(model).split("(")[0]
        print(model_name)

        processing_pipeline = Pipeline([
                ('feature_filter', FilterFeatures()),
                ('scaler', StandardScaler()),
                ('selector', VarianceThreshold()),
                ('model', model)
            ])

        # defining optimisation criteria here, this could be user defined in future.
        full_pipeline = GridSearchCV(processing_pipeline, model_param_dict[model], cv=5,
                            scoring=['neg_root_mean_squared_error', 'r2'],
                            refit='r2',
                            return_train_score=True,
                            verbose=2)

        full_pipeline.fit(x_train, y_train)

        # best model from cv search
        best_params = full_pipeline.best_params_
        best_model = full_pipeline.best_estimator_

        # save best models
        pickle.dump(
            best_model,
            open(output_path + "/" + output_label + "_" + model_name + "_" + target_var + ".pickle", "wb"),
        )

        # evaluate model
        train_rmse, train_rmse_sd, test_rmse, train_r2, train_r2_sd, test_r2, train_predictions, test_predictions = evaluate_model(full_pipeline, best_model, x_train, y_train, x_test, y_test)
        
        # model parameters and metrics to MLflow
        log_gridsearch_results_to_mlflow(full_pipeline, model_name, output_label)

        # keep track of best performing model in terms of r2 for eval plots
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_evaluation_model = full_pipeline
            # now just input into eval metrics function

        # output model results
        output_evaluation_metrics_and_plots(user_evaluation_model, best_evaluation_model, final_model, full_pipeline, model_name, best_params, target_var, target_df, id_col, original_df, x_train, y_train, x_test, y_test, train_predictions, test_predictions, train_rmse, train_rmse_sd, test_rmse, train_r2, train_r2_sd, test_r2, output_label, output_path, col_label_map, pd_y_label, shap_plots, shap_id_keys, index_mapping)
    return 