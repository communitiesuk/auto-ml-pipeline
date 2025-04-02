import git
import os
import pickle
import datetime
import sys


# setting path for those using Spyder
repo = git.Repo(".", search_parent_directories=True)
os.chdir(repo.working_tree_dir)
sys.path.append(repo.working_tree_dir)


import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from joblib import effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import set_config

from src.visualise.regression_evaluation_plots import create_regression_evaluation_plots
from src.visualise.classification_evaluation_plots import create_classification_evaluation_plots

# set config to track feature names after transformations
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


def preprocess_features(
    df: pd.DataFrame, cols_to_drop: list = None, encode_categoricals: bool = True
) -> pd.DataFrame:
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
    def __init__(self, filter_features=False, feature_filter_list=None):  #
        self.filter_features = filter_features
        self.feature_filter_list = feature_filter_list

    def fit(self, x, y=None):
        return self  # nothing else to do

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


def evaluate_model(
    full_pipeline: object,
    best_model: object,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple:
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
    # model predictions
    train_predictions = best_model.predict(x_train)
    test_predictions = best_model.predict(x_test)
    # dictionary to store eval metrics
    eval_metrics = {}
    # if classifier, output classification eval metrics
    if is_classifier(best_model):
        eval_metrics["train_f1"] = round(cv_results["mean_test_f1_macro"][best_model_idx], 2)
        eval_metrics["train_accuracy"] = round(cv_results["mean_test_accuracy"][best_model_idx], 2)
        eval_metrics["test_f1"] = round(f1_score(y_test, test_predictions), 2)
        eval_metrics["test_accuracy"] = round(accuracy_score(y_test, test_predictions), 2)
        eval_metrics["test_precision"] = round(precision_score(y_test, test_predictions), 2)
        eval_metrics["test_recall"] = round(recall_score(y_test, test_predictions), 2)
        # add ml flow bit
        # log test performance to MLflow
        mlflow.log_metric("test_accuracy", eval_metrics["test_accuracy"])
        mlflow.log_metric("test_f1", eval_metrics["test_f1"])
        mlflow.log_metric("test_precision", eval_metrics["test_precision"])
        mlflow.log_metric("test_recall", eval_metrics["test_recall"])
    else:
        # else output regression metrics
        # training metrics from CV test sets
        eval_metrics["train_rmse"] = round(-cv_results["mean_test_neg_root_mean_squared_error"][best_model_idx], 2)
        eval_metrics["train_rmse_sd"] = round(cv_results["std_test_neg_root_mean_squared_error"][best_model_idx], 3)
        eval_metrics["train_r2"] = round(cv_results["mean_test_r2"][best_model_idx], 2)
        eval_metrics["train_r2_sd"] = round(cv_results["std_test_r2"][best_model_idx], 2)
        eval_metrics["test_rmse"] = round(np.sqrt(mean_squared_error(y_test, test_predictions)), 2)
        eval_metrics["test_r2"] = round(r2_score(y_test, test_predictions), 2)
        # log test performance to MLflow
        mlflow.log_metric("test_r2", eval_metrics["test_r2"])
        mlflow.log_metric("test_rmse", eval_metrics["test_rmse"])
    return (
        eval_metrics,
        train_predictions,
        test_predictions,
    )


def output_evaluation_metrics_and_plots(
    user_evaluation_model: str,
    best_evaluation_model: str,
    final_model: str,
    full_pipeline: object,
    model_name: str,
    best_params: dict,
    target_var: str,
    target_df: pd.DataFrame,
    id_col: str,
    original_df: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    eval_metrics: dict,
    output_label: str = "",
    output_path: str = "",
    col_label_map: dict = {},
    shap_id_keys: list = [],
    index_mapping: dict = {},
) -> None:
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
    - shap_id_keys (list, optional): List for rows to create shap plots for.
    - index_mapping (dict, optional): a mapping dictionary: original_df index -> (x_train or x_test, index)

    Returns: None
    """
    # create an empty csv for the results
    filename = f"{output_path}/{output_label}_regression_model_summary.csv"
    if os.path.isfile(filename):
        all_models_evaluation_df = pd.read_csv(filename)
    else:
        all_models_evaluation_df = pd.DataFrame()
    # remove feature list hyperparameter from output
    try:
        del best_params["feature_filter__feature_filter_list"]
    except KeyError:
        pass
    model_evaluation_dict = {
        "time": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "model": model_name,
        "params": str(best_params),
        "target_variable": target_var
        }
    # create evaluation plots
    feature_importance_model = full_pipeline.best_estimator_.named_steps["model"]

    if is_classifier(full_pipeline.best_estimator_.named_steps["model"]):
        model_evaluation_dict.update(eval_metrics)
        print("Creating classification evaluation plots")

    else:
        model_evaluation_dict.update(eval_metrics)
        model_evaluation_dict.update({
            "target_mean": target_df.mean(),
            "test_RMSE_perc_mean": (eval_metrics["test_rmse"] / target_df.mean()).round(2),
        })
    model_evaluation_df = pd.DataFrame([model_evaluation_dict])
    output = pd.concat([all_models_evaluation_df, model_evaluation_df])
    output.drop_duplicates().to_csv(filename, index=False)
    
    if model_name == user_evaluation_model:
        if is_classifier(feature_importance_model):
            print("Creating classification evaluation plots")
            create_classification_evaluation_plots(
                full_pipeline,
                feature_importance_model,
                target_var,
                id_col,
                original_df,
                x_train,
                y_train,
                x_test,
                y_test,
                train_predictions,
                test_predictions,
                output_label,
                output_path,
                col_label_map,
                shap_id_keys,
                index_mapping,
            )
        else:
            print("Creating regression plots")
            create_regression_evaluation_plots(
                full_pipeline,
                feature_importance_model,
                target_var,
                id_col,
                original_df,
                x_train,
                y_train,
                x_test,
                y_test,
                train_predictions,
                test_predictions,
                output_label,
                output_path,
                col_label_map,
                shap_id_keys,
                index_mapping,
            )
    # if no user defined model then create plots for best performing model
    elif user_evaluation_model == "":
        if model_name == final_model:
            best_model = best_evaluation_model.best_estimator_.named_steps["model"]
            train_predictions = best_evaluation_model.predict(x_train)
            test_predictions = best_evaluation_model.predict(x_test)
            print("The best performing model is: " + str(best_model))
            if is_classifier(best_model):
                print("Creating classification evaluation plots")
                create_classification_evaluation_plots(
                    best_evaluation_model,
                    best_model,
                    target_var,
                    id_col,
                    original_df,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    train_predictions,
                    test_predictions,
                    output_label,
                    output_path,
                    col_label_map,
                    shap_id_keys,
                    index_mapping,
                )
            else:
                print("Creating regression plots")
                create_regression_evaluation_plots(
                    best_evaluation_model,
                    best_model,
                    target_var,
                    id_col,
                    original_df,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    train_predictions,
                    test_predictions,
                    output_label,
                    output_path,
                    col_label_map,
                    shap_id_keys,
                    index_mapping,
                ) 
    return


def log_results_to_mlflow(model_name: str, output_label: str = "") -> None:
    """
    Log all randomised search cv results and hyperparameters to MLflow.

    Parameters:
    - model_name (str): A name to identify the model in the MLflow logs.
    - output_label (str): A label to add to the output files saved.
    """
    # set experiment id as output label if present, default experiment if not
    if output_label:
        mlflow.set_experiment(output_label)
    # autolog hyperparams and eval metrics
    mlflow.sklearn.autolog(
        log_input_examples=True,
        max_tuning_runs=50,
        log_post_training_metrics=False,
        extra_tags={"model_name": model_name},
    )
    return


def model_pipeline(
    model_param_dict: dict,
    target_var: str,
    target_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    id_col: str,
    original_df: pd.DataFrame,
    output_label: str = "",
    output_path: str = "",
    col_label_map: dict = {},
    user_evaluation_model: str = "",
    shap_id_keys: list = [],
    custom_pre_processing_steps: list = [],
) -> None:
    """
    Perform randomised search cross-validation for multiple regression models.

    Parameters:
    - model_param_dict (dict): Dictionary containing regression models and their hyperparameter grids.
    - target_var (str): Name of the target variable.
    - target_df (pd.DataFrame): Original target df.
    - feature_df (pd.DataFrame): Original feature df.
    - id_col (str): Name of the unique id variable for each row in the dataset.
    - original_df (str): Original full feature and target df with id col.
    - x_train (pd.DataFrame): Training input data.
    - y_train (np.ndarray): Training target data.
    - x_test (pd.DataFrame): Test input data.
    - y_test (np.ndarray): Test target data.
    - output_label (str): A label to add to the output files saved.
    - output_path (str): A path to the directory where the output files will be saved.
    - col_label_map (dict): A map of shortened feature names for the evaluation plots
    - user_evaluation_model (str, optional): User defined model to use when creating evaluation plots.
      If not defined, evaluation plots will be created for the best performing model.
    - shap_id_keys (list, optional): List for rows to create shap plots for.
    - custom_pre_processing_steps (list, optional): List of user provided steps for custom pre-processing pipeline


    Returns: None
    """
    # create test set of 20%
    x_train, x_test, y_train, y_test = train_test_split(
        feature_df, target_df, test_size=0.20, random_state=36
    )

    # print number of cores available for parallel processing
    print(f"Number of cores available for parallel processing: {effective_n_jobs(-1)}")

     # Create a mapping dictionary for Shap plots: original_df index -> (x_train or x_test, index)
    # Get the indices for the train and test sets
    train_indices = x_train.index
    test_indices = x_test.index
    index_mapping = {}
    # Fill the mapping for train indices
    for idx in train_indices:
        index_mapping[idx] = ("train", train_indices.get_loc(idx))

    # Fill the mapping for test indices
    for idx in test_indices:
        index_mapping[idx] = ("test", test_indices.get_loc(idx))

    # initialise evaluation metric checkers to track best performing model (r2 for regression, f1 for binary classification)
    # final model name to trigger evaluation chart plotting
    final_model = str(list(model_param_dict.keys())[-1]).split("(")[0]

    for model in model_param_dict.keys():
        # check if classifer, if not regression
        if is_classifier(model):
            scoring_metrics = ["f1_macro", "accuracy"]
            best_scorer = "test_f1"
            best_score = 0

        elif is_regressor(model):
            scoring_metrics = ["neg_root_mean_squared_error", "r2"]
            best_scorer = "test_r2"
            best_score = -100

        else:
            raise Exception("Model type not recognised")

        model_name = str(model).split("(")[0]
        print(model_name)

        # apply custom pre_processing steps, else use default processing pipeline
        if custom_pre_processing_steps:
            steps = custom_pre_processing_steps.copy()
            steps.append(("model", model))
            processing_pipeline = Pipeline(steps)
        else:
            processing_pipeline = Pipeline(
                [
                    ("feature_filter", FilterFeatures()),
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )

        # log model parameters and metrics to MLflow
        log_results_to_mlflow(model_name, output_label)

        with mlflow.start_run(run_name=output_label + "_" + model_name) as run:
            # defining optimisation criteria here, this could be user defined in future.
            full_pipeline = RandomizedSearchCV(
                processing_pipeline,
                model_param_dict[model],
                cv=5,
                n_iter=150,
                scoring=scoring_metrics,
                refit=scoring_metrics[0],
                return_train_score=True,
                verbose=2,
                n_jobs=-1,
            )

            full_pipeline.fit(x_train, y_train)

            # best model from cv search
            best_params = full_pipeline.best_params_
            best_model = full_pipeline.best_estimator_

            # save best models
            pickle.dump(
                best_model,
                open(
                    output_path
                    + "/"
                    + output_label
                    + "_"
                    + model_name
                    + "_"
                    + target_var
                    + ".pickle",
                    "wb",
                ),
            )

            # evaluate model
            (
                eval_metrics,
                train_predictions,
                test_predictions,
            ) = evaluate_model(
                full_pipeline, best_model, x_train, y_train, x_test, y_test
            )

        # keep track of best performing model in terms of r2 or f1 score for eval plots
        test_score = eval_metrics[best_scorer]
        if test_score > best_score:
            best_score = test_score
            best_evaluation_model = full_pipeline
            # now just input into eval metrics function

        # output model results
        output_evaluation_metrics_and_plots(
            user_evaluation_model,
            best_evaluation_model,
            final_model,
            full_pipeline,
            model_name,
            best_params,
            target_var,
            target_df,
            id_col,
            original_df,
            x_train,
            y_train,
            x_test,
            y_test,
            train_predictions,
            test_predictions,
            eval_metrics,
            output_label,
            output_path,
            col_label_map,
            shap_id_keys,
            index_mapping,
        )
    return
