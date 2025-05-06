# Set up Git repository and path
import git
import os

repo = git.Repo(".", search_parent_directories=True)
os.chdir(repo.working_tree_dir)
import sys

sys.path.append(repo.working_tree_dir)


from typing import Any, Tuple, Dict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

from src.visualise.regression_evaluation_plots import create_feature_sign_dict, create_permutation_feature_importance_plot

# matplotlib font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def create_confusion_matrix(y_test,
                          test_predictions,
                          target_var: str,
                          output_path: str,
                          output_label: str = "",
                          group_names=["True Neg","False Pos","False Neg","True Pos"],
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    y_test:        target test values

    test_predictions: predicted target test values

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''
    # generate confusion matrix values
    cf = confusion_matrix(y_test, test_predictions)

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    plt.savefig(f"{output_path}/{output_label}_confusion_matrix_{target_var}.png")
    return


def create_precision_recall_curve(
    full_pipeline: Any, 
    x_test: pd.DataFrame, 
    y_test: np.ndarray, 
    target_var: str, 
    output_path: str, 
    output_label: str = ""
) -> None:
    """
    Generates and saves a Precision-Recall (PR) curve plot for a binary classifier.

    Args:
        full_pipeline (Pipeline): A fitted scikit-learn pipeline containing a classifier.
        x_test (pd.DataFrame): The test feature set.
        y_test (np.ndarray): The true labels for the test set.
        target_var (str): The target variable name for labeling the output file.
        output_path (str): The directory where the PR curve plot will be saved.
        output_label (str, optional): A label to prefix the output file. Defaults to an empty string.

    Returns:
        None: The function saves the PR curve plot as a PNG file.
    """
    y_scores = full_pipeline.predict_proba(x_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recalls, precisions, color='darkorange', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_tick_params(bottom=True, top=False)
    ax.yaxis.set_tick_params(left=True, right=False)
    fig.legend(bbox_to_anchor=(0.9, 0.87))
    fig.savefig(f"{output_path}/{output_label}_pr_curve_{target_var}.png")
    return


def create_precision_recall_vs_threshold(
    full_pipeline: Any, 
    x_test: pd.DataFrame, 
    y_test: np.ndarray, 
    target_var: str, 
    output_path: str, 
    output_label: str = ""
) -> None:
    """
    Generates and saves a plot showing precision and recall as a function of the decision threshold.

    Args:
        full_pipeline (Pipeline): A fitted scikit-learn pipeline containing a classifier.
        x_test (pd.DataFrame): The test feature set.
        y_test (np.ndarray): The true labels for the test set.
        target_var (str): The target variable name for labeling the output file.
        output_path (str): The directory where the plot will be saved.
        output_label (str, optional): A label to prefix the output file. Defaults to an empty string.

    Returns:
        None: The function saves the precision-recall vs threshold plot as a PNG file.
    """
    y_scores = full_pipeline.predict_proba(x_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(thresholds, precisions[:-1], "darkorange", label="Precision")
    ax.plot(thresholds, recalls[:-1], "#4575b4", label="Recall")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel("Score")
    ax.set_xlabel("Decision Threshold")
    ax.set_title("Precision and Recall Scores as a function of the decision threshold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_tick_params(bottom=True, top=False)
    ax.yaxis.set_tick_params(left=True, right=False)
    fig.legend(bbox_to_anchor=(0.9, 0.87))
    fig.savefig(f"{output_path}/{output_label}_pr_threshold_{target_var}.png")
    return


def create_classification_evaluation_plots(
    full_pipeline: Any,
    model: Any,
    target_var: str,
    id_col: str,
    original_df: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    train_predictions: pd.Series,
    test_predictions: pd.Series,
    output_path: str,
    output_label: str = "",
    col_labels: dict = {},
    shap_id_keys: list = [],
    index_mapping: dict = {},
    cat_features: list = [],
) -> None:
    """
    Generates multiple plots for model evaluation including feature importance, actual vs. predicted, residuals, and partial dependence plots.

    Args:
        full_pipeline (Any): The full preprocessing and modeling pipeline.
        model (Any): The trained machine learning model.
        target_var (str): The target variable name.
        id_col (str): Name of the unique id variable for each row in the dataset.
        original_df (str): Original full feature and target df with id col.
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        train_predictions (pd.Series): Predictions for the training data.
        test_predictions (pd.Series): Predictions for the test data.
        output_path (str): A path to the directory where the output files will be saved.
        output_label (str, optional): Label to prepend to the output filename. Defaults to "".
        col_labels (dict, optional): Column labels for the features. Defaults to empty dict.
        shap_id_keys (list, optional): List for rows to create shap plots for.
        index_mapping (dict, optional): a mapping dictionary: original_df index -> (x_train or x_test, index)

    Returns:
        None
    """
    print(output_path)
    feature_sign_dict, feature_diff_dict = create_feature_sign_dict(
        full_pipeline, x_train, cat_features
    )
    create_permutation_feature_importance_plot(
        full_pipeline,
        x_test,
        y_test,
        target_var,
        feature_sign_dict,
        col_labels,
        output_path,
        output_label,
    )

    create_confusion_matrix(
        y_test,
        test_predictions,
        target_var,
        output_path,
        output_label,
    )
    create_precision_recall_curve(
        full_pipeline, 
        x_test, 
        y_test,
        target_var,
        output_path,
        output_label,
        )
    create_precision_recall_vs_threshold(
        full_pipeline, 
        x_test, 
        y_test,
        target_var,
        output_path,
        output_label,
    )
    return
