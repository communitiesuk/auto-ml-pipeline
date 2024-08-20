# Set up Git repository and path
import git
import os
repo = git.Repo('.', search_parent_directories=True)
os.chdir(repo.working_tree_dir)
import sys
sys.path.append(repo.working_tree_dir)

from typing import Any, Tuple
import shap
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from sklearn import tree
from sklearn.inspection import partial_dependence, PartialDependenceDisplay, permutation_importance

from src.visualise.scatter_chart import scatter_chart

# matplotlib font sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def create_partial_dependence_plots(model: Any, x_train: pd.DataFrame, target_var: str, output_path: str, output_label: str = "",  col_labels: dict={}, pd_y_label: str = "", feature_diff_dict: dict={}) -> None:
    """
    Generates partial dependence plots (PDPs) for a given machine learning model.

    Args:
        model (Any): The trained machine learning model or pipeline. Its type depends on the specific library used (e.g., sklearn.ensemble.RandomForestRegressor).
        x_train (pd.DataFrame): The training data used to fit the model.
        target_var (str): The name of the target variable in the training data.
        output_path (str): A path to the directory where the output files will be saved.
        output_label (str, optional): A label to prepend to the output filename. Defaults to "".
        col_labels (dict): A map of shortened feature names for the plots
        pd_y_label (str, optional): A label the y axis of the PD plots.
        feature_diff_dict (dict): The difference values between the first data point in the and the last data point in the series for each feature.

    Returns:
        None

    This function creates a grid of partial dependence plots, visualising the marginal effect of each feature in
    x_train on the target variable. 

    The plots are saved to a PNG image file with a descriptive filename based on the `output_label` and `target_var`.

    Raises:
        ValueError: If the `target_var` is not found in the training data columns.
    """
    # Compute Rows required to plot each feature
    n_cols = 3
    n_features = len(x_train.columns)
    n_rows = (n_features + n_cols - 1) // n_cols  # This ensures the ceiling of n_features / n_cols

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20

    # define feature order based on max abs difference from diff_list
    features_sorted_differences = sorted(feature_diff_dict.values())
    features_sorted = sorted(feature_diff_dict, key=feature_diff_dict.get)

    # if features less than number of cells in grid, add dummy axes and remove later
    dummy_count = 0
    while n_features < n_cols * n_rows:
        # iterate
        n_features += 1
        dummy_count += 1
    # Create main figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 11))
    axs = axs.flatten()  # Flatten the axs array to iterate easily

    # if dummy plots present, remove from axes grid
    if dummy_count != 0:
        for j in range(0, dummy_count):
            # remove axes objects for dummies
            fig.delaxes(axs[n_features - j - 1])
        # select non empty axes only
        axs = axs[0: -dummy_count]

    display = PartialDependenceDisplay.from_estimator(model, x_train, features_sorted, ax=axs, n_cols=n_cols, line_kw={"color": "#4575b4"})

    # Create PartialDependenceDisplay for each feature and apply custom formatting
    for i, feature in enumerate(features_sorted):
        # apply formatting after plotting
        new_title = col_labels.get(feature, feature)
        axs[i].set_xlabel(new_title)
        if i % n_cols == 0:
            axs[i].set_ylabel(pd_y_label)
        else: 
            axs[i].set_ylabel('')
        # add target difference annotation

        x_data = axs[i].get_lines()[0].get_xdata()
        y_data = axs[i].get_lines()[0].get_ydata()
        arr = mpatches.FancyArrowPatch((x_data[-1], y_data[0]), (x_data[-1], y_data[-1]),
                               arrowstyle='|-|')
        axs[i].add_patch(arr)
        axs[i].annotate(int(round(abs(features_sorted_differences[i]), 0)), (2, .5), xycoords=arr, ha='left', va='center', fontsize=12)

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].xaxis.set_tick_params(bottom=True, top=False)
        axs[i].yaxis.set_tick_params(left=True, right=False)
        plt.setp(display.deciles_vlines_, visible=False)
    
    plt.setp(display.deciles_vlines_, visible=False)
    
    fig.suptitle('Partial Dependence Plots')
    fig.tight_layout()
    fig.savefig(f"{output_path}/{output_label}_pd_plots_{target_var}.png")
    return


def create_feature_sign_dict(model: Any, x_train: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Creates a dictionary indicating the sign (positive or negative) of the relationship between each feature and the target variable.

    Args:
        model (Any): The trained machine learning model or pipeline. 
        x_train (pd.DataFrame): The training data used to fit the model.

    Returns:
        feature_sign_dict (dict): A dictionary where keys are feature names (from `x_train.columns`) and values are +1 (positive relationship) or -1 (negative relationship).
        feature_diff_dict (dict): The difference values between the first data point in the and the last data point in the series for each feature.

    This function analyzes the partial dependence plots (implicitly generated) to infer the sign of the impact of each
    feature on the target variable. It assumes that a positive (negative) change in the PDP line from the beginning to the end
    indicates a positive (negative) relationship between the feature and the target.

    Raises:
        ValueError: If the `target_var` is not found in the training data columns.
    """
    feature_sign_dict = {}
    feature_diff_dict = {}
    select_features_list = x_train.columns
    for i, feature in enumerate(select_features_list):
        y_data = PartialDependenceDisplay.from_estimator(model, x_train, [select_features_list[i]]).lines_[0, 0].get_ydata()
        diff = y_data[len(y_data) - 1] - y_data[0]
        if diff > 0:
            feature_sign_dict[feature] = 1
        else:
            feature_sign_dict[feature] = -1
        feature_diff_dict[feature] = diff
    return  feature_sign_dict, feature_diff_dict


def create_feature_importance_plot(model, x_train, target_var, sign_dict, col_labels, output_label: str ="", output_path: str= ""):
    global_importances = pd.DataFrame(data={"Importance": model.feature_importances_, "Feature": x_train.columns})
    global_importances.sort_values(by="Importance", ascending=True, inplace=True)
    global_importances["sign"] = global_importances["Feature"].map(sign_dict).astype(str)
    global_importances["Feature"] = global_importances["Feature"].replace(col_labels)
    fig = px.bar(global_importances, y="Importance", x="Feature", color="sign", title="Global Feature Importance", color_discrete_map={"1": "#4575b4", "-1": "#d73027"}, labels=col_labels)
    fig.update_layout(
        xaxis_categoryorder = 'total ascending',
        showlegend=False,
        height=750,
        width=1000,
    )
    fig.write_html(f"{output_path}/{output_label}_tree_feature_importance_{target_var}.html")
    return


def create_permutation_feature_importance_plot(model: Any, x_test: pd.DataFrame, y_test: pd.Series, target_var: str, sign_dict: dict, col_labels: dict, output_path: str, output_label: str = "") -> None:
    """
    Creates permutation feature importance plot for any given model (not just tree based models).

    Args:
        model (Any): The trained machine learning model.
        x_test (pd.DataFrame): The test data used for evaluation.
        y_test (pd.Series): The true target values for the test data.
        target_var (str): The target variable name.
        sign_dict (Dict[str, int]): Dictionary of feature names and their signs.
        col_labels (Dict[str, str]): Dictionary of column labels for the features.
        output_label (str, optional): A label to prepend to the output filename. Defaults to "".
        output_path (str): A path to the directory where the output files will be saved.

    Returns:
        None
    """
    # get permutation importances array
    feature_importances = permutation_importance(model, x_test, y_test,
                           n_repeats=30,
                           random_state=0)
    # sort by most important features
    sorted_idx = feature_importances.importances_mean.argsort()

    # join bootstrapped importance arrays together for plotting
    combined_df = pd.DataFrame()
    for i, col in enumerate(x_test.columns[sorted_idx]):
        temp_df = pd.DataFrame()
        # create temp df for each feature
        temp_df["importances"] = feature_importances.importances[sorted_idx][i]
        temp_df["variable"] = col
        # concat together and plot result using px box
        combined_df = pd.concat([temp_df, combined_df], axis=0)

    combined_df["sign"] = combined_df["variable"].map(sign_dict).astype(str)
    combined_df["variable"] = combined_df["variable"].replace(col_labels) 
    # plotly box plot
    fig = px.box(combined_df, y="variable", x="importances", color="sign", title="Permutation Feature Importance", color_discrete_map={"1": "#4575b4", "-1": "#d73027"}, points=False)
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="grey")
    fig.update_layout(
    yaxis_title="",
    xaxis_title="Importance"
    )
    fig.update_layout(yaxis_categoryorder = 'total ascending')
    fig.update_layout(showlegend=False)
    fig.update_layout(
        height=750,
        width=1000,
    )
    fig.write_html(f"{output_path}/{output_label}_permutation_feature_importance_{target_var}.html")
    return


def create_actual_vs_predicted_scatter(y_train: pd.Series, y_test: pd.Series, train_predictions: pd.Series, test_predictions: pd.Series, target_var: str, output_path: str, output_label: str = "") -> None:
    """
    Generates an actual vs. predicted scatter plot for both training and testing datasets.

    Args:
        y_train (pd.Series): Actual values for the training data.
        y_test (pd.Series): Actual values for the test data.
        train_predictions (pd.Series): Predicted values for the training data.
        test_predictions (pd.Series): Predicted values for the test data.
        target_var (str): The name of the target variable.
        output_path (str): A path to the directory where the output files will be saved.
        output_label (str, optional): Label to prepend to the output filename. Defaults to "".

    Returns:
        None
    """
    # create actual vs predicted plot
    actual_vs_predicted_test = pd.DataFrame(data={"Actual": y_test, "Predicted": test_predictions})
    actual_vs_predicted_test["Type"] = "Test"
    actual_vs_predicted_train = pd.DataFrame(data={"Actual": y_train, "Predicted": train_predictions})
    actual_vs_predicted_train["Type"] = "Train"
    actual_vs_predicted = pd.concat([actual_vs_predicted_test, actual_vs_predicted_train], axis=0)
    fig = scatter_chart(data=actual_vs_predicted , x_var="Actual", y_var="Predicted", 
                        x_label='Actual', y_label="Predicted", hover_labels='Actual', title="Predicted vs Actual Values " + target_var , 
                        colour_col="Type", trend_line=None)
    # add y = x line
    # find start and end coords for the y = x line by finding the min and max Actual values from the training set
    x_min = pd.concat([actual_vs_predicted_train['Actual'], actual_vs_predicted_test['Actual']]).min()
    x_max = pd.concat([actual_vs_predicted_train['Actual'], actual_vs_predicted_test['Actual']]).max()
    fig.update_traces(opacity=0.7)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[x_min, x_max],
        mode="lines",
        line=go.scatter.Line(color="gray"),
        name='y = x',
        line_dash='dash',
        showlegend=True))
    fig.update_layout(
        height=750,
        width=1000,
    )
    fig.write_html(f"{output_path}/{output_label}_actual_vs_predicted_scatter_{target_var}.html")
    return


def create_residuals_plot(y_train: pd.Series, y_test: pd.Series, train_predictions: pd.Series, test_predictions: pd.Series, target_var: str, output_path: str, output_label: str = "") -> go.Figure:
    """
    Generates a residuals plot for both training and testing datasets.

    Args:
        y_train (pd.Series): Actual values for the training data.
        y_test (pd.Series): Actual values for the test data.
        train_predictions (pd.Series): Predicted values for the training data.
        test_predictions (pd.Series): Predicted values for the test data.
        target_var (str): The name of the target variable.
        output_path (str): A path to the directory where the output files will be saved.
        output_label (str, optional): Label to prepend to the output filename. Defaults to "".

    Returns:
        go.Figure: Plotly figure object.
    """
    # create actual vs predicted plot
    actual_vs_predicted_test = pd.DataFrame(data={"Actual": y_test, "Predicted": test_predictions})
    actual_vs_predicted_test['Residuals'] = actual_vs_predicted_test['Predicted'] - actual_vs_predicted_test['Actual']
    actual_vs_predicted_test['Type'] = 'Test'
    actual_vs_predicted_train = pd.DataFrame(data={"Actual": y_train, "Predicted": train_predictions})
    actual_vs_predicted_train['Residuals'] = actual_vs_predicted_train['Predicted'] - actual_vs_predicted_train['Actual']
    actual_vs_predicted_train['Type'] = 'Train'
    actual_vs_predicted = pd.concat([actual_vs_predicted_test, actual_vs_predicted_train], axis=0)
    fig = scatter_chart(data=actual_vs_predicted , x_var="Actual", y_var="Residuals", 
                        x_label='Actual', y_label="Residuals", hover_labels='Actual', title="Residuals vs Actual Values " + target_var, 
                        colour_col="Type", trend_line=None)
    # add y = 0 line
    # find start and end x coords for the y = 0 line by finding the min and max Actual values from the training set
    x_min = pd.concat([actual_vs_predicted_train['Actual'], actual_vs_predicted_test['Actual']]).min()
    x_max = pd.concat([actual_vs_predicted_train['Actual'], actual_vs_predicted_test['Actual']]).max()
    fig.update_traces(opacity=0.7)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[0, 0],
        mode="lines",
        line=go.scatter.Line(color="gray"),
        name='y = 0',
        line_dash='dash',
        showlegend=True))
    fig.update_layout(
        height=750,
        width=1000,
    )
    fig.write_html(f"{output_path}/{output_label}_residuals_scatter_{target_var}.html")
    return fig


def create_tree_plot(model: Any, x_train: pd.DataFrame, target_var: str, output_path: str, output_label: str = "") -> None:
    """
    Creates a plot of the decision tree structure.

    Args:
        model (Any): The trained machine learning model (should be a tree-based model).
        x_train (pd.DataFrame): The training data used to fit the model.
        target_var (str): The target variable name.
        output_path (str): A path to the directory where the output files will be saved.
        output_label (str, optional): Label to prepend to the output filename. Defaults to "".

    Returns:
        None
    """
    if not hasattr(model, 'estimators_'):
        raise ValueError("Model must be a tree-based ensemble with 'estimators_' attribute.")
    
    feature_names = list(x_train.columns)
    target_name = [target_var]
    fig, axes = plt.subplots(nrows = 1, ncols=1, figsize=(12, 12), dpi=600)
    tree.plot_tree(model.estimators_[0],
                feature_names=feature_names, 
                class_names=target_name,
                filled=False,
                impurity=False,
                fontsize=16)
    fig.savefig(f"{output_path}/{output_label}_tree_diagram_{target_var}.png")
    return


def create_shap_plots(id_col: str, original_df: pd.DataFrame, 
                      x_test: pd.DataFrame, x_train: pd.DataFrame,
                      full_pipeline: any, target_var: str, shap_id_keys: list[str],
                      output_path: str, output_label: str = "") -> None:
    """
    Creates SHAP force plots for specified IDs.

    Args:
        id_col: The column name containing the unique IDs.
        original_df: The original dataframe containing all data.
        x_test: The test dataset.
        x_train: The training dataset.
        full_pipeline: The fitted pipeline used for preprocessing and modeling.
        target_var: The target variable for the model.
        shap_id_keys: A list of IDs for which to create SHAP plots.
        output_path (str): A path to the directory where the output files will be saved.
        output_label (str, optional): Label to prepend to the output filename. Defaults to "".

    Returns:
        None
    """
    # init for plotting
    shap.initjs()
    model = full_pipeline.best_estimator_.named_steps['model']
    if hasattr(model, 'estimators_'):
        explainer = shap.TreeExplainer(model)
    else: 
        print("shap plots not available for non-tree based models")
        return
    for id in shap_id_keys:
        # get index for each id shap_id_keys
        row_index = original_df.index.get_loc(original_df[original_df[id_col] == id].index[0])
        # use pipeline without model step to transform test data
        try:
            row_data = x_test.loc[[row_index]]
            x_transformed = full_pipeline.best_estimator_[:-1].transform(x_test.loc[[row_index]])
        except(KeyError):
            print(f"warning, id: {id} is in training set rather than test set")
            row_data = x_train.loc[[row_index]]
            x_transformed = full_pipeline.best_estimator_[:-1].transform(x_train.loc[[row_index]])
        # get shap value
        shap_values = explainer.shap_values(x_transformed)
        # create plot
        shap.force_plot(explainer.expected_value, shap_values[0], row_data, 
                        show=False, matplotlib=True, text_rotation=45, contribution_threshold=0.035).savefig(f'{output_path}/{output_label}_shap_plot_{target_var}_{id}.png', bbox_inches='tight', dpi=300)
    return 


def create_model_evaluation_plots(full_pipeline: Any, model: Any, target_var: str, id_col: str, original_df: pd.DataFrame,
                                  x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, 
                                  train_predictions: pd.Series, test_predictions: pd.Series, 
                                  output_path: str ,output_label: str = "", col_labels: dict = {}, pd_y_label: str = "", 
                                  shap_plots: bool=False, shap_id_keys: list=[]) -> None:
    """
    Generates multiple plots for model evaluation including feature importance, actual vs. predicted, residuals, and partial dependence plots.

    Args:
        full_pipeline (Any): The full preprocessing and modeling pipeline.
        model (Any): The trained machine learning model.
        target_var (str): The target variable name.
        id_col (str): Name of the unique id variable for each row in the dataset.
        target_df (str): Original full feature and target df with id col.
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        train_predictions (pd.Series): Predictions for the training data.
        test_predictions (pd.Series): Predictions for the test data.
        output_path (str): A path to the directory where the output files will be saved.
        output_label (str, optional): Label to prepend to the output filename. Defaults to "".
        col_labels (dict, optional): Column labels for the features. Defaults to empty dict.
        pd_y_label (str, optional): Y-axis label for partial dependence plots. Defaults to "".
        shap_plots (bool, optional): Toggle to create shap plots for rows specified by shap_id_keys list.
        shap_id_keys (list, optional): List for rows to create shap plots for.

    Returns:
        None
    """
    feature_sign_dict, feature_diff_dict = create_feature_sign_dict(full_pipeline, x_train)
    create_permutation_feature_importance_plot(full_pipeline, x_test, y_test, target_var, feature_sign_dict, col_labels, output_label, output_path)
    create_actual_vs_predicted_scatter(y_train, y_test, train_predictions, test_predictions, target_var, output_label, output_path)
    create_residuals_plot(y_train, y_test, train_predictions, test_predictions, target_var, output_label, output_path)
    create_partial_dependence_plots(full_pipeline, x_train, target_var, output_label, output_path, col_labels, pd_y_label, feature_diff_dict)
    if shap_plots:
        create_shap_plots(id_col, original_df, x_test, x_train, full_pipeline, target_var, shap_id_keys, output_label, output_path) 
    return   

