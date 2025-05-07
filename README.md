# Automatic machine learning pipeline

We have created a multi-stage modelling pipeline which can be applied to any regression, or binary classification machine learning problem. The following guidance explains how to run the pipeline using your own data.

The pipeline is implemented using the scikit-learn library.

<img src="https://github.com/communitiesuk/auto-ml-pipeline/blob/main/src/utils/modelling_pipeline_diagram.PNG" width="700">

For any suggestions or feedback, please contact sean.ogara@communities.gov.uk, or raise a GitHub issue.

## Contents 

  * [Contents](#contents)
  * [First time setup](#first-time-setup)
    + [Installing the virtual environment](#installing-the-virtual-environment)
  * [Running pipeline with new data](#running-pipeline-with-new-data)
    + [Read in your data](#read-in-your-data)
    + [Specify columns to drop](#specify-columns-to-drop)
    + [Model configuration](#model-configuration)
    + [Run the pipeline](#run-the-pipeline)
    + [Interpretation of results](#interpretation-of-results)
    + [Automatic logging with MLflow](#automatic-logging-with-mlflow)
  * [Optional steps](#optional-steps)
    + [Select specific features](#select-specific-features)
    + [Provide custom pre-processing steps](#provide-custom-pre-processing-steps)
    + [Create tidy variable labels](#create-tidy-variable-labels)
## First time setup
### Installing the virtual environment

In the terminal run: 

```
conda env create --file environment.yaml
```

And then activate the virtual environment by running:

```
conda activate auto-ml
```

## Running pipeline with new data

### Read in your data

The pipeline uses tabular input data where each row represents a single observation (e.g. a local authority), while each column represents a specific attribute or variable related to the observation (e.g. the healthly life expectancy in the local authority).

The first step in the pipeline is to read in your data to a Pandas dataframe (including both feature and target columns) and remove any duplicate rows or rows containing NAs if necessary.

Alternatively you can use scikit-learn imputers to replace missing values in your data. See the [Provide custom pre-processing steps](#provide-custom-pre-processing-steps) section for instructions of how to incorporate imputers into your pipeline.

```python
data_path = "path/to/your/data.csv"

regression_data = pd.read_csv(data_path)
regression_data = regression_data.drop_duplicates()
regression_data = regression_data.dropna()
```

### Specify columns to drop

The next step is to remove any columns that you want to remove from your modelling pipeline. These can be identifier columns that will not help the model learning process (e.g local authority codes). Or they could be variables that will not provide helpful conclusions about your research question (e.g. using GVA in 2020 would not provide much insight when predicting the underlying drivers of GVA in 2021).

The target variable list should contain all of the variables that you want to use as target/dependant variables in your modelling loop. 

```python
drop_variables = ['unwanted_col1', 'unwanted_col2', 'unwanted_col3']
target_var_list = ["target_variable_1", "target_variable_2"]
```

### Model configuration

Next you need to define the list of models and the hyperparameter search space to use in the pipeline.

The example below shows some common models and some example hyperparameters from the scikit-learn library.

```python
model_param_dict = {
        LinearRegression(): {
            "feature_filter__filter_features":  [True],
            "feature_filter__feature_filter_list": [select_features_list]
        },
        Lasso(): {
            "model__fit_intercept": [True, False],
            "model__alpha": loguniform(1e-4, 1), 
            "feature_filter__filter_features":  [True],
            "feature_filter__feature_filter_list": [select_features_list]
        },
        RandomForestRegressor(): {
            "model__max_depth": randint(1, 100),
            "model__max_features": [1, 0.5, "sqrt"],
            "model__min_samples_leaf": randint(1, 20),
            "model__min_samples_split": randint(2, 20),
            "model__n_estimators": randint(5, 300),
            "feature_filter__filter_features":  [True],
            "feature_filter__feature_filter_list": [select_features_list]
        },
        XGBRegressor(): {
            "model__max_depth": randint(2, 20),
            "model__learning_rate": loguniform(1e-4, 0.1),
            "model__subsample": uniform(0.3, 0.7),
            "model__n_estimators": int_loguniform(5, 5000),
            "feature_filter__filter_features":  [True],
            "feature_filter__feature_filter_list": [select_features_list]
        },
    }
```

You can add new models by adding them to the model_param_dict object with the corresponding hyperparameters that you'd like to optimise for. Please see the [scikit-learn documentation](https://scikit-learn.org/stable/api/index.html) for more example model architectures that you can use in the pipeline.

### Run the pipeline

The following code shows an example of running the pipeline.

First the feature data is preprocessed by dropping the specified drop columns and target variable and performing one-hot dummy encoding. This step can be omitted if one-hot encoding is not desirable. The target data is separate from the rest of the data for use in the pipeline.

The main model pipeline function is called which will train and evaluate the models for each of the model types specified in the model_param_dict dictionary. The output_label variable is used to specify a label to add to each of the output files for the pipeline run. Within the main pipeline code, the data is split into a training and test set (80%/20%).

This pipeline loop is then repeated for each variable in the target_var_list.

```python
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
        id_col="msoa11cd",
        original_df=regression_data,
        scoring_metrics=["r2", "neg_root_mean_squared_error"],
        output_path="your/output/path",
        output_label="msoa_demo",
        col_label_map=col_labels,
        user_evaluation_model=user_model,
        shap_id_keys=["E02000266", "E02000503"],
        custom_pre_processing_steps=pre_processing_pipeline_steps,
    )
```
The final parameters show below are optional. Their usage is described in [Optional steps](#optional-steps). If you do not want to use them simply remove them from the function call and run the pipeline in the same way as above.
```python
        col_label_map=col_labels,
        user_evaluation_model=user_model,
        shap_id_keys=["E02000266", "E02000503"],
        custom_pre_processing_steps=pre_processing_pipeline_steps,
```
### Interpretation of results

After running the pipeline, the model evaluation metrics and plots will be saved to the specified output file path. These results will include:

The best hyperparameters for each model.
Training and test RMSE scores (Root Mean Squared Error) and standard deviations (from CV evaluation).
Training and test R^2 scores and standard deviations.
Feature importance plots (for tree-based models).
Other diagnostic plots to evaluate model performance.

Use these results to understand the performance of each model and make informed decisions about which model best fits your data and use case.

### Automatic logging with MLflow

The pipeline incorporates MLflow for model performance logging. To access the user interface after running the pipeline, open a terminal, CD to the project directory, and type the following command in the terminal:

```python
mlflow ui
```

The model runs will be sorted into experiments based on the specified output_label parameter.

See the [MLflow docs](https://mlflow.org/docs/latest/index.html) for more details.

## Optional steps

### Select specific features

You can create a list of features to filter the input data as a hyperparameter step which can be used to see if a smaller subset of features improves model performance. 

```python
select_features_list = ["feature1", "feature2", "feature3"]

# add hyperparams to model_param_dict
model_param_dict = {
    LinearRegression(): {
        "feature_filter__filter_features": True,
        "feature_filter__feature_filter_list": select_features_list
    }
}
```
### Provide custom pre-processing steps

You can add custom pre-processing steps to the pipeline using the pre_processing_pipeline_steps list variable. This works with scalers and imputers from scikit-learn. 

To use custom steps, pass the pre_processing_pipeline_steps to the model_pipeline function using the custom_pre_processing_steps parameter.

If the custom_pre_processing_steps parameter is removed the default pre-processing pipeline will be used: FilterFeatures(), StandardScaler()

```python

# custom pre-processing pipeline - remove to use default pre-processing pipeline: FilterFeatures(), StandardScaler()
pre_processing_pipeline_steps = [
    ("feature_filter", FilterFeatures()),
    ("knn_imputer", KNNImputer()),
    ("scaler", MinMaxScaler()),
]
```
### Custom scoring metrics
The scoring_metrics parameter can be used to specify the criteria for which the model training should by optimised and evaluated against. A list of one or more metrics can be provided and the first metric in the list will be used to optimise the hyperparameter search algorithm. Any other metrics in the list will be evaluated during model training but will not be used for optimisation. 

The default scoring metrics for a regression pipeline run are:

```
scoring_metrics = ["r2", "neg_root_mean_squared_error"]
```

The default scoring metrics for a classification pipeline run are: 

```
scoring_metrics = ["f1", "accuracy"]
```

### Create tidy variable labels

The col_labels dictionary can be used to create cleaner or shorter variable names for use in the evaluation plots.

```python
col_labels = {
    "long feature label 1": "Short label 1",
    "long feature label 2": "Short label 2",
    "long feature label 3": "Short label 3"
}
```

### Specify model type to evaluate 

optional - user specified model for evaluation plots. e.g. user_model = "Lasso"

To evaluate a specific model type, pass the user_model to the model_pipeline() function using the user_evaluation_model parameter.

If user_evaluation_model is removed or user_model is left blank, the best performing model will be used for the evaluation plots

```python
user_model = "Lasso"
```

### Shap plots - local feature importance

The pipeline can also create shap force plots for specified observations in the data. These plots allow you to understand the contributions of different features to the model's predictions.

To create shap plots set shap_id_keys to a list containing the IDs of the observations you would like to create force plots for. The keys should correspond to values found in the id_col parameter that is passed to the main ml_pipeline function.

See the [shap docs](https://shap.readthedocs.io/en/latest/generated/shap.plots.force.html) for more details.

```python
shap_id_keys=["E02000266", "E02000503"]
 ```
