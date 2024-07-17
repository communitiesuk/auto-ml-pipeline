# Predictive modelling pipeline guidance

We have created a multi-stage modelling pipeline which can be applied to any regression or classificaiton machine learning problem. The following guidance explains how to run the pipeline using your own data.

The pipeline is implemented using the scikit-learn library.

<img src="https://github.com/communitiesuk/SDU_real_time_indicators/blob/main/src/storytelling/devo_bua_regression_analysis/pipeline_diagram.PNG" width="700">

## Contents 

  * [First time setup](#first-time-setup)
    + [Installing the virtual environment](#installing-the-virtual-environment)
  * [Running pipeline with new data](#running-pipeline-with-new-data)
    + [Read in your data](#read-in-your-data)
    + [Specify columns to drop](#specify-columns-to-drop)
    + [Model configuration](#model-configuration)
    + [Run the pipeline](#run-the-pipeline)
    + [Interpretation of results](#interpretation-of-results)

## First time setup
### Installing the virtual environment

In the terminal run: 

```
conda env create --file environment.yml
```

And then activate the virtual environment by running:

```
conda activate rti
```

## Running pipeline with new data

### Read in your data

The pipeline uses tabular input data where each row represents a single observation (e.g. a local authority), while each column represents a specific attribute or variable related to the observation (e.g. the healthly life expectancy in the local authority).

The first step in the pipeline is to read in your data to a Pandas dataframe (including both feature and target columns) and remove any duplicate rows or rows containing NAs.

```python
data_path = "path/to/your/data.csv"

main_df = pd.read_csv(data_path)
main_df = main_df.drop_duplicates()
main_df = main_df.dropna()
```

### Specify columns to drop

The next step is to remove any columns that you want to remove from your modelling pipeline. These can be identifier columns that will not help the model learning process (e.g local authority codes). Or they could be variables that will not provide helpful conclusions about your research question (e.g. using GVA in 2020 would not provide much insight when predicting the underlying drivers of GVA in 2021).

The target variable list should contain all of the variables that you want to use as target variables in your modelling loop. 

```python
geography_variables = ['geo_col1', 'geo_col2']
drop_variables = ['unwanted_col1', 'unwanted_col2', 'unwanted_col3']
target_var_list = ["target_variable_1", "target_variable_2"]
```

### Model configuration

Next you need to define the list of models and the hyperparameter search space to use in the pipeline.

The example below shows some common models and some example hyperparameters from the scikit learn library.

```python
model_param_dict = { 
    LinearRegression(): {
        'feature_filter__filter_features':  [True],
        'feature_filter__feature_filter_list': [select_features_list]
        },
    Lasso(): {
        'model__fit_intercept': [True, False],
        'model__alpha': [0.001, 0.01, 0.1, 0.5, 1],
        'feature_filter__filter_features': [True],
        'feature_filter__feature_filter_list': [select_features_list]
        },
    DecisionTreeRegressor(): {
            'model__max_depth': [None, 2, 5, 10, 25, 50],
            'model__max_features': ['sqrt','auto', None],
            'model__min_samples_leaf': [1, 2, 4, 6, 10],
            'model__min_samples_split': [2, 5, 10],
            'feature_filter__filter_features': [True],
            'feature_filter__feature_filter_list': [select_features_list]
        }
}
```

### Run the pipeline

The following code shows an example of running the pipeline.

First the data is preprocessed by dropping specified columns and performing dummy encoding. The data is then split into a traning and test set.

The main model pipeline function is called which will train and evaluate the models for each of the model types specified in the model_param_dict dictionary. The output_label variable is used to specifiy a label to add to each of the output files for the pipeline run.

This pipeline loop is then repeated for each variable in the target_var_list.

```python
for target_var in target_var_list:
    # Preprocess the data
    cols_to_drop = list(set([target_var] + drop_variables + geography_variables))
    features = preprocess_features(df=main_df, cols_to_drop=cols_to_drop)
    target_df = preprocess_target(df=main_df, target_col=target_var)

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, target_df, test_size=0.20, random_state=36)

    # Run the model pipeline
    model_grid_cv_pipeline(model_param_dict, target_var, target_df, x_train, y_train, x_test, y_test, 
    output_label="output_label", col_label_map=col_labels)
```

### Interpretation of results

After running the pipeline, the model evaluation metrics and plots will be saved to the specified output file path. These results will include:

The best hyperparameters for each model.
Training and test RMSE scores (Root Mean Squared Error) and standard deviations (from CV evaluation).
Training and test R^2 scores and standard deviations.
Feature importance plots (for tree-based models).
Other diagnostic plots to evaluate model performance.

Use these results to understand the performance of each model and make informed decisions about which model best fits your data and use case.

### Optional steps: select features and shortened labels for evaluation plots

You can create a list of features to filter the input data as a hyperparameter step which can be used to see if a smaller subset of features improves model performance. 

The col_labels dictionary can be used to create cleaner or shorter variable names for use in the evaluation plots.

```python
select_features_list = ["feature1", "feature2", "feature3"]

col_labels = {
    "feature1": "Short label 1",
    "feature2": "Short label 2",
    "feature3": "Short label 3"
}
```
