# Object Oriented Pipeline for Machine Learning (Classification)







## Project Structure

- **`artifacts/`**: The **artifacts** folder is the central location where all project outputs, including logs and trained models, are stored.
    - `logs`: This folder stores all the log files generated during model training and evaluation. Each log file is named based on the model name and the date it was created. 
    - `models`: This folder contains all the trained models. Each model is saved with a filename that includes the model name and the date it was created.
 
- **`feature_analysis/`**: The **feature_analysis** folder contains scripts used to analyze the relationships and dependencies between features in the dataset
    - **`calculate_chisquare.py`**: This script calculates chi-square scores for categorical features in the dataset. The class **Chi2_Calculation** takes in input data, the processed target variable, and configuration options to compute chi-square values or p-values based on the settings. It helps in determining the dependency of features on the target variable.
    - **`calculate_correlation.py`**: This script computes the correlation matrix for numeric features in the dataset. The class **CorrelationCoefficient** analyzes the strength and direction of correlations between features, logging insights on strong positive correlations, highly correlated features, and the top N most correlated feature pairs.
 
- **`feature_engineering/`**: The **feature_engineering** folder contains scripts that implement various feature engineering techniques. 
    - **`feature_engineering_call_methods.py`**: This script provides a method to call feature engineering functions dynamically based on their names.
    - **`feature_engineering_combo.py`**: This script contains the actual implementations of feature engineering logic. 

- **`feature_importance/`**: The **feature_importance** folder contains scripts used to calculate the importance of features using model-based feature selection, sequential feature selection, and SHAP values.
    - **`calculate_model_fs.py`**: This script calculates feature importance based on different model types (linear or tree-based models). It computes and displays feature importance scores using coefficients or feature importance attributes, depending on the model type.
    - **`calculate_sfs.py`**: This script performs Sequential Feature Selection (SFS) to determine feature importance. The **CalculateSfsImportance** class allows for forward and backward sequential feature selection, logging selected features and their corresponding importance scores.
    - **`SHAP.py`**: This script calculates SHAP (SHapley Additive exPlanations) values to evaluate the contribution of each feature to the model's output. The **CalculateSHAPValues** class computes SHAP values and provides insights such as ranked feature importance, specific sample analysis and mean absolute SHAP values.
 
- **`launch/`**: The **launch** folder contains scripts used to initiate the options for machine learning models with specified configurations.
    - **`train.sh`**:  This bash script is used to launch the training process by running the `train.py` script with specified arguments for running the pipeline. {There is no necessary to change the Base_Options and Train_Options for changing the arguments. Instead, parse the arguments in this shell script and run it}.
 
