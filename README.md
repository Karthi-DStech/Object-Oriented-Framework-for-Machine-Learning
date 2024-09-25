# Object Oriented Pipeline for Machine Learning (Classification)

This project is a comprehensive, object-oriented machine-learning pipeline designed to streamline the entire process of data analysis, preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

## See Artifacts -> Logs -> (Any logged Model) for better understanding. 

The goal is to provide a flexible, scalable, and reusable framework that fits a wide range of ML classification workloads. With 97% of the coding already handled, this pipeline can be fully controlled via command-line arguments, making it highly efficient and easy to use.

This OOP-based pipeline simplifies the work of data scientists and machine learning engineers, enabling them to train, evaluate, and track models efficiently, while also facilitating seamless collaboration and model sharing within teams.

<img width="908" alt="Screenshot 2024-09-25 at 05 15 57" src="https://github.com/user-attachments/assets/2869d99a-7100-4f76-af14-af11f0f58956">


## Key Features

**"An all-in-one machine learning pipeline that allows you to control the entire machine learning process using Python command-line arguments, requiring minimal coding (97% coding-free) and no need for new implementations."**

- **Automated Data Preprocessing**: Includes handling missing values, feature encoding, and customizable feature engineering.
- **Model Agnostic**: Supports a wide range of classification algorithms such as Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forests, and boosting models like XGBoost, LightGBM, and CatBoost.
- **Hyperparameter Tuning**: Uses Optuna for hyperparameter optimization, allowing models to achieve their best performance.
- **Feature Importance & Analysis**: Includes methods for calculating chi-square scores, correlation analysis, sequential feature selection, model-based feature selection and SHAP Values. 
- **Logging and Model Saving**: Provides comprehensive logging of model training and evaluation, with the ability to save models and logs for future reference.

## Technologies Used

- **Python**: Core language for the project.
- **scikit-learn**: For implementing machine learning models and data transformations.
- **Optuna**: For hyperparameter tuning.
- **pandas, NumPy**: For data manipulation and analysis.
- **SHAP**: For calculating feature importance.
- **Logging**: Custom logging system for tracking model performance and data processing steps.


## Project Structure

- **`artifacts/`**:
The **artifacts** folder is the central location where all project outputs, including logs and trained models, are stored.

    - `logs`:
      This folder stores all the log files generated during model training and evaluation. Each log file is named based on the model name and the date it was created.
      
    - `models`:
      This folder contains all the trained models. Each model is saved with a filename that includes the model name and the date it was created.
 
- **`feature_analysis/`**:
The **feature_analysis** folder contains scripts used to analyze the relationships and dependencies between features in the dataset.

    - **`calculate_chisquare.py`**:
      This script calculates chi-square scores for categorical features in the dataset. The class **Chi2_Calculation** takes in input data, the processed target variable, and configuration options to compute chi-square values or p-values based on the settings. It helps in determining the dependency of features on the target variable.
      
    - **`calculate_correlation.py`**:
      This script computes the correlation matrix for numeric features in the dataset. The class **CorrelationCoefficient** analyzes the strength and direction of correlations between features, logging insights on strong positive correlations, highly correlated features, and the top N most correlated feature pairs.
 
- **`feature_engineering/`**:
The **feature_engineering** folder contains scripts that implement various feature engineering techniques.

    - **`feature_engineering_call_methods.py`**:
      This script provides a method to call feature engineering functions dynamically based on their names.
      
    - **`feature_engineering_combo.py`**:
      This script contains the actual implementations of feature engineering logic. 

- **`feature_importance/`**:
The **feature_importance** folder contains scripts used to calculate the importance of features using model-based feature selection, sequential feature selection, and SHAP values.

    - **`calculate_model_fs.py`**:
      This script calculates feature importance based on different model types (linear or tree-based models). It computes and displays feature importance scores using coefficients or feature importance attributes, depending on the model type.
    - **`calculate_sfs.py`**:
      This script performs Sequential Feature Selection (SFS) to determine feature importance. The **CalculateSfsImportance** class allows for forward and backward sequential feature selection, logging selected features and their corresponding importance scores.
      
    - **`SHAP.py`**:
      This script calculates SHAP (SHapley Additive exPlanations) values to evaluate the contribution of each feature to the model's output. The **CalculateSHAPValues** class computes SHAP values and provides insights such as ranked feature importance, specific sample analysis and mean absolute SHAP values.
 
- **`launch/`**:
The **launch** folder contains scripts used to initiate the options for machine learning models with specified configurations.

    - **`train.sh`**:
      This bash script is used to launch the training process by running the `train.py` script with specified arguments for running the pipeline. **{There is no necessary to change the Base_Options and Train_Options for changing the arguments. Instead, parse the arguments in this shell script and run it}**.
 

- **`models/`**:
The **models** folder contains the implementations of various machine learning models used in the project. Each script corresponds to a different algorithm, providing a clear structure for training and evaluation.
    
    - **`adaptive_boost.py`**:  
      This script implements the Adaptive Boosting (AdaBoost) algorithm, which combines weak learners (e.g., decision trees) to create a strong classifier by iteratively adjusting the weights of incorrectly classified instances.
    
    - **`base_model.py`**:  
      This script provides a base class for machine learning models, defining common methods for model training, evaluation, and hyperparameter tuning. It serves as a foundation for other model implementations and all the models will inherit the BaseModel class for using the functionality. 
    
    - **`cat_boost.py`**:  
      Implements the CatBoost algorithm, which is a gradient-boosting decision tree algorithm tailored to handle categorical features more efficiently without requiring explicit pre-processing.
    
    - **`decision_tree.py`**:  
      This script implements the Decision Tree Classifier, a non-parametric supervised learning method used for classification and regression. It builds a tree-like model of decisions based on feature values.
    
    - **`gradient_boost.py`**:  
      Implements the Gradient Boosting algorithm, which builds an ensemble of weak prediction models (like decision trees) by optimizing the loss function.
    
    - **`knn.py`**:  
      This script implements the K-Nearest Neighbors (KNN) algorithm, a simple, instance-based learning method used for classification and regression by finding the nearest neighbors of a query point.
    
    - **`light_gbm.py`**:  
      Implements the LightGBM (Light Gradient Boosting Machine) algorithm, which is designed for fast training speed and low memory usage, especially suited for large datasets.
    
    - **`logistic_regression.py`**:  
      This script implements the Logistic Regression algorithm, a statistical model used for binary classification that predicts the probability of a categorical outcome.
    
    - **`model_wrapper.py`**:  
      This script defines a `ModelWrapper` class that serves as a wrapper for machine learning models, allowing them to be seamlessly integrated into scikit-learn pipelines. It extends the functionality of the `BaseEstimator` and `ClassifierMixin` classes from scikit-learn, providing a consistent interface for fitting, predicting, and accessing model parameters.
    
    - **`random_forest.py`**:  
      Implements the Random Forest algorithm, an ensemble method that builds multiple decision trees and merges them to obtain a more accurate and stable prediction.
    
    - **`svm.py`**:  
      This script implements the Support Vector Machine (SVM) algorithm, which is used for both classification and regression tasks by finding a hyperplane that best divides a dataset into classes.
    
    - **`xgboost.py`**:  
      Implements the XGBoost algorithm, a scalable and efficient implementation of gradient boosting that is optimized for speed and performance in large datasets.

- **`options/`**:
  The **options** folder contains configuration scripts that define the various options and parameters used for training models, feature engineering, and other processes.
  
    - **`base_options.py`**:
      This script defines the base options used across all experiments. It uses Python's **argparse** library to handle command-line arguments.
      
    - **`train_options.py`**:
      This script extends the base options and adds specific configuration settings for training machine learning models through the pipeline.

- **`parameters/`**:
  The **parameters** folder contains scripts that define hyperparameters for various machine learning models. These hyperparameters are used during model training to optimize the model performance through trial-based optimization frameworks (Optuna).

    - **`hyperparameters.py`**:
      This script defines the **Hyperparameters** class, which provides hyperparameter configurations for different classification models. Each method in the class corresponds to a specific model and returns a dictionary of hyperparameters.

- **`process/`**:
  The **process** folder contains scripts related to data preprocessing, data splitting, and transformations before model training. 

    - **`preprocessing.py`**:
      This script defines the **DataProcessor** class, which provides various methods for loading, overall preprocessing, and saving the dataset for further implementation in the ML pipeline.
      
    - **`train_test_split.py`**:
       This script defines the **TrainTestProcessor** class, which handles the splitting of the processed data into training and testing sets and implements unit testing for the overall preprocessed data.

- **`utils/`**:
  The **utils** folder contains utility scripts that provide general-purpose functionalities, such as logging and saving models during the model training and evaluation process.
  
    - **`logs.py`**:
      This script defines the **Logger** class, which is used to log information throughout the data processing, model training, and model evaluation stages.
      
    - **`save_utils.py`**:
      This script provides a utility function, **save_model_and_logs()**, to save the trained model and associated logs to the artifacts folder.

- **`call_methods.py/`**:
This script defines functions to dynamically create models and retrieve hyperparameters based on the model name. Key functionalities include:

    - `make_network(network_name, *args, **kwargs)`: This function imports and returns the appropriate model class based on the specified `network_name`.
      
    - `make_params(param_name, *args, **kwargs)`: This function imports and returns the hyperparameters for the specified `param_name`, facilitating easy tuning for models.
      
- **`train.py/`**:
 The **train.py** script orchestrates the entire machine learning pipeline, from data preprocessing to model training, tuning, evaluation, and saving the results. 


