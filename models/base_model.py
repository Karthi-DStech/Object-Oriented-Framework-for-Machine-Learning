import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import time
import optuna
from feature_importance.calculate_model_fs import calculate_feature_importance
from feature_importance.SHAP import CalculateSHAPValues


class BaseModel:
    """
    This class provides base model functionality for training,
    evaluation, tuning, and saving models.
    """

    def __init__(self, logger, opt):
        """
        Initialize the BaseModel object with the logger and options.

        Parameters
        ----------
        logger : Logger
            The logger instance for logging information.

        opt : Namespace
            Object containing the experiment options.
        """
        self.model = None
        self._model_name = "BaseModel"
        self.logger = logger
        self._opt = opt
        self._model_params = {}
        self._model_type = None
        self.training_evaluation = []
        self.tuning_evaluation = []

    @property
    def name(self) -> str:
        """
        Returns the name of the network
        """
        return self._model_name

    @property
    def model_type(self) -> str:
        """
        Returns the model type (e.g., 'tree', 'linear').
        """
        return self._model_type

    def train(self, model, X_train, y_train):
        """
        This method trains the model using the given training data.

        Parameters
        ----------
        model : object
            The model to train.

        X_train : DataFrame
            The training features.

        y_train : DataFrame
            The training target.

        Raises
        ------
        ValueError
            If the model or training data is None.
            If an error occurs during training.
        """
        if model is None:
            raise ValueError("Model cannot be None.")
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None.")

        try:
            start_time = time.time()
            self.logger.update_log("model_training", "model_name", self._model_name)
            self.logger.update_log(
                "model_training",
                "training_started",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            )

            model.fit(X_train, y_train)
            self.model = model

            end_time = time.time()
            duration = end_time - start_time
            self.logger.update_log(
                "model_training",
                "training_completed",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            )
            self.logger.update_log(
                "model_training", "training_duration", f"{duration:.2f} seconds"
            )

            print("Model training completed.")
            self.logger.update_log("model_training", "status", "completed")

        except Exception as e:
            raise ValueError(f"An error occurred during training: {e}")

    def evaluate(self, X_test, y_test, section="model_evaluation"):
        """
        This method evaluates the model using the given test data.

        Parameters
        ----------
        X_test : DataFrame
            The testing features.

        y_test : DataFrame
            The testing target.

        section : str
            The section to log the evaluation results.

        Raises
        ------
        ValueError
            If the model is None.
            If the test data is None.
            If an error occurs during evaluation.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if X_test is None or y_test is None:
            raise ValueError("Test data cannot be None.")

        try:
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test, predictions, average="macro", zero_division=0
            )
            recall = recall_score(y_test, predictions, average="macro", zero_division=0)
            f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

            # To handle zero-division issues in classification_report.
            # Remove the parameter if necessary.

            report = classification_report(y_test, predictions, zero_division=0)

            cm = confusion_matrix(y_test, predictions)

            if cm.shape == (2, 2):
                # For binary classification
                tn, fp, fn, tp = cm.ravel()
            else:
                # For multiclass classification
                tn = fp = fn = tp = None

            print("\nModel evaluation completed.")
            print(f"\nAccuracy: {accuracy} ")
            print(f"\n Classification Report:\n{report}")

            self.logger.update_log(section, "accuracy", accuracy)
            self.logger.update_log(section, "classification_report", report)

            # Logs if the confusion matrix is binary
            if cm.shape == (2, 2):
                total_predictions = len(predictions)
                correct_predictions = tn + tp
                wrong_predictions = fp + fn

                self.logger.update_log(section, "accuracy", total_predictions)
                self.logger.update_log(section, "accuracy", correct_predictions)
                self.logger.update_log(section, "accuracy", wrong_predictions)

            evaluation = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            if section == "model_evaluation":
                self.training_evaluation.append(evaluation)

            elif section == "model_tuning_evaluation":
                self.tuning_evaluation.append(evaluation)

            return evaluation

        except Exception as e:
            raise ValueError(f"An error occurred during evaluation: {e}")

    def shap_calculation(self, X_test, model_type):

        if self._opt.calculate_SHAP:

            try:
                actual_model = (
                    self.model.model if hasattr(self.model, "model") else self.model
                )

                shap_calculator = CalculateSHAPValues(
                    model=actual_model,
                    X_train=None,
                    X_test=X_test,
                    logger=self.logger,
                    opt=self._opt,
                )

                model_type = self.model_type
                if model_type is None:
                    raise ValueError("Model type is not defined.")

                # Calculate and save SHAP plots
                shap_calculator.save_shap_plots(model_type)

            except Exception as e:
                raise ValueError(
                    f"An error occurred during SHAP value calculation: {e}"
                )

        else:
            self.logger.update_log(
                "SHAP", "Status", "Skipped calculating SHAP values..."
            )
            print("\nSkipping SHAP calculations...\n")

    def model_tuning(
        self,
        model_class,
        get_params_func,
        X_train,
        y_train,
        X_test,
        y_test,
        n_trials,
    ):
        """
        This method tunes the model using Optuna for hyperparameter tuning.

        Parameters
        ----------
        model_class : object
            The model class to tune.

        get_params_func : callable
            The function to get the hyperparameters for tuning.

        X_train : DataFrame
            The training features.

        y_train : DataFrame
            The training target.

        X_test : DataFrame
            The testing features.

        y_test : DataFrame
            The testing target.

        n_trials : int
            The number of trials for tuning using Optuna.

        Raises
        ------
        ValueError
            If the model class is None.
            If get_params_func is not callable.
            If training or testing data is None.
        """
        if model_class is None:
            raise ValueError("Model class cannot be None.")
        if not callable(get_params_func):
            raise ValueError("get_params_func must be a callable function.")
        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise ValueError("Training and testing data cannot be None.")

        start_time = time.time()
        self.logger.update_log(
            "model_tuning",
            "tuning_started",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        )
        self.logger.update_log(
            "model_tuning", "method", "Optuna for hyperparameter tuning"
        )
        self.logger.update_log("model_tuning", "n_trials", n_trials)

        def objective(trial):
            """
            This function defines the objective for Optuna to optimize.

            Parameters
            ----------
            trial : Trial
                The Optuna trial object.

            Returns
            -------
            float
                The accuracy score for the model.

            Raises
            ------
            ValueError
                If an error occurs during model training or prediction.
            """
            # Get hyperparameters from external function
            params = get_params_func(trial)

            # Create the model with the current hyperparameters
            model = model_class(**params)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            return accuracy

        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            print("\nBest hyperparameters:", study.best_params)
            self.logger.update_log(
                "model_tuning", "best_hyperparameters", study.best_params
            )

            self.model = model_class(**study.best_params)
            self.model.fit(X_train, y_train)
            end_time = time.time()
            duration = end_time - start_time
            self.logger.update_log(
                "model_tuning",
                "tuning_completed",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            )
            self.logger.update_log(
                "model_tuning", "tuning_duration", f"{duration:.2f} seconds"
            )
            print("\nModel tuning completed.\n")
            self.logger.update_log("model_tuning", "status", "completed")

            # Evaluate the model after tuning
            self.evaluate(X_test, y_test, section="model_tuning_evaluation")

        except Exception as e:
            raise ValueError(f"An error occurred during model tuning: {e}")

    def calculate_feature_importance(self, X, model_type, section="feature_importance"):
        """
        This method calculates the feature importance for the trained model.

        Parameters
        ----------
        X : DataFrame
            The features.

        model_type : str
            The type of the model ('linear' or 'tree').

        section : str
            The section to log the feature importance results.

        Raises
        ------
        ValueError
            If the model is None.
            If the feature importance is not enabled.
        """

        if self._opt.feature_importance:

            if model_type == "linear" or model_type == "tree":

                self.logger.update_log(
                    section, "feature importance calculation status", "started"
                )
                print("\nCalculating feature importance...")

                technique = (
                    "Correlation Coefficients"
                    if model_type == "linear"
                    else "Plot Tree Importance"
                )

                self.logger.update_log(
                    section, "feature importance technique", technique
                )

                feature_importance = calculate_feature_importance(
                    self.model, X, model_type, top_n=self._opt.top_n
                )
                self.logger.update_log(
                    section, "feature importance calculation", "completed"
                )
                self.logger.update_log(
                    section, "number of features selected", self._opt.top_n
                )
                self.logger.update_log(
                    section, "feature importance", f"\n{feature_importance}\n"
                )

            else:
                print("Feature importance is not supported for other types of models.")

        else:
            self.logger.update_log(
                "model_training", "feature importance status", "not enabled"
            )
            self.logger.update_log(
                "model_tuning", "feature importance status", "not enabled"
            )
            print("Feature importance is not enabled.")

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self._model_params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
        """
        for key, value in params.items():
            self._model_params[key] = value
        return self

    def analysis_and_save_model(self, model_path=None):
        """
        This method saves the trained model to the given path.

        Parameters
        ----------

        model_path : str
            The path to save the model.

        Raises
        ------
        ValueError
            If the model is None.
            If the model path is None.
            If an error occurs during saving the model.
        """
        if self.model is None:
            raise ValueError("There is no trained model to save.")
        if model_path is None:
            raise ValueError("Model path cannot be None.")

        try:

            def format_to_percentage(value):
                return f"{value * 100:.2f}%"

            if len(self.training_evaluation) > 0:
                training_eval = self.training_evaluation[-1]
            else:
                training_eval = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

            if len(self.tuning_evaluation) > 0:
                tuning_eval = self.tuning_evaluation[-1]
            else:
                tuning_eval = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

            # Create a dictionary for before evaluation log (as percentages)
            before_evaluation_log = {
                "accuracy": format_to_percentage(training_eval["accuracy"]),
                "precision": format_to_percentage(training_eval["precision"]),
                "recall": format_to_percentage(training_eval["recall"]),
                "f1": format_to_percentage(training_eval["f1"]),
            }

            # Create a dictionary for after evaluation log (as percentages)
            after_evaluation_log = {
                "accuracy": format_to_percentage(tuning_eval["accuracy"]),
                "precision": format_to_percentage(tuning_eval["precision"]),
                "recall": format_to_percentage(tuning_eval["recall"]),
                "f1": format_to_percentage(tuning_eval["f1"]),
            }

            # Log before evaluation data
            self.logger.update_log(
                "Model Performance Analysis & Saving",
                "Evaluation_before_tuning",
                before_evaluation_log,
            )

            # Log after evaluation data
            self.logger.update_log(
                "Model Performance Analysis & Saving",
                "Evaluation_after_tuning",
                after_evaluation_log,
            )

            # Save the model
            pd.to_pickle(self.model, model_path)
            print(f"\nModel saved to {model_path} successfully. \n")
            self.logger.update_log(
                "Model Performance Analysis & Saving", "model_path", model_path
            )
        except Exception as e:
            raise ValueError(f"An error occurred while saving the model: {e}")
