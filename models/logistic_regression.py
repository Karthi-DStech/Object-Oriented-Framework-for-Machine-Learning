from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel
from models.model_wrapper import ModelWrapper


class LogisticRegressionModel(BaseModel):
    """
    This class provides methods for training, evaluating, and tuning
    a Logistic Regression model.

    Parameters
    ----------
    This class inherits parameters from the BaseModel class.
    """

    def __init__(self, logger, opt):
        super().__init__(logger, opt)
        self._model_name = "LogisticRegression"
        self._model_type = "linear"
        self._trained_model = None

    def get_model(self, **kwargs):
        """
        This method returns a Logistic Regression instance.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments.
        """

        return ModelWrapper(LogisticRegression, **kwargs)

    def train(self, X_train, y_train, **kwargs):
        """
        This method trains the model using the given training data.

        Parameters
        ----------
        X_train : DataFrame
            The training features.

        y_train : DataFrame
            The training target.

        **kwargs : dict
            Arbitrary keyword arguments.

        execute
        -------
        model : object
            The trained model.

        if feature_importance : DataFrame
            The feature importance of the model.

        else : str
            Feature importance will be skipped

        raise
        -----
        ValueError
            If the model is not trained.
        """
        model = self.get_model(**kwargs)
        super().train(model, X_train, y_train)
        self._trained_model = model

        self.calculate_feature_importance(
            X_train, model_type="linear", section="model_training"
        )

    def model_tuning(self, get_params_func, X_train, y_train, X_test, y_test, n_trials):
        """
        This method tunes the model using Optuna.

        Parameters
        ----------
        get_params_func : function
            The function to get the model parameters
            using Dynamic Imports Technique.

        X_train : DataFrame
            The training features.

        y_train : DataFrame
            The training target.

        X_test : DataFrame
            The testing features.

        y_test : DataFrame
            The testing target.

        n_trials : int
            The number of trials for tuning the model.

        execute
        -------
        model : object
            The tuned model.

        if feature_importance : DataFrame
            The feature importance of the model.

        else : str
            Feature importance will be skipped

        raise
        -----
        ValueError
            If the model is not tuned.
        """
        super().model_tuning(
            model_class=LogisticRegression,
            get_params_func=get_params_func,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_trials=n_trials,
        )
        self._trained_model = self.model

        self.calculate_feature_importance(
            X_train, model_type="linear", section="model_tuning"
        )
