from sklearn.ensemble import RandomForestClassifier
from models.base_model import BaseModel
from models.model_wrapper import ModelWrapper


class RandomForestModel(BaseModel):
    """
    This class provides methods for training, evaluating, and tuning
    a Random Forest model.

    Parameters
    ----------
    This class inherits parameters from the BaseModel class.
    """

    def __init__(self, logger, opt):
        """
        Initialize the RandomForestModel object with the logger and options.

        Parameters
        ----------
        logger : Logger
            The logger instance for logging information.

        opt : Namespace
            Object containing the experiment options.
        """
        super().__init__(logger, opt)
        self._model_name = "RandomForestClassifier"
        self._model_type = "tree"
        self._trained_model = None

    def get_model(self, **kwargs):
        """
        This method returns a Random Forest instance.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        return ModelWrapper(RandomForestClassifier, **kwargs)

    def train(self, X_train, y_train, **kwargs):
        """
        This method trains the model.

        Parameters
        ----------
        X_train : DataFrame
            The training features.

        y_train : DataFrame
            The training target.
        """
        model = self.get_model(**kwargs)
        super().train(model, X_train, y_train)

        self._trained_model = model

        self.calculate_feature_importance(
            X_train, model_type="tree", section="model_training"
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
        """
        super().model_tuning(
            model_class=RandomForestClassifier,
            get_params_func=get_params_func,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_trials=n_trials,
        )

        self._trained_model = self.model

        self.calculate_feature_importance(
            X_train, model_type="tree", section="model_tuning"
        )
