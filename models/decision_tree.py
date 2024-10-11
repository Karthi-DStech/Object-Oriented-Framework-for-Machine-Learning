from sklearn.tree import DecisionTreeClassifier
from models.base_model import BaseModel
from models.model_wrapper import ModelWrapper


class DecisionTreeModel(BaseModel):
    """
    This class provides methods for training, evaluating, and tuning
    a Decision Tree model.

    Parameters
    ----------
    This class inherits parameters from the BaseModel class.
    """

    def __init__(self, logger, opt):
        """
        Initialize the DecisionTreeModel object with the logger and options.

        Parameters
        ----------
        logger : Logger
            The logger instance for logging information.

        opt : Namespace
            Object containing the experiment options.
        """
        super().__init__(logger, opt)
        self._model_name = "DecisionTreeClassifier"
        self._model_type = "tree"
        self._trained_model = None

    def get_model(self, **kwargs):
        """
        This method returns a Decision Tree instance.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        DecisionTreeClassifier
            A Decision Tree instance.
        """
        return ModelWrapper(DecisionTreeClassifier, **kwargs)

    def train(self, X_train, y_train, **kwargs):
        """
        This method trains the decision tree model.

        Parameters
        ----------
        X_train : DataFrame
            The training features.

        y_train : DataFrame
            The training target.

        **kwargs : dict
            Arbitrary keyword arguments.
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
            model_class=DecisionTreeClassifier,
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
