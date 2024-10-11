from sklearn.neighbors import KNeighborsClassifier
from models.base_model import BaseModel


class KnnModel(BaseModel):
    """
    This class provides methods for training, evaluating, and tuning
    a KNeighbors model.

    Parameters
    ----------
    This class inherits parameters from the BaseModel class.
    """

    def __init__(self, logger, opt):
        """
        This method initializes the KnnModel object with the logger and options.

        Parameters
        ----------
        logger : Logger
            The logger instance for logging information.

        opt : Namespace
            Object containing the experiment options.
        """
        super().__init__(logger, opt)
        self._model_name = "KNeighborsClassifier"
        self._model_type = "neighbors"

    def get_model(self, **kwargs):
        """
        This method returns a KNeighbors instance.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        KNeighborsClassifier
            A KNeighbors instance.
        """
        return KNeighborsClassifier(**kwargs)

    def train(self, X_train, y_train, **kwargs):
        """
        This method trains the KNeighbors model.

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
            model_class=KNeighborsClassifier,
            get_params_func=get_params_func,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_trials=n_trials,
        )
