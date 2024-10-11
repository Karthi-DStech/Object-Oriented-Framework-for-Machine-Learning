from sklearn.base import BaseEstimator, ClassifierMixin


class ModelWrapper(BaseEstimator, ClassifierMixin):
    """
    This class wraps a model to be used in scikit-learn pipelines.

    Parameters
    ----------
    BaseEstimator : object
        The base class for all estimators in scikit-learn.

    ClassifierMixin : object
        Mixin class for all classifiers in scikit-learn.

    model_class : object
        The model class to wrap.

    **kwargs : dict
        Arbitrary keyword arguments.

    """

    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs
        self.model = model_class(**kwargs)

    def fit(self, X, y):
        """
        This method fits the model to the given training data.

        Parameters
        ----------
        X : DataFrame
            The training features.

        y : DataFrame
            The training target.

        Returns
        -------
        self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        This method predicts the target using the given features.

        parameters
        ----------
        X : DataFrame
            The features.

        Returns
        -------
        predictions : array
            The predicted target.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        This method predicts the probability of the target using the given features.

        parameters
        ----------
        X : DataFrame
            The features.

        Returns
        -------
        prediction probability : array
            The predicted probability of the target.
        """
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        """
        This method returns the parameters for the model.

        parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for the model and
            contained subobjects that are estimators.

        returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self.kwargs

    def set_params(self, **params):
        """
        This method sets the parameters for the model.

        parameters
        ----------
        **params : dict
            Estimator parameters.

        returns
        -------
        self
        """
        model_class = params.pop("model_class", self.model_class)
        if model_class != self.model_class:
            self.model_class = model_class
            self.model = model_class(**params)
        else:
            self.kwargs.update(params)
            self.model.set_params(**params)
        return self

    def __getattr__(self, name):
        """
        This method delegate attribute access to the underlying model.

        parameters
        ----------
        name : str
            The attribute name.

        returns
        -------
        attribute : object
            The attribute of the model.
        """
        return getattr(self.model, name)

    def __sklearn_clone__(self):
        """
        This method creates custom clone method for
        sklearn's clone function.

        parameters
        ----------
        None

        returns
        -------
        model : object
            The cloned model.
        """
        return ModelWrapper(self.model_class, **self.kwargs)
