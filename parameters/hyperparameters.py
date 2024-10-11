class Hyperparameters:
    """
    This class defines hyperparameters for different classification models.

    Parameters
    ----------
    None

    Attributes
    ----------
    _param_name : str
        The name of the hyperparameters to create.

    """

    def __init__(self):
        self._param_name = None

    def get_logistic_regression_params(self, trial):
        """
        Define and return hyperparameters for Logistic Regression.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Logistic Regression.
        """
        self._param_name = "LogisticRegression"
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
        }
        return params

    def get_knn_params(self, trial):
        """
        Define and return hyperparameters for K-Nearest Neighbors.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for K-Nearest Neighbors.
        """
        self._param_name = "KNeighborsClassifier"
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 50),
            "p": trial.suggest_int("p", 1, 2),
        }
        return params

    def get_svc_params(self, trial):
        """
        Define and return hyperparameters for Support Vector Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Support Vector Classifier.
        """
        self._param_name = "SVC"
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "degree": trial.suggest_int("degree", 2, 5),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
        }
        return params

    def get_decision_tree_params(self, trial):
        """
        Define and return hyperparameters for Decision Tree Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Decision Tree Classifier.
        """
        self._param_name = "DecisionTreeClassifier"
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        return params

    def get_random_forest_params(self, trial):
        """
        Define and return hyperparameters for Random Forest Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Random Forest Classifier.
        """

        self._param_name = "RandomForestClassifier"
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]
            ),
        }
        return params

    def get_ada_boost_params(self, trial):
        """
        Define and return hyperparameters for AdaBoost Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for AdaBoost Classifier.
        """
        self._param_name = "AdaBoostClassifier"
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"]),
        }
        return params

    def get_gradient_boosting_params(self, trial):
        """
        Define and return hyperparameters for Gradient Boosting Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for Gradient Boosting Classifier.
        """
        self._param_name = "GradientBoostingClassifier"
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        return params

    def get_xgboost_params(self, trial):
        """
        Define and return hyperparameters for XGBoost Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for XGBoost Classifier.
        """
        self._param_name = "XGBClassifier"
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }

        return params

    def get_cat_boost_params(self, trial):
        """
        Define and return hyperparameters for CatBoost Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for CatBoost Classifier.
        """
        self._param_name = "CatBoostClassifier"
        params = {
            "iterations": trial.suggest_int("iterations", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "verbose": 0,
        }

        return params

    def get_light_gbm_params(self, trial):
        """
        Define and return hyperparameters for LightGBM Classifier.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object provided by Optuna.

        Returns
        -------
        dict
            A dictionary of hyperparameters for LightGBM Classifier.
        """
        self._param_name = "LGBMClassifier"
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 31, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "verbose": -1,
        }

        return params
