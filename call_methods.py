def make_network(network_name, *args, **kwargs):
    """
    This fucntion import and return model based
    on the network_name provided.

    Parameters
    ----------
    network_name : str
        The name of the network to create.

    *args : list
        List of arguments to pass to the network.

    **kwargs : dict
        Dictionary of keyword arguments to pass to the network.

    Raises
    ------
    NotImplementedError
        If the network_name is not implemented.
    """

    if network_name.lower() == "logisticregression":
        from models.logistic_regression import LogisticRegressionModel

        network = LogisticRegressionModel(*args, **kwargs)
        return network

    elif network_name.lower() == "kneighborsclassifier":
        from models.knn import KnnModel

        network = KnnModel(*args, **kwargs)
        return network

    elif network_name.lower() == "svc":
        from models.svm import SvmModel

        network = SvmModel(*args, **kwargs)
        return network

    elif network_name.lower() == "decisiontreeclassifier":
        from models.decision_tree import DecisionTreeModel

        network = DecisionTreeModel(*args, **kwargs)
        return network

    elif network_name.lower() == "randomforestclassifier":
        from models.random_forest import RandomForestModel

        network = RandomForestModel(*args, **kwargs)
        return network

    elif network_name.lower() == "adaboostclassifier":
        from models.adaptive_boost import AdaBoostModel

        network = AdaBoostModel(*args, **kwargs)
        return network

    elif network_name.lower() == "gradientboostingclassifier":
        from models.gradient_boost import GradientBoostModel

        network = GradientBoostModel(*args, **kwargs)
        return network

    elif network_name.lower() == "xgbclassifier":
        from models.xgboost import XgBoostModel

        network = XgBoostModel(*args, **kwargs)
        return network

    elif network_name.lower() == "catboostclassifier":
        from models.cat_boost import CatBoostModel

        network = CatBoostModel(*args, **kwargs)
        return network

    elif network_name.lower() == "lgbmclassifier":
        from models.light_gbm import LightGbmModel

        network = LightGbmModel(*args, **kwargs)
        return network

    else:

        raise NotImplementedError(f"Model {network_name} not implemented")


def make_params(param_name, *args, **kwargs):
    """
    This function import and return hyperparameters
    based on the param_name provided.

    Parameters
    ----------
    param_name : str
        The name of the hyperparameters to create.

    *args : list
        List of arguments to pass to the hyperparameters.

    **kwargs : dict
        Dictionary of keyword arguments to pass to the hyperparameters.

    Raises
    ------
    NotImplementedError
        If the param_name is not implemented
    """
    if param_name.lower() == "logisticregression":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_logistic_regression_params
        return get_params_func

    elif param_name.lower() == "kneighborsclassifier":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_knn_params
        return get_params_func

    elif param_name.lower() == "svc":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_svc_params
        return get_params_func

    elif param_name.lower() == "decisiontreeclassifier":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_decision_tree_params
        return get_params_func

    elif param_name.lower() == "randomforestclassifier":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_random_forest_params
        return get_params_func

    elif param_name.lower() == "adaboostclassifier":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_ada_boost_params
        return get_params_func

    elif param_name.lower() == "gradientboostingclassifier":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_gradient_boosting_params
        return get_params_func

    elif param_name.lower() == "xgbclassifier":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_xgboost_params
        return get_params_func

    elif param_name.lower() == "catboostclassifier":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_cat_boost_params
        return get_params_func

    elif param_name.lower() == "lgbmclassifier":
        from parameters.hyperparameters import Hyperparameters

        hyperparams = Hyperparameters(*args, **kwargs)
        get_params_func = hyperparams.get_light_gbm_params
        return get_params_func

    else:
        raise NotImplementedError(f"Make parameters for {param_name} not implemented")

