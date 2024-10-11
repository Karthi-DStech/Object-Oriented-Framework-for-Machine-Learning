import numpy as np
import pandas as pd
from tabulate import tabulate


def calculate_feature_importance(model, X, model_type, top_n=None):
    """
    Calculate and return feature importance for different model types.

    Parameters
    ----------
    model : object
        The trained model.

    X : DataFrame
        The features.

    model_type : str
        Type of the model ('linear' or 'tree').

    top_n : int, optional
        Number of top features to display. If None, display all features.

    Options
    -------
    model_type (linear):
        - model.coef_[0] : Coefficients of the linear model.

    model_type (tree):
        - model.feature_importances_ : Feature importances of the tree model.

    Returns
    -------
    table : str
        The table containing the sorted feature importance.
    """
    if model is None:
        raise ValueError("Model is not trained. Please train the model.")

    try:
        if model_type == "linear":
            feature_importance = np.abs(model.coef_[0])
        elif model_type == "tree":
            feature_importance = model.feature_importances_
        else:
            raise ValueError(
                "Unsupported model type for feature importance calculation."
            )

        feature_names = (
            X.columns
            if isinstance(X, pd.DataFrame)
            else [f"Feature {i}" for i in range(len(feature_importance))]
        )
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importance}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )

        if top_n is not None:
            feature_importance_df = feature_importance_df.head(top_n)

        table = tabulate(feature_importance_df, headers="keys", tablefmt="pretty")
        print(table)

        return table

    except Exception as e:
        raise ValueError(f"An error occurred while calculating feature importance: {e}")
