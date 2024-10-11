from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.base import clone
from tabulate import tabulate
import pandas as pd


class CalculateSfsImportance:
    """
    This class provides methods for calculating feature importance using
    Sequential Feature Selection (SFS).

    Parameters
    ----------
    model : object
        The trained model.

    logger : Logger
        The logger instance for logging information.

    opt : Namespace
        Object containing the experiment options.
    """

    def __init__(self, model, logger, opt):
        self.logger = logger
        self._opt = opt
        self.model = model

    def perform_sfs(self, X_train, y_train, model_params=None, tuning_phase=None):
        """
        This method performs Sequential Feature Selection (SFS) to calculate
        the feature importance.

        Parameters
        ----------
        X_train : DataFrame
            The training features.

        y_train : DataFrame
            The training target.

        model_params : dict, optional
            The parameters for the model (default is None).

        tuning_phase : str, optional
            The tuning phase (default is None).

        Returns
        -------
        table : str
            The feature importance table.

        Raises
        ------
        ValueError
            If an error occurs while performing SFS.

        """
        # Setting variables for logging
        phase = "Before Tuning" if tuning_phase == "before" else "After Tuning"

        if self._opt.sequential_feature_selector:
            print("\nPerforming Sequential Feature Selection \n")

            if self.model is None:
                raise ValueError("Model is not trained. Please train the model.")

            try:
                self.logger.update_log(f"SFS {phase}", "Status", "Started")

                self.logger.update_log(
                    f"SFS {phase}", "Method of SFS", self._opt.sfs_k_features
                )

                self.logger.update_log(f"SFS {phase}", "Verbose", self._opt.sfs_verbose)

                self.logger.update_log(f"SFS {phase}", "Scoring", self._opt.sfs_scoring)

                self.logger.update_log(
                    f"SFS {phase}", "Cross Validation", self._opt.sfs_cv
                )

                base_model = self.model.model
                if model_params:
                    clone_model = clone(base_model).set_params(**model_params)
                else:
                    clone_model = clone(base_model)

                clone_model = clone(base_model)
                sfs = SFS(
                    clone_model,
                    k_features=self._opt.sfs_k_features,
                    forward=self._opt.sfs_direction,
                    floating=False,
                    verbose=self._opt.sfs_verbose,
                    scoring=self._opt.sfs_scoring,
                    cv=self._opt.sfs_cv,
                    n_jobs=-1,
                )
                sfs.fit(X_train, y_train)

                selected_features = list(sfs.k_feature_names_)
                feature_importances = pd.DataFrame(
                    {
                        "Feature": selected_features,
                        "Importance Score": [sfs.k_score_] * len(selected_features),
                    }
                )

                feature_importances.sort_values(
                    by="Importance Score", ascending=False, inplace=True
                )

                self.logger.update_log(
                    f"SFS {phase}",
                    "Number of features selected",
                    len(selected_features),
                )

                if (
                    self._opt.sfs_n_features is not None
                    and self._opt.sfs_n_features < len(feature_importances)
                ):
                    feature_importances = feature_importances.head(
                        self._opt.sfs_n_features
                    )

                    table = tabulate(
                        feature_importances, headers="keys", tablefmt="pretty"
                    )

                else:
                    table = tabulate(
                        feature_importances, headers="keys", tablefmt="pretty"
                    )

                print("\nFeature Selection Completed Successfully!\n")
                print(table)

                self.logger.update_log(
                    f"SFS {phase}", "SFS Feature Importance Table", f"\n{table}\n"
                )
                return table

            except Exception as e:
                raise ValueError(f"An error occurred while performing SFS: {e}")

        else:

            print("\nSequential Feature Selection will be skipped\n")
            self.logger.update_log("SFS", "Status", "Skipped calculating SFS....")
