from sklearn.feature_selection import chi2
import pandas as pd
import json


class Chi2_Calculation:
    """
    This class calculates chi-square scores for categorical
    features in the dataset.

    Parameters
    ----------
    chi2_data : pandas DataFrame
        The input data containing categorical features.

    processed_data : pandas DataFrame
        The processed data containing the target variable.

    opt : object
        The configuration options object.

    logger : object
        The logger object to log the results.
    """

    def __init__(self, chi2_data, processed_data, opt, logger):
        self.chi2_data = chi2_data
        self.processed_data = processed_data
        self._opt = opt
        self._logger = logger
        self.y = None

    def setup_target(self):
        """
        This method sets up the target variable 'y' for
        chi-square calculation.
        """
        try:
            self.y = self.processed_data[self._opt.target_column]
        except KeyError:
            raise ValueError(
                f"Target column '{self._opt.target_column}' not found in the processed data."
            )

    # XOR: Only one should be True
    def validate_options(self):
        """
        This method validates the configuration options
        for chi-square calculation.
        """
        if not (self._opt.chi2_values ^ self._opt.chi2_pvalues):
            raise ValueError(
                "Invalid configuration: You must set either 'chi2_values' or 'chi2_pvalues' to True, but not both."
            )

    def calculate_chi2(self):
        """
        This method calculates chi-square scores for the
        categorical features in the dataset.

        Returns
        -------
        chi2_scores_df : pandas DataFrame
            The DataFrame containing chi-square scores for
            categorical features.

        chi2_pvalues_df : pandas DataFrame
            The DataFrame containing chi-square p-values for
            categorical features.

        Raises
        ------
        ValueError
            If the target variable 'y' is not set.
        """
        if not self._opt.calculate_chi2:
            print(
                "\nChi2 calculation is not enabled. Please enable it in the config file.\n"
            )
            return None

        if self.y is None:
            raise ValueError(
                "Target variable 'y' is not set. Please ensure the target column is correctly set up."
            )

        self.validate_options()

        print("\nCalculating chi2 scores....")

        self._logger.update_log(
            "Chi_Square_Scores", "Status", "Chi2 scores calculation started"
        )

        try:
            chi2_scores = chi2(self.chi2_data, self.y)

            categorical_columns = list(self.chi2_data.columns)

            # Return chi-square values
            if self._opt.chi2_values:

                self._logger.update_log(
                    "Chi_Square_Scores", "Values Utilised", "chi2_values"
                )

                self._logger.update_log(
                    "Chi_Square_Scores",
                    "chi2_values info",
                    "Higher the value, more dependent the feature is on the target.",
                )

                chi2_scores_df = pd.DataFrame(
                    chi2_scores[0], index=self.chi2_data.columns, columns=["chi2_score"]
                )

                print(
                    f"\nCategorical columns with chi-square calculated: {categorical_columns}"
                )

                # Logging as a formatted JSON string for better readability
                chi2_scores_dict = chi2_scores_df.to_dict()["chi2_score"]

                formatted_chi2_scores = json.dumps(chi2_scores_dict, indent=4)

                print(f"\nChi2 scores:\n{formatted_chi2_scores}")

                self._logger.update_log(
                    "Chi_Square_Scores",
                    "Chi square categorical columns",
                    categorical_columns,
                )
                self._logger.update_log(
                    "Chi_Square_Scores", "chi2_scores", chi2_scores_dict
                )

                return chi2_scores_df

            # Return chi-square p-values
            elif self._opt.chi2_pvalues:

                self._logger.update_log(
                    "Chi_Square_Scores", "Values Utilised", "p_values"
                )

                self._logger.update_log(
                    "Chi_Square_Scores",
                    "p_values info",
                    "Lower the value, more dependent the feature is on the target.",
                )

                chi2_pvalues_df = pd.DataFrame(
                    chi2_scores[1],
                    index=self.chi2_data.columns,
                    columns=["chi2_pvalue"],
                )

                print(
                    f"\nCategorical columns with chi-square calculated: {categorical_columns}"
                )

                chi2_pvalues_dict = chi2_pvalues_df.to_dict()["chi2_pvalue"]
                formatted_chi2_pvalues = json.dumps(chi2_pvalues_dict, indent=4)
                print(f"\nChi2 p-values:\n{formatted_chi2_pvalues}")

                self._logger.update_log(
                    "Chi_Square_Scores",
                    "Chi square categorical columns",
                    categorical_columns,
                )
                self._logger.update_log(
                    "Chi_Square_Scores", "chi2_pvalues", chi2_pvalues_dict
                )

                return chi2_pvalues_df

            else:
                raise ValueError(
                    "Please configure either chi2_values or chi2_pvalues in the config options."
                )

        except Exception as e:
            raise ValueError(f"Error occurred during chi-square calculation: {e}")

    def get_chi2_scores(self):
        """
        This method returns the chi-square scores for the
        categorical features in the dataset.
        """
        self.setup_target()
        return self.calculate_chi2()
