import json
from tabulate import tabulate


class CorrelationCoefficient:
    """
    This class calculates the correlation matrix and provides
    insights on the correlation between features.

    Parameters
    ----------
    data : pandas DataFrame
        The input data containing features.

    logger : object
        The logger object to log the results.

    opt : object
        The configuration options object.
    """

    def __init__(self, data, logger, opt):
        self._opt = opt
        self.data = data
        self.logger = logger
        self.correlation_matrix = None

    def calculate_correlation(self):
        """
        This method calculates the correlation matrix for the
        numeric features in the dataset.

        Returns
        -------
        correlation_matrix : pandas DataFrame
            The correlation matrix of the numeric features.

        Raises
        ------
        ValueError
            If no numeric data is found in the dataset.
        """

        if not self._opt.calculate_correlation:
            print(
                "\nCorrelation calculation is not enabled. Please enable it in the config file."
            )
            return None

        if self.data is None:
            raise ValueError(
                "Data is not set. Please ensure the data is correctly set up."
            )

        print("\nCalculating correlation matrix....")

        self.logger.update_log(
            "Correlation_Analysis", "Status", "Calculating correlation matrix"
        )

        numeric_data = self.data.select_dtypes(include=["number"])

        if numeric_data.empty:
            raise ValueError("No numeric data found in the dataset.")

        try:
            # Calculate the correlation matrix
            self.correlation_matrix = numeric_data.corr()

            print("\nCorrelation matrix calculated successfully.")

            self._log_correlation_insights(numeric_data)

            return self.correlation_matrix

        except Exception as e:
            raise ValueError(f"Error calculating correlation matrix: {str(e)}")

    def _strong_positive_correlation(self):
        """
        This method identifies strong positive correlations
        between features and logs the results.

        Options
        -------
        threshold : float
            The threshold value to identify strong positive
            correlations.
        """

        corr_pairs = self._prepare_corr_pairs()

        threshold = self._opt.strong_positive_threshold
        self.logger.update_log(
            "Correlation_Analysis", "strong_positive_threshold", threshold
        )
        strong_pos_corr = corr_pairs[corr_pairs["Correlation"] > threshold].sort_values(
            by="Correlation", ascending=False
        )

        if not strong_pos_corr.empty:
            strong_pos_corr_dict = strong_pos_corr.to_dict(orient="records")
            strong_pos_corr_json = json.dumps(strong_pos_corr_dict, indent=4)

            print(
                f"\nStrong positive correlations (>{threshold}):\n{strong_pos_corr_json}"
            )

            self.logger.update_log(
                "Correlation_Analysis",
                "strong_positive_correlations",
                strong_pos_corr_json,
            )

    def _highly_correlated_features(self):
        """
        This method identifies highly correlated features
        and logs the results.

        Options
        -------
        high_corr_threshold : float
            The threshold value to identify highly correlated
            features.
        """

        corr_pairs = self._prepare_corr_pairs()
        high_corr_threshold = self._opt.high_corr_threshold

        self.logger.update_log(
            "Correlation_Analysis", "highly_correlation_threshold", high_corr_threshold
        )

        highly_corr = corr_pairs[
            (corr_pairs["Correlation"] > high_corr_threshold)
            | (corr_pairs["Correlation"] < -high_corr_threshold)
        ].sort_values(by="abs_correlation", ascending=False)

        if not highly_corr.empty:
            highly_corr_dict = highly_corr.to_dict(orient="records")
            highly_corr_json = json.dumps(highly_corr_dict, indent=4)

            print(
                f"\nHighly correlated features (>{high_corr_threshold} or <{-high_corr_threshold}):\n{highly_corr_json}"
            )

            self.logger.update_log(
                "Correlation_Analysis", "highly_correlated_features", highly_corr_json
            )

    def _top_n_most_correlated_pairs(self):
        """
        This method identifies the top N most correlated pairs
        and logs the results.

        Options
        -------
        top_n_most_correlated : int
            The number of most correlated pairs to display.
        """

        corr_pairs = self._prepare_corr_pairs()
        top_n = corr_pairs.sort_values(by="abs_correlation", ascending=False).head(
            self._opt.top_n_most_correlated
        )
        if not top_n.empty:
            top_n_table = tabulate(top_n, headers="keys", tablefmt="grid")

            print(
                f"\nTop {self._opt.top_n_most_correlated} most correlated pairs: \n{top_n_table}"
            )

            self.logger.update_log(
                "Correlation_Analysis",
                f"Top {self._opt.top_n_most_correlated} most_correlated_pairs",
                f"\n{top_n_table}",
            )

    def _prepare_corr_pairs(self):
        """
        This method prepares the correlation pairs for analysis.

        Returns
        -------
        corr_pairs : pandas DataFrame
            The correlation pairs of features.
        """

        corr_matrix = self.correlation_matrix
        corr_pairs = corr_matrix.unstack().reset_index()
        corr_pairs.columns = ["Feature1", "Feature2", "Correlation"]
        corr_pairs = corr_pairs[corr_pairs["Feature1"] != corr_pairs["Feature2"]]

        # Remove duplicate pairs -> (e.g., Feature1-Feature2 and Feature2-Feature1)

        corr_pairs["abs_correlation"] = corr_pairs["Correlation"].abs()
        corr_pairs = corr_pairs.drop_duplicates(subset=["abs_correlation"])

        return corr_pairs

    def _log_correlation_insights(self, numeric_data):
        """
        This method logs insights on the correlation between features.

        Parameters
        ----------
        numeric_data : pandas DataFrame
            The numeric data containing features
        """
        # Call each function to log separate insights

        self._strong_positive_correlation()
        self._highly_correlated_features()
        self._top_n_most_correlated_pairs()
