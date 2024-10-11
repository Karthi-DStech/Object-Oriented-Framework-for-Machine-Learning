import shap
from tabulate import tabulate
import numpy as np
import pandas as pd


class CalculateSHAPValues:
    """
    This class calculates SHAP values for a given model and dataset.

    Parameters
    ----------
    model : object
        The trained model object

    X_train : DataFrame
        The training data

    X_test : DataFrame
        The testing data

    logger : object
        The logger object

    opt : object
        The options object for the experiment.
    """

    def __init__(self, model, X_train, X_test, logger, opt):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.logger = logger
        self._opt = opt

    def _calculate_shap_values(self, model_type):
        """
        This method calculates SHAP values for the given model and dataset.

        Parameters
        ----------
        Parameters
        ----------
        model_type : str
            The type of the model ('tree', 'linear').

        Returns
        -------
        shap_values : array
            The SHAP values for the model and dataset.

        Raises
        ------
        ValueError
            If an error occurs while calculating SHAP values.
        """

        if model_type == "tree":

            try:
                print("Calculating SHAP values for Tree Based Models...\n")
                actual_model = (
                    self.model.model if hasattr(self.model, "model") else self.model
                )

                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(self.X_test)
                self.logger.update_log(
                    "SHAP", "Status", "SHAP values calculated successfully"
                )
                print("SHAP values calculated successfully\n")
                return shap_values
            except Exception as e:
                raise ValueError(
                    f"An error occurred while calculating SHAP values: {e}"
                )

        else:
            print(
                "SHAP values are supported or configured only for Tree-based models...\n"
            )
            self.logger.update_log(
                "SHAP",
                "Status",
                "SHAP values are supported or configured only for tree-based models",
            )
            return None

    def _format_shap_values_as_table(self, shap_values):
        """
        This method formats SHAP values as a table for logging.

        Parameters
        ----------
        shap_values : array
            The SHAP values for the model and dataset.

        Returns
        -------
        table_str : str
            The SHAP values formatted as a table.

        shap_value_df : DataFrame
            The SHAP values formatted as a DataFrame.

        Raises
        ------
        ValueError
            If an error occurs while formatting SHAP values as table.
        """

        try:

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            table = []
            feature_names = (
                self.X_test.columns
                if hasattr(self.X_test, "columns")
                else range(self.X_test.shape[1])
            )

            for i in range(len(feature_names)):
                row = [feature_names[i]] + list(shap_values[:, i])
                table.append(row)

            headers = ["Feature"] + [
                f"Sample {i+1}" for i in range(shap_values.shape[0])
            ]

            table_str = tabulate(table, headers, tablefmt="grid")

            self.logger.update_log("SHAP", "Values as Table", f"\n{table_str}")
            print("SHAP values logged as table successfully\n")

            # Convert the table to a DataFrame
            shap_value_df = pd.DataFrame(table, columns=headers)
            print("SHAP values converted to DataFrame successfully\n")
            print(shap_value_df.head())

            return table_str, shap_value_df

        except Exception as e:
            raise ValueError(
                f"An error occurred while formatting SHAP values as table: {e}"
            )

    def _calculate_mean_absolute_shap(self, shap_values):
        """
        This method calculates the mean absolute SHAP values
        for the given model and dataset.

        Parameters
        ----------
        shap_values : array
            The SHAP values for the model and dataset.

        Returns
        -------
        table_str : str
            The mean absolute SHAP values formatted as a table.

        Raises
        ------
        ValueError
            If an error occurs while calculating mean absolute SHAP values.
        """

        try:

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

            table = []
            feature_names = (
                self.X_test.columns
                if hasattr(self.X_test, "columns")
                else range(self.X_test.shape[1])
            )

            for i in range(len(feature_names)):
                row = [feature_names[i], mean_abs_shap_values[i]]
                table.append(row)

            headers = ["Feature", "Mean Absolute SHAP Value"]

            table_str = tabulate(table, headers, tablefmt="grid")

            self.logger.update_log(
                "SHAP", "Mean Absolute SHAP Values", f"\n{table_str}"
            )
            print("Mean Absolute SHAP values logged as table successfully\n")

            return table_str

        except Exception as e:
            raise ValueError(
                f"An error occurred while calculating mean absolute SHAP values: {e}"
            )

    def _ranked_mean_absolute_values(self, shap_values):
        """
        This method ranks the mean absolute SHAP values
        for the given model and dataset.

        Parameters
        ----------
        shap_values : array
            The SHAP values for the model and dataset.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an error occurs while ranking mean absolute SHAP values.
        """

        try:
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

            # Handle 1D and 2D cases
            if mean_abs_shap_values.ndim == 1:
                mean_abs_shap_values = mean_abs_shap_values.reshape(-1, 1)

            feature_names = (
                self.X_test.columns
                if hasattr(self.X_test, "columns")
                else range(self.X_test.shape[1])
            )

            # Create a list of tuples (feature_name, mean_absolute_shap_value)
            feature_importance = [
                (feature_names[i], mean_abs_shap_values[i])
                for i in range(len(feature_names))
            ]

            # Sort the list by the sum of mean absolute SHAP values across all classes
            feature_importance.sort(key=lambda x: np.sum(x[1]), reverse=True)

            if self._opt.shap_n_features_ranked is None:
                try:
                    top_shap_features = len(feature_importance)
                    self.logger.update_log(
                        "SHAP",
                        f"All {top_shap_features} Features Ranked",
                        "All features ranked by mean absolute SHAP values",
                    )
                except Exception as e:
                    raise ValueError(
                        f"An error occurred while logging ranked mean absolute SHAP values: {e}"
                    )

            else:
                try:
                    top_shap_features = self._opt.shap_n_features_ranked
                    self.logger.update_log(
                        "SHAP",
                        f"Top {top_shap_features} Features Ranked",
                        f"Top {top_shap_features} features are chosen for ranked by mean absolute SHAP values",
                    )
                except Exception as e:
                    raise ValueError(
                        f"An error occurred while logging ranked mean absolute SHAP values: {e}"
                    )

            # Select only the top N features
            top_features = feature_importance[:top_shap_features]

            table = []
            for feature, values in top_features:
                row = [feature] + list(values)
                table.append(row)

            headers = ["Feature"] + [
                f"Class {i+1}" for i in range(mean_abs_shap_values.shape[1])
            ]

            table_str = tabulate(table, headers, tablefmt="grid")
            self.logger.update_log(
                "SHAP",
                f"Top {top_shap_features} Ranked Mean Absolute SHAP Values",
                f"\n{table_str}",
            )
            print(
                f"Top {top_shap_features} ranked mean absolute SHAP values logged successfully\n"
            )
        except Exception as e:
            raise ValueError(
                f"An error occurred while logging ranked mean absolute SHAP values: {e}"
            )

    def _specific_sample_shap_values(self, shap_value_df):
        """
        This method logs SHAP values for specific samples.

        Parameters
        ----------
        shap_value_df : DataFrame
            The DataFrame containing SHAP values.

        Returns
        -------
        specific_samples_df : DataFrame
            The DataFrame containing SHAP values for specific samples.

        Raises
        ------
        ValueError
            If an error occurs while logging SHAP values for specific samples.
        """

        if self._opt.shap_specific_samples is not None:
            sample_numbers = self._opt.shap_specific_samples

            try:

                # Convert 1-based to 0-based sample numbers and get the column names
                sample_columns = [f"Sample {i}" for i in sample_numbers]

                # Extract the rows corresponding to the specified sample columns
                specific_samples_df = shap_value_df[["Feature"] + sample_columns]

                # Convert to a string table format using tabulate
                table_str = tabulate(
                    specific_samples_df,
                    headers="keys",
                    tablefmt="grid",
                    showindex=False,
                )

                # Log the table
                self.logger.update_log(
                    "SHAP",
                    f"SHAP Values for Samples {sample_numbers}",
                    f"\n{table_str}",
                )
                print(f"SHAP values for samples {sample_numbers} logged successfully\n")

                return specific_samples_df

            except Exception as e:
                raise ValueError(
                    f"An error occurred while retrieving SHAP values for specific samples: {e}"
                )

        else:
            print(
                "No specific sample numbers provided. Skipping logging of specific sample SHAP values\n"
            )
            self.logger.update_log(
                "SHAP",
                "SHAP Values for Samples",
                "No specific sample numbers provided. \
                \n  If required, mention specific sample numbers in the options file.",
            )
            return None

    def save_shap_plots(self, model_type):
        """
        This method executes all the logic for SHAP calculation.

        Parameters
        ----------
        model_type : str
            The type of the model ('tree', 'linear', etc.)
        """

        if self._opt.calculate_SHAP:

            shap_values = self._calculate_shap_values(model_type)

            if model_type == "tree":

                self.logger.update_log(
                    "SHAP", "Status", "Started calculating SHAP values..."
                )

                table_str, shap_value_df = self._format_shap_values_as_table(
                    shap_values
                )

                self._specific_sample_shap_values(shap_value_df)

                self._calculate_mean_absolute_shap(shap_values)

                self._ranked_mean_absolute_values(shap_values)
