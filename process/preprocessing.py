import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tabulate import tabulate
import io

from feature_engineering.feature_engg_call_methods import call_feature_engineering


class DataProcessor:
    """
    This class provides methods for loading, preprocessing and saving data.
    """

    def __init__(self, file_path, logger, opt):
        """
        Initialize the DataProcessor object with the file path.

        Parameters
        ----------
        file_path : str
            The path to the data file.

        logger : Logger
            The logger instance for logging information.

        opt : Namespace
            The namespace object containing the experiment options.
        """
        self.file_path = file_path
        self.data = None
        self.logger = logger
        self._opt = opt

    def load_data(self):
        """
        Load the data from the specified file path.

        Parameters
        ----------
        None

        raises
        ------
        ValueError
            If the file path is not a string or does not end with '.csv'.
            If the data loading was not successful.
        """
        assert isinstance(self.file_path, str), "The file path must be a string."
        assert self.file_path.endswith(".csv"), "The file path must end with '.csv'."

        try:
            self.data = pd.read_csv(self.file_path)
            num_samples = self.data.shape[0]
            num_features = self.data.shape[1]
            missing_values_exist = self.data.isnull().values.any()
            missing_values_status = "Yes" if missing_values_exist else "No"

            buffer = io.StringIO()
            self.data.info(buf=buffer)
            info_str = buffer.getvalue()

            self.logger.update_log("data_loading", "data_loaded", True)
            self.logger.update_log("data_loading", "total_samples", num_samples)
            self.logger.update_log("data_loading", "total_features", num_features)
            self.logger.update_log(
                "data_loading", "missing_values", missing_values_status
            )
            self.logger.update_log("data_loading", "dataframe_info", info_str)

            print("\n" + "-" * 40)
            print("-" * 40 + "\n")
            print("Data loaded successfully. \n")

        except Exception as e:
            raise ValueError(f"Failed to load data from {self.file_path}: {e}")

        finally:
            if self.data is None:
                raise ValueError("Data loading was not successful.")

    @staticmethod
    def wrap_text(self, text_list):
        """
        This method Wrap text to a specified width for better formatting in logs.

        Parameters
        ----------
        text_list : list of str
            List of text strings to be wrapped.

        Returns
        -------
        wrapped_text : str
            The wrapped text.
        """
        import textwrap

        wrapped_text = []
        for text in text_list:
            wrapped_text.extend(textwrap.wrap(text, width=self._opt.wrapper_width))
        return "\n".join(wrapped_text)

    def change_column_dtype(self, dtype_dict):
        """
        Change the data type of specified columns.

        Parameters
        ----------
        dtype_dict : dict
            Dictionary where keys are column names and values are the target data types.

        raises
        ------
        ValueError
            If data is not loaded before calling this method.
            If a specified column does not exist in the data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        if dtype_dict == ["None"] or dtype_dict is None:
            print("\nNo datatype should be changed.\n")
            return

        for column, dtype in dtype_dict.items():
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' does not exist in the data.")
            try:
                if dtype == "datetime64":
                    self.data[column] = pd.to_datetime(self.data[column])
                else:
                    self.data[column] = self.data[column].astype(dtype)
                print(f"\nColumn '{column}' has been converted to '{dtype}' type")
            except Exception as e:
                raise ValueError(
                    f"Failed to convert column '{column}' to '{dtype}': {e}"
                )

        self.logger.update_log("data_processing", "dtype_changes", dtype_dict)

    def separate_data_types(self):
        """
        Separate columns into categorical, numeric, and other data types.

        Parameters
        ----------
        None

        Returns
        -------
        categorical_cols : list of str
            List of categorical columns.

        numeric_cols : list of str
            List of numeric columns.

        other_cols : list of str
            List of other columns.

        raises
        ------
        ValueError
            If data is not loaded before calling this method.
        """
        if self.data is None:
            raise ValueError(
                "Data not loaded in separate_data_types. Please call load_data() first."
            )

        categorical_cols = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numeric_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
        other_cols = self.data.select_dtypes(
            exclude=["number", "object", "category"]
        ).columns.tolist()

        print(f"\nCategorical columns: {categorical_cols}")
        print(f"\nNumeric columns: {numeric_cols}")
        if not other_cols:
            other_cols = None
            print(f"\nOther columns: {None}\n")
        else:
            print(f"Other columns: {other_cols}\n")

        self.logger.update_log(
            "data_processing", "categorical_columns", categorical_cols
        )
        self.logger.update_log("data_processing", "numeric_columns", numeric_cols)
        self.logger.update_log("data_processing", "other_columns", other_cols)

        return (
            categorical_cols,
            numeric_cols,
            other_cols,
        )

    def check_missing_values(self):
        """
        Check for missing values in the DataFrame.

        Parameters
        ----------
        None

        returns
        -------
        missing_values : Series
            Series containing the count of missing values in each column.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        missing_values = self.data.isnull().sum()
        missing_values_table = pd.DataFrame(missing_values, columns=["Missing Values"])

        print("\nMissing values in each column:\n")
        print(tabulate(missing_values_table, headers="keys", tablefmt="pretty"))

        self.logger.update_log(
            "data_processing", "missing_values", missing_values.to_dict()
        )
        return missing_values

    def impute_missing_values(self, imputation_dict, method_type):
        """
        Impute missing values in the specified columns using the given methods.

        Parameters
        ----------
        imputation_dict : dict
            Dictionary where keys are column names and values are
            tuples of the form (method, fill_value).

        Available methods:
            'mean', 'median', 'mode', 'fillna'.

        raises
        ------
        ValueError
            If data is not loaded before calling this method.
            If an invalid imputation method is provided.
            If 'fillna' method is used without providing fill_value.
        """
        if self.data is None:
            raise ValueError(
                "Data is not loaded. Please load the data first using load_data method."
            )

        if not self.data.isnull().values.any():
            print("\nThere are no missing values. No imputation needed.\n")
            self.logger.update_log(
                "data_processing",
                "imputation_needed",
                "No missing values. Imputation not needed.",
            )
            return

        method_type = self._opt.missing_values_imputation_method

        if method_type == "imputation_dictionary":
            self.logger.update_log(
                "data_processing", "imputation_method_type", "imputation_dictionary"
            )

            try:
                for column, (method, fill_value) in imputation_dict.items():
                    if method == "mean":
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                    elif method == "median":
                        self.data[column].fillna(
                            self.data[column].median(), inplace=True
                        )
                    elif method == "mode":
                        self.data[column].fillna(
                            self.data[column].mode()[0], inplace=True
                        )
                    elif method == "fillna":
                        if fill_value is None:
                            raise ValueError(
                                "fill_value must be provided when using 'fillna' method."
                            )
                        self.data[column].fillna(fill_value, inplace=True)
                    else:
                        raise ValueError(
                            "Invalid imputation method. Choose from 'mean', 'median', 'mode', 'fillna'."
                        )
                print(
                    "\nMissing values in columns have been imputed using specified methods.\n"
                )

                self.logger.update_log(
                    "data_processing",
                    "imputation_methods",
                    {k: v[0] for k, v in imputation_dict.items()},
                )

            except Exception as e:
                raise ValueError(f"Error occurred during missing value imputation: {e}")

        elif method_type == "global_imputation":
            self.logger.update_log(
                "data_processing", "imputation_method_type", "global_imputation"
            )

            try:
                if self._opt.global_imputation_method == "mean":
                    self.data.fillna(self.data.mean(), inplace=True)
                elif self._opt.global_imputation_method == "median":
                    self.data.fillna(self.data.median(), inplace=True)
                elif self._opt.global_imputation_method == "mode":
                    self.data.fillna(self.data.mode().iloc[0], inplace=True)
                elif self._opt.global_imputation_method == "fillna":
                    if self._opt.global_fill_value is None:
                        raise ValueError(
                            "fill_value must be provided when using 'fillna' method."
                        )
                    self.data.fillna(self._opt.global_fill_value, inplace=True)
                else:
                    raise ValueError(
                        "Invalid imputation method. Choose from 'mean', 'median', 'mode', 'fillna'."
                    )
                print(
                    f"\nMissing values in columns have been imputed using global method using {self._opt.global_imputation_method}.\n"
                )

                self.logger.update_log(
                    "data_processing",
                    "imputation_methods",
                    self._opt.global_imputation_method,
                )

            except Exception as e:
                raise ValueError(f"Error occurred during missing value imputation: {e}")

        else:
            raise ValueError(
                "Invalid imputation method type. Choose from 'imputation_dictionary', 'global_imputation'."
            )

    def multivariate_impute(self):
        """
        Perform multivariate imputation based on custom conditions.
        Placeholder for future custom logic.
        """
        # Template for multivariate imputation based on custom conditions.
        # Add the conditions here.
        pass

    def feature_engineering(self):
        """
        This method calls the feature engineering function based
        on the user's choice.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        if self._opt.calculate_feature_engg:
            try:
                feature_engineering_func = call_feature_engineering(
                    self._opt.feature_engg_name, self.data, self.logger, self._opt
                )
                print(
                    f"\nExecuting feature engineering: {self._opt.feature_engg_name}\n"
                )
                self.data = feature_engineering_func()
                print(
                    f"Feature engineering '{self._opt.feature_engg_name}' completed successfully.\n"
                )

                # ----- ADD OTHER FEATURE ENGINEERING METHODS HERE -----

            except Exception as e:
                raise ValueError(f"Error occurred during feature engineering: {e}")

        else:
            print("\nNo feature engineering to be performed....\n")
            self.logger.update_log(
                "data_processing",
                "feature_engineering",
                "No feature engineering performed or required...",
            )

    def drop_columns(self):
        """
        Drop specified columns from the dataset.

        Parameters
        ----------
        columns_to_drop : list of str
            List of columns to be dropped from the dataset.

        raises
        ------
        ValueError
            If data is not loaded before calling this method
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        self.data.drop(columns=self._opt.drop_columns, inplace=True)
        print(f"Columns {self._opt.drop_columns} have been dropped.\n")
        self.logger.update_log(
            "data_processing", "dropped_columns", self._opt.drop_columns
        )

    def get_chi_square_data(self):
        """
        This method processes the data for chi square test.

        Returns
        -------
        chi2_data : DataFrame
            The data processed for chi square test.

        raises
        ------
        ValueError
            If an error occurs during chi square data processing.
        """

        chi2_data = None

        if self._opt.calculate_chi2:

            try:

                chi2_cat_cols = self.data.select_dtypes(
                    include=["object", "category"]
                ).columns

                chi2_data = self.data[chi2_cat_cols]

                le = LabelEncoder()

                for column in chi2_data.columns:
                    chi2_data.loc[:, column] = le.fit_transform(chi2_data[column])

                return chi2_data

            except Exception as e:
                raise ValueError(
                    f"Error occurred during chi square data processing: {e}"
                )

        else:
            return print(
                "Chi square calculation is not enabled. Please enable it in the config file."
            )

    def encode_columns(
        self,
        label_encode_columns,
        one_hot_encode_columns,
    ):
        """
        Encode the defined columns using label encoding and one hot encoding.

        Parameters
        ----------
        label_encode_columns : list of str
            List of columns to be label encoded.

        one_hot_encode_columns : list of str
            List of columns to be one hot encoded.

        do_label_encode : bool
            Whether to perform label encoding.

        do_one_hot_encode : bool
            Whether to perform one hot encoding.

        Returns
        -------
        Column encoded data : DataFrame
            The data with the specified columns encoded.
        """
        # Label Encoding
        if self._opt.do_label_encode:

            label_encoded_columns = []

            for column in label_encode_columns:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column])
                label_encoded_columns.append(column)

            print(f"\nColumns {label_encoded_columns} have been label encoded.\n")

        else:
            print("\nNo label encoding to be performed.\n")
            self.logger.update_log(
                "data_processing",
                "label_encoded_columns",
                "No label encoding performed.",
            )

        # One-Hot Encoding
        if self._opt.do_one_hot_encode:

            try:

                ohe = OneHotEncoder(sparse_output=False, drop="first")

                encoded_features = ohe.fit_transform(self.data[one_hot_encode_columns])
                encoded_df = pd.DataFrame(
                    encoded_features,
                    columns=ohe.get_feature_names_out(one_hot_encode_columns),
                )

                self.data = self.data.drop(columns=one_hot_encode_columns).reset_index(
                    drop=True
                )
                self.data = pd.concat([self.data, encoded_df], axis=1)

                print(f"Columns {one_hot_encode_columns} have been one-hot encoded. \n")

                self.logger.update_log(
                    "data_processing",
                    "encoded_columns",
                    {
                        "label_encoded": label_encode_columns,
                        "one_hot_encoded": one_hot_encode_columns,
                    },
                )

            except Exception as e:
                raise ValueError(f"Error occurred during one-hot encoding: {e}")

        else:
            print("\nNo one-hot encoding to be performed.\n")
            self.logger.update_log(
                "data_processing",
                "one_hot_encoded_columns",
                "No one-hot encoding performed.",
            )

    def process_and_save(
        self,
        imputation_dict=None,
        label_encode_columns=None,
        one_hot_encode_columns=None,
        dtype_dict=None,
        feature_engg_names=None,
        calculate_chi2=None,
    ):
        """
        Process the data by loading, checking missing values, imputing missing values,
        encoding columns, and saving the processed data into a new DataFrame.

        Parameters
        ----------
        imputation_dict : dict
            Dictionary where keys are column names and values are tuples of the form (method, fill_value).

        label_encode_columns : list of str, optional
            List of columns to be label encoded.

        one_hot_encode_columns : list of str, optional
            List of columns to be one hot encoded.

        dtype_dict : dict, optional
            Dictionary where keys are column names and values are the target data types.

        feature_engg_names : list of str, optional
            List of feature engineering functions to apply.

        Returns
        -------
        processed_data : DataFrame
            The processed data.
        """
        self.load_data()
        missing_values = self.check_missing_values()
        self.separate_data_types()
        self.impute_missing_values(
            imputation_dict, method_type=self._opt.missing_values_imputation_method
        )
        self.multivariate_impute()
        self.check_missing_values()

        if dtype_dict:
            self.change_column_dtype(dtype_dict)

        self.feature_engineering()

        if calculate_chi2:
            chi2_data = self.get_chi_square_data()

        if label_encode_columns or one_hot_encode_columns:
            self.encode_columns(label_encode_columns, one_hot_encode_columns)

        if self._opt.drop_columns:
            self.drop_columns()

        remaining_columns = self.data.shape[1]
        print(
            f"\nFinal number of columns before train_test_split: {remaining_columns}\n"
        )
        self.logger.update_log(
            "data_processing", "final_number_of_columns", remaining_columns
        )
        print(f"\nFinal columns before train_test_split: {self.data.columns}\n")
        self.logger.update_log("data_processing", "final_columns", self.data.columns)

        processed_data = self.data.copy()
        return processed_data, missing_values, chi2_data
