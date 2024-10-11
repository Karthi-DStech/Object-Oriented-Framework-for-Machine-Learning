import pandas as pd


class FeatureEngineeringLogics:
    """
    This class contains methods for feature engineering on the input data.

    Parameters
    ----------
    None
    """

    def __init__(self, data, logger, opt):
        """
        Initialize the FeatureEngineering object with the logger instance.

        Parameters
        ----------
        data : pd.DataFrame
            The input dataframe.
        logger : Logger
            The logger instance for logging information.
        opt : Namespace
            The namespace object containing the experiment options.
        """
        self.data = data
        self.logger = logger
        self._opt = opt

    def calculate_total_days(self):
        """
        Calculate the total days admitted based on the
        Date of Admission and Discharge Date columns.

        Returns
        -------
        pd.DataFrame
            The dataframe with the Total Days Admitted column added.
        """

        self.name = "calculate_total_days"

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        try:

            date_admission_col = self._opt.starting_date_ctd
            discharge_date_col = self._opt.ending_date_ctd

            if (
                date_admission_col in self.data.columns
                and discharge_date_col in self.data.columns
            ):
                self.data["Total Days Admitted"] = (
                    pd.to_datetime(self.data[discharge_date_col])
                    - pd.to_datetime(self.data[date_admission_col])
                ).dt.days
                print("\nColumn 'Total Days Admitted' has been created.\n")

            if "Total Days Admitted" in self.data.columns:
                self.logger.update_log(
                    "data_processing",
                    "feature_engineering",
                    "Total Days Admitted created",
                )
            else:
                self.logger.update_log(
                    "data_processing", "feature_engineering", "Date columns missing..."
                )

            return self.data

        except Exception as e:
            raise ValueError(f"Error occurred during total days calculation: {e}")

    def separate_date_columns(self):
        """
        Separate the specified date-time column into separate columns
        for year, month, day, hour, minute, and day of the week.

        Returns
        -------
        pd.DataFrame
            The dataframe with the new date-related columns added.
        """

        self.name = "separate_date_columns"

        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        try:

            date_column = self._opt.date_column_sdc

            if date_column in self.data.columns:

                # Convert the column to datetime format if it's not already
                self.data[date_column] = pd.to_datetime(
                    self.data[date_column], errors="coerce"
                )

                # Extracting the date-time components conditionally
                if self.data[date_column].dt.year.isnull().sum() == 0:
                    self.data["Year"] = self.data[date_column].dt.year

                if self.data[date_column].dt.month.isnull().sum() == 0:
                    self.data["Month"] = self.data[date_column].dt.month

                if self.data[date_column].dt.day.isnull().sum() == 0:
                    self.data["Day"] = self.data[date_column].dt.day

                if self.data[date_column].dt.hour.isnull().sum() == 0:
                    self.data["Hour"] = self.data[date_column].dt.hour

                # Extracting day of the week since it's commonly available
                self.data["Day of Week"] = self.data[date_column].dt.dayofweek

                # Only add minute if it exists in the data
                if self.data[date_column].dt.minute.isnull().sum() == 0:
                    self.data["Minute"] = self.data[date_column].dt.minute

                print(
                    "\nDate columns separated into Year, Month, Day, Hour, Day of Week, and Minute (if applicable).\n"
                )

                self.logger.update_log(
                    "data_processing", "feature_engineering", "Date columns separated"
                )
            else:
                self.logger.update_log(
                    "data_processing",
                    "feature_engineering",
                    f"Date column '{date_column}' not found",
                )

            return self.data

        except Exception as e:
            raise ValueError(f"Error occurred during separate date columns: {e}")

    # ----- Add other feature engineering methods here as needed -----
