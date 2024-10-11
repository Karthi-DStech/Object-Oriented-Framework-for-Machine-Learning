from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


class TrainTestProcessor:
    """
    This class provides methods for splitting the data into training
    and testing sets and applying standard scaling.
    """

    def __init__(self, data, logger, opt):
        """
        Initialize the TrainTestProcessor object with the data and target column.

        Parameters
        ----------
        data : DataFrame
            The processed data.

        logger : Logger
            The logger instance for logging information.

        opt : Namespace
            Object containing the experiment options.
        """
        self._opt = opt
        self.data = data
        self.logger = logger

    def split_data(self, test_size):
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        test_size : float, optional
            The proportion of the dataset to include in the test split (default is 0.2).

        Returns
        -------
        X_train, X_test, y_train, y_test : DataFrame
            The training and testing sets for the features and target.
        """
        X = self.data.drop(columns=[self._opt.target_column])
        y = self.data[self._opt.target_column]
        self.logger.update_log(
            "train_test_split", "target_column", self._opt.target_column
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self._opt.test_size, random_state=self._opt.random_state
        )
        print(f"Data split into train and test sets with test size = {test_size} \n")

        # Log split information
        self.logger.update_log("train_test_split", "test_size", test_size)
        self.logger.update_log("train_test_split", "train_size", len(X_train))
        self.logger.update_log("train_test_split", "test_size_count", len(X_test))

        return X_train, X_test, y_train, y_test

    def standard_scale(self, X_train, X_test):
        """
        Apply standard scaling to the features.

        Parameters
        ----------
        X_train : DataFrame
            The training set features.
        X_test : DataFrame
            The testing set features.

        Returns
        -------
        X_train_scaled, X_test_scaled : DataFrame
            The scaled training and testing set features.
        """
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("Features have been standard scaled.\n")

        # Log scaling information
        self.logger.update_log("train_test_split", "scaling", "StandardScaler")

        return X_train_scaled, X_test_scaled

    def process(self):
        """
        Process the data by splitting into training and testing sets and optionally applying standard scaling.

        Parameters
        ----------
        None

        Returns
        -------
        Scaling applied:
            If scaling is applied, returns the scaled training and
            testing sets for the features and target.

        No scaling:
            If scaling is not applied, returns the training and
            testing sets for the features and target.
        """
        X_train, X_test, y_train, y_test = self.split_data(
            test_size=self._opt.test_size
        )

        if self._opt.scale_data:
            X_train, X_test = self.standard_scale(X_train, X_test)

            self.logger.update_log("train_test_split", "scaling", "StandardScaler")
            print("Data has been scaled.\n")
            return X_train, X_test, y_train, y_test
        else:
            self.logger.update_log("train_test_split", "scaling", "None")
            return X_train, X_test, y_train, y_test

    def final_checks(self, X_train, X_test, y_train, y_test):
        """
        Perform final checks to ensure that the data is correctly processed.

        Parameters
        ----------
        X_train : DataFrame
            The training set features.
        X_test : DataFrame
            The testing set features.
        y_train : DataFrame
            The training set target.
        y_test : DataFrame
            The testing set target.
        scale_data : bool
            Whether scaling was applied to the features.

        Raises
        ------
        AssertionError
            If any of the checks fail.
        """
        assert len(X_train) > 0, "X_train should not be empty"
        assert len(X_test) > 0, "X_test should not be empty"
        assert len(y_train) > 0, "y_train should not be empty"
        assert len(y_test) > 0, "y_test should not be empty"
        assert (
            X_train.shape[1] == X_test.shape[1]
        ), "Number of features in X_train and X_test should be the same"
        assert (
            y_train.shape[0] == X_train.shape[0]
        ), "X_train and y_train should have the same number of samples"
        assert (
            y_test.shape[0] == X_test.shape[0]
        ), "X_test and y_test should have the same number of samples"

        print("All checks passed successfully.")

        print("-" * 40 + "\n")
        print("Machine Learning Training Begins... \n")

        # Log final checks
        self.logger.update_log(
            "train_test_split", "checks", "All checks passed successfully"
        )
