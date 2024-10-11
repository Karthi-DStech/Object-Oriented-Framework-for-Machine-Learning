import argparse
from typing import Dict
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseOptions:
    """
    This class defines options used during all types of experiments.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseOptions class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._parser = argparse.ArgumentParser()
        self._none_or_int = self.none_or_int
        self.initialized = False

    def initialize(self) -> None:
        self._parser.add_argument(
            "--experiment_name",
            type=str,
            default="Predicting the Stage of Liver Cirrhosis",
            help="Name of the experiment",
        ),

        self._parser.add_argument(
            "--data_path",
            type=str,
            default="/Users/karthik/Desktop/cirrhosis.csv",
            help="Path to the data file",
        )

        self._parser.add_argument(
            "--wrapper_width",
            type=int,
            default=50,
            help="Width of the wrapper for the logs",
        )

        self._parser.add_argument(
            "--saved_model_path",
            type=str,
            default="/Users/karthik/My-Github-Repos/ml-indus/artifacts/models/",
            help="Path to save the trained model",
        )

        self._parser.add_argument(
            "--log_path",
            type=str,
            default="/Users/karthik/My-Github-Repos/ml-indus/artifacts/logs/",
            help="Path to save the logs",
        )

        self.initialized = True

    def parse(self):
        """
        Parses the arguments passed to the script

        Parameters
        ----------
        None

        Returns
        -------
        opt: argparse.Namespace
            The parsed arguments
        """
        if not self.initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        args = vars(self._opt)
        self._print(args)

        return self._opt

    def _print(self, args: Dict) -> None:
        """
        Prints the arguments passed to the script

        Parameters
        ----------
        args: dict
            The arguments to print

        Returns
        -------
        None
        """
        print("------------ Options -------------")
        for k, v in args.items():
            print(f"{str(k)}: {str(v)}")
        print("-------------- End ---------------")

    def none_or_int(value):
        """
        This function checks if the value is an integer or None.

        Parameters
        ----------
        value : str
            The value to check.

        Returns
        -------
        int
            The integer value.

        None
            If the value is None.
        """
        if value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value: {value}")
