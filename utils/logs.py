import json
from datetime import datetime


class Logger:
    """
    This class is used to log information during the data processing,
    model training, and model evaluation stages.

    Parameters
    ----------
    opt : Namespace
        The namespace object containing the experiment options.

    Attributes
    ----------
    log_info : dict
        A dictionary to store the log information.

    """

    # ----- Add new section of the logs here ----->

    def __init__(self, opt):
        self._opt = opt
        self.log_info = {
            "data_loading": {},
            "data_processing": {},
            "Correlation_Analysis": {},
            "Chi_Square_Scores": {},
            "train_test_split": {},
            "model_training": {},
            "model_evaluation": {},
            "model_tuning": {},
            "model_tuning_evaluation": {},
            "Model Performance Analysis & Saving": {},
        }
        self.start_time = datetime.now()

    def update_log(self, section, key, value):
        """
        This method updates the log information with the given section, key, and value.

        Parameters
        ----------
        section : str
            The section of the log to update.

        key : str
            The key within the section to update.

        value : str, int, float, dict
            The value to update in the log.

        """
        if section in self.log_info:
            if isinstance(self.log_info[section], dict):
                self.log_info[section][key] = value
            else:
                print(f"Error: Section {section} is not a dictionary.")
        else:
            self.log_info[section] = {key: value}

    def save_log(self, file_name):
        """
        This method saves the log information to a JSON file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the log information.
        """
        self.log_info["log_created_at"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(file_name, "w") as f:
            json.dump(self.log_info, f, indent=4)
        print(f"Log saved to {file_name}")

    def save_pretty_log(self, file_name="pretty_log.txt"):
        """
        This method saves the log information to a text file in a pretty format.

        Parameters
        ----------
        file_name : str
            The name of the file to save the log information
        """
        self.log_info["log_created_at"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(file_name, "w") as f:
            f.write(self.format_pretty_log())
        print(f"Logs saved to {file_name}")

    def format_pretty_log(self):
        """
        This method formats the log information in a pretty format.

        Returns
        -------
        str
            A string containing the formatted log information.
        """
        pretty_log = []
        pretty_log.append("=" * 90)
        pretty_log.append(
            " " * 10 + f"{self._opt.experiment_name} LOG REPORT" + " " * 10
        )
        pretty_log.append("=" * 90)
        pretty_log.append(f"\nLog created at: {self.log_info['log_created_at']}")
        pretty_log.append(
            f"\nModel Utilised for the Experiment: {self._opt.model_name}\n"
        )
        pretty_log.append("=" * 90)

        for section, entries in self.log_info.items():
            if section != "log_created_at":
                if isinstance(entries, dict):
                    pretty_log.append(f"{section.replace('_', ' ').upper()}:\n")
                    for key, value in entries.items():
                        pretty_log.append("")
                        if isinstance(value, dict):
                            pretty_log.append(
                                f"  {key.replace('_', ' ').capitalize()}:\n"
                            )
                            for sub_key, sub_value in value.items():
                                pretty_log.append(f"    {sub_key}: {sub_value}")
                        else:
                            pretty_log.append(
                                f"  {key.replace('_', ' ').capitalize()}: {value}"
                            )
                        pretty_log.append("")
                    pretty_log.append("-" * 80)
                else:
                    pretty_log.append(
                        f"{section.replace('_', ' ').upper()}: {entries}\n"
                    )

        pretty_log.append("=" * 80)

        return "\n".join(pretty_log)
