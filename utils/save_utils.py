import os
from datetime import datetime


def save_model_and_logs(model, logger, opt):
    """
    Save the model, logs, and SHAP plots with date and time.

    Parameters
    ----------
    model : object
        The trained model to save.
    logger : Logger
        The logger instance to save logs.
    opt : Namespace
        The namespace object containing the experiment options.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = opt.model_name

    model_filename = f"{model_name}_{now}_model.pkl"
    log_filename = f"{model_name}_{now}_Logs.txt"

    # Ensure the directories exist
    os.makedirs(opt.saved_model_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)

    # Save the model
    model.analysis_and_save_model(os.path.join(opt.saved_model_path, model_filename))

    # Save the logs
    logger.save_pretty_log(os.path.join(opt.log_path, log_filename))
