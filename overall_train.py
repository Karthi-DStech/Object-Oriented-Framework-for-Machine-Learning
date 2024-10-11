from options.train_options import TrainOptions
from utils.overall_setup import overall_training_setup

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run() -> None:
    """
    Run the overall training process for multiple models and log their performance.
    """
    # Create an instance of TrainOptions
    train_options = TrainOptions()

    # Initialize the parser to populate options
    train_options.initialize()

    # Parse the training options
    opt = TrainOptions().parse()

    # Get all the model names available from TrainOptions
    model_choices = train_options._parser._option_string_actions["--model_name"].choices

    print("Available Model Choices:")

    for model in model_choices:
        print(f"Training model: {model}")

        opt.model_name = model

        model_instance = overall_training_setup(opt)

        if model_instance and model_instance.training_evaluation:

            training_evaluation = model_instance.training_evaluation[-1]
            print(f"\nTraining evaluation for {model}: {training_evaluation}\n")

        if model_instance and model_instance.tuning_evaluation:
            # Access the tuning evaluation data without logging
            tuning_evaluation = model_instance.tuning_evaluation[-1]
            print(f"\nTuning evaluation for {model}: {tuning_evaluation}\n")

    # Print that the training process is complete
    print(f"\nTraining process for {model} is complete.\n")


if __name__ == "__main__":
    run()
