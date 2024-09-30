"""
Main script to train and save the model.
"""

from src.data import load_data, split_data
from src.models import (
    create_model,
    evaluate_model,
    save_metadata,
    save_model,
    train_model,
)


def main() -> None:
    """
    Main function to train and save the model.

    This function loads the data, splits it into training and testing sets,
    creates and trains a model, evaluates it, and then saves the model and metadata.
    """
    # pylint: disable=invalid-name

    # Load data
    x, y, feature_names, target_names = load_data()

    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Create and train model
    model = create_model()
    trained_model = train_model(model, x_train, y_train)

    # Evaluate model
    train_score, test_score = evaluate_model(
        trained_model, x_train, y_train, x_test, y_test
    )
    print(f"Train score: {train_score:.2f}")
    print(f"Test score: {test_score:.2f}")

    # Save model and metadata
    save_model(trained_model)
    save_metadata(feature_names, target_names)

    print("Model trained and saved successfully.")


if __name__ == "__main__":
    main()
