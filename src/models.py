"""
Model training script with modular functions for better testability.

This script contains functions for creating, training, evaluating, saving, and loading
a BaseEstimator model for the Iris dataset. It also includes functions for
saving and loading associated metadata.
"""

import os
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = "models"


def create_model(n_estimators: int = 100, random_state: int = 42) -> BaseEstimator:
    """
    Create a BaseEstimator model.

    Parameters
    ----------
    n_estimators : int, optional
        The number of trees in the forest, by default 100
    random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples used
        when building trees and the sampling of the features to consider when
        looking for the best split at each node, by default 42

    Returns
    -------
    BaseEstimator
        An instance of BaseEstimator with the specified parameters
    """
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


def train_model(
    model: BaseEstimator, x_train: np.ndarray, y_train: np.ndarray
) -> BaseEstimator:
    """
    Train the model on the given data.

    Parameters
    ----------
    model : BaseEstimator
        The model to be trained
    X_train : np.ndarray
        The training input samples
    y_train : np.ndarray
        The target values

    Returns
    -------
    BaseEstimator
        The trained model
    """
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """
    Evaluate the model and return train and test scores.

    Parameters
    ----------
    model : BaseEstimator
        The trained model to evaluate
    X_train : np.ndarray
        The training input samples
    y_train : np.ndarray
        The target values for training data
    X_test : np.ndarray
        The testing input samples
    y_test : np.ndarray
        The target values for testing data

    Returns
    -------
    Tuple[float, float]
        A tuple containing the train score and test score
    """
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    return train_score, test_score


def save_model(
    model: RandomForestClassifier, filename: str = "iris_model.joblib"
) -> None:
    """
    Save the trained model to a file.

    Parameters
    ----------
    model : RandomForestClassifier
        The trained model to save
    filename : str, optional
        The name of the file to save the model to, by default "iris_model.joblib"
    """
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)


def save_metadata(feature_names: List[str], target_names: List[str]) -> None:
    """
    Save feature names and target names to files.

    Parameters
    ----------
    feature_names : List[str]
        The names of the features
    target_names : List[str]
        The names of the target classes
    """
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    target_names_path = os.path.join(MODELS_DIR, "target_names.joblib")

    joblib.dump(feature_names, feature_names_path)
    joblib.dump(target_names, target_names_path)


def load_model(
    model_filename: str = "iris_model.joblib",
    feature_filename: str = "feature_names.joblib",
    target_filename: str = "target_names.joblib",
) -> Tuple[RandomForestClassifier, List[str], List[str]]:
    """
    Load the model and associated metadata.

    Parameters
    ----------
    model_filename : str, optional
        The name of the file containing the saved model, by default "iris_model.joblib"
    feature_filename : str, optional
        The name of the file containing the feature names, by default "feature_names.joblib"
    target_filename : str, optional
        The name of the file containing the target names, by default "target_names.joblib"

    Returns
    -------
    Tuple[RandomForestClassifier, List[str], List[str]]
        A tuple containing the loaded model, feature names, and target names
    """
    model_path = os.path.join(MODELS_DIR, model_filename)
    feature_names_path = os.path.join(MODELS_DIR, feature_filename)
    target_names_path = os.path.join(MODELS_DIR, target_filename)

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    target_names = joblib.load(target_names_path)
    return model, feature_names, target_names
