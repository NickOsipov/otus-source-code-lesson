"""
Module for loading and splitting the Iris dataset.

This module provides functions to load the Iris dataset and split it into
training and testing sets.
"""

from typing import List, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# pylint: disable=no-member
def load_data() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the Iris dataset.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str], List[str]]
        A tuple containing:
        - Features (X): 2D array of shape (n_samples, n_features)
        - Target (y): 1D array of shape (n_samples,)
        - Feature names: List of strings describing each feature
        - Target names: List of strings describing each target class
    """
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    X : np.ndarray
        The input samples.
    y : np.ndarray
        The target values.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split,
        by default 42.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing train-test split of inputs:
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
