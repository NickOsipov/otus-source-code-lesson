"""
Module: metrics.py
Description: This module contains functions to calculate metrics.
"""

from typing import List, Union


def calculate_mean(
    numbers: List[Union[int, float]],
    flag: bool = False,
    plus_target: Union[int, float] = 0,
) -> float:
    """
    Function for calculating the mean of a list of numbers.
    Returns the mean of the numbers.

    Parameters
    ----------
    numbers : List[Union[int, float]]
        List of numbers to calculate the mean from.
    flag : bool, optional
        Flag to add a target to the numbers, by default False
    plus_target : Union[int, float], optional
        Target to add to the numbers

    Returns
    -------
    float
        The mean of the numbers.

    Notes
    -----
    The mean is calculated as the sum of the numbers
    divided by the count of the numbers.

    Examples
    --------
    >>> calculate_mean([1, 2, 3, 4, 5])
    3.0
    """

    if flag:
        numbers = [x + plus_target for x in numbers]

    return sum(numbers) / len(numbers)
