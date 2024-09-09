"""
Test module for the metrics module.
"""

from typing import List

import pytest
from src.metrics import calculate_mean


@pytest.mark.parametrize(
    "numbers, expected",
    [
        ([1, 2], 1.5),
        ([1, 2, 3], 2.0),
        ([1, 2, 3, 4, 5], 3.0),
    ],
)
def test_calculate_mean(numbers: List[int], expected: float) -> None:
    """
    Test the calculate_mean function.
    """
    assert calculate_mean(numbers) == expected
