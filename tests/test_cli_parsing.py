"""Regression tests for command-line value parsing."""

import argparse

import pytest

from monte_carlo_single_instance import str2bool


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "YeS"])
def test_str2bool_true_values(value):
    assert str2bool(value) is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "No"])
def test_str2bool_false_values(value):
    assert str2bool(value) is False


def test_str2bool_rejects_unknown_value():
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")
