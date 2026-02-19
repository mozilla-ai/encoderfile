import pytest
import encoderfile_py


def test_sum_as_string():
    assert encoderfile_py.sum_as_string(1, 1) == "2"
