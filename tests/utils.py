import unittest
from collections import abc


def assert_len(container, expected_len, msg=None):
    """Asserts that an object has the expected length.
    Args:
      container: Anything that implements the collections.abc.Sized interface.
      expected_len: The expected length of the container.
      msg: Optional message to report on failure.
    """
    assert isinstance(
        container, abc.Sized
    ), "Expected a Sized object, got: " "{!r}".format(type(container).__name__) + (
        "" if msg is None else msg
    )
    assert len(container) == expected_len, "{} has length of {}, expected {}.".format(
        unittest.util.safe_repr(container), len(container), expected_len
    ) + ("" if msg is None else msg)
