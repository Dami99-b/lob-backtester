import pytest

# Tests for code you haven't written yet are marked `todo`: they are expected to
# fail with NotImplementedError. Any other error is a real failure. Delete the
# marker from a test once its code exists.
todo = pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="not implemented yet")
