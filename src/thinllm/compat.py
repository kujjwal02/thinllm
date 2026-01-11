"""Compatibility utilities for different Python versions."""

import sys
from enum import Enum

# StrEnum is only available in Python 3.11+
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """String enum for Python < 3.11 compatibility."""


__all__ = ["StrEnum"]
