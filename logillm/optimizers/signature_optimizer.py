"""SignatureOptimizer - deprecated alias for COPRO.

This module provides backward compatibility for code that expects
a SignatureOptimizer class. New code should use COPRO directly.
"""

import warnings

from .copro import COPRO


class SignatureOptimizer(COPRO):
    """Deprecated alias for COPRO optimizer.

    This class is provided for backward compatibility.
    New code should use COPRO directly.

    Example:
        # Old way (deprecated)
        optimizer = SignatureOptimizer(metric=accuracy_metric)

        # New way (preferred)
        optimizer = COPRO(metric=accuracy_metric)
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SignatureOptimizer is deprecated and will be removed in a future version. "
            "Use COPRO instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["SignatureOptimizer"]
