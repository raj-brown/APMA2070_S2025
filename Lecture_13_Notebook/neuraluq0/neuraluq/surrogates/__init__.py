"""Package for surrogates module."""

__all__ = [
    "Surrogate",
    "FNN",
    "Identity",
]

from .surrogate import Surrogate, Identity
from ..backend import backend_name


if backend_name == "jax":
    from .fnn_jax import FNN
elif backend_name in ["tensorflow", "tensorflow.compat.v1"]:
    from .fnn_tf import FNN
else:
    raise NotImplementedError(
        "Backend {} does not support fully-connected neural network.".format(
            backend_name
        )
    )
