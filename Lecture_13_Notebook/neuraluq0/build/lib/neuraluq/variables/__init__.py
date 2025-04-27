"""Package for variables module."""

__all__ = ["fnn", "const"]


from ..backend import backend_name
from .variables import _Samplable


if backend_name in ["tensorflow", "tensorflow.compat.v1"]:
    from . import fnn_tf as fnn
    from . import const_tf as const
elif backend_name == "jax":
    from . import fnn_jax as fnn
else:
    raise NotImplementedError(
        "Backend {} does not support fully-connected neural network.".format(
            backend_name
        )
    )
