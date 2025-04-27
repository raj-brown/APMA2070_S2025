"""Package for method modules."""

__all__ = ["Inference", "HMC"]

from .inference import Inference
from .mcmc import HMC, NUTS
