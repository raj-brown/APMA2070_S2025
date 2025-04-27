"""
This file contains configurations of the library. Currently it supports backend setting
(tensorflow & jax), data type, device for training (CPU & GPU).
"""


from . import backend

# choose and set backend and backend name
# TODO: dangerous behavior. To be done in safer ways.
backend.backend_name = backend.get_backend_name()
backend.tf, backend.jax, backend.tfp = backend.set_backend(backend.backend_name)

# after backend is chosen and set, import other modules
from . import likelihoods
from . import variables
from . import surrogates
from . import inferences
from . import utils


from .models import Model
from .process import Process
