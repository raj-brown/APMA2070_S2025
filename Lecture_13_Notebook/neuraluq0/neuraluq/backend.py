"""
This file contains backend configuration of the library. Currently it supports three backends, 
Tensorflow 1, Tensorflow 2, and JAX, each of which has a specific compatible Tensorflow Probability 
library.
"""


import os
import json


# store backend name, interfaces to necessary libraries here
backend_name = None
tf, jax, tfp = None, None, None


def set_backend(backend_name):
    """Sets the backend, given `backend_name`."""
    if backend_name not in ["tensorflow", "tensorflow.compat.v1", "jax"]:
        raise ValueError("Backend name {} is not supported.".format(backend_name))

    tf, jax, tfp = None, None, None

    if backend_name == "tensorflow":
        import tensorflow as tf
        import tensorflow_probability as tfp
    elif backend_name == "tensorflow.compat.v1":
        import tensorflow.compat.v1 as tf

        tf.disable_eager_execution()

        import tensorflow_probability as tfp
    elif backend_name == "jax":
        import jax
        import tensorflow_probability.substrates.jax as tfp
    else:
        raise ValueError("Backend {} is not supported.".format(backend_name))

    return tf, jax, tfp


def get_backend_name():
    """Fetches the backend name from user-specified file."""
    if not os.path.exists("./config.json"):
        print(
            "Backend is chosen to be the default, tensorflow.",
            "You can customize the backend by change it in the 'config.json' file,",
            "which can be created or found in your current directory.",
        )
        backend_name = "tensorflow"
    else:
        with open("./config.json", "r") as file:
            config_dict = json.load(file)
            backend_name = config_dict.get("backend")
        if backend_name not in ["tensorflow", "tensorflow.compat.v1", "jax"]:
            raise ValueError("Backend name {} is not supported.".format(backend_name))
    return backend_name
