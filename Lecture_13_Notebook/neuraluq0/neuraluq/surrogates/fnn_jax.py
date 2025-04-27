from ..backend import jax
from .surrogate import Surrogate


import flax


from typing import Any
from jax.flatten_util import ravel_pytree


class FNN(Surrogate):
    """Fully-connected neural network that works as a Python callable."""

    def __init__(
        self,
        layers,
        activation=jax.numpy.tanh,
    ):
        self.L = len(layers) - 1
        self.nn = _FNN(layers=layers, activation=activation)
        # initialize nn and set pytree reconstruction function
        params = self.nn.init(jax.random.PRNGKey(0), jax.numpy.ones([1, layers[0]]))
        _, self.pytree_fn = ravel_pytree(params)

    def __call__(self, inputs, var_list):
        return self.forward(inputs, var_list)

    def _forward(self, inputs, _flat_params):
        params = self.pytree_fn(_flat_params)
        return self.nn.apply(params, inputs)

    def forward(self, inputs, var_list):
        flat_params = var_list[0]

        if len(flat_params.shape) == 1:
            # for inference
            outputs = self._forward(inputs, flat_params)
        elif len(flat_params.shape) == 2:
            # for testing
            outputs = jax.vmap(self._forward, in_axes=(0, 0), out_axes=(0))(
                inputs, flat_params
            )
        else:
            raise ValueError("Shape is not supported.")

        return outputs


class _FNN(flax.linen.Module):
    """Fully-connected neural network, using flax."""

    layers: Any
    activation: Any = jax.numpy.tanh

    def setup(self):
        initializer = jax.nn.initializers.zeros

        self.denses = []
        for unit in self.layers[1:]:
            self.denses += (
                flax.linen.Dense(
                    features=unit,
                    kernel_init=jax.nn.initializers.glorot_normal(),
                    bias_init=jax.nn.initializers.zeros,
                ),
            )

    def __call__(self, inputs):
        outputs = inputs
        for linear in self.denses[:-1]:
            outputs = self.activation(linear(outputs))
        return self.denses[-1](outputs)
