from ..backend import tfp
from .variables import _Samplable


import jax


class Samplable(_Samplable):
    """
    Samplable type variable of a fully-connected neural network with independent
        Normal distributions.
    """

    def __init__(
        self, layers, mean, sigma=0.1,
    ):
        super().__init__()
        self._num_tensors = 1
        dims = 0
        for i in range(len(layers) - 1):
            dims += layers[i] * layers[i + 1] + layers[i + 1]
        self._initial_values = [jax.numpy.zeros(shape=[dims,])]
        self.dist = tfp.distributions.Normal(loc=mean, scale=sigma)

    def log_prob(self, samples):
        return jax.numpy.sum(self.dist.log_prob(samples[0]))
