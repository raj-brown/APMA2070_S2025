from ..backend import tf, tfp
from .variables import _Samplable


class Samplable(_Samplable):
    """
    Samplable type variable of a fully-connected neural network with independent
        Normal distributions.
    """

    def __init__(
        self, layers, mean, sigma=0.1, initializer=None,
    ):
        super().__init__()

        mean, sigma = tf.constant(mean, tf.float32), tf.constant(sigma, tf.float32)
        self._num_tensors = 2 * (len(layers) - 1)
        # TODO: support other initializers
        w_init = tf.keras.initializers.zeros()
        b_init = tf.keras.initializers.zeros()

        self._initial_values = []
        self.dists = []
        for i in range(len(layers) - 1):
            shape = [layers[i], layers[i + 1]]
            # add one axis before axis 0, for MCMC sampler
            self._initial_values += [w_init(shape=shape, dtype=tf.float32)]
            _mean, _sigma = mean * tf.ones(shape=shape), sigma * tf.ones(shape=shape)
            self.dists += [tfp.distributions.Normal(loc=_mean, scale=_sigma)]
            # self.dists += [tfp.distributions.Laplace(loc=mean, scale=sigma)]
        for i in range(len(layers) - 1):
            shape = [1, layers[i + 1]]
            # add one axis before axis 0, for MCMC sampler
            self._initial_values += [b_init(shape=shape, dtype=tf.float32)]
            _mean, _sigma = mean * tf.ones(shape=shape), sigma * tf.ones(shape=shape)
            self.dists += [tfp.distributions.Normal(loc=_mean, scale=_sigma)]

    def log_prob(self, samples):
        # For NN, `samples` is a list of tensors, each one which is of shape [N, :, :], where N
        # is the sample size. Hence, the log probability should be of shape [N]
        _log_prob = tf.zeros(shape=[samples[0].shape[0]])
        for s, dist in zip(samples, self.dists):
            _log_prob += tf.reduce_sum(dist.log_prob(s), axis=[-1, -2])
        return _log_prob

    def sample(self, sample_shape=[]):
        return [dist.sample(sample_shape=sample_shape) for dist in self.dists]
