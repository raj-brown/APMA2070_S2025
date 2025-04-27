from ..backend import tf, tfp
from .variables import _Samplable


class Samplable(_Samplable):
    """
    Samplable type variable of a fully-connected neural network with independent
        Normal distributions.
    """

    def __init__(
        self, mean, sigma=1.0, shape=[], initializer=None,
    ):
        super().__init__()

        mean, sigma = tf.constant(mean, tf.float32), tf.constant(sigma, tf.float32)
        self._num_tensors = 1
        if initializer is None:
            init = tf.keras.initializers.zeros()
        else:
            init = initializer
        
        self._initial_values = [init(shape=mean.shape)]
        if mean.shape == []:
            # make sure constant has at least 2 dimension
            self._initial_values = [self.initial_values[0][None, None, ...]]
        self.dist = tfp.distributions.Normal(loc=mean, scale=sigma)

    def log_prob(self, samples):
        # Note: here, because a constant is considered, `samples` is a list of only
        # one element._log_prob = tf.zeros(shape=[samples[0].shape[0]])
        return tf.reduce_sum(self.dist.log_prob(samples[0]))

    def sample(self, sample_shape):
        return [self.dist.sample(sample_shape=sample_shape)]