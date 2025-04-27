from ..backend import tf
from .surrogate import Surrogate


class FNN(Surrogate):
    """Fully-connected neural network that works as a Python callable."""

    def __init__(
        self, layers, activation=tf.tanh, input_transform=None, output_transform=None,
    ):
        self.L = len(layers) - 1
        self.activation = activation
        self._input_transform = input_transform
        self._output_transform = output_transform

    def __call__(self, inputs, var_list):
        return self.forward(inputs, var_list)

    def forward(self, inputs, var_list):
        w, b = var_list[: self.L], var_list[self.L :]

        if self.input_transform is not None:
            outputs = self.input_transform(inputs)
        else:
            outputs = inputs

        for i in range(self.L - 1):
            outputs = self.activation(tf.matmul(outputs, w[i]) + b[i])
        outputs = tf.matmul(outputs, w[-1]) + b[-1]

        if self.output_transform is not None:
            outputs = self.output_transform(outputs)

        return outputs
