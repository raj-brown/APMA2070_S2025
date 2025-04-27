import numpy as np


from .backend import backend_name, tf, jax, tfp


class Loss:
    """Base class for all likelihoods and losses"""

    def __init__(self):
        self._inputs = None
        self._targets = None
        self._processes = []
        self._in_dims = None
        self._out_dims = None
        self._equation = None
        self._trainable_variables = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def processes(self):
        return self._processes

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def equation(self):
        return self._equation


class Normal(Loss):
    """Independent Normal distribution for likelihood over all observations"""

    def __init__(
        self,
        inputs,
        targets,
        processes,
        equation=None,
        in_dims=None,
        out_dims=None,
        sigma=0.1,
    ):
        """Initializes distribution"""
        super().__init__()

        if not isinstance(processes, list):
            processes = [processes]
        self._processes = processes

        self._in_dims = in_dims if in_dims is not None else len(processes) * [None]
        self._out_dims = out_dims if out_dims is not None else len(processes) * [None]

        self._equation = equation
        # build the distribution

        if backend_name in ["tensorflow", "tensorflow.compat.v1"]:
            self._equation = equation

            self._inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
            self._targets = tf.convert_to_tensor(targets, dtype=tf.float32)
            self.sigma = tf.constant(sigma, dtype=tf.float32)

            def _log_prob(x):
                return tf.reduce_sum(
                    -tf.math.log(tf.math.sqrt(2 * np.pi) * self.sigma)
                    - x ** 2 / self.sigma ** 2 / 2
                )

        elif backend_name == "jax":
            if equation is not None:
                self._equation = jax.vmap(equation, in_axes=(0, None), out_axes=(0))

            self._inputs = inputs
            self._targets = targets
            self.sigma = sigma

            def _log_prob(x):
                return jax.numpy.sum(
                    -jax.numpy.log(jax.numpy.sqrt(2 * np.pi) * self.sigma)
                    - x ** 2 / self.sigma ** 2 / 2
                )

        else:
            raise NotImplementedError(
                "Backend {} is not supported.".format(backend_name)
            )

        self._log_prob = _log_prob

    def log_prob(self, global_var_dict):
        # it's necessary to declare inputs here, to prevent bugs in MCMC methods
        inputs = tf.constant(self.inputs, tf.float32)
        targets = tf.constant(self.targets, tf.float32)
        if self.equation is None:
            # for direct observation, where only a single process is present.
            # TODO: support in_dims and out_dims
            p = self.processes[0]
            outputs = p.surrogate(inputs, global_var_dict[p.key])
        else:
            # for observation through PDE
            # TODO: support in_dims and out_dims
            args = []  # arguments of the equation

            def make_fn(process):
                # This is very important!! Making functions this way can prevent modifications
                # from global variables.
                return lambda x: process.surrogate(x, global_var_dict[process.key])

            for p in self.processes:
                # TODO: figure this out and delete old ones.
                # careful!!!! overwritten happens.
                # args += [lambda x: p.surrogate(x, global_var_dict[p.key])]
                args += [make_fn(p)]
            outputs = self.equation(inputs, *args)
        return self._log_prob(outputs - targets)
