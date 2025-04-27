import numpy as np


from .process import GlobalProcesses
from .backend import backend_name, tf, jax


class Model:
    """
    A model collects processes and likelihoods, and then performs inferences.
    """

    def __init__(self, processes, likelihoods):
        # create a instance of GlobalProcess, for control over all used processes
        self.global_processes = GlobalProcesses()
        self.global_processes.update(processes)

        self.processes = processes
        self.likelihoods = likelihoods

        print("Supporting backend " + str(backend_name) + "\n")
        if backend_name == "tensorflow.compat.v1":
            self.session = tf.Session()

        self.trainable_variables = []
        for p in processes:
            # warning: use key to collect trainable_variables
            if p.trainable_variables is not None:
                self.trainable_variables += p.trainable_variables

        self.method = None

    def compile(self, method):
        """Compiles the Bayesian model with a method"""
        # different method has different way of compiling
        print("Compiling a {} method\n".format(method.method_type))
        if method.method_type == "MCMC":
            self._compile_mcmc(method)
        else:
            raise NotImplementedError(
                "Support for {} to be implemented.".format(method.method_type)
            )

    def run(self):
        """Performs posterior estimate over the Bayesian model with a compiled method"""
        if self.method is None:
            raise ValueError("Model has not been compiled with a method.")
        sess = None
        if backend_name == "tensorflow.compat.v1":
            sess = self.session
        return self.method.sampling(sess)

    def predict(self, inputs, samples, processes, equation=None):
        """
        Performs prediction over `processes` at `inputs`, given posterior samples stored in
        `samples` as a Python dictionary. Every process in `processes` needs to be stored in
        the model.
        If `pde_fn` is not None, then the prediction is on the quantity defined by `pde_fn`,
        and `processes` has be stored in order in a list such that, together with `inputs`,
        it matches the arguments of `pde_fn`.
        If `pde_fn` is None, then the prediction is on all processes in `processes`.
        """
        sample_size = samples[0].shape[0]
        # TODO: support Variational and Trainable setups.
        # convert to tensor first, for tensorflow.compat.v1 and further computation on
        # derivatives.
        if backend_name in ["tensorflow", "tensorflow.compat.v1"]:
            inputs = tf.constant(inputs, tf.float32)
            inputs = tf.tile(inputs[None, ...], [sample_size, 1, 1])
            samples = [tf.constant(s, tf.float32) for s in samples]
        elif backend_name == "jax":
            inputs = jax.numpy.array(inputs, jax.numpy.float32)
            inputs = jax.numpy.tile(inputs[None, ...], [sample_size, 1, 1])

        # assign samples to processes
        samples_dict = self.global_processes.assign(samples)

        if equation is None:
            predictions = [p.surrogate(inputs, samples_dict[p.key]) for p in processes]
        else:
            # Same thing happens in the likelihood function
            # Be careful on wrapping functions! It is very important! Remember figuring this out!

            def make_fn(process):
                # This is very important!! Making functions this way can prevent modifications
                # from global variables.
                return lambda x: process.surrogate(x, samples_dict[process.key])

            args = []
            for p in processes:
                # args += [lambda x: p.surrogate(x, samples_dict[p.key])]
                args += [make_fn(p)]
            if backend_name == "tensorflow":
                predictions = [tf.function(equation)(inputs, *args)]
            elif backend_name == "tensorflow.compat.v1":
                predictions = [equation(inputs, *args)]
            elif backend_name == "jax":
                predictions = [
                    jax.vmap(equation, in_axes=(1, None), out_axes=(1))(inputs, *args)
                ]

        if backend_name == "tensorflow":
            predictions = [v.numpy() for v in predictions]
        elif backend_name == "tensorflow.compat.v1":
            predictions = self.session.run(predictions)

        return predictions

    def _compile_mcmc(self, method):
        """Compiles the model with a MCMC-type inference method"""
        # build log posterior function
        def log_posterior_fn(*var_list):
            # The computation of log probabilistic density of posterior distribution is
            # decomposed into three steps.
            # Step 1: assign var_list to corresponding processes, e.g. BNNs, constants.
            # Step 2: for each process, compute its prior
            # Step 3: for each likelihood, compute its likelihood
            global_var_dict = self.global_processes.assign(var_list)
            if backend_name in ["tensorflow", "tensorflow.compat.v1"]:
                log_prior = []
                for key, p in self.global_processes.processes.items():
                    log_prior += [tf.reduce_sum(p.prior.log_prob(global_var_dict[key]))]
                log_prior = tf.reduce_sum(log_prior)

                log_likeli = tf.reduce_sum(
                    [lh.log_prob(global_var_dict) for lh in self.likelihoods]
                )
            elif backend_name == "jax":
                log_prior = []
                for key, p in self.global_processes.processes.items():
                    log_prior += [jax.numpy.sum(p.prior.log_prob(global_var_dict[key]))]
                log_prior = jax.numpy.sum(jax.numpy.array(log_prior),)

                log_likeli = jax.numpy.sum(
                    jax.numpy.array(
                        [lh.log_prob(global_var_dict) for lh in self.likelihoods]
                    )
                )
            return log_prior + log_likeli

        # compile the method
        method.make_sampler(log_posterior_fn, self.global_processes.initial_values)
        # assign the method
        self.method = method
