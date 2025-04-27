"""This file contains MCMC sampling inference methods, e.g. Hamiltonian Monte Carlo."""


from .inference import Inference
from ..backend import backend_name, tf, jax, tfp


class MCMC(Inference):
    """Base class for all MCMC inference methods."""

    def __init__(self):
        super().__init__()
        self._params = {"seed": None}
        self.method_type = "MCMC"
        self.mcmc_kernel = None
        self.trace_fn = lambda _, pkr: pkr

    def set_sampler(self, init_state):
        if self.mcmc_kernel is None:
            raise ValueError("MCMC kernel is not found.")

        def sampler():
            samples, results = tfp.mcmc.sample_chain(
                num_results=self.params["num_samples"],
                num_burnin_steps=self.params["num_burnin"],
                current_state=init_state,
                kernel=self.mcmc_kernel,
                trace_fn=self.trace_fn,
                seed=self.params["seed"],
            )
            return samples, results

        # set sampler
        if backend_name == "tensorflow":
            # tf.function makes the function executed in graph mode
            self.sampler = tf.function(sampler)
        elif backend_name == "tensorflow.compat.v1":
            self.sampler = sampler
        elif backend_name == "jax":
            self.sampler = jax.jit(sampler)
        else:
            raise NotImplementedError(
                "Backend {} is not supported for MCMC inference methods.".format(
                    backend_name
                )
            )

    def sampling(self, sess=None):
        """Perform sampling with Hamiltonian Monte Carlo."""
        print("sampling from posterior distribution ...\n")
        if self.sampler is None:
            raise ValueError("Sampler is not found.")
        if sess is not None:
            samples, results = sess.run(self.sampler())
        else:
            samples, results = self.sampler()
        print("Finished sampling from posterior distribution ...\n")
        return samples, results


class HMC(MCMC):
    """Adaptive Hamiltonian Monte Carlo inference method."""

    def __init__(
        self, num_samples, num_burnin, init_time_step=0.1, leapfrog_step=30, seed=None,
    ):
        super().__init__()
        if backend_name in ["tensorflow", "tensorflow.compat.v1"]:
            seed = seed
        elif backend_name == "jax":
            seed = 0 if seed is None else seed
            seed = jax.random.PRNGKey(seed)
        else:
            raise NotImplementedError(
                "Backend {} is not supported.".format(backend_name)
            )

        self._params.update(
            {
                "num_samples": int(num_samples),
                "num_burnin": int(num_burnin),
                "init_time_step": init_time_step,
                "leapfrog_step": leapfrog_step,
                "seed": seed,
            }
        )
        # to compute acceptance rate
        self.trace_fn = lambda _, pkr: pkr.inner_results.is_accepted

    def make_sampler(self, target_log_prob_fn, init_state):
        self.mcmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                num_leapfrog_steps=self.params["leapfrog_step"],
                step_size=self.params["init_time_step"],
            ),
            num_adaptation_steps=int(self.params["num_burnin"] * 0.8),
        )
        self.set_sampler(init_state)


class NUTS(MCMC):
    """No-U-Turn MCMC method."""

    def __init__(self, num_samples, num_burnin, time_step, seed=None):
        super().__init__()
        self._params.update(
            {
                "num_samples": num_samples,
                "num_burnin": num_burnin,
                "time_step": time_step,
                "seed": seed,
            }
        )

    def make_sampler(self, target_log_prob_fn, init_state):
        """Initializes a MCMC chain."""
        self.mcmc_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn, step_size=self.params["time_step"],
        )
        self.set_sampler(init_state)


class NoUTurn(MCMC):
    """Adaptive No-U-Turn MCMC method."""

    def __init__(self, num_samples, num_burnin, time_step, seed=None):
        super().__init__()
        self._params.update(
            {
                "num_samples": num_samples,
                "num_burnin": num_burnin,
                "time_step": time_step,
                "seed": seed,
            }
        )

    def make_sampler(self, target_log_prob_fn, init_state):
        """Initializes a MCMC chain."""
        sampler = tfp.mcmc.TransformedTransitionKernel(
            tfp.mcmc.NoUTurnSampler(target_log_prob_fn=posterior, step_size=1.0),
            bijector=tfp.bijectors.Identity(),
        )
        self.mcmc_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn, step_size=self.params["time_step"],
        )
        self.set_sampler(init_state)

    def mcmc(self, x, W_g_z, b_g_z, W_g_x, b_g_x, model):
        prior_ll_model = Prior_LL(self.z_dim, self.noise)
        prior_z = prior_ll_model.prior()

        def posterior(z):
            likelihood_sc, likelihood_sa = prior_ll_model.likelihood(
                x, z, W_g_z, b_g_z, W_g_x, b_g_x, model
            )
            return (
                tf.reduce_sum(prior_z.log_prob(z))
                + tf.reduce_sum(likelihood_sc.log_prob(self.sc_train))
                + tf.reduce_sum(likelihood_sa.log_prob(self.sa_train))
            )

        sampler = tfp.mcmc.TransformedTransitionKernel(
            tfp.mcmc.NoUTurnSampler(target_log_prob_fn=posterior, step_size=1.0),
            bijector=self.bijector_list,
        )

        adaptive_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=sampler,
            num_adaptation_steps=int(self.num_burnin * 0.8),
            target_accept_prob=0.6,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                inner_results=pkr.inner_results._replace(step_size=new_step_size)
            ),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
        )

        samples, _ = tfp.mcmc.sample_chain(
            num_results=self.num_samples,
            num_burnin_steps=self.num_burnin,
            current_state=self.z_init,
            kernel=adaptive_hmc,
        )

        return samples
