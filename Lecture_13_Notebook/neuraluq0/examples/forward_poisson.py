"""NeuralUQ for 1-D Poisson equation (forward), from B-PINN paper."""


# See also this paper for reference:
# B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time


import neuraluq as neuq
from neuraluq.backend import backend_name, tf, jax, tfp


def load_data():
    data = sio.loadmat("../dataset/Poisson_forward.mat")
    x_test, u_test, f_test = data["x_test"], data["u_test"], data["f_test"]
    x_u_train, u_train = data["x_u_train"], data["u_train"]
    x_f_train, f_train = data["x_f_train"], data["f_train"]
    return x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test


# def pde_fn(x, u_fn):
#     # for backend Tensorflow, using tf.gradients to compute gradients,
#     # compatible with graph/tf.function mode
#     D, k = 0.01, 0.7
#     u = u_fn(x)
#     u_x = tf.gradients(u, x)[0]
#     u_xx = tf.gradients(u_x, x)[0]
#     return D * u_xx + k * tf.tanh(u)


def pde_fn(x, u_fn):
    # for backend Tensorflow, using tf.GradientTape() to compute gradients,
    # compatible with both graph/tf.function and eager modes
    D, k = 0.01, 0.7
    with tf.GradientTape() as g_xx:
        g_xx.watch(x)
        with tf.GradientTape() as g_x:
            g_x.watch(x)
            u = u_fn(x)
        u_x = g_x.gradient(u, x)
    u_xx = g_xx.gradient(u_x, x)
    return D * u_xx + k * tf.tanh(u)


# def pde_fn(x, u_fn):
#     # for backend Tensorflow, using tf.GradientTape() to compute gradients,
#     # compatible with both graph/tf.function and eager modes
#     D, k = 0.01, 0.7

#     def u_x_fn(x):
#         u, vjp_fn = jax.vjp(u_fn, x, has_aux=False)
#         return vjp_fn(jax.numpy.ones_like(x))[0], u

#     u_x, vjp_fn, u = jax.vjp(u_x_fn, x, has_aux=True)
#     u_xx = vjp_fn(jax.numpy.ones_like(x))[0]
#     # u = u_fn(x)
#     return D * u_xx + k * jax.numpy.tanh(u)


if __name__ == "__main__":
    ################## Load data and specify some hyperparameters ####################
    x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test = load_data()
    layers = [1, 50, 50, 1]

    ####################### Build model and perform inference ########################
    # All models share the same general procedure:
    # Step 1: build surrogate, e.g. a fully-connected neural network, using [surrogates]
    # Step 2: build prior and/or posterior using [variables]
    # Step 3: build process, based the surrogate, prior and/or posterior, using [Process]
    # Step 4: build likelihood, given noisy measurements, using [likelihoods]
    # Step 5: build model using [models]
    # Step 6: create an inference method and assign it to the model using [inferences]
    # Step 7: perform posterior sampling using [model.run]

    process = neuq.Process(
        surrogate=neuq.surrogates.FNN(layers=layers),
        prior=neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1),
    )
    likelihood_u = neuq.likelihoods.Normal(
        inputs=x_u_train,
        targets=u_train,
        processes=[process],
        sigma=0.1,
    )
    likelihood_f = neuq.likelihoods.Normal(
        inputs=x_f_train,
        targets=f_train,
        processes=[process],
        equation=pde_fn,
        sigma=0.1,
    )

    model = neuq.models.Model(
        processes=[process],
        likelihoods=[likelihood_u, likelihood_f],
    )
    method = neuq.inferences.HMC(
        num_samples=1000,
        num_burnin=1000,
        init_time_step=0.01,
        leapfrog_step=50,
        seed=6666,
    )
    model.compile(method)
    t0 = time.time()
    samples, results = model.run()
    print("Acceptance rate: %.3f \n" % (np.mean(results)))  # if HMC is used
    print("Elapsed: ", time.time() - t0, " with backend {}".format(backend_name))

    ################################# Predictions ####################################
    (u_pred,) = model.predict(x_test, samples, processes=[process])
    (f_pred,) = model.predict(x_test, samples, processes=[process], equation=pde_fn)
    ############################### Postprocessing ###################################
    mu = np.mean(u_pred, axis=0)
    std = np.std(u_pred, axis=0)
    plt.fill_between(
        x_test.flatten(), (mu + 2 * std).flatten(), (mu - 2 * std).flatten(), alpha=0.3
    )
    plt.plot(x_test, mu, "--")
    plt.plot(x_test, u_test)
    plt.plot(x_u_train, u_train, "o")
    plt.show()

    mu = np.mean(f_pred, axis=0)
    std = np.std(f_pred, axis=0)
    plt.fill_between(
        x_test.flatten(),
        (mu + 2 * std).flatten(),
        (mu - 2 * std).flatten(),
        alpha=0.3,
        label="2 std",
    )
    plt.plot(x_test, mu, "--", label="mean")
    plt.plot(x_test, f_test, label="reference")
    plt.plot(x_f_train, f_train, "o", label="measurements")
    plt.legend()
    plt.show()
