import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


import neuraluq as neuq
from neuraluq.backend import backend_name, tf, jax, tfp


def load_data():
    x_test = np.linspace(-1, 1, 64).reshape([-1, 1])
    f_test = 0.5 * x_test * np.sin(2 * np.pi * x_test)
    u_test = np.arctanh(f_test)

    idx = np.random.choice(64, 64, replace=False)[:6]
    x_train = x_test[idx]
    f_train = f_test[idx] + 0.05 * np.random.normal(size=x_train.shape)
    return x_train, f_train, x_test, u_test, f_test


def equation(x, u_fn):
    """
    Computes the left hand side of the equation.

    Args:
        x: The locations where measurements are obtained.
        u_fn: The surrogate to the solution.
    Returns:
        f: The approximated left hand side.
    """
    # for the equation: tanh(u) = 0.5x * sin(2pi x)
    u = u_fn(x)
    # lhs = tf.tanh(u)
    lhs = jax.numpy.tanh(u)
    return lhs


if __name__ == "__main__":
    ################## Load data and specify some hyperparameters ####################
    np.random.seed(1234)
    x_train, f_train, x_test, u_test, f_test = load_data()
    layers = [1, 50, 50, 1]

    process = neuq.Process(
        surrogate=neuq.surrogates.FNN(layers=layers),
        prior=neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1),
    )
    likelihood_f = neuq.likelihoods.Normal(
        inputs=x_train,
        targets=f_train,
        processes=[process],
        equation=equation,
        sigma=0.05,
    )

    model = neuq.models.Model(
        processes=[process],
        likelihoods=[likelihood_f],
    )
    method = neuq.inferences.HMC(
        num_samples=1000,
        num_burnin=1000,
        init_time_step=0.01,
        leapfrog_step=50,
        seed=6666,
    )
    model.compile(method)
    samples, results = model.run()
    print("Acceptance rate: %.3f \n" % (np.mean(results)))  # if HMC is used

    ################################# Predictions ####################################
    (u_pred,) = model.predict(x_test, samples, processes=[process])
    (f_pred,) = model.predict(x_test, samples, processes=[process], equation=equation)
    ############################### Postprocessing ###################################
    mu = np.mean(u_pred, axis=0)
    std = np.std(u_pred, axis=0)
    plt.fill_between(
        x_test.flatten(),
        (mu + 2 * std).flatten(),
        (mu - 2 * std).flatten(),
        alpha=0.3,
        label="2 std",
    )
    plt.plot(x_test, mu, "--", label="mean")
    plt.plot(x_test, u_test, label="reference")
    plt.legend()
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
    plt.plot(x_train, f_train, "o", label="measurements")
    plt.legend()
    plt.show()
