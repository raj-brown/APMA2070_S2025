import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time


import neuraluq as neuq
from neuraluq.backend import backend_name, tf, jax, tfp


def load_data():
    data = sio.loadmat("../dataset/func_train.mat")
    x_u_train = data["x_train"]
    u_train = data["y_train"]
    x_test = data["x_test"]
    u_test = data["y_test"]
    return x_u_train, u_train, x_test, u_test


if __name__ == "__main__":
    ################## Load data and specify some hyperparameters ####################
    x_train, u_train, x_test, u_test = load_data()
    layers = [1, 50, 50, 1]

    surrogate = neuq.surrogates.FNN(layers=layers)
    prior = neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1)
    process_u = neuq.Process(surrogate=surrogate, prior=prior)
    likelihood = neuq.likelihoods.Normal(
        inputs=x_train,
        targets=u_train,
        processes=[process_u],
        sigma=0.1,
    )

    model = neuq.Model(processes=[process_u], likelihoods=[likelihood])

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
    print("Elapsed ", time.time() - t0, " with {}".format(backend_name))

    ################################# Predictions ####################################
    (outputs,) = model.predict(
        inputs=x_test,
        samples=samples,
        processes=[process_u],
    )

    mu = np.mean(outputs, axis=0)
    std = np.std(outputs, axis=0)
    plt.plot(x_test, mu, "--")
    plt.plot(x_train, u_train, "o")
    plt.fill_between(
        x_test.flatten(), (mu + 2 * std).flatten(), (mu - 2 * std).flatten(), alpha=0.3
    )
    plt.plot(x_test, u_test)
    plt.show()
