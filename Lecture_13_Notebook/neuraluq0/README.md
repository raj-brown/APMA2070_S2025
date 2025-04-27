# NeuralUQ0
A library for uncertainty quantification in neural differential equations, complementary to NeuralUQ library.

This library focuses on Bayesian neural networks for differential equations with Markov chain Monte Carlo (MCMC) inference methods, and aims to provide useful tools across deep learning platforms, including Tensorflow 1, Tensorflow 2 and JAX.

# Installation
**NeuralUQ** requires the following dependencies to be installed:

- All backends: TensorFlow Probability 0.18.0
- Backend Tensorflow 1 or 2: Tensorflow 2.10.0
- Backend JAX: JAX 0.3.20 and Flax 0.6.0

Then install with `python`:

```
$ python setup.py install
```

# The Team
NeuralUQ was developed by [Zongren Zou](https://github.com/ZongrenZou) and [Xuhui Meng](https://github.com/XuhuiM), under the supervision of Professor George Em Karniadakis at Division of Applied Mathematics, Brown University. 

The library is currently maintained by [Zongren Zou](https://github.com/ZongrenZou) and [Xuhui Meng](https://github.com/XuhuiM).

# Cite NeuralUQ

[@misc{zou2022neuraluq, <br />
    title={NeuralUQ: A comprehensive library for uncertainty quantification in neural differential equations and operators}, <br />
    author={Zongren Zou, Xuhui Meng, Apostolos F Psaros, and George Em Karniadakis}, <br />
    year={2022}, <br />
    eprint={2208.11866}, <br />
    archivePrefix={arXiv}, <br />
    primaryClass={cs.LG} <br />
}](http://arxiv.org/abs/2208.11866)
