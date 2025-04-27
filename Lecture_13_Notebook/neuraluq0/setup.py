"""Temporary setup file."""

from setuptools import setup, find_packages

setup(
    name="NeuralUQ",
    version="v0.1.0-beta",
    description="A library for uncertainty quantification in neural differential equations",
    author="Zongren Zou & Xuhui Meng",
    author_email="zongren_zou@brown.edu, xuhui_meng@hust.edu.cn",
    keywords=[
        "Scientific machine learning",
        "Uncertainty Quantification",
        "Deep learning",
        "Neural networks",
        "Differential equations",
        "Bayesian inference",
        "Neural ODEs/PDEs",
    ],
    packages=find_packages(),
)
