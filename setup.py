from setuptools import setup, find_packages

setup(
    name="quasidots",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "numba",
        "aeon",
        "sktime",
    ],
)