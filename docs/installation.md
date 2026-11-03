# Installation

RabMV needs Python 3.12 or later and depends only on NumPy, pandas, SciPy and scikit-learn.

    python -m pip install rabmv

From a clone of the repository (to run the tests, the simulations or the examples):

    python -m pip install -e ".[dev]"            # tests and linting
    python -m pip install -e ".[dev,development]" # also the simulations and the development harness
    python -m pytest

The `development` extra installs InterpretML's EBM 0.7.8, imbalanced-learn, Optuna, DEAP, Matplotlib and
tabulate, with the versions used in the published results.
