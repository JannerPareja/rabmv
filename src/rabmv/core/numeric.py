"""Numerical helpers shared by the RabMV core (unchanged from EBMP v1)."""
import numpy as np

LOGIT_MAX = 30.0


def sigmoid(z):
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def log_loss(y, z, w):
    """Mean weighted logistic loss for logits ``z`` (one column per bag).

    Parameters
    ----------
    y : numpy.ndarray of shape (n,)
        0/1 class.
    z : numpy.ndarray of shape (n, K)
        Logits.
    w : numpy.ndarray of shape (n, K)
        Weights (0 = the case does not count for that bag).

    Returns
    -------
    numpy.ndarray of shape (K,)
    """
    loss = np.logaddexp(0.0, z) - y[:, None] * z               # Bernoulli negative log-likelihood
    return (loss * w).sum(0) / np.maximum(w.sum(0), 1e-12)
