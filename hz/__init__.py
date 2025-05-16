__author__ = 'tcw'
"""
A collection of functions to generate synthetic data from real labeled data itself.
The purpose being to have data that resembles the real data, but with more or less force structure in order to
allow for controlled model assessment.
"""

from numpy import *


def label_means(X, y):
    XX = X.copy()
    for yy in unique(y):
        lidx = y == yy
        mean_X = mean(X[lidx, :], axis=0)
        XX[lidx, :] = mean_X
    return XX




def label_variances(X, y):
    """
    Calculate the variance of features for each label in a labeled dataset.

    Parameters:
        X (numpy.ndarray): The input features, assumed to be a 2D array.
        y (numpy.array): The corresponding labels for the dataset.

    Returns:
        numpy.ndarray: An array where each row corresponds to the variance of features for each label.

    Example:
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 1, 0, 1])
        >>> label_variances(X, y)
        array([[4., 4.],
               [4., 4.]])
    """
    import numpy as np
    variances = np.zeros((len(np.unique(y)), X.shape[1]))
    for i, label in enumerate(np.unique(y)):
        label_indices = np.where(y == label)
        variances[i, :] = np.var(X[label_indices], axis=0)
    return variances




def label_medians(X, y):
    """
    Calculate the median of features for each label in a labeled dataset.

    Parameters:
        X (numpy.ndarray): The input features, assumed to be a 2D array.
        y (numpy.array): The corresponding labels for the dataset.

    Returns:
        numpy.ndarray: An array where each row corresponds to the median of features for each label.

    Example:
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 1, 0, 1])
        >>> label_medians(X, y)
        array([[3., 4.],
               [5., 6.]])
    """
    import numpy as np
    medians = np.zeros((len(np.unique(y)), X.shape[1]))
    for i, label in enumerate(np.unique(y)):
        label_indices = np.where(y == label)
        medians[i, :] = np.median(X[label_indices], axis=0)
    return medians


def label_standard_deviations(X, y):
    """
    Calculate the standard deviation of features for each label in a labeled dataset.

    Parameters:
        X (numpy.ndarray): The input features, assumed to be a 2D array.
        y (numpy.array): The corresponding labels for the dataset.

    Returns:
        numpy.ndarray: An array where each row corresponds to the standard deviation of features for each label.

    Example:
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 1, 0, 1])
        >>> label_standard_deviations(X, y)
        array([[2., 2.],
               [2., 2.]])
    """
    import numpy as np
    std_devs = np.zeros((len(np.unique(y)), X.shape[1]))
    for i, label in enumerate(np.unique(y)):
        label_indices = np.where(y == label)
        std_devs[i, :] = np.std(X[label_indices], axis=0)
    return std_devs
