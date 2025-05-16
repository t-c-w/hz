# hz
A collection of functions to generate synthetic data from real labeled data itself.

To install:	```pip install hz```

## Overview
The `hz` package provides a suite of functions designed to analyze labeled datasets by computing statistical measures across different labels. This can be particularly useful for understanding the characteristics of data subsets and for generating synthetic data that mimics real datasets. The functions included allow you to calculate means, variances, medians, and standard deviations for features within each label of a dataset.

## Functions

### `label_means`
Computes the mean of features for each label in a labeled dataset. This can be useful to create a simplified representation of the dataset where each group of labels is represented by its mean feature values.

**Usage Example:**
```python
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])
means = hz.label_means(X, y)
print(means)
```

### `label_variances`
Calculates the variance of features for each label in a labeled dataset. This function helps in understanding the dispersion of data points from the mean within each label group.

**Parameters:**
- `X` (numpy.ndarray): The input features, assumed to be a 2D array.
- `y` (numpy.array): The corresponding labels for the dataset.

**Returns:**
- numpy.ndarray: An array where each row corresponds to the variance of features for each label.

**Usage Example:**
```python
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])
variances = hz.label_variances(X, y)
print(variances)
```

### `label_medians`
Computes the median of features for each label in a labeled dataset. This function is useful for understanding the central tendency of the data without the influence of outliers.

**Usage Example:**
```python
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])
medians = hz.label_medians(X, y)
print(medians)
```

### `label_standard_deviations`
Calculates the standard deviation of features for each label in a labeled dataset. This function provides insights into the amount of variation or dispersion of the dataset features within each label.

**Usage Example:**
```python
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])
std_devs = hz.label_standard_deviations(X, y)
print(std_devs)
```

## Installation
You can install the `hz` package directly from PyPI:
```bash
pip install hz
```

This package requires `numpy` to be installed in your Python environment, as it is heavily used for all mathematical computations within the package.