Dimensionality estimation
===========================================

This repository contains code to apply different crtieria to choose the optimal number of Principal Components (PCs) for estimating the dimensionality of a data matrix. It also allow to generate a data matrix as random linear combination of a subset of latent variables with known properties (such as size, the number of underlying latent variables, the amount of noise etc.) to test the criteria of the repository as well as other methods. For a full explanation of the data generation procedure and the implemented criteria refer to the article below.

Upon usage, please cite:
> FE Vaccari, S Diomedi, E Bettazzi, M Filippini, M De Vitis, K Hadjidimitrakis, P Fattori<br>
> **More or less latent variables  in the high-dimensional data space? That is the question**<br>
> bioRxiv 2024.11.28.625854
> (doi: https://doi.org/10.1101/2024.11.28.625854)


## MATLAB

Add the Matlab subfolder to the Matlab search path.

Example code in `dimensionality_DEMO.m` generates data and analyze the resulting data matrix with all the criteria provided in the package plotting the results.

Note: the package was implemented on MATLAB 2024b. Backward compatibility with older versions is not guaranteed.


# Dimensionality-estimation Repository Documentation

## Overview
This repository contains functions for estimating and analyzing the dimensionality of data matrices using various methods, particularly focused on Principal Component Analysis (PCA).

## Main Functions

### 1. `eval_num_PCs_cross_val.m`
**Purpose**: Performs cross-validation to select the optimal number of Principal Components.
- Input: Data matrix, maximum number of PCs to evaluate
- Output: Optimal number of PCs and detailed statistics
- Supports multiple cross-validation schemes (rows, columns, rows&columns, bi-cross)
- Uses R² as performance metric

### 2. `parallel_analysis.m`
**Purpose**: Estimates significant dimensions through comparison with shuffled data.
- Compares real data eigenvalues with shuffled versions
- Uses quantile thresholding for dimension selection
- Useful for determining statistically significant components

### 3. `participation_ratio.m`
**Purpose**: Calculates the participation ratio of the eigenspectrum.
- Measures variance distribution across principal components
- Higher values indicate more evenly distributed eigenspectrum
- Reference: Gao, et al., 2017

### Data Generation Functions

#### `simulate_data_matrix.m`
**Purpose**: Generates synthetic data for testing dimensionality estimation methods.
- Supports multiple generation methods
- Configurable number of variables (neurons) to be generate, length of the time series, number of latent components
- Configurable noise levels and distributions
- Customizable decay factors
- Customizable degree of non-linearity

##### `generate_random_sine_series.m`
- Creates matrix of sine waves with different frequencies
- Frequencies increase exponentially

##### `generate_random_units_from_PCs.m`
- Generates random latent variables based on real PCs
- Uses precomputed PC scores

##### `generate_random_dynamical_series.m`
- Implements dynamical systems (Lorenz96 system)
- Includes stochastic components

##### `generate_random_time_series.m`
- Generates random latent variables

### Utility Functions

##### `fit_tau.m`
**Purpose**: Fits exponential decay to explained variance of PCs.
- Computes explained variance of first PCs
- Fits data to exponential model

##### `fit_power_law.m`
**Purpose**: Fits power law to explained variance of PCs.
- Computes explained variance of first PCs
- Fits data to power-law model



## References
For implementation details and theoretical background, refer to:
> FE Vaccari, S Diomedi, E Bettazzi, M Filippini, M De Vitis, K Hadjidimitrakis, P Fattori
> "More or less latent variables in the high-dimensional data space? That is the question"
> bioRxiv 2024.11.28.625854

## Support
Email francesco.vaccari6@unibo.it (MATLAB) with any questions.
