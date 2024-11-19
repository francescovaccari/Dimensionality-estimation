Dimensionality estimation
===========================================

This repository contains code to apply different crtieria to choose the optimal number of Principal Components (PCs) for estimating the dimensionality of a data matrix. It also allow to generate a data matrix as random linear combination of a subset of latent variables with known properties (such as size, the number of underlying latent variables, the amount of noise etc.) to test the criteria of the repository as well as other methods. For a full explanation of the data generation procedure and the implemented criteria refer to the article below.

Upon usage, please cite:
> FE Vaccari, S Diomedi, E Bettazzi, M Filippini, M De Vitis, K Hadjidimitrakis, P Fattori<br>
> **More or less latent variables  in the high-dimensional data space? That is the question**<br>
> Journal yyyy, [link here]<br>
> (arXiv link: [link here])

## Use of the package

Refer to each function header to be aware of input/output variables.


### MATLAB

Add the Matlab subfolder to the Matlab search path.

Example code in `dimensionality_DEMO.m` generates data and analyze the resulting data matrix with all the criteria provided in the package plotting the results.

Note: the package was implemented on MATLAB 2024b. Backward compatibility with older versions is not guaranteed.


## Support

Email francesco.vaccari6@unibo.it (MATLAB) with any questions.
