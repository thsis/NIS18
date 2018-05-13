# Numerical Introductory Seminar

# TODOS:

## algorithms

* Normalize eigenvectors in `jacobi` and `qr_method`.

## models

* Implement PCA.
* Implement LDA.


## Random Number Generation:

* Mersenne-Twister

## Lecture 6.5.18

Tests for normality (test your residuals):

* Kolmogorov-Smirnov
* Shapiro
* Anderson

## Presentation

* Eigenfaces - you've got to be kidding.


## Failure in tests

Seems the issue was just a rounding error. Initially I computed the step by calculating

$ Q_{i} A_{i} Q_{i} \prime $

explicitly, which can be simplified to

$ R_i Q_i $.

Also raising the number of maximum iterations seems to have fixed the issue. Maybe accelerating the Convergence of the algorithm will also help.
