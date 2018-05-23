# Numerical Introductory Seminar 2018

## Getting started with Python
To run the `Python` code from this repository you may need to install additional modules. You can do this by opening a terminal (`CTRL-ALT-T` on `Linux` or `MacOS`) or a command line (on `Windows` just push `HOME` and start typing `run`, you should see a black icon).

First you want to change into your directory:
```
cd __path_to_your_cloned_repository__
```
Next you want to install the packages from the `requirements.txt`-file:
```
pip install -r requirements.txt
```

Thats all. After that you can run the python scripts by running:
```
python __name_of_script__.py
```
Note that this might be different if you are using `IDE`'s like `PyCharm`, `IDLE`, `Spyder`, etc.
<h1 id="algorithms.eigen">algorithms.eigen</h1>


Algorithms for solving eigenvalue problems.

1. Compute diagonalization of 2x2 matrices via jacobi iteration.
2. Generalize Jacobi iteration for symmetric matrices.

<h2 id="algorithms.eigen.jacobi2x2">jacobi2x2</h2>

```python
jacobi2x2(A)
```

Diagonalize a 2x2 matrix through jacobi step.
Solve: U' A U = E s.t. E is a diagonal matrix.

* Parameters:
    + A - 2x2 numpy array.
* Returns:
    + A - 2x2 diagonal numpy array

<h2 id="algorithms.eigen.jacobi">jacobi</h2>

```python
jacobi(X, precision=1e-06, debug=False)
```

Compute Eigenvalues and Eigenvectors for symmetric matrices.

* Parameters:
    + X - 2D numpy ndarray which represents a symmetric matrix
    + precision - float in (0, 1). Convergence criterion.

* Returns:
    + A - 1D numpy array with eigenvalues sorted by absolute value
    + U - 2D numpy array with associated eigenvectors (column).

<h2 id="algorithms.eigen.qrm">qrm</h2>

```python
qrm(X, maxiter=15000, debug=False)
```

Compute Eigenvalues and Eigenvectors using the QR-Method.

* Parameters:
    + X: square numpy ndarray.
* Returns:
    + Eigenvalues of A.
    + Eigenvectors of A.

<h2 id="algorithms.eigen.qrm2">qrm2</h2>

```python
qrm2(X, maxiter=15000, debug=False)
```

First compute similar matrix in Hessenberg form, then compute the
Eigenvalues and Eigenvectors using the QR-Method.

* Parameters:
    + X: square numpy ndarray.
* Returns:
    + Eigenvalues of A.
    + Eigenvectors of A.

<h2 id="algorithms.eigen.qrm3">qrm3</h2>

```python
qrm3(X, maxiter=15000, debug=False)
```

First compute similar matrix in Hessenberg form, then compute the
Eigenvalues and Eigenvectors using the QR-Method.

* Parameters:
    + X: square numpy ndarray.
* Returns:
    + Eigenvalues of A.
    + Eigenvectors of A.

<h1 id="algorithms.helpers">algorithms.helpers</h1>


<h2 id="algorithms.helpers.hreflect1D">hreflect1D</h2>

```python
hreflect1D(x)
```

Calculate Householder reflection: Q = I - 2*uu'.

* Parameters:
    + X: numpy array.

* Returns:
    + Qx: reflected vector.
    + Q: Reflector (matrix).

<h2 id="algorithms.helpers.qr_factorize">qr_factorize</h2>

```python
qr_factorize(X, offset=0)
```

Compute QR factorization of X s.t. QR = X.

* Parameters:
    + X: square numpy ndarray.
    + offset: (int) either 0 or 1. If offset is unity: compute Hessenberg-
              matrix.

* Returns:
    + Q: square numpy ndarray, same shape as X. Rotation matrix.
    + R: square numpy ndarray, same shape as X. Upper triangular matrix if offset is 0, Hessenberg-matrix if offset is 1.

<h1 id="models.lda">models.lda</h1>


<h2 id="models.lda.LDA">LDA</h2>

```python
LDA(self)
```

Define class for Linear Discriminant Analysis of a rank 2 array of data.

* Attributes:
    + data:
    + generalized eigenvectors:
    + generalized eigenvalues:
    + rotated data:
    + inertia:
* Methods:
    + fit(X):
    + plot(dim):
    + scree():

<h3 id="models.lda.LDA.fit">fit</h3>

```python
LDA.fit(self, X, g)
```

Fit Linear Discriminant model to data.

* Parameters:
    + X: numpy ndarray of shape (n, m). Contains data and labels of observations
    + g: integer indicating the column of labels.

* Defines Attributes:
    + data: copy of X without label column.
    + groups: 1D numpy array of labels.
    + Sb: 2D numpy array of Between-Class-Scatter-Matrix.
    + Sw: 2D numpy array of Within-Class-Scatter-Matrix.
    + eigenvalues: Fitted eigenvalues for generalized eigenvalue problem.
    + eigenvectors: 2D numpy array (m, m). Contains eigenvectors.
    + rotated_data: 2D numpy array (n, m). Contains projected points.
    + inertia: 1D numpy array. Contains calculated inertia.


<h3 id="models.lda.LDA.scree">scree</h3>

```python
LDA.scree(self)
```

Return scree plot for the supplied data.

<h3 id="models.lda.LDA.plot">plot</h3>

```python
LDA.plot(self, x, y)
```

Return 2-dimensional plot for 2 Linear Discriminants.

* Parameters:
    + x: Integer indicating which Linear Discriminant to plot on
         x-axis.
    + y: Integer indicating which Linear Discriminant to plot on
         y-axis.

<h1 id="models.pca">models.pca</h1>


<h2 id="models.pca.PCA">PCA</h2>

```python
PCA(self)
```

Define class for Principal Component Analysis of a rank 2 array of data.

* Attributes:
    + data:
    + eigenvectors:
    + eigenvalues:
    + rotated data:
    + inertia:
* Methods:
    + fit(X):
    + plot(dim):
    + scree():

<h3 id="models.pca.PCA.fit">fit</h3>

```python
PCA.fit(self, X, norm=True)
```

Fit PCA to data.

* Parameters:
    + X:    numpy 2D-array. Contains data to be fit.
    + norm: boolean. If `True`, the default, use correlation matrix. Else use the covariance matrix.

* Returns Attributes:
    + data:         Centered data.
    + rotated_data: Rotated data. Premultiplied with the matrix of Eigenvectors
    + cov:          Covariance/correlation matrix of the data.
    + eigenvectors: Eigenvectors of `cov`.
    + eigenvalues:  Eigenvalues of `cov`.
    + inertia:      Proportion of explained variance by each component.

<h3 id="models.pca.PCA.scree">scree</h3>

```python
PCA.scree(self)
```

Return scree plot for the supplied data.

<h3 id="models.pca.PCA.plot">plot</h3>

```python
PCA.plot(self, x, y)
```

Return 2-dimensional plot for 2 Pricipal Components.

* Parameters:
    + x: Integer indicating which Pricipal Component to plot on x-axis.
    + y: Integer indicating which Pricipal Component to plot on y-axis.
