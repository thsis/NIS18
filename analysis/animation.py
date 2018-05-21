"""
Animate the QR-Method.

Visualize the performed chasing accross iterations

Code for heatmap was inspired by: https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
Code for animations was inspired by:
https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from algorithms import helpers
from scipy import linalg as lin

np.random.seed(42)
X = np.random.uniform(low=-1.0, high=1.0, size=(10, 10))
X += X.T

# Construct empty matrix for background.
Empty = np.empty(shape=X.shape)
Empty[:] = np.nan

# Set up figure and axis
fig = plt.figure()
ax = plt.axes(xlim=(Empty.shape[0]-0.5, -0.5),
              ylim=(-0.5, Empty.shape[1]-0.5))
plt.ylabel("")
plt.xlabel("")
hm = ax.imshow(Empty, cmap=plt.get_cmap('PiYG'), vmin=-2.0, vmax=2.0)
ax.figure.colorbar(hm)
ax.set_yticks([])
ax.set_xticks([])
ax.grid(True)
fig.suptitle("Demonstrate QR-Method on a 10x10 matrix")


# Initialize background for each frame
def init():
    """Initialize fig."""
    hm.set_data(Empty)
    return hm


# Define animation behavior
def animate(i, delay=9):
    """
    Update axes of fig.
        - i: step of current iteration. Will be updated by FuncAnimation-class.
        - delay: int. Number of frames to delay the initial matrix and the
          first iteration.
    """
    global A
    if i < delay:
        A = X
        ax.set_title("Iteration: 0")
    elif i == delay:
        A = next(iterated_qrm)
        ax.set_title("Iteration: 1")
    elif delay < i < 2 * delay:
        ax.set_title("Iteration: 1")
    else:
        A = next(iterated_qrm)
        ax.set_title("Iteration: " + str(i - 2*delay))

    hm.set_data(np.round(A, 3))
    return hm


def demonstrate_qrm(X, maxiter=5000):
    """
    Create generator for transformed matrices after applying the QR-Method.

    Parameters:
        - X: 2D-numpy array. Matrix whose eigenvalues should be computed.
        - maxiter: maximum number of iterations to be performed.
    Yields:
        - T: 2D-numpy array. Similar matrix to X.
    """
    n, m = X.shape
    assert n == m

    # First stage: transform to upper Hessenberg-matrix.
    T = lin.hessenberg(X)

    conv = False
    k = 0

    # Second stage: perform QR-transformations.
    while (not conv) and (k < maxiter):
        k += 1
        Q, R = helpers.qr_factorize(T - T[n-1, n-1] * np.eye(n))
        T = R.dot(Q) + T[n-1, n-1] * np.eye(n)

        conv = np.alltrue(np.isclose(np.tril(T, k=-1), np.zeros((n, n))))
        yield T


iterated_qrm = demonstrate_qrm(X, maxiter=10000)

anim = animation.FuncAnimation(fig, animate,
                               init_func=init, frames=130, interval=20)
anim.save('analysis/qrm_symmetric.mp4', fps=3,
          extra_args=['-vcodec', 'libx264'])
plt.show()
