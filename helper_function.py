import numpy as np
from scipy.linalg import eigh

def solve_vertical_modes(N2, z):
    """
    Solve vertical eigenmodes.

    Parameters
    ----------
    N2 : (nz,) np.ndarray
    z  : (nz,) np.ndarray

    Returns
    -------
    eigvals : (nmodes,)
    eigvecs : (nz, nmodes)
    mask    : (nz,)
    B       : (nz, nz)
    """

    N2 = np.asarray(N2)
    z = np.asarray(z)

    # mask = np.isfinite(N2)
    # z = z[mask]
    # N2 = N2[mask]

    n = len(z)

    # coefficient
    k = 1.0 / N2
    k_face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])

    # grid spacing
    z_face = np.zeros(n + 1)

    z_face[1:-1] = 0.5 * (z[:-1] + z[1:])

    z_face[0] = z[0] - 0.5 * (z[1] - z[0])
    z_face[-1] = z[-1] + 0.5 * (z[-1] - z[-2])

    dz = z_face[1:] - z_face[:-1]

    # operator matrix
    A = np.zeros((n, n))

    for i in range(1, n - 1):
        dzm = z[i] - z[i - 1]
        dzp = z[i + 1] - z[i]

        km = k_face[i - 1]
        kp = k_face[i]

        A[i, i - 1] = km / dzm
        A[i, i]     = -(km / dzm + kp / dzp)
        A[i, i + 1] = kp / dzp

    # boundaries
    kp = k_face[0]
    A[0, 0] = -kp / (z[1] - z[0])
    A[0, 1] = kp / (z[1] - z[0])

    km = k_face[-1]
    A[-1, -2] = km / (z[-1] - z[-2])
    A[-1, -1] = -km / (z[-1] - z[-2])

    # print(A.shape)

    # mass matrix
    B = np.diag(dz)

    # eigenproblem
    eigvals, eigvecs = eigh(-A, B)

    # keep = eigvals > 0
    # eigvals = eigvals[keep]
    # eigvecs = eigvecs[:, keep]

    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    ###
    # # checking normalisation and orthogonality
    # print(np.diag(eigvecs.T @ B @ eigvecs)) # 1 if normalised
    # print(eigvecs.T @ B @ eigvecs) # off diagonals zero if orthogonal
    ###

    # return eigvals, eigvecs, mask, B
    return eigvals, eigvecs, B