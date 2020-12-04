# ======================================================================================================================
# Author:       Jordan Leiker
# GTid:         903453031
# Class:        MATH6644 - Iterative Methods for Solving Systems of Equations
# Date:         02/1/2020
# ======================================================================================================================

import numpy as np
import utility as util

# General Purpose Conjugate Gradient
def conjugateGradient(A, b, tol, maxIters):
    (m, n) = np.shape(A)

    # Calculate Initial Residual / Direction
    x_k = np.zeros((n, 1))
    r_k = b - A @ x_k
    t_km2 = 0

    # Initialize Loop exit conditions
    err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
    numIters = 0

    while (err > tol) and (numIters < maxIters):
        t_km1 = r_k.T @ r_k
        if numIters == 0:
            p_k = r_k
        else:
            B_k = (t_km1)/(t_km2)
            p_k = r_k + B_k*p_k
        w_k = A @ p_k
        a_k = t_km1/(p_k.T @ w_k)
        x_k = x_k + a_k * p_k
        r_k = r_k - a_k * w_k
        t_km2 = t_km1

        # Stopping Condition Checks
        # Note: r_0 = b, because r_0 = b - Ax_0 where x_0 = 0
        err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
        numIters += 1

    return x_k, numIters

# General Purpose Pre-Conditioned Conjugate Gradient
def preconditionedConjugateGradient(A, M, b, tol, maxIters):
    (m, n) = np.shape(A)

    # Calculate Initial Residual / Direction
    x_k = np.zeros((n, 1))
    r_k = b - np.matmul(A, x_k)
    t_km2 = 0

    # Initialize Loop exit conditions
    err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
    numIters = 0

    while (err > tol) and (numIters < maxIters):
        z_k = M @ r_k
        t_km1 = z_k.T @ r_k
        if numIters == 0:
            p_k = z_k
        else:
            B_k = (t_km1)/(t_km2)
            p_k = z_k + B_k*p_k
        w_k = A @ p_k
        a_k = t_km1/(p_k.T @ w_k)
        x_k = x_k + a_k * p_k
        r_k = r_k - a_k * w_k
        t_km2 = t_km1

        # Stopping Condition Checks
        # Note: r_0 = b, because r_0 = b - Ax_0 where x_0 = 0
        err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
        numIters += 1

    return x_k, numIters

# Conjugate Gradient for Toeplitz Matrices
# and Circulant Preconditioners. This method uses
# only FFT's for matrix multiplies
#    For matrix multiplies of A @ x, the matrix A is embedded
#    in a 2n x 2n circulant matrix, and x is zero padded to 2n.
#    Then the FFT method is used:
#       c = A @ x = F^-1 @ L @ F @ x
#    or c = ifft( diag(L) * fft(x) )
def cg_Toep_FFT(A, b, tol, maxIters):
    (m, n) = np.shape(A)

    # Calculate Initial Residual / Direction
    x_k = np.zeros((n, 1))
    r_k = b - util.toepMatMul(A, x_k)
    t_km2 = 0

    # Initialize Loop exit conditions
    err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
    numIters = 0

    while (err > tol) and (numIters < maxIters):
        t_km1 = r_k.T @ r_k
        if numIters == 0:
            p_k = r_k
        else:
            B_k = (t_km1)/(t_km2)
            p_k = r_k + B_k*p_k
        w_k = util.toepMatMul(A, p_k)
        a_k = t_km1/(p_k.T @ w_k)
        x_k = x_k + a_k * p_k
        r_k = r_k - a_k * w_k
        t_km2 = t_km1

        # Stopping Condition Checks
        # Note: r_0 = b, because r_0 = b - Ax_0 where x_0 = 0
        err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
        numIters += 1

    return x_k, numIters

# Pre-Conditioned Conjugate Gradient for Toeplitz Matrices
# and Circulant Preconditioners. This method uses
# only FFT's for matrix multiplies
#    For preconditioner (circulant) multipies:
#       c = A @ x = F^-1 @ L @ F @ x
#    or c = ifft( diag(L) * fft(x) )
#
#    For matrix multiplies of A @ x, the matrix A is embedded
#    in a 2n x 2n circulant matrix, and x is zero padded to 2n.
#    Then the method above (i.e. FFT method) is used.
def preCG_ToepCirc_FFT(A, M, b, tol, maxIters):
    (m, n) = np.shape(A)

    # Calculate Initial Residual / Direction
    x_k = np.zeros((n, 1))
    r_k = b - util.toepMatMul(A, x_k)
    t_km2 = 0

    # Initialize Loop exit conditions
    err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
    numIters = 0

    while (err > tol) and (numIters < maxIters):
        z_k = util.circMatMul(M,r_k)
        t_km1 = z_k.T @ r_k
        if numIters == 0:
            p_k = z_k
        else:
            B_k = (t_km1)/(t_km2)
            p_k = z_k + B_k*p_k
        w_k = util.toepMatMul(A, p_k)
        a_k = t_km1/(p_k.T @ w_k)
        x_k = x_k + a_k * p_k
        r_k = r_k - a_k * w_k
        t_km2 = t_km1

        # Stopping Condition Checks
        # Note: r_0 = b, because r_0 = b - Ax_0 where x_0 = 0
        err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
        numIters += 1

    return x_k, numIters

# More Efficient version of cg_Toep_FFT.
# Conceptually the same as cg_Toep_FFTmin, except unnecessary calculations have been removed
# from the loop. Now:
#    1. Toep -> Circulant is only done once, before entering the loop
#    2. Calculating the Circulant matric diagonal coefficients is only done once, before entering the loop
#
# Originally circMatMul computed the diagonal coefficients each time it was called, this was unnecessary
# FFT's. The original toepMatMul created the 2n x 2n circulant matrix from the Toeplitz matrix each time it was
# called and then also called the inefficient circMatMul.
def cg_Toep_FFTmin(A, b, tol, maxIters):
    (m, n) = np.shape(A)

    # Concert nxn Toeplitz to a 2nx2n Circlant matrix (so we can use FFTs instead of matmuls)
    A2 = util.toep2Circ(A)

    # Diagonalize (via FFT because it is circulant) the main Matrix, A2
    L2 = np.fft.fft(A2[:, 0])

    # Calculate Initial Residual / Direction
    x_k = np.zeros((n, 1))
    x_k2 = np.concatenate((np.ravel(x_k), np.zeros(x_k.shape[0]))) # zero pad x_k because A is now 2nx2n
    te = np.ravel(util.circMatMulMin(L2, x_k2))
    r_k = b - np.atleast_2d(te[0:x_k.shape[0]]).T
    t_km2 = 0

    # Initialize Loop exit conditions
    err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
    numIters = 0

    while (err > tol) and (numIters < maxIters):
        t_km1 = r_k.T @ r_k
        if numIters == 0:
            p_k = r_k
        else:
            B_k = (t_km1)/(t_km2)
            p_k = r_k + B_k*p_k
        p_k2 = np.concatenate((np.ravel(p_k), np.zeros(p_k.shape[0]))) # zero pad p_k because A is now 2nx2n
        te = np.ravel(util.circMatMulMin(L2, p_k2))
        w_k = np.atleast_2d(te[0:x_k.shape[0]]).T
        a_k = t_km1/(p_k.T @ w_k)
        x_k = x_k + a_k * p_k
        r_k = r_k - a_k * w_k
        t_km2 = t_km1

        # Stopping Condition Checks
        # Note: r_0 = b, because r_0 = b - Ax_0 where x_0 = 0
        err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
        numIters += 1

    return x_k, numIters

# More efficient preCG_ToepCirc.
# Identical efficiency improvements as the CG_Toep_FFTmin. See comment there.
def preCG_ToepCirc_FFTmin(A, M, b, tol, maxIters):
    (m, n) = np.shape(A)

    # Concert nxn Toeplitz to a 2nx2n Circlant matrix (so we can use FFTs instead of matmuls)
    A2 = util.toep2Circ(A)

    # Diagonalize (via FFT because they are circulant) the main Matrix, A2, and the preconditioner, M
    L2 = np.fft.fft(A2[:, 0])
    LM = np.fft.fft(M[:, 0])

    # Calculate Initial Residual / Direction
    x_k = np.zeros((n, 1))
    x_k2 = np.concatenate((np.ravel(x_k), np.zeros(x_k.shape[0]))) # zero pad x_k because A is now 2nx2n
    te = np.ravel(util.circMatMulMin(L2, x_k2))
    r_k = b - np.atleast_2d(te[0:x_k.shape[0]]).T
    t_km2 = 0

    # Initialize Loop exit conditions
    err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
    numIters = 0

    while (err > tol) and (numIters < maxIters):
        z_k = util.circMatMulMin(LM,r_k)
        t_km1 = z_k.T @ r_k
        if numIters == 0:
            p_k = z_k
        else:
            B_k = (t_km1)/(t_km2)
            p_k = z_k + B_k*p_k
        p_k2 = np.concatenate((np.ravel(p_k), np.zeros(p_k.shape[0]))) # zero pad p_k because A is now 2nx2n
        te = np.ravel(util.circMatMulMin(L2, p_k2))
        w_k = np.atleast_2d(te[0:x_k.shape[0]]).T
        a_k = t_km1/(p_k.T @ w_k)
        x_k = x_k + a_k * p_k
        r_k = r_k - a_k * w_k
        t_km2 = t_km1

        # Stopping Condition Checks
        # Note: r_0 = b, because r_0 = b - Ax_0 where x_0 = 0
        err = np.linalg.norm(r_k, 2) / np.linalg.norm(b, 2)
        numIters += 1

    return x_k, numIters