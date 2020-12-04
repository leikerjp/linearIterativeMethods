import numpy as np
import scipy.linalg as sp

# Matrix Mult for Circulant A matrix. This method uses
# only FFT's for matrix multiplies
#       c = A @ x = F^-1 @ L @ F @ x
#    or c = ifft( diag(L) * fft(x) )
# Note: diag(L) is just the fft of col0 of A
def circMatMul(A,x):
    return np.atleast_2d(np.fft.ifft(np.fft.fft(A[:, 0]) * np.fft.fft(np.ravel(x)))).T

# Minimal Circulant Matrix Multiply. This function expects to be given the diagonalized
# values, L, for the circulant matrix as a result of diagonalization via Fourier Matrices.
# Other than eliminating the calculation of the diagonalized value, this routine is identical
# to cricMatMul
def circMatMulMin(L,x):
    return np.atleast_2d(np.fft.ifft(L * np.fft.fft(np.ravel(x)))).T

# Function To Embed an nxn Toeplitz Matrix inside
# a 2nx2n Circulant Matrix. The original Toepltiz matrix
# is located in the top left nxn block of the new matrix
def toep2Circ(T):
    # Create a col
    a = T[:,0]
    b = np.zeros(1)
    c = np.flip(T[0,1:T.shape[1]])
    col = np.concatenate((a,b,c))
    # Create a row
    a = T[0,:]
    b = np.zeros(1)
    c = np.flip(T[1:T.shape[0],0])
    row = np.concatenate((a,b,c))
    return sp.toeplitz(col,row)

# Matrix Mult for Toeplitz Matrices.
# For matrix multiplies of A @ x, the matrix A is embedded
# in a 2n x 2n circulant matrix, and x is zero padded to 2n.
# Then the circulant matrix mult method (i.e. FFT method) is used.
def toepMatMul(A,x):
    # Need to unravel because indexing 0:n on last line only works for n, arrays (not n,1)
    b = np.ravel(circMatMul(toep2Circ(A), np.concatenate((np.ravel(x), np.zeros(x.shape[0])))))
    return np.atleast_2d(b[0:x.shape[0]]).T
