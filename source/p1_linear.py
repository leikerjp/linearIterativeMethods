# ======================================================================================================================
# Author:       Jordan Leiker
# GTid:         903453031
# Class:        MATH6644 - Iterative Methods for Solving Systems of Equations
# Date:         03/19/2020
#
# Title:        Project 1 (Linear)
# ======================================================================================================================

import numpy as np
import scipy.linalg as sp
import conjugateGradient as cg
import matplotlib.pyplot as plt
import time
import pandas

desiredWidth = 400
pandas.set_option('display.width', desiredWidth)
pandas.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desiredWidth)


# Function to generate Test Toeplitz Matrix
def genToeplitzTest(n):
    col = np.arange(1,n+1)
    row = np.concatenate((np.array([col[0]]),np.arange(col[n-1]+1,col[n-1]+n)))
    return sp.toeplitz(col,row)

# Function to generate Toeplitz A
# i.e. ak = |k+1|^(-p) where p = 2, 1, 1/10, 1/100
def genToeplitzA(n, p):
    col = np.zeros(n)
    for k in range(n):
        col[k] = (np.abs(k+1))**(np.float(-p))
    return sp.toeplitz(col,col[0:])

# Function to generate Toeplitz B
# i.e. ak = 1/(2pi)*integral(f(theta)*exp(-iktheta) dtheta) for k = 0, +/- 1, +/-2, etc
# Note: f(theta) = theta^4 + 1 for -pi <= theta <= pi
def genToeplitzB(n):
    theta = np.linspace(-np.pi,np.pi,1000)
    f = (theta**4 + 1)
    col = np.fft.fft(f,n)
    return sp.toeplitz(col,col[0:])

# Function to generate G. Strangs circulant preconditioner:
# cj =  a_j     0<= j <= [n/2]
#       a_j-n   [n\2] < j < n
#       c_n+j   0 < -j < n
def genStrangPreconditioner(A):
    M = np.zeros(A.shape)
    M = M.astype(np.complex)
    n = M.shape[0]
    n2 = np.round(n/2)
    for l in range(n):
        for k in range(n):
            j = (k-l)
            if (j >= 0) and (j <= n2):
                M[k,l] = A[j,0]
            elif (j > n2) and (j < n):
                M[k,l] = A[0,n-j] #note: n-j == abs(j-n) in this case
            else:
                M[k,l] = M[n+j,0]
    return M

# Function to generate G. Strangs circulant preconditioner:
# cj =  ((n-j)aj + j*a_j-n)/n     0<= j < n
#       c_n+j   0 < -j < n
def genChanPreconditioner(A):
    M = np.zeros(A.shape)
    M = M.astype(np.complex)
    n = M.shape[0]
    for l in range(n):
        for k in range(n):
            j = (k - l)
            if (j == 0):
                M[k,l] = ((n-j)*A[j, 0])/n
            elif (j > 0) and (j < n):
                M[k,l] = ((n-j)*A[j,0] + j*A[0,n-j])/n  #note: n-j == abs(j-n) in this case
            else:
                M[k,l] = M[n+j,0]
    return M


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Setup Environment
    # ------------------------------------------------------------------------------------------------------------------
    matrix = "TA" # TA | TB | TEST
    PRINT_ENABLE = False
    PLOT_ENABLE = True
    p = 1/100 # 2, 1, 1/10, 1/100
    maxIters = 1e4
    tol = 1e-6
    #nlist = np.array([2,3,4,5,6,7,8])
    #nlist = np.array([50,100,150,200])
    nlist = np.arange(50,1650,50)
    #nlist = np.array([50,100,200,400,800,1600])
    nsize = nlist.shape[0]
    # For arrays:
    # :,0 = CG w/ matrix multiplies
    # :,1 = CG w/ FFT's
    # :,2 = PCG w/ Strang w/ matrix multiplies
    # :,3 = PCG w/ Strang w/ FFT's
    # :,4 = PCG w/ Chan w/ matrix multiplies
    # :,5 = PCG w/ Chan w/ FFT's
    times = np.zeros((nsize,6))
    iters = np.zeros((nsize,6))

    # ------------------------------------------------------------------------------------------------------------------
    # Iterate over all the matrix sizes in nlist
    for idx in range(nsize):
        n = nlist[idx]
        print("Now performing calculations for n=" + "{}".format(n))

        # --------------------------------------------------------------------------------------------------------------
        # Generate Matrices / vectors
        # --------------------------------------------------------------------------------------------------------------
        # Select Toeplitz A or Toeplitz B (never do both for the sake of time / plotting convenience)
        if matrix == "TA":
            A = genToeplitzA(n,p)
        elif matrix == "TB":
            A = genToeplitzB(n)
        else:
            A = genToeplitzTest(n)

        if PRINT_ENABLE:
            print("A=")
            print(A)
            eigval, eigvec = np.linalg.eig(A)
            print("A eig vals=")
            print(eigval)
            # print("A condition number =")
            # print(np.linalg.cond(A))
            # print("row sum, minus diagonal")
            # B = A - np.diag(np.diag(A))
            # for j in range(A.shape[0]):
            #     print(np.sum(B[j,:]))

        # generate circulant pre-conditioners
        Mstrang = genStrangPreconditioner(A)
        Mchan = genChanPreconditioner(A)

        if PRINT_ENABLE:
            print("Strang Precon. M=")
            print(Mstrang)
            print("Chan Precon. M=")
            print(Mchan)

        # Generate Random b vector for system to solve with
        #b = np.random.rand(n,1)
        b = np.ones((n,1))

        # --------------------------------------------------------------------------------------------------------------
        # Solve using each method
        # --------------------------------------------------------------------------------------------------------------
        # TEST: Solve using inverse
        if PRINT_ENABLE:
            x = (np.linalg.inv(A) @ b)
            print("Inverse, x=\t\t\t\t" + "{}".format(x[0:3].T))

        # --------------------------------------------------------------------------------------------------------------
        # Solve using CG (w/ matrix multiplies)
        start = time.time()
        x, iters[idx,0] = cg.conjugateGradient(A, b, tol, maxIters)
        end = time.time()
        times[idx, 0] = end - start
        if PRINT_ENABLE:
            print("CG, x=\t\t\t\t\t" + "{}".format(x[0:3].T))

        # Solve using CG (w/ FFTs, w/o matrix multiplies - efficient)
        start = time.time()
        x, iters[idx, 1] = cg.cg_Toep_FFTmin(A, b, tol, maxIters)
        end = time.time()
        times[idx, 1] = end - start
        if PRINT_ENABLE:
            print("CG (fftmim), x=\t\t\t" + "{}".format(x[0:3].T))

        # --------------------------------------------------------------------------------------------------------------
        # Solve using PCG w/ Strang (w/ matrix multiplies)
        start = time.time()
        x, iters[idx,2] = cg.preconditionedConjugateGradient(A, Mstrang, b, tol, maxIters)
        end = time.time()
        times[idx,2] = end-start
        if PRINT_ENABLE:
            print("PCG-Strang, x=\t\t\t" + "{}".format(x[0:3].T))

        # Solve using PCG w/ Strang (w/ FFTs, w/o matrix multiplies - efficient)
        start = time.time()
        x, iters[idx, 3] = cg.preCG_ToepCirc_FFTmin(A, Mstrang, b, tol, maxIters)
        end = time.time()
        times[idx, 3] = end - start
        if PRINT_ENABLE:
            print("PCG-Strang (fftmin), x=\t" + "{}".format(x[0:3].T))

        # --------------------------------------------------------------------------------------------------------------
        # Solve using PCG /w Chan
        start = time.time()
        x, iters[idx, 4] = cg.preconditionedConjugateGradient(A, Mchan, b, tol, maxIters)
        end = time.time()
        times[idx,4] = end-start
        if PRINT_ENABLE:
            print("PCG-Chan, x=\t\t\t" + "{}".format(x[0:3].T))

        # Solve using PCG w/ Chan (w/ FFTs, w/o matrix multiplies - efficient)
        start = time.time()
        x, iters[idx, 5] = cg.preCG_ToepCirc_FFTmin(A, Mchan, b, tol, maxIters)
        end = time.time()
        times[idx, 5] = end - start
        if PRINT_ENABLE:
            print("PCG-Chan (fftmin), x=\t" + "{}".format(x[0:3].T))

    # ------------------------------------------------------------------------------------------------------------------
    # Display Info / Plot
    # ------------------------------------------------------------------------------------------------------------------
    print("\nConjugate Gradient (no precon.):")
    # print("Time to Converge\t\tNumber of Iterations")
    data = np.vstack((times[:,0],iters[:,0],times[:,1],iters[:,1])).T
    print(pandas.DataFrame(data,nlist,columns=['Times (matmul)','Iters (matmul)','Times (FFT)','Iters (FFT)']))

    print("\nPreconditioned Conjugate Gradient /w Strang Precon.")
    data = np.vstack((times[:,2],iters[:,2],times[:,3],iters[:,3])).T
    print(pandas.DataFrame(data,nlist,columns=['Times (matmul)','Iters (matmul)','Times (FFT)','Iters (FFT)']))

    print("\nPreconditioned Conjugate Gradient /w Chan Precon.")
    data = np.vstack((times[:,4],iters[:,4],times[:,5],iters[:,5])).T
    print(pandas.DataFrame(data,nlist,columns=['Times (matmul)','Iters (matmul)','Times (FFT)','Iters (FFT)']))

    # Figure comparing each "time to converge vs N" to "n*log(n)" and "N^2"
    ops_nlogn = nlist * np.log(nlist)
    ops_npp2 = nlist**2

    fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey='row')

    ax1.plot(nlist,times[:,0], label="CG w/ MatMul (time)", color='blue')
    ax1.plot(nlist,times[:,1], label="CG w/ FFT (time)", color ='green')
    ax1_2 = ax1.twinx()
    ax1_2.plot(nlist, ops_nlogn, label="Nlog(N)", color ='red')
    ax1_2.plot(nlist, ops_npp2, label="N^2", color ='purple')
    ax1_2.set_yticklabels([])
    ax1.set(xlabel='N', ylabel='Time',
            title='CG: Comparison of Calc Time, NlogN, and N^2')
    ax1.legend()
    ax1_2.legend()

    ax2.plot(nlist,times[:,2], label="PCG-Strang w/ MatMul (time)", color='blue')
    ax2.plot(nlist,times[:,3], label="PCG-Strang w/ FFT (time)", color ='green')
    ax2_2 = ax2.twinx()
    ax2_2.plot(nlist, ops_nlogn, label="Nlog(N)", color ='red')
    ax2_2.plot(nlist, ops_npp2, label="N^2", color ='purple')
    ax2_2.set_yticklabels([])
    ax2.set(xlabel='N', ylabel='',
            title='PCG-Strang: Comparison of Calc Time, NlogN, and N^2')
    ax2.legend()
    ax2_2.legend()

    ax3.plot(nlist,times[:,4], label="PCG-Chan w/ MatMul (time)", color='blue')
    ax3.plot(nlist,times[:,5], label="PCG-Chan w/ FFT (time)", color ='green')
    ax3_2 = ax3.twinx()
    ax3_2.plot(nlist, ops_nlogn, label="Nlog(N)", color ='red')
    ax3_2.plot(nlist, ops_npp2, label="N^2", color ='purple')
    ax3_2.set(ylabel='NlogN / N^2')
    ax3.set(xlabel='N', ylabel='',
            title='PCG-Chan: Comparison of Calc Time, NlogN, and N^2')
    ax3.legend()
    ax3_2.legend()

    ax1.set_ylim(top=1.1*np.max(times))
    fig.tight_layout()


    if PLOT_ENABLE:
        plt.show()