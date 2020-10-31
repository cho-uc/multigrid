# Multigrid application (adapted from Multigrid solvers by sussmanm@math.pitt.edu)

import numpy as np

# perform one v-cycle on the matrix A
def vcycle(A,f):
    sizeF = np.size(A,axis=0);
    # size for direct inversion < 15
    if sizeF < 15:
        v = np.linalg.solve(A,f)
        return v
    
    # Pre-smoothing -------------------------------------
    N1 = 5;     # number of Gauss-Seidel iterations before coarsening
    v = np.zeros(sizeF);
    for numGS in range(N1):
        for k in range(sizeF):
            v[k] = (f[k] - np.dot(A[k,0:k], v[0:k]) \
                -np.dot(A[k,k+1:], v[k+1:]) ) / A[k,k];
    
    # construct interpolation operator from next coarser to this mesh
    # next coarser has ((n-1)/2 + 1 ) points
    assert(sizeF%2 ==1)
    sizeC = int((sizeF-1)/2 +1)
    
    print("sizeF = ", sizeF)
    print("sizeC = ", sizeC)
    # create prolongation matrix
    P = np.zeros((sizeF,sizeC));
    for k in range(sizeC):
        P[2*k,k] = 1;           # copy these points
    for k in range(sizeC-1):
        P[2*k+1,k] = .5;        # average these points
        P[2*k+1,k+1] = .5;
    # compute defect
    residual = f - np.dot(A,v)
    
    # project defect onto coarser mesh (restriction)
    residC = np.dot(P.transpose(),residual)
    
    # Find coarser matrix (sizeC X sizeC)
    AC = np.dot(P.transpose(),np.dot(A,P))
    
    vC = vcycle(AC,residC); # recursive to coarser level, until reach bottom
    
    # extend to this mesh
    v = np.dot(P,vC)
    
    # Post-smoothing -------------------------------------
    N2 = 5; # number of Gauss-Seidel iterations after coarsening
    for numGS in range(N2):
        for k in range(sizeF):
            v[k] = (f[k] - np.dot(A[k,0:k], v[0:k]) \
            -np.dot(A[k,k+1:], v[k+1:]) ) / A[k,k];
    
    return v   

if __name__ == "__main__":
    #N = 2**9+1
    N = 2**8+1
    x = np.linspace(0,1,N);
    h = x[1]-x[0]
    
    # tridiagonal matrix
    A = np.diag(2.*np.ones(N)) - np.diag(np.ones(N-1), 1)
    A = A/h**2
    f = np.ones(N, dtype=float) #rhs
    udirect = np.linalg.solve(A, f) # correct solution
    
    u = np.zeros(N) # initial guess
    print("N = ", + N)
    print("x = ")
    print(x)
    print("A = ")
    print(A)
    print("f = ")
    print(f)
    print("udirect = ")
    print(udirect)
    
    
    for iters in range(100):
        r = f - np.dot(A,u)
        if np.linalg.norm(r)/np.linalg.norm(f) < 1.e-10:
            break
        du = vcycle(A, r)
        u += du
        print ("step %d, rel error= %e"% \
            (iters+1, np.linalg.norm(u-udirect)/np.linalg.norm(udirect) ))
    
    
    