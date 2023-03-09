import numpy as np

import partie1

""" Creates a vector with a single non-zero element (index i)"""
def vect_ei(n,i):
    vect = np.zeros(n)
    vect[i] = 1
    return vect

def to_bidiag(BD):
    n = np.shape(BD)[0]
    m = np.shape(BD)[1]
    Qleft = np.identity(n)
    Qright = np.identity(n)
    for i in range(n):
        Q1 = partie1.householder_mat(BD[i:n,i], vect_ei(n, i))
        Qleft = Qleft@Q1
        BD = Q1@BD
        if i != (m - 2):
            Q2 = partie1.householder_mat(BD[i,(i+1):m], vect_ei(m-1, i))
            Qright = Q2@Qright
            BD = BD@Q2
    return (Qleft, BD, Qright)

if __name__ == "__main__":
    A = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
    print(A[0,(0+1):np.shape(A)[1]])
    print(to_bidiag(A))

    # A = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6]])
    # print(to_bidiag(Q1,Q2,A))
