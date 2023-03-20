import numpy as np

import partie1

""" Creates a vector with a single non-zero element (index i)"""
def vect_ei(n,i):
    vect = np.zeros(n)
    vect[i] = 1
    return vect



def padWithId(Q, n):
    k = Q.shape[0]
    pad_width = ((n-k, 0), (n-k, 0))  # la taille du padding
    M_pad = np.pad(Q, pad_width, mode='constant')  # la matrice avec le padding
    for i in range(n-k):
        M_pad[i,i] = 1
    return M_pad

def to_bidiag(BD):
    n, m = BD.shape
    Qleft = np.identity(n)
    Qright = np.identity(m)
    for i in range(n):
        Q1 = partie1.householder_mat(BD[i:n,i], vect_ei(n-i, 0) * np.linalg.norm(BD[i:n,i]))
        Q1 = padWithId(Q1,n)
        Qleft = Qleft@Q1
        BD = Q1@BD
        if i==m-1 :
            break
        if i != (m - 2) :
            Q2 = partie1.householder_mat(BD[i,(i+1):m], vect_ei(m-i-1, 0)* np.linalg.norm(BD[i,(i+1):m]))
            Q2 = padWithId(Q2,m)
            Qright = Q2@Qright
            BD = BD@Q2
        print(Qleft @ BD @ Qright)
    return (Qleft, BD, Qright)




def to_bidiag_opti(BD):
    n, m = BD.shape
    Qleft = np.identity(n)
    Qright = np.identity(m)
    for i in range(n):
        Q1 = partie1.produit_mat_opti(BD[i:n,i], vect_ei(n-i, 0) * np.linalg.norm(BD[i:n,i]), Qleft[i:,i:])
        Qleft[i:,i:] = Q1
        BD = Qleft@BD
        if i==m-1 :
            break
        if i != (m - 2) :
            Q2 = partie1.produit_mat_opti(BD[i,(i+1):m], vect_ei(m-i-1, 0)* np.linalg.norm(BD[i,(i+1):m]), Qright[i+1:,i+1:])
            Qright[i+1:,i+1:] = Q2
            BD = BD@Qright
        print(Qleft @ BD @ Qright)
    return (Qleft, BD, Qright)

if __name__ == "__main__":
    A = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print(to_bidiag(A)[1])

    print("\n\n\n")

    A = np.array([[1,2,3], [3,4,5], [5,6,7], [7,8,9]])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print(to_bidiag(A)[1])

    print("\n\n\n")

    A = np.array([[1,2,3], [3,4,5], [5,6,7], [7,8,9]])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print(to_bidiag_opti(A)[1])
