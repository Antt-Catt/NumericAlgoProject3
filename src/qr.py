import numpy as np

#Entrée : U et V matrices identités, S matrice bidiagonale
#Sortie : (U,S,V) mmatrices de décomposition SVD, avec S diagonale
def transfo_qr(U, V, S):
    for i in range(1000):
        (Q1, R1) = np.linalg.qr(S.T)
        (Q2, R2) = np.linalg.qr(R1.T)
        S = R2
        U = U@Q2
        V = Q1.T@V
    return (U, S, V)

if __name__ == "__main__":
    #np.set_printoptions(formatter={'float': lambda x: "{0:0.01f}".format(x)})
    U = np.identity(4)
    V = np.identity(4)
    S = np.array([[1,2,0,0],[0,5,8,0],[0,0,3,2],[0,0,0,2]])
    print(transfo_qr(U,V,S)[0])
    print(transfo_qr(U,V,S)[1])
    print(transfo_qr(U,V,S)[2])
