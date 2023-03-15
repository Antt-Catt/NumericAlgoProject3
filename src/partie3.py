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

def change_U_S(U, S):
    n, m = S.shape
    for i in range(m):
        if (res[1])[i][i] < 0:
            for j in range(n):
                 (res[0])[j][i] = -(res[0])[j][i]
            (res[1])[i][i] = -(res[1])[i][i]
    return (res[0], res[1])
        

if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True)
    U = np.eye(4,4)
    V = np.eye(4,4)
    S = np.array([[1,2,0,0],[0,3,4,0],[0,0,5,6],[0,0,0,7]])

    res = transfo_qr(U,V,S)

    new = change_U_S(res[0], res[1]) # U et S modifiées pour que elts de S soient > 0 et décroissants
    print(res[0], "U")
    print(res[1], "S")
    print(res[2], "V")

    print("UxSxV =", new[0]@new[1]@res[2])
    print("BD =", S)
