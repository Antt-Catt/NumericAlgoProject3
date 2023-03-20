import numpy as np
import matplotlib.pyplot as plt
import partie1
import partie2

#Entrée : U et V matrices identités, S matrice bidiagonale
#Sortie : (U,S,V) mmatrices de décomposition SVD, avec S diagonale
def transfo_qr(U, V, S):
    conv_array = []
    for i in range(100):
        diff = 0 # initialise le compteur d'elt extradiagonaux
        for j in range(np.shape(S)[0]):
            for k in range(np.shape(S)[1]):
                if j != k:
                    if S[j][k] > 10**-5 or S[j][k] < -10**-5: # verifie si l'elt extradiagonal est non-negligeable (precision de 10^-5 ici)
                        diff += 1
        conv_array.append(diff)
        (Q1, R1) = np.linalg.qr(S.T)
        (Q2, R2) = np.linalg.qr(R1.T)
        S = R2
        U = U@Q2
        V = Q1.T@V
    plt.plot(conv_array)
    plt.yticks(range(int(min(conv_array)), int(max(conv_array))+1)) # graduations entières
    plt.xlabel("Itérations")
    plt.ylabel("Nombre d'EENN")
    plt.show()
    return (U, S, V)

def change_U_S(U, S):
    n, m = S.shape
    for i in range(m):
        if (res[1])[i][i] < 0:
            for j in range(n):
                 (res[0])[j][i] = -(res[0])[j][i]
            (res[1])[i][i] = -(res[1])[i][i]
    return (res[0], res[1])

def qr_simpl(A):
    m, n = A.shape
    Q = np.eye(n)
    R = A.copy()
    for j in range(n):
        u = R[j:,j]
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            continue
        v = np.zeros(len(u))
        v[0] = norm_u
        H = np.eye(m)
        H[j:, j:] = partie1.householder_mat(u, v)
        Q = np.dot(Q, H)
        R = np.dot(H, R)
    return Q, R

def transfo_qr_simpl(U, V, S):
    for i in range(1000):
        (Q1, R1) = qr_simpl(S.T)
        (Q2, R2) = qr_simpl(R1.T)
        S = R2
        U = U@Q2
        V = Q1.T@V
    return (U, S, V)

if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True)
    U = np.eye(4,4)
    V = np.eye(4,4)
    S = np.array([[1,2.5,0,0],[0,11,4,0],[0,0,5,6],[0,0,0,7]])

    res = transfo_qr(U,V,S)

    new = change_U_S(res[0], res[1]) # U et S modifiées pour que elts de S soient > 0 et décroissants
    for i in range(3):
        if (new[1][i][i] < new[1][i+1][i+1]) or (new[1][i+1][i+1] < 0):
            print("S ne respecte pas la propriete des elts diagonaux > 0 et decroissants") 
    
    print(res[0], "U")
    print(res[1], "S")
    print(res[2], "V")

    print("UxSxV =", new[0]@new[1]@res[2])
    print("BD =", S)

    test1 = np.linalg.qr(S)
    test2 = qr_simpl(S)
    print(test1[0], "np")
    print(test2[0], "moi")
    print(test1[1], "np")
    print(test2[1], "moi")

    U = np.eye(10, 10)
    V = np.eye(10, 10)
    S = partie2.to_bidiag(np.random.rand(10,10))[1]
    res = transfo_qr(U,V,S)
