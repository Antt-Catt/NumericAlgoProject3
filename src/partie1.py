import numpy as np
import matplotlib.pyplot as plt

#img_full = plt.imread("p3_takeoff_base.png")
#n, m, colors = img_full.shape
#print(n,m,colors)

#PARTIE 1: MATRICE DE HOUSEHOLDER

#1- CONSRUCTION DE LA MATRICE DE HOUSEHOLDER

#1-1 Construction du vecteur N
def vect_N(U, V) :
    S=U-V
    norme=np.linalg.norm(S)
    N=S/norme
    return N

#1-2 La matrice de Householder

def householder_mat(U, V) :
    if np.all(U == V) or np.all(U == -V):
        return np.eye(len(U))
    N = vect_N(U, V)
    H = np.eye(len(U)) - 2*np.outer(N, N)
    return H

#2-1 Produit d'une matrice de Householder par un vecteur

#Version primaire (non optimale)
def produit(U, V, vect) :
    return np.dot(householder_mat(U, V), vect)

#Version optimale
def produit_opti(U, V, vect) :
    N = vect_N(U, V)
    return vect - 2*np.dot(np.transpose(vect),N)*N

#2-1 Produit d'une matrice de Householder par une matrice

def produit_mat_opti(U, V, M):
    N = vect_N(U, V)
    R = np.empty((np.shape(M)[0],0))
    for i in range(np.shape(M)[1]):
        vect = M[:,i] - 2*np.dot(np.transpose(M[:,i]),N)*N
        column = np.reshape(vect, (np.shape(M)[0], 1))
        R = np.hstack((R, column))
    return R

if __name__ == "__main__":

    #1-3 Test de fonctionnement
    U=np.array([[3],[4],[0]])
    V=np.array([[0],[0],[5]])
    T=np.array([[3],[6],[1]])
    H=householder_mat(U, V)
    # H(U)=V
    # print(H.dot(T))


    #2-1 Test du produit par HH
    # print(produit(U, V, T))
    # print(produit_opti(U, V, T))


    #2-2 Test du produit matriciel par HH
    K = np.hstack((U, V, T))
    # print(K)
    # print(H@K)
    # print(produit_mat_opti(U, V, K))

    A = np.array([[1,2,3], [3,4,5], [5,6,7], [7,8,9]])
    print(np.shape(A))
    print(A[:,0])
    ei = np.array([1, 0, 0, 0])
    print(produit_mat_opti(A[0:4,0], ei * np.linalg.norm(A[0:4,0]), A))

    

    #+end_src
