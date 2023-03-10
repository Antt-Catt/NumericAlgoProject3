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
    N=vect_N (U, V)
    H=np.eye(len(U))-2*N*np.transpose(N)
    return H

#2-1 Produit d'une matrice de Householder par un vecteur

def produit(U, V, vect) :
    return householder_mat(U, V).dot(vect)

def produit_opti(U, V, vect) :
    N = vect_N(U, V)
    return vect - 2*np.dot(np.transpose(vect),N)*N

#2-1 Produit d'une matrice de Householder par une matrice

def produit_mat_opti(U, V, M):
    N = vect_N(U, V)
    R = np.empty((np.shape(M)[0],0))
    for i in range(np.shape(M)[0]):
        column = np.reshape(M[:,i], (np.shape(M)[0], 1))
        vect = column - 2*np.dot(np.transpose(column),N)*N
        R = np.hstack((R, vect))
    return R

if __name__ == "__main__":

    #1-3 Test de fonctionnement
    U=np.array([[3],[4],[0]])
    V=np.array([[0],[0],[5]])
    T=np.array([[3],[6],[1]])
    H=householder_mat(U, V)
    # H(U)=V
    print(H.dot(T))


    #2-1 Test du produit par HH
    # print(produit(U, V, T))
    # print(produit_opti(U, V, T))

    #2-2 Test du produit matriciel par HH
    K = np.hstack((U, V, T))
    print(K)
    print(H@K)
    print(produit_mat_opti(U, V, K))

    #+end_src