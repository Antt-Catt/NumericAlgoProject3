import numpy as np
import matplotlib.pyplot as plt

#img_full = plt.imread("p3_takeoff_base.png")
#n, m, colors = img_full.shape
#print(n,m,colors)

#PARTIE 1: MATRICE DE HOUSEHOLDER

#Question préliminaire

def vect_N (U, V) :
    S = U - V
    norme = np.linalg.norm(S)
    N = S / norme
    return N

#La matrice de Householder

def householder_mat(U, V) :
    N = vect_N (U, V)
    H = np.eye(len(U)) - 2*N*np.transpose(N)
    return H

#Test de fonctionnement

U=np.array([[3],[4],[0]])
V=np.array([[0],[0],[5]])
H=householder_mat(U, V)
# H(U)=V
print(H.dot(U))
#+end_src
