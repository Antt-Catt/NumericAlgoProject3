import os
import numpy as np
import matplotlib.pyplot as plt


def compress(A, k) :
    #On commence déjà par décomposer A en valeurs singulières
    U, S, V = np.linalg.svd(A)

    #On trie ensuite les valeurs diagonales de S en ordre décroissant
    S = np.diag(S)
    ssv = sorted(np.diag(S), reverse=True)

    #On traite le cas k = la taille de A
    if k == A.shape[0] :
        return A
    else :
        #D'après l'énoncé, on doit annuler les termes diagonaux de S d'indices strictement supérieurs à k
        S_comp = np.zeros(A.shape[0])
        for i in range(k) :
            s_comp[i,i] = ssv[i]
        A_comp = U*(S_comp)*V
        return A_comp
