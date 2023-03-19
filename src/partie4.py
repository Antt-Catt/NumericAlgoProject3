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
    if k >= A.shape[0] :
        return A
    else :
        #D'après l'énoncé, on doit annuler les termes diagonaux de S d'indices strictement supérieurs à k
        S_comp = np.zeros(A.shape)
        for i in range(k+1) :
            S_comp[i,i] = ssv[i]
        A_comp = U@(S_comp)@V
        return A_comp


def compressRGB(img_full, k) :
    img_comp = np.array([compress(img_full[:,:,0], k), compress(img_full[:,:,1], k), compress(img_full[:,:,2], k)])
    img_comp = np.swapaxes(img_comp, 0, 1)
    img_comp = np.swapaxes(img_comp, 1, 2)
    return img_comp



if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    img_full = plt.imread("../files/p3_takeoff_base.png")
    n, m, colors = img_full.shape
    print(n,m,colors)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print(img_full[:,:,0])
    print(img_full[:,:,1])
    print(img_full[:,:,2])

    k=299
    img_comp = compressRGB(img_full, k)
    print(img_comp[:,:,0])
    print(img_comp[:,:,1])
    print(img_comp[:,:,2])

    print()
    for x in np.nditer(img_comp):
        if x>=1 :
            print(x, end=' ')
    plt.imsave("../files/p3_compress.png", img_comp)
