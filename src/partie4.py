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
    R = img_full[:,:,0]
    G = img_full[:,:,1]
    B = img_full[:,:,2]
    img_comp = np.array([compress(R, k), compress(G, k), compress(B, k)])
    img_comp = np.swapaxes(img_comp, 0, 1)
    img_comp = np.swapaxes(img_comp, 1, 2)

    compressed = (np.minimum(img_comp, 1.0) * 0xff).astype(np.uint8)
    return compressed

def map_compression_efficiency(img_full):
    x = list()
    y = list()
    original_size = os.stat("../files/p3_takeoff_base.png").st_size
    for k in range(0,101,5):
        x.append(k)
        img_comp = compressRGB(img_full, k)
        plt.imsave("../files/p3_compress.png", img_comp)
        y.append(os.stat("../files/p3_compress.png").st_size / original_size)
    plt.plot(x, y)
    plt.title("Ratio de compression en fonction du rang k")
    plt.ylabel("Ratio taille_compressée/taille_originale")
    plt.xlabel("Rang de compression k")
    plt.show()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    img_full = plt.imread("../files/p3_takeoff_base.png")
    n, m, colors = img_full.shape
    print(n,m,colors)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

    # relative_rank = 0.2
    # k = int(relative_rank * min(img_full.shape[0], img_full.shape[1]))

    k = 9
    print("max rank = %d" % k)

    img_comp = compressRGB(img_full, k)

    map_compression_efficiency(img_full)
    print("Map ok !")

    # for x in np.nditer(img_comp):
    #     if x>=1 :
    #         print(x, end=' ')
    plt.imsave("../files/p3_compress.png", img_comp)
