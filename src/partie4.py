import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    img_full = plt.imread("../files/p3_takeoff_base.png")
    n, m, colors = img_full.shape
    print(n,m,colors)

    

def compression(S, r, n):
    for i in range(n):
        if (S[i][i] > r):
            S[i][i] = 0

