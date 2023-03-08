import numpy as np

def to_bidiag(Qleft, Qright, BD):
    for i in range(np.shape(BD)[0]):
        Q1 = #create HH where U = B[i:n,i] and V = ...
        Qleft = Qleft@Q1
        BD = Q1@BD
        if i != (np.shape(A)[1] - 2):
            Q2 = #create HH where U = B[i,(i+i):m] and V = ...
            Qright = Q2@Qright
            BD = BD@Q2
    return (Qleft, BD, Qright)

if __name__ == "__main__":
    Q2 = np.identity(4)
    Q1 = np.identity(4)
    A = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]])
    print(to_bidiag(Q1,Q2,A))
