import numpy as np


#紧奇异矩阵分解
#其中rank(A) = r  r<=min(m,n) 称 A=Ur∑rVr (V为转置之后的)
#Ur指的是原来U的前r列
#∑r指的是原来∑的前r个对角元素  其秩与原矩阵的秩相等
#Vr指的是原来V的前r列
if __name__ == '__main__':
    data = np.mat([[1,2,3,4,5,6,7,8,9],
                [5,6,7,8,9,0,8,6,7],
                [9,0,8,7,1,4,3,2,1],
                [6,4,2,1,3,4,2,1,5]])

    k= np.linalg.matrix_rank(data)
    print(k)
    u , s ,v = np.linalg.svd(data,full_matrices=True)
    u = u[:,:k]
    s = s[:k]
    v = v[:k,:] #已转置的矩阵
    print(s)
    matrix_s = np.diag(s)
    matrix_s = np.mat(matrix_s)
    data_CSVD = np.dot(np.dot(u,matrix_s),v)
    print(data)
    print(data_CSVD)