import numpy as np
#截断奇异值矩阵分解
#指的是对弈奇异矩阵分解中只取最大的k个值  k<r (r是A的秩)
#其中rank(A) = r  0<k<r 称 A=Uk∑kVk (V为转置之后的)
#A≈Uk∑kVk
#Uk指的是原来U的前k列
#∑k指的是原来∑的最大的k个对角元素  是k阶对角矩阵
#Vk指的是原来V的前k列
#当k越小  则和原矩阵差的越大    而紧奇异值分解是无损压缩

if __name__ == '__main__':
    data = np.mat([[1,2,3,4,5,6,7,8,9],
                [5,6,7,8,9,0,8,6,7],
                [9,0,8,7,1,4,3,2,1],
                [6,4,2,1,3,4,2,1,5]])

    rank= np.linalg.matrix_rank(data)
    k = np.random.randint(1,rank)
    print(rank,k)
    u , s ,v = np.linalg.svd(data,full_matrices=True)
    u = u[:,:k]
    print(s)
    #numpy 中 argsort是从大到小  list中 sort(reverse=True) [1,23].sort(reverse)
    s.argsort()
    s = s[:k]
    v = v[:k,:] #已转置的矩阵
    matrix_s = np.diag(s)
    matrix_s = np.mat(matrix_s)
    data_TSVD = np.dot(np.dot(u,matrix_s),v)
    print(data)
    print(data_TSVD)