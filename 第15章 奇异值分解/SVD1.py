import numpy as np

if __name__ == '__main__':
    A = np.array([[0,1],[1,1],[1,0]])
    ATA = np.dot(A.T,A)
    AAT = np.dot(A,A.T)
    lamATA,vectATA = np.linalg.eig(ATA) #求特征值  特征向量
    lamAAT,vectAAT = np.linalg.eig(AAT)

    bate1,bate2=np.sqrt(lamATA)
    sum = np.zeros((vectAAT.shape[1],vectATA.shape[0]))
    sum[0,0],sum[1,1]=bate1,bate2
    res = np.dot(np.dot(vectAAT,sum),vectATA)
    # print(res,A)

    #python 的mat函數
    x = np.random.rand(3,3) #x是array的类型
    y = np.mat(x) #将x的类型转成了矩阵类型
    print(type(y))

    a = np.mat([[1,2,3],[4,5,6]])
    #sigma是一个向量，其中包括的是奇异值，如果想重构a  必须将simga恢复成正确的结构
    u,simga,v = np.linalg.svd(a)
    # print(u)
    # print(simga)
    # print(v)


    #奇异矩阵分解
    data = np.mat([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9, 0, 8, 6, 7],
                [9, 0, 8, 7, 1, 4, 3, 2, 1],
                [6, 4, 2, 1, 3, 4, 2, 1, 5]])
    print ("data的维度：", np.shape(data))
    U, sigma, VT = np.linalg.svd(data, full_matrices=False)
    print("svd近似")
    print("-----------------1-U----------------------")
    print(U)
    print(np.shape(U))
    print("-----------------1-Σ-----------------------")
    print(sigma)
    print(np.shape(sigma))
    print("------------------1-VT---------------------")
    print(VT)
    print(np.shape(VT))
    print("svd分解")
    print("------------------2-U---------------------")
    U, sigma, VT = np.linalg.svd(data, full_matrices=True)
    print(U)
    print(np.shape(U))
    print("-------------------2---------------------")
    print(sigma)
    print(np.shape(sigma))
    print("-------------------2-VT--------------------")
    print(VT)
    print(np.shape(VT))



# 函数：np.linalg.svd(a,full_matrices=1,compute_uv=1)。
#
# 参数：
#
# a是一个形如(M,N)矩阵
#
# full_matrices的取值是为0或者1，默认值为1，这时u的大小为(M,M)，v的大小为(N,N) 。否则u的大小为(M,K)，v的大小为(K,N) ，K=min(M,N)。
#
# compute_uv的取值是为0或者1，默认值为1，表示计算u,s,v。为0的时候只计算s。
#
# 返回值：
#
# 总共有三个返回值u,s,v
# u大小为(M,M)，s大小为(M,N)，v大小为(N,N)。
#
# A = u*s*v
# 其中s是对矩阵a的奇异值分解。s除了对角元素不为0，其他元素都为0，并且对角元素从大到小排列。
# s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值。

#
# 有几点需要注意的地方：
# 1. python中的svd分解得到的VT就是V的转置，这一点与matlab中不一样，matlab中svd后得到的是V，如果要还原的话还需要将V转置一次，而Python中不需要。
# 2. Python中svd后得到的sigma是一个行向量，Python中为了节省空间只保留了A的奇异值，所以我们需要将它还原为奇异值矩阵。同时需要注意的是，比如一个5*5大小的矩阵的奇异值只有两个，但是他的奇异值矩阵应该是5*5的，所以后面的我们需要手动补零，并不能直接使用diag将sigma对角化。
#  def svd_():
#      Data=loadExData();
#      U,Sigma,VT=linalg.svd(Data);
#      print('SVD分解Sigma的结果为:',Sigma);
#      return U,Sigma,VT;
#  """
#  array([  9.72140007e+00,   5.29397912e+00,   6.84226362e-01,
#           1.52344501e-15,   2.17780259e-16])
#  可以看到最后两个值很小,于是就可以将最后两个值去掉了
#  """
#  # 近似重构原始矩阵
#  def reconstructMat():
#      U,Sigma,VT=svd_();
#      Sig3=mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]]);
#      reconMat=U[:,:3]*Sig3*VT[:3,:];
#      return reconMat;
