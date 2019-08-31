#### 主要性质：
   1.$A^TA=(U \sum V^T)^T(U \sum V^T)=V (\sum)^T\sum V^T$
        V的列向量是 $A^TA$的特征向量
   $AA^T=(U \sum V^T)(U \sum V^T)^T=U \sum(\sum)^T U^T$
      U的列向量是 $AA^T$的特征向量
      $\sum$ 的奇异值是 $A^TA$和$AA^T$的特征值的平方
   
2.由$A = U \sum V^T$可知：$AV = U\sum $
则对于第j列：$Av_j = \sigma_ju_j$
类似的，由$A = U \sum V^T$可知：$ A^T U= V(\sum)^T$
则对于第j列:$A^Tu_j = \sigma_jv_j$          (j=1,2,....,n)
$A^Tu_j = 0$          (j=n+1,....,m)
              
3.在奇异值分解中，奇异值是唯一的，而矩阵U，V不是唯一的

4.矩阵A和∑的秩相等，等于正奇异值的个数(包括重复的正奇异值)

#### 奇异值分解的计算
求解过程：
  1.首先求解$A^TA$的特征值和特征向量
        计算$W=A^TA$
        求特征值和特征向量 (np.linalig.eig())
        将特征值大小排序
  2.构建n阶正交矩阵V
      将特征向量单位化，得到单位向量 v1,v2,...,vn 构成n阶正交矩阵V
  3.构建m x n对角矩阵∑
      计算奇异值$\sigma_i = \sqrt{\lambda_i}$   i=1,....,n
      构造对角矩阵∑=np.zero((U.shape[1],V.shape[0])) + diag(σ1,σ2,...,σn)
  4.求m阶正交矩阵U
      对A的前r个正奇异值，令 $ u_j = \frac{1}{\sigma_j} A v_j$  j=1,2,...,r
      求$ A^T$的零空间的一组标准正交基${u_{r+1},...,u_{m} }$
  5.得到奇异值分解    $A=U \sum V^T$


#### 费罗贝尼乌斯范数
定义：对于矩阵A的费罗贝尼乌斯范数:  $||A||_F = (\sum_{i=1}^m \sum_{j=1}^n a_{i j})^{ \frac{1}{2}}$

所以对于A的奇异值分解有：$|A||_F = (\sigma_1^2 +\sigma_2^2 + ...+\sigma_n^2 )^{\frac{1}{2}}$

#### 矩阵的最优近似
 $||A-X||_F = (\sigma_{k+1}^2 +\sigma_{k+2}^2 + ...+\sigma_n^2)^{\frac{1}{2}}$
其中 称X是矩阵A在弗洛贝尼乌斯范数意义下的最优近似
 $||A-X||_F = (\sigma_{k+1}^2 +\sigma_{k+2}^2 + ...+\sigma_n^2)^{\frac{1}{2}} =||A-A^{'}||_F $
其中$A^{'} = U∑^{'} V^T$是达到最优值得一个矩阵


#### 矩阵的外积展开式
$U∑ = [\sigma_1u_1,\sigma_2u_2,...,\sigma_nu_n]$
$V^T = \begin{bmatrix}v_1^T \\\\v_2^T \\\\ .\\\\ v_n^T \end{bmatrix}$
$A = \sigma_1u_1v_1^T+...+\sigma_nu_nv_n^T$
$A = \sum_{k=1}^n A_k = \sum_{k=1}^n \sigma_ku_kv_k^T$






































