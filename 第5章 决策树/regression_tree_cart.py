# -*- coding: utf-8 -*-
#李航《统计学习方法》第五章习题2 最小二乘回归树

import numpy as np

y = np.array([4.5, 4.75, 4.91, 5.34, 5.8, 7.05, 7.9, 8.23, 8.7, 9])

# def average(list):
#     ave = 0
#     for i in list:
#         ave += i
#     if len(list)>0:
#         ave = ave/len(list)
#     else:
#         ave = 0
#     return ave

def CART(start, end,y):
    if (end-start)>1:
        result = []
        for i in range(start,end+1,1):
            c1 = [np.average(y[start:i+1])]   #左子树平均值
            c2 = [np.average(y[i+1:end+1])]   #右子树平均值
            y1 = y[start:i+1]
            y2 = y[i+1:end+1]
            result.append((sum((y1-c1)**2)+sum((y2-c2)**2)))  #计算平方误差损失
        index1 = np.argmin(result) + start     #每一步的切分点，argmin返回最值所在的索引
        print (index1,'---',np.average(y[start:index1+1]),'---',np.average(y[index1+1:end+1]))
        CART(start,index1,y)   #对左子树生成
        CART(index1+1,end,y)   #对右子树生成
    else:
        return None

CART(0,9,y)