#通过Gini系数进行二分类树的构建，目标生成分类树
from numpy import *

def creatDataSet():
    dataset=[['青年','否','否','一般','否'],
             ['青年','否','否','好','否'],
             ['青年','是','否','好','是'],
             ['青年','是','是','一般','是'],
             ['青年','否','否','一般','否'],
             ['中年','否','否','一般','否'],
             ['中年','否','否','好','否'],
             ['中年','是','是','好','是'],
             ['中年','否','是','非常好','是'],
             ['中年','否','是','非常好','是'],
             ['老年','否','是','非常好','是'],
             ['老年','否','是','好','是'],
             ['老年','是','否','好','是'],
             ['老年','是','否','非常好','是'],
             ['老年','否','否','一般','否']]
    label=['年龄','有工作','有自己的房子','信贷情况']
    return dataset,label

#计算Gini系数 ,计算最后一列
def gini(dataSet):
    numsamples = len(dataSet)
    classcount = {}
    for data in dataSet:
        classname = data[-1]
        if classname not in classcount.keys():
            classcount[classname] = 0
        classcount[classname] +=1
    gini = 1.0
    for i in classcount:
        pro =float(classcount[i])/numsamples
        pro = -1*pow(pro,2)
        gini +=pro
    return gini


def splitdata(dataSet,axis,value):
    """
    划分数据 将某个特征属于某个值的分为一类 其余的分为一类
    :param dataSet:  数据集
    :param axis: 某个特征的维度(索引)
    :param value: 特征对应的值
    :return: 两个划分后的数据集
    """
    redataSet1 = []
    redataSet2 = []
    for FeaVec in dataSet:
        reFeaVec = FeaVec[:axis]
        reFeaVec.extend(FeaVec[axis + 1:])
        if FeaVec[axis] == value:
            redataSet1.append(reFeaVec)
        elif FeaVec[axis] != value:
            redataSet2.append(reFeaVec)
    return redataSet1,redataSet2


def choosebsetfeature(dataSet):
    """
        通过计算不同特征的Gini系数来选择最优的特征
    :param dataSet: 数据集
    :return: 返回Gini 系数最小的特征的索引和值
    """
    numfeature = len(dataSet[0])-1
    bestgini = inf  #最好的gini值
    bestfeature = -1  #Gini值最小的特征
    bestvalue =-1   #特征对应的值
    for i in range(numfeature):
        features = [data[i] for data in dataSet]
        uniquefea = list(set(features))  #获取第i个特征的所有属性值
        for value in uniquefea:
            split1,split2 = splitdata(dataSet,i,value)
            basegini = gini(split1) * (len(split1)/len(dataSet)) + gini(split2)*(len(split2)/len(dataSet))
            if basegini < bestgini:
                bestgini=basegini
                bestfeature = i
                bestvalue=value
    return bestfeature,bestvalue


def majority(classList):
    """
    在cart算法中的第1 第2步 中需要计算种类中值最多的值
    :param classList: 类列的所有值
    :return: 返回不同种类的字典
    """
    classcount = {}
    for v in classList:
        if v not in classcount.keys():
            classcount[v] =0
        classcount[v] +=1

    return classcount


def BulidTree(dataSet,labels):
    """
    构建分类树
    :param dataSet:数据集
    :param labels: 标签列
    :return: 返回构建好的树的根节点
    """
    classList = [i[-1] for i in dataSet]
    # 如果所有的分类是同一类 返回这一类的名称
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果没有其他的特征，返回类别列中 最多的类别
    if len(dataSet[0]) ==1:
        return majority(classList)

    #通过gini系数选择最好的特征
    bestfeature,bestvalue = choosebsetfeature(dataSet)
    bestlabel = labels[bestfeature]
    #通过字典存储这棵树
    MyTree = {bestlabel:{}}
    #删除这个特征，用剩下的特征构建树
    del(labels[bestfeature])
    #去除这一特征列后的数据
    data1 ,data2 = splitdata(dataSet,bestfeature,bestvalue)
    #获取这一类的所有取值,对其构建左右子树， 与value相同的在左子树  不同的在右子树
    features = [example[bestfeature] for example in dataSet]
    uniqueFea = list(set(features))
    for value in uniqueFea:
        if bestvalue == value:
            MyTree[bestlabel][value] = BulidTree(data1,labels)
        else:
            MyTree[bestlabel][value] =BulidTree(data2,labels)
    return MyTree



if __name__ == '__main__':
    dataSet,labels = creatDataSet()
    MyTree = BulidTree(dataSet,labels)
    print(MyTree)