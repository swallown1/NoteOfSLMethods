import numpy as np
import operator


def getdata():
    fr = open('data.txt', encoding='utf-8')
    lense = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses =[]
    for i in lense:
        for j in i:
            lenses.append(j.split(' '))
    return lenses

#多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #operator.itemgetter(1) 选择第一个域的值 相当于定义了一个函数
    #sorted可以对list或者iterator进行排序 true降序排列
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


def classLeaf(dataSet):
    # 基于多数投票，判断节点叶子的类别
    classList = [i[-1] for i in dataSet]
    return majorityCnt(classList)

def classErr(dataSet,n):
    type = [i[-1] for i in dataSet]
    c = type
    classes = len(set(type))
    samples = len(type)
    print(samples)

if __name__ == '__main__':
    data = getdata()
    dataSet = np.array(data)
    classErr(data,2)


