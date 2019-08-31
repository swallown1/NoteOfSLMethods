import numpy as np
import math
import treePlotter

def getdata():
    fr = open('data.txt', encoding='utf-8')
    lense = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses =[]
    for i in lense:
        for j in i:
            lenses.append(j.split(' '))
    return lenses
def CreatDataSet():
	dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataSet,labels

'''
按照特定特征划分数据集，axis 特征的维度，value 特征的值
'''
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featvec in dataSet:
        if featvec[axis] == value:
        # 一下两句，是将这个特征去除了
            reduceFeatVec = featvec[:axis]
            reduceFeatVec.extend(featvec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


#计算数据集的熵
def calEntropy(dataSet,feature =-1):
    # 增加一个feature参数，默认 -1，也就是计算最后一列类别的熵
    # 如果改变数值，可以计算信息增益比中，数据集D关于每个特征的经验熵
    numsamples = len(dataSet)
    labelCouts = {}
    for featureVect in dataSet:
        currlabel = featureVect[feature]
        if currlabel not in labelCouts.keys():
            labelCouts[currlabel] = 0
            labelCouts[currlabel] +=1
        else:
            labelCouts[currlabel] +=1
        Entroy = 0.0

        for key in labelCouts:
            prob = float(labelCouts[key])/numsamples
            Entroy -= prob * math.log(prob,2)
    return  Entroy

def calfeatureEntory(dataSet):
    #計算每个单独的特征的熵
    numsampels = len(dataSet)
    # featureCount = {}
    featureEntory = []
    for feature in range(len(dataSet[0])-1):
        Entroy = calEntropy(dataSet,feature=feature)
        featureEntory.append(Entroy)
    return featureEntory

'''
使用信息增益计算最好的特征,返回特征列的索引
这实现起来很有意思啊
info 计算条件尚的方法，默认id3,也可以设置为C54
'''
def chosseBestFeatureToSplit(dataSet,info ='id3'):
    numFeatures = len(dataSet[0])-1
    #根据不同的算法 除以不同的Ha(D) 即每个特征的熵
    if info == 'id3':
        featureEntory = np.ones((len(dataSet[0])-1,1))
    else:
        featureEntory = calfeatureEntory(dataSet)
    baseFeature = -1
    baseEntory = calEntropy(dataSet)
    baseinfoGain= 0.0  #存储不同特征的信息增益率
    #id这一列不要了
    for i in range(1,numFeatures):
        feature = [example[i] for example in dataSet]
        uniquefeature = set(feature) #用set创建无序不重复集合
        newEntory = 0.0  #计算的是条件熵
        for value in uniquefeature:
            #按照特征进行划分数据集
            subDataSet = splitDataSet(dataSet,i,value)
            #计算H(Di)
            pro = len(subDataSet) / float(len(dataSet))
            FeatureEntory = calEntropy(subDataSet)
            newEntory +=pro * FeatureEntory
        #计算信息增益
        infoGain = baseEntory - newEntory
        #计算信息增益率，如果是id3 由于分母是1 所以相当于没处理
        infoGain /= featureEntory[i]
        if infoGain > baseinfoGain:
            baseinfoGain = infoGain
            # 存储信息增益率最高的索引
            baseFeature = i
    return baseFeature
#返回种类最多的分类
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

#这里的数是使用python中的字典进行保存的  不是c++中的指针
def createTree(dataSet,lables,info='id3'):
    classList = [example[-1] for example in dataSet]
    #如果 所有类只有一类，返回单节点树  步骤1
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果没有特征列  返回种类最多的节点  步骤2
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    #计算信息增益  步骤3
    bestFeature = chosseBestFeatureToSplit(dataSet,info) #最好特征的索引
    bestFeatureLabel = lables[bestFeature]
    MyTree = {bestFeatureLabel:{}}
    #删除添加进树的特征
    del(lables[bestFeature])
    '''
        第[5]步 及 第[6]步，计算特征每个取值的集，构建子结点，然后递归前面几步构造树
    	没有对阈值的计算，如果计算的话，可以在chosseBestFeatureToSplit（）设置阈值，如果小于阈值，
    	返回一个特征标号和一个标志，然后构造单节点树即可
    '''
    featVet = [samples[bestFeature] for samples in dataSet]
    uniqueFeature = set(featVet)
    #递归创建决策树
    for value in uniqueFeature:
        #因为上面已经将加入决策树的特征删除过了。
        subLabels = lables[:]
        MyTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)

    return MyTree

def storeTree(Tree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(Tree,fw)
    fw.close()

def loadTree(filename):
    import pickle
    fr = open(filename,"rb")
    return pickle.load(fr)

def buildTree():
    lense = getdata()
    lensesLabels = ['age', 'job', 'house', 'xindai']
    lenseTree = createTree(lense, lensesLabels, info='c45')
    return lenseTree

def classfiy(inputTree,featureLabels,testVet):
    first = inputTree.keys()[0]
    second = inputTree[first]
    featureindex = featureLabels.index(first)
    for key  in second.keys():
        if testVet[featureindex] == key:
            #判断是否是叶子节点
            if type(second[key]).__name__ == 'dict':
                classlable = classfiy(second[key],featureLabels,testVet)
            else:
                classlable = second[key]
    return classlable

if __name__ == '__main__':

    # treePlotter.createPlot(lenseTree)
    # lenseTree = buildTree()
    # storeTree(lenseTree,'./tree.txt')
    tree = loadTree('./tree.txt')
    treePlotter.createPlot(tree)