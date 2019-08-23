import pandas as pd
import unittest
import numpy as np

class Naive_Bayes(object):
    def __init__(self,lambda_):
        self.lambda_ = lambda_
        self.classes = None
        self.prior = None
        self.class_prior = None
        self.class_count=None

    def fit(self,x,y):
        self.classes = np.unique(y)
        #加了表格框
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        #统计不同类别的个数
        self.class_count = y[y.columns[0]].value_counts()
        #不同类别的占比
        self.class_prior = self.class_count / y.shape[0]

        self.prior = dict()

        for idx in x.columns:
            for j in self.classes:
                p_x_y = x[(y==j).values][idx].value_counts()
                for i in p_x_y.index:
                    self.prior[(idx, i, j)] = p_x_y[i] / self.class_count[j]

    def predict(self,X):
        rst = []
        for class_ in self.classes:
            py = self.class_prior[class_]
            pxy = 1
            for idx, x in enumerate(X):
                pxy *= self.prior[(idx, x, class_)]

            rst.append(py * pxy)
        return self.classes[np.argmax(rst)]


if __name__ == '__main__':
    data = pd.read_csv('./data.txt',header=None,sep=',')
    x = data[data.columns[0:2]]
    y = data[data.columns[2]]
    nb = Naive_Bayes(1)
    nb.fit(x,y)
    rst = nb.predict([2, "S"])
    print(rst)

    a = np.array([[1,0,0],[0,2,4]])
    print(a.T,a)
    print(np.dot(a.T,a))
