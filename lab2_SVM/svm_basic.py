from classifier import classifier
from svmMLiA import smoPK, calcWs
from numpy import dot as np_dot, nditer

class svm_basic(classifier):
    def __init__(self, c=1, toler=0.001, maxIter=50):
        self.alpha = None
        self.b = None
        self.w = None
        self.C = c
        self.toler = toler
        self.maxIter = maxIter

    def fit(self, X, Y):
        self.b, self.alpha = smoPK(X, Y, self.C, self.toler, self.maxIter)
        self.w = calcWs(self.alpha, X, Y)

    def predict(self, X):
        result = np_dot(self.w.T, X.T) + self.b
        for x in nditer(result, op_flags=['readwrite']):
            if x >= 0:
                x[...] = 1
            else:
                x[...] = -1
        return result
