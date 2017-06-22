import numpy as np
import datetime
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import math


# 梯度下降法 特征缩放


# sigmoid函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# y=wx, w,x皆为向量,且x0=1
class Predictor:
    def __init__(self, y, x, times, rate):
        self.y = y
        self.x = x
        self.times = times
        self.rate = rate

        self.mu = []
        self.sigma = []
        # 增加一列x0=1
        self.x, self.mu, self.sigma = self._featureScaling_(x)
        self.x = np.c_[np.ones((self.x.shape[0], 1)), self.x]
        self.theta = np.zeros([self.x.shape[1], 1])
        self.m = self.x.shape[0]
        # print('weight:\n', self._weight_)

    def getResult(self, inputDatas):
        inputDatas.astype('float64')
        # 特征缩放
        inputDatas = (inputDatas - self.mu) / self.sigma
        data = np.c_[np.ones([inputDatas.shape[0], 1]), inputDatas].astype('float64')
        return np.dot(data, self.theta)

    # 特征缩放
    def _featureScaling_(self, inputDatas):
        mu = np.mean(inputDatas, axis=0)
        sigma = np.std(inputDatas, axis=0)
        return (inputDatas - repmat(mu, np.shape(inputDatas)[0], 1)) / repmat(sigma, np.shape(inputDatas)[0],
                                                                              1), mu, sigma

    def train(self):
        for i in range(self.times):
            h = sigmoid(np.dot(self.x, self.theta))
            self.costFunction(h)
            self._updateWeight(h)
        print('theta:', self.theta)

    def costFunction(self, h):
        cost = -(1 / self.m * np.sum(np.log(h).T.dot(self.y) + np.log(1 - h).T.dot(1 - self.y)))
        print('cost', cost)
        pass

    def _updateWeight(self, h):
        # 根据梯度下降公式修正theta
        grad = self.x.T.dot((h - self.y)) / self.m
        self.theta -= self.rate * grad
        print('grad', grad)
        # print('weight:', self._weight_)


if __name__ == '__main__':
    loadtxt = np.loadtxt('ex2data1.txt', delimiter=',')
    zero = np.where(loadtxt == 0)[0]
    one = np.where(loadtxt == 1)[0]
    plt.plot(loadtxt[zero][:, 0], loadtxt[zero][:, 1], 'r+')
    plt.plot(loadtxt[one][:, 0], loadtxt[one][:, 1], 'b.')
    # plt.show()
    Y = np.transpose(np.matrix(loadtxt[:, 2]))
    X = np.matrix(loadtxt[:, 0:2])
    predictor = Predictor(Y, X, 1000, 0.1)
    predictor.train()
    print('result', sigmoid(predictor.getResult(np.matrix([45, 85]))))
    # ys = []
    # for i in predictor._featureScaling_(np.matrix(loadtxt[:, 0]).T)[0]:
    #     ys.append((predictor.theta.T[0, 0] + predictor.theta.T[0, 1] * i[0, 0]) / -predictor.theta.T[0, 2])
    #
    # print(ys)
    # # print(x)
    # sigma_ = np.matrix(ys).T * predictor.mu[0, 1] + predictor.sigma[0, 1]
    # print(sigma_.T)
    # plt.plot(loadtxt[:, 0], sigma_)
    plot_x = (np.matrix([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2]) - predictor.sigma[0, 0]) / predictor.mu[0, 0]
    print([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
    plot_y = (-1. / predictor.theta[2]) * (predictor.theta[1] * (plot_x) + predictor.theta[0])
    print(plot_y * predictor.mu[0, 1] + predictor.sigma[0, 1])
    plt.plot([28.60326323428011, 100.86943574220611], [-21.54564318, -100.55773148])
    plt.show()
