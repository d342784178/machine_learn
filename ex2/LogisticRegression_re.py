import numpy as np
import datetime
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


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
        self.lmd = 1

        self.mu = []
        self.sigma = []
        # 增加一列x0=1
        # self.x, self.mu, self.sigma = self._featureScaling_(x)
        self.x = np.c_[np.ones((self.x.shape[0], 1)), self.x]
        self.theta = np.zeros([self.x.shape[1], 1])
        self.m = self.x.shape[0]
        # print('weight:\n', self._weight_)

    def getResult(self, inputDatas):
        inputDatas.astype('float64')
        # 特征缩放
        # inputDatas = (inputDatas - self.mu) / self.sigma
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
        cost = -(1 / self.m * np.sum(np.log(h).T.dot(self.y) + np.log(1 - h).T.dot(1 - self.y))) + self.lmd / (
            2 * self.m) * np.sum(np.square(self.theta))
        print('cost', cost)
        pass

    def _updateWeight(self, h):
        # 根据梯度下降公式修正theta
        grad = self.x.T.dot((h - self.y)) / self.m + self.lmd / self.m * self.theta
        grad[0, 0] -= self.lmd / self.m * self.theta[0, 0]
        self.theta -= self.rate * grad
        print('grad', grad)
        # print('weight:', self._weight_)


def mapFeature(x1, x2):
    degree = 7
    out = np.ones([1, 28])
    for i in range(degree):
        for j in range(i):
            out[0, i * 7 + j] = x1 ^ (i - j) * x2 ^ j
    return out


if __name__ == '__main__':
    loadtxt = np.loadtxt('ex2data2.txt', delimiter=',')
    zero = np.where(loadtxt == 0)[0]
    one = np.where(loadtxt == 1)[0]
    plt.plot(loadtxt[zero][:, 0], loadtxt[zero][:, 1], 'r+')
    plt.plot(loadtxt[one][:, 0], loadtxt[one][:, 1], 'b.')
    Y = np.transpose(np.matrix(loadtxt[:, 2]))
    X = np.matrix(loadtxt[:, 0:2])
    predictor = Predictor(Y, X, 1000, 0.05)
    predictor.train()

    # 绘制决策边界
    x1Min = X[:, 0].min()
    x1Max = X[:, 0].max()
    x2Min = X[:, 1].min()
    x2Max = X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1Min, x1Max),
                           np.linspace(x2Min, x2Max))
    ploy = StandardScaler()
    transform = ploy.fit_transform(np.c_[xx1.ravel(), xx2.ravel()])
    h = sigmoid(np.c_[np.ones([transform.shape[0], 1]), transform].dot(predictor.theta))
    h = h.reshape(xx1.shape)
    #TODO 画出图像
    plt.contour(xx1, xx2, h, [0.5], colors='b', linewidth=.5)
    plt.show()
