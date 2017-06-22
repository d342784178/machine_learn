import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.matlib import repmat


# 梯度下降法 特征缩放

# y=wx, w,x皆为向量,且x0=1
class Predictor:
    def __init__(self, y, x, times, rate):
        self.y = y
        self.x = x
        self.times = times  # 训练次数
        self.rate = rate  # 学习速率

        self._outs_ = []
        self.mu = []  # 平均值
        self.sigma = []  # 标准差
        # 增加一列x0=1
        self.x = self._featureScaling_(x)
        self.x = np.c_[np.ones((self.x.shape[0], 1)), self.x]
        self.theta = np.zeros([self.x.shape[1], 1])
        self.m = self.x.shape[0]  # 输入纬度
        # print('weight:\n', self._weight_)

    def getResult(self, inputDatas):
        inputDatas.astype('float64')
        # 特征缩放
        data = (inputDatas - self.mu) / self.sigma
        data = np.c_[[[1]], data].astype('float64')
        return np.sum(np.dot(data, self.theta))

    # 特征缩放
    def _featureScaling_(self, inputDatas):
        self.mu = np.mean(inputDatas, axis=0)
        self.sigma = np.std(inputDatas, axis=0)
        return (inputDatas - repmat(self.mu, np.shape(inputDatas)[0], 1)) / repmat(self.sigma, np.shape(inputDatas)[0],
                                                                                   1);

    def train(self):
        for i in range(self.times):
            out = np.dot(self.x, self.theta)
            # 打印成本函数
            print(1 / (2 * self.m) * np.sum(np.square((out - self.y))))
            self._updateWeight(out)
        print('weight:', self.theta)

    def _updateWeight(self, out):
        # 根据梯度下降公式得到修正w
        self.theta = self.theta + self.rate * np.dot(np.transpose(self.x),
                                                     (self.y - out)) / self.m
        self._outs_.append(out)
        # print('weight:', self._weight_)

    # 显示2d变化轨迹图
    def show(self, inputs, save=False):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        #  询问图形在屏幕上的尺寸和DPI（每英寸点数）。
        #  注意当我们把图形储存成一个文件时，我们需要再另外提供一个DPI值
        print('fig size: {0} DPI, size in inches {1}'.format(
            fig.get_dpi(), fig.get_size_inches()))

        ax.scatter(np.transpose(inputs[:, 0]).tolist(), np.transpose(self.y[:, 0]).tolist())
        # 画出一个维持不变（不会被重画）的散点图和一开始的那条直线。
        line, = ax.plot(inputs, self._outs_[0], 'r-', linewidth=2)

        def update(i):
            label = 'train {0} time'.format(i - 1)
            # 更新直线和x轴（用一个新的x轴的标签）。
            # 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
            line.set_ydata(self._outs_[i])
            ax.set_xlabel(label)
            return line, ax

        # FuncAnimation 会在每一帧都调用“update” 函数。
        # 在这里设置一个10帧的动画，每帧之间间隔200毫秒
        anim = FuncAnimation(fig, update, frames=self._outs_.__len__(), interval=1, repeat=False)
        if save:
            anim.save('line.gif', dpi=80, writer='imagemagick')
        else:
            plt.show()


if __name__ == '__main__':
    start = datetime.datetime.utcnow().timestamp()
    # inputs = np.array([[1], [2], [3], [4]])
    # lables = np.array([[1], [2], [3], [4]])
    # predicotor1 = Predictor(inputs, lables, 500,
    #                         0.03)
    # predicotor1.train()
    # print(predicotor1.getResult(np.array([1])))
    # print(predicotor1.getResult(np.array([2])))
    # print(predicotor1.getResult(np.array([3])))
    # print(predicotor1.getResult(np.array([4])))
    #
    ex1Data = np.loadtxt('ex1data2.txt', delimiter=',')
    inputs = np.matrix(ex1Data[:, 0:2])
    lables = np.transpose(np.matrix(ex1Data[:, 2]))
    predicotor1 = Predictor(lables, inputs, 2000,
                            0.01)
    predicotor1.train()
    print(predicotor1.getResult(np.matrix([1650, 3])))
    # print(predicotor1.getResult(np.matrix([1650, 3])))
    end = datetime.datetime.utcnow().timestamp()
    print('time:', end - start)
    # predicotor1.show(inputs)
