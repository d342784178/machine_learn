import random, datetime
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


# 梯度下降法

def sum(a, b):
    return a + b


def aaaa(func, *args):
    return map(func, list(zip(*args)))


# y=w0*b+w1*x, w0=1
class Predictor:
    def __init__(self, lables, inputDatas, times, rate):
        self.lables = lables
        self.inputDatas = inputDatas
        self.times = times
        self.rate = rate

        self._out_ = []
        self._weight_ = [random.random() for i in range(1)]
        self.bais = random.random()
        self._outs_ = []
        print('weight:', self._weight_)
        print('bais:', self.bais)

    def getResult(self, inputData):
        return self._weight_[0] * inputData[0] + self.bais

    def train(self):
        for i in range(self.times):
            self.trainOnce()

    def trainOnce(self):
        for data in zip(self.lables, self.inputDatas):
            self.caculate(data[0], data[1])
        self.updateWeight(self.lables, self.rate, self.inputDatas)

    def caculate(self, lable, data):
        self._out_.append(reduce(sum, aaaa(lambda a: a[0] * a[1], data, self._weight_)) + self.bais)
        # print('out:', self._out_)

    def updateWeight(self, lables, rate, inputData):
        self._weight_ = list(map(lambda w: w - rate * reduce(sum, list(
                aaaa(lambda a: (a[0] - a[1]) * a[2][0], self._out_, lables, inputData))), self._weight_))
        self.bais = self.bais - rate * reduce(sum, list(
                aaaa(lambda a: a[0] - a[1], self._out_, lables)))
        self._outs_.append(self._out_)
        self._out_ = []
        # print('weight:', self._weight_)
        # print('bais:', self.bais)

    def show(self, save=False):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        #  询问图形在屏幕上的尺寸和DPI（每英寸点数）。
        #  注意当我们把图形储存成一个文件时，我们需要再另外提供一个DPI值
        print('fig size: {0} DPI, size in inches {1}'.format(
                fig.get_dpi(), fig.get_size_inches()))

        # 画出一个维持不变（不会被重画）的散点图和一开始的那条直线。
        ax.scatter(self.inputDatas, self.lables)
        line, = ax.plot(self.inputDatas, self._outs_[0], 'r-', linewidth=2)

        def update(i):
            label = 'timestep {0}'.format(i)
            print(label)
            # 更新直线和x轴（用一个新的x轴的标签）。
            # 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
            line.set_ydata(self._outs_[i - 1])
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
    predictor = Predictor([1, 2, 3, 4], [[1], [2], [3], [4]], 500, 0.01)
    predictor.train()
    print(predictor.getResult([1]))
    print(predictor.getResult([2]))
    print(predictor.getResult([3]))
    print(predictor.getResult([4]))
    end = datetime.datetime.utcnow().timestamp()
    print('time:', end - start)

    predictor.show()
    # print(list(aaaa(lambda a: a, [1], [2], [3])))
    # print(list(aaa(lambda a: a, [1], [2], [3])))
