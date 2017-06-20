import pandas
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 梯度下降法 使用矩阵

# y=wx, w,x皆为向量,且x0=1
class Predictor:
    def __init__(self, lables, inputDatas, times, rate):
        self.lables = lables
        self.inputDatas = inputDatas
        self.times = times
        self.rate = rate

        self._outs_ = []
        self._columns_average = []
        self._columns_s = []
        # 增加一列x0=1
        datas_ = np.c_[np.ones((inputDatas.shape[0], 1)), self.inputDatas]
        self.inputDatas = self._featureScaling_(datas_)
        self._weight_ = np.random.rand(self.inputDatas.shape[1], 1)
        self._out_ = np.zeros((inputDatas.shape[0], 1))
        self.m = self.inputDatas.shape[0]
        # print('weight:\n', self._weight_)

    def getResult(self, inputDatas):
        # inputDatas.dtype='float64'
        inputDatas.astype('float64')
        # 增加
        data = np.c_[[[1]], [inputDatas]].astype('float64')
        for i in range(data.shape[1]):
            i_ = data[:, i]
            data[:, i] = self._scaling(i_, self._columns_average[i], self._columns_s[i])
        return np.sum(np.dot(data, self._weight_))

    # 特征缩放
    def _featureScaling_(self, inputDatas):
        self._columns_average = np.mean(inputDatas, axis=0)
        for i in range(inputDatas.shape[1]):
            column = inputDatas[:, i]
            s = np.max(column) - np.min(column)
            self._columns_s.append(s)
            scaling = self._scaling(column, self._columns_average[0, i], s)
            inputDatas[:, i] = scaling
        return inputDatas

    def _scaling(self, column, average, delta):
        return (column - average) / delta if delta > 0 else column

    def train(self):
        for i in range(self.times):
            self._out_ = np.dot(self.inputDatas, self._weight_)
            self._updateWeight()
        print('weight:', self._weight_)

    def _updateWeight(self):
        # 根据梯度下降公式得到修正w
        self._weight_ = self._weight_ + self.rate * np.dot(np.transpose(self.inputDatas),
                                                           (self.lables - self._out_)) / self.m
        self._outs_.append(self._out_)
        self._out_ = np.zeros((self.inputDatas.shape[0], 1))
        # print('weight:', self._weight_)

    # 显示2d变化轨迹图
    def show(self, inputs, save=False):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        #  询问图形在屏幕上的尺寸和DPI（每英寸点数）。
        #  注意当我们把图形储存成一个文件时，我们需要再另外提供一个DPI值
        print('fig size: {0} DPI, size in inches {1}'.format(
                fig.get_dpi(), fig.get_size_inches()))

        ax.scatter(inputs, self.lables)
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
    ex1Data = pandas.read_csv('ex1data1.txt')
    inputs = np.transpose(np.matrix(ex1Data['x']))
    lables = np.transpose(np.matrix(ex1Data['y']))
    predicotor1 = Predictor(lables, inputs, 1000,
                            0.1)
    predicotor1.train()
    end = datetime.datetime.utcnow().timestamp()
    print('time:', end - start)
    predicotor1.show(inputs)
