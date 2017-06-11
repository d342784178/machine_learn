import random
from functools import reduce


# 批梯度下降法

def sum(a, b):
    return a + b


def aaaa(func, *args):
    return map(func, list(zip(*args)))


# y=w0*b+w1*x, w0=1
class Predictor:
    def __init__(self):
        self._out_ = 0;
        self._weight_ = [random.random() for i in range(1)]
        self.bais = random.random()
        print('weight:', self._weight_)
        print('bais:', self.bais)

    def getResult(self, inputDatas):
        return self._weight_[0] * inputDatas[0] + self.bais

    def train(self, lables, inputDatas, times, rate):
        for i in range(times):
            self.trainOnce(lables, inputDatas, rate)

    def trainOnce(self, lables=None, inputDatas=None, rate=None):
        for data in zip(lables, inputDatas):
            self.caculate(data[0], data[1])
            self.updateWeight(data[0], rate, data[1])

    def caculate(self, lable, data):
        self._out_ = (reduce(sum, aaaa(lambda a: a[0] * a[1], data, self._weight_)) + self.bais)
        # print('out:', self._out_)

    def updateWeight(self, lable, rate, inputData):
        self._weight_ = list(map(lambda w: w + rate * (lable - self._out_) * inputData[0], self._weight_))
        self.bais = self.bais + rate * (lable - self._out_)
        self._out_ = []
        print('weight:', self._weight_)
        print('bais:', self.bais)


if __name__ == '__main__':
    predictor = Predictor()
    predictor.train([1, 2, 3, 4], [[1], [2], [3], [4]], 1000, 0.01)
    print(predictor.getResult([1]))
    print(predictor.getResult([2]))
    print(predictor.getResult([3]))
    print(predictor.getResult([4]))

    # print(list(aaaa(lambda a: a, [1], [2], [3])))
    # print(list(aaa(lambda a: a, [1], [2], [3])))
