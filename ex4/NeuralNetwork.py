#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'神经网络'

__author__ = 'DLJ'

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import scipy.io as sio
from PIL import Image as img
from scipy.optimize import fmin_tnc
import os
import time


def exeTime(func):
    """ 耗时计算装饰器
    """

    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print('耗时', func.__name__, time.time() - t0)
        return back

    return newFunc


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidGradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


@exeTime
def costFunction(X, y, theta, input_layer_size, hidden_layer_size, out_layer_size, m, lmd):
    Y = np.zeros([m, out_layer_size])
    for ii in range(m):
        Y[ii, y[ii, 0] - 1] = 1
    # 向前推导
    theta_1 = np.asmatrix(theta[0, 0:hidden_layer_size * (input_layer_size + 1)]).reshape(
        hidden_layer_size, (input_layer_size + 1))
    theta_2 = np.asmatrix(theta[0, hidden_layer_size * (input_layer_size + 1)::]).reshape(
        out_layer_size, (hidden_layer_size + 1))

    X = X.reshape(m, input_layer_size)
    a1 = X
    a2 = sigmoid(np.c_[np.ones([m, 1]), a1].dot(theta_1.T))
    a3 = sigmoid(np.c_[np.ones([m, 1]), a2].dot(theta_2.T))

    # 计算成本函数
    cost = np.multiply(-Y, np.log(a3)) - np.multiply((1 - Y), np.log(1 - a3))
    regular = lmd / (2 * m) * (np.sum(np.square(theta_1[:, 1::])) + np.sum(np.square(theta_2[:, 1::])))
    J = 1 / m * np.sum(cost[:]) + regular

    # 反向传播
    Delta_2 = np.zeros(theta_2.shape)
    Delta_1 = np.zeros(theta_1.shape)
    for ii in range(m):
        a_1 = np.matrix(X[ii, :]).T
        a_1 = np.r_[np.ones([1, 1]), a_1]
        a_2 = sigmoid(theta_1.dot(a_1))
        a_2 = np.r_[np.ones([1, 1]), a_2]
        a_3 = sigmoid(theta_2.dot(a_2))

        # 输出层 误差项
        delta_3 = a_3 - np.matrix(Y[ii, :]).T
        # 隐藏层 误差项
        delta_2 = np.multiply(theta_2.T.dot(delta_3)[1::, :],
                              sigmoidGradient(theta_1.dot(a_1)))

        Delta_2 += delta_3.dot(a_2.T)
        Delta_1 += delta_2.dot(a_1.T)
    # 正规化
    Telta_1_temp = np.c_[np.zeros([theta_1.shape[0], 1]), theta_1[:, 1::]]
    Telta_2_temp = np.c_[np.zeros([theta_2.shape[0], 1]), theta_2[:, 1::]]
    # theta_1的梯度
    J_grad_1 = 1 / m * Delta_1 + lmd / m * Telta_1_temp
    # theta_2的梯度
    J_grad_2 = 1 / m * Delta_2 + lmd / m * Telta_2_temp

    J_grad = np.c_[J_grad_1.ravel(), J_grad_2.ravel()]
    return J, J_grad


def initTheta(l_in, l_out):
    epsilon = 0.12
    W = np.random.random([l_out, l_in + 1]) * 2 * epsilon - epsilon
    return W


def debugInitWeight(f_out, f_in):
    W = np.random.random([f_out, f_in])
    return W


# 梯度校验
def checkGradient(lmd):
    input_layer_size = 3
    hidden_layer_size = 5
    out_layer_size = 3
    m = 5
    lmd = 1
    Theta1 = initTheta(input_layer_size, hidden_layer_size)
    Theta2 = initTheta(hidden_layer_size, out_layer_size)
    X = debugInitWeight(m, input_layer_size)
    y = (np.ones([m, 1]) * out_layer_size).astype('int32')

    theta = np.asmatrix(np.r_[Theta1.ravel().T, Theta2.ravel().T])

    espilon = 1e-4
    gradApprox = np.zeros([1, theta.shape[1]])
    for i in range(theta.shape[1]):
        left = np.matrix(theta)
        right = np.matrix(theta)
        left[0, i] = theta[0, i] + espilon
        right[0, i] = theta[0, i] - espilon
        cost_left, grad_left = costFunction(np.asmatrix(X), y, left, input_layer_size, hidden_layer_size,
                                            out_layer_size,
                                            m, lmd)
        cost_right, grad_right = costFunction(np.asmatrix(X), y, right, input_layer_size, hidden_layer_size,
                                              out_layer_size, m,
                                              lmd)
        espilon_ = (cost_left - cost_right) / (2 * espilon)
        gradApprox[0, i] = espilon_
        # print(espilon_)
    cost, grad = costFunction(np.matrix(X), y, theta, input_layer_size, hidden_layer_size, out_layer_size, m, lmd)
    # print(gradApprox)
    # print(grad)
    # print(gradApprox - grad)
    delta = np.sum(np.square(gradApprox - grad))
    # print(delta)
    return delta < 1e-9


def predict(X, y, theta, input_layer_size, hidden_layer_size, out_layer_size, m, lmd):
    Y = np.zeros([m, out_layer_size])
    for ii in range(m):
        Y[ii, y[ii, 0] - 1] = 1
    # 向前推导
    theta_1 = np.asmatrix(theta[0, 0:hidden_layer_size * (input_layer_size + 1)]).reshape(
        hidden_layer_size, (input_layer_size + 1))
    theta_2 = np.asmatrix(theta[0, hidden_layer_size * (input_layer_size + 1)::]).reshape(
        out_layer_size, (hidden_layer_size + 1))

    X = X.reshape(m, input_layer_size)
    a1 = X
    a2 = sigmoid(np.c_[np.ones([m, 1]), a1].dot(theta_1.T))
    a3 = sigmoid(np.c_[np.ones([m, 1]), a2].dot(theta_2.T))
    return np.where(a3 > 0.5)[1] + 1


def checkResult(X, y, theta, input_layer_size, hidden_layer_size, out_layer_size, m, lmd):
    pass


if __name__ == '__main__':
    # gradient = checkGradient(1)
    # print('梯度检验成功' if gradient else '梯度检验失败')

    input_layer_size = 400
    hidden_layer_size = 25
    out_layer_size = 10
    data1 = sio.loadmat('ex4data1.mat')
    X = np.matrix(data1['X'])
    y = np.matrix(data1['y'])
    print('dataSize:', X.shape)
    m = X.shape[0]
    print('m', m)
    # data2 = sio.loadmat('ex4weights.mat')
    # theta_1 = data2['Theta1']
    # theta_2 = data2['Theta2']
    theta = None
    if os.path.exists('./theta.mat'):
        theta = sio.loadmat('./theta.mat')['theat']
    else:
        theta_1 = initTheta(input_layer_size, hidden_layer_size)
        theta_2 = initTheta(hidden_layer_size, out_layer_size)
        theta = np.matrix(np.r_[theta_1.ravel(), theta_2.ravel()])

    image = img.fromarray(X[1000, :].reshape(20, 20).T * 255)
    # image.show()


    # J, grad, result = costFunction(X, y, theta, input_layer_size, hidden_layer_size,
    #                                out_layer_size, m, lmd=1)
    # print(J)
    # print(grad)

    # rate = 1
    # trainTimes = 5000
    # for i in range(trainTimes):
    #     J, grad = costFunction(X, y, theta, input_layer_size, hidden_layer_size,
    #                            out_layer_size, m, lmd=1)
    #     print('J', J)
    #     theta -= rate * grad
    #     sio.savemat('./theta.mat', {'theat': theta})
    # print(theta)
    # sio.savemat('./theta.mat', {'theat': theta})

    # image = img.fromarray(X[0:1000, :].reshape(20, 20).T * 255)
    # image.show()
    result = predict(X[0:50, :], y, theta, input_layer_size, hidden_layer_size, out_layer_size, m=50, lmd=1)
    print(result)
    # print(np.matrix(result).T - y[0:1000, :])
