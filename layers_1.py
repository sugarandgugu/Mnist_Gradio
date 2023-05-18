# encoding: utf-8
import numpy as np
import struct
import os
import time

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.d_weight = None  # 添加 d_weight 属性
        self.d_bias = None  # 添加 d_bias 属性
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
        # show_matrix(self.weight, 'fc weight ')
        # show_matrix(self.bias, 'fc bias ')
    def forward(self, input):
        start_time = time.time()
        self.input = input
        self.output = np.dot(self.input, self.weight) + self.bias
        show_matrix(self.output, 'fc out ')
        show_time(time.time() - start_time, 'fc forward time: ')
        return self.output
    def backward(self, top_diff):
        bottom_diff = np.dot(top_diff, self.weight.T)
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        return bottom_diff
    
    def get_gradient(self):
        return self.d_weight, self.d_bias

    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
        show_matrix(self.weight, 'fc update weight ')
        show_matrix(self.bias, 'fc update bias ')

    def load_param(self, weight, bias):
        # assert self.weight.shape == weight.shape
        # assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
        # show_matrix(self.weight, 'fc weight ')
        # show_matrix(self.bias, 'fc bias ')
    def save_param(self):
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')
        return self.weight, self.bias

class Sigmod(object):
    def __init__(self):
        self.output = None
        print('\tSigmod layer.')
    def forward(self, input):
        
        # output = spc.expit(input)
        output = 1.0 / (1.0 + np.exp(-input))
        self.output = output
        #output = 0.5 * (1 + np.tanh(0.5 * input))
        return output
    
    def backward(self, top_diff):
        
        bottom_diff = top_diff * self.output * (1 - self.output)
        return bottom_diff
# 131-767-715
class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')
    def forward(self, input):
        start_time = time.time()
        self.input = input
        in_mask = input <= 0 
        # output = np.maximum(0, self.input)
        output = input.copy()
        output[in_mask] = 0
        show_matrix(output, 'relu out')
        show_time(time.time() - start_time, 'relu forward time: ')
        return output
    def backward(self, top_diff):
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff

class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        exp_sum = np.sum(input_exp, axis=1, keepdims=True)
        self.prob = input_exp / exp_sum
        return self.prob
    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self): 
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size

        return bottom_diff
    
if __name__ == '__main__':
    net = FullyConnectedLayer(100,200)
    print(net.init_param())

