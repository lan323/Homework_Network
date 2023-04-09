# -*- coding: UTF-8 -*-
import numpy as np
import os
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def process_data(data):  
    data_in = [np.reshape(x, (784, 1)) for x in data[0]]
    data_re = [make_vector(kind) for kind in data[1]]
    data = list(zip(data_in, data_re))
    return data

def make_vector(kind):
    #将 train_data[1] 转化为一个列矩阵
    vector = np.zeros((10, 1))
    vector[kind] = int(1)
    return vector

def load_data():
    filename = open("C:/Users/admin/Desktop/data/mnist.pkl", "rb")
    train_data, v_data, test_data = pickle.load(filename, encoding="latin1")
    filename.close()
    train_data = process_data(train_data)
    v_data = process_data(v_data)
    test_data = process_data(test_data)
    return train_data, v_data, test_data

def sig(inp):
    temp = 1 / (1 + np.exp(-inp))
    return temp
def sig_delta(inp):
    temp = sig(inp) * (1-sig(inp))
    return temp
def relu(inp):
    temp = np.maximum(0, inp)
    return temp
def relu_delta(inp):
    return inp > 0
def soft(inp):
    temp = np.exp(inp) / np.sum(np.exp(inp))
    return temp

def calculate(network, b_w_tool_realize, train_data, v_data, epos):

    updating_accuracy = 0
    train_loss_list = []
    v_loss_list = []
    accuracy_list = []

    for e in range(epos):
        v_loss = 0
        ture_result = []
        for inp, kind_ in v_data:
            out = network.forth(inp)
            # 因为经过softmax函数，很多值为0，所以全部+0.1，再取log
            v_loss += np.where(kind_ == 1, -np.log(out+0.1), 0).sum()
            ture_result.append(np.argmax(out) == np.argmax(kind_))
        train_loss = 0
        B = [train_data[k:k + b_w_tool_realize.bsize] for k in range(0, len(train_data), b_w_tool_realize.bsize)]
        
        accuracy = sum(ture_result) / 100.0
        accuracy_list.append(accuracy)

        if accuracy > updating_accuracy:
            updating_accuracy = accuracy
            np.savez_compressed(file=(f"{network.network_size[1]}_{b_w_tool_realize.lr}_{b_w_tool_realize.w_para_decay}.npz"),weights=network.weights, biases=network.biases,network_l1_biases=network.network_l1_biases,network_l2_biases=network.network_l2_biases)
        v_loss /= len(v_data)
        v_loss_list.append(v_loss)

        for b in B:
            
            b_w_tool_realize.make_b_w_zero()
    
            for inp, kind_ in b:
                out = network.forth(inp)
                train_loss += np.where(kind_ == 1, -np.log(out+0.1), 0).sum()
                
                last_E = out - kind_
                bias_delta, weighs_delta = network.back(last_E)
                
                b_w_tool_realize.change_b_w(bias_delta, weighs_delta)
                
            b_w_tool_realize.reset_b_w()

        num_train_data = len(train_data)
        train_loss = train_loss/num_train_data 
        train_loss_list.append(train_loss)

        print(f"第 {e + 1}个运算过程的准确度为百分之{accuracy} ")
    result_frame = {"train_loss": train_loss_list,"validate_loss": v_loss_list,"validate_accuracy": accuracy_list}
    pd.DataFrame(result_frame).to_csv(f'{network.network_size[1]}_{b_w_tool_realize.lr}_{b_w_tool_realize.w_para_decay}.csv', )

    return updating_accuracy


def test_score_accuracy(network, test_data):
    ture_result = []

    for inp, kind_ in test_data:
        out = network.forth(inp)
        ture_result.append(np.argmax(out) == np.argmax(kind_))

    score = sum(ture_result)
    accuracy =score / 100
    print("测试集的准确率为：",accuracy)
    

class build_network:

    def __init__(self, network_size):
        self.network_size = network_size
        self.network_l_count = len(network_size)
        self.weights = [np.array([0])] + [np.random.randn(column, row) for column, row in zip(network_size[1:], network_size[:-1])]
        self.biases = [np.array([0])] + [np.random.randn(row, 1) for row in network_size[1:]]
        
        self.network_l1_biases = [np.zeros(bias.shape) for bias in self.biases]
        self.network_l2_biases = [np.zeros(bias.shape) for bias in self.biases]

    def forth(self, inp):
        self.network_l2_biases[0] = inp
        for i in range(1, self.network_l_count):
            self.network_l1_biases[i] = self.weights[i].dot(self.network_l2_biases[i - 1]) + self.biases[i]
            if i == self.network_l_count - 1:
                self.network_l2_biases[i] = soft(self.network_l1_biases[i])
            else:
                self.network_l2_biases[i] = relu(self.network_l1_biases[i])
        return self.network_l2_biases[-1]

    def back(self, last_E):
        temp_bias = [np.zeros(bias.shape) for bias in self.biases]
        temp_weigh = [np.zeros(weight.shape) for weight in self.weights]

        temp_bias[-1] = last_E
        temp_weigh[-1] = last_E.dot(self.network_l2_biases[-2].transpose())

        for i in range(self.network_l_count-2,0,-1):
            last_E = np.multiply(self.weights[i+1].transpose().dot(last_E),relu_delta(self.network_l1_biases[i]))
            temp_bias[i] = last_E
            temp_weigh[i] = last_E.dot(self.network_l2_biases[i - 1].transpose())
        return temp_bias, temp_weigh



class b_w_tool:

    def __init__(self, network, a, w_para_decay, bsize):
        self.network = network
        self.lr = a
        self.w_para_decay = w_para_decay
        self.bsize = bsize

        self.temp_bias = [np.zeros(bias.shape) for bias in self.network.biases]
        self.temp_weigh = [np.zeros(weight.shape) for weight in self.network.weights]

    def reset_b_w(self):
        self.network.weights = [(1 - self.lr * self.w_para_decay) * w - (self.lr / self.bsize) * dw for w, dw in zip(self.network.weights, self.temp_weigh)]
        self.network.biases = [(1 - self.lr * self.w_para_decay) * b - (self.lr / self.bsize) * db for b, db in zip(self.network.biases, self.temp_bias)]

    def change_b_w(self, bias_delta, weighs_delta):
        self.temp_bias = [nb + dnb for nb, dnb in zip(self.temp_bias, bias_delta)]
        self.temp_weigh = [nw + dnw for nw, dnw in zip(self.temp_weigh, weighs_delta)]

    def make_b_w_zero(self):
        self.temp_bias = [np.zeros(bias.shape) for bias in self.network.biases]
        self.temp_weigh = [np.zeros(weight.shape) for weight in self.network.weights]



if __name__ == '__main__':
    structure_list = [[784, 40, 10], [784, 50, 10], [784, 60, 10], [784, 70, 10]]
    a_s = [1e-2,2e-2,5e-3]
    w_para_decay_list = [0,1e-2,2e-2,5e-3]
    bsize=16;epos=5
    train_data, v_data, test_data = load_data()
    print("导入数据成功！")
    final_result_setting = {'accuracy': 0}
    for structure in structure_list:
        for a in a_s:
            for w_para_decay in w_para_decay_list:
                network = build_network(structure)
                b_w_tool_realize = b_w_tool(network, a, w_para_decay, bsize)
                accuracy = calculate(network, b_w_tool_realize, train_data, v_data, epos)
                if accuracy > final_result_setting['accuracy']:
                    final_result_setting['accuracy'] = accuracy
                    final_result_setting['structure'] = structure
                    final_result_setting['a'] = a
                    final_result_setting['w_para_decay'] = w_para_decay
                print(f"目前的结构为:{structure},当前的学习率为:{a},当前的权重衰退系数为: {w_para_decay}，最终准确率为：{accuracy}")
    print((f"最优的结构为:{final_result_setting['structure']},当前的学习率为:{final_result_setting['a']},当前的权重衰退系数为: {final_result_setting['w_para_decay']}，最终准确率为：{final_result_setting['accuracy']}"))

    file_ = pd.read_csv(f"{final_result_setting['structure'][1]}_{final_result_setting['a']}_{final_result_setting['w_para_decay']}.csv")
    file_[['train_loss','validate_loss']].plot()
    plt.xlabel("运行次数")
    plt.ylabel("损失")

    file_[['validate_accuracy']].plot()
    plt.xlabel("运行次数")
    plt.ylabel("准确率")
