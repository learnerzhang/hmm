#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08/08/2017 10:56 AM
# @Author  : zhangzhen
# @Site    : 
# @File    : demo_hhm.py
# @Software: PyCharm
import numpy as np


class HMM:
    """
    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector
    """
    def __init__(self, A, B, pi):
        self.A = A  # 状态转移矩阵
        self.B = B  # 观测概率
        self.pi = pi

    def simulate(self, T):
        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        """根据当前状态生成下一个状态"""
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0], :])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t - 1], :])
            observations[t] = draw_from(self.B[states[t], :])
        return observations, states

    def _forward(self, obs_seq):
        # 取A = N x N
        N = self.A.shape[0]
        T = len(obs_seq)

        F = np.zeros((N, T))
        # alpha = pi*b
        F[:, 0] = self.pi * self.B[:, obs_seq[0]]
        for t in range(1, T):
            for n in range(N):
                # 计算第t时，第n个状态的前向概率
                F[n, t] = np.dot(F[:, t - 1], (self.A[:, n])) * self.B[n, obs_seq[t]]
        return F

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        X = np.zeros((N, T))
        # 表示X矩阵的最后一列
        X[:, -1:] = 1
        for t in reversed(range(T - 1)):
            for n in range(N):
                # 边权值为a_ji
                X[n, t] = sum(X[:, t + 1] * self.A[n, :] * self.B[:, obs_seq[t + 1]])
        return X

    def viterbi(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        prev = np.zeros((T-1, N), dtype=int)
        V = np.zeros((N, T))
        V[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                # 计算delta(j)*a_ji
                seq_probs = V[:, t - 1] * self.A[:, n] * self.B[n, obs_seq[t]]
                # 记录最大状态转移过程
                prev[t - 1, n] = np.argmax(seq_probs)
                V[n, t] = max(seq_probs)
        return V, prev

    def build_viterbi_path(self, prev, last_state):
        """
        returns a state path ending in last_state in reverse order.
        """
        T = len(prev)
        yield(last_state)
        # 从T-1开始，每次下降1
        for i in range(T - 1, -1, -1):
            yield (prev[i, last_state])
            last_state = prev[i, last_state]

    def state_path(self, obs_seq):
        V, prev = self.viterbi(obs_seq)
        # build state path with greatest probability
        last_state = np.argmax(V[:, -1])
        path = list(self.build_viterbi_path(prev, last_state))
        return V[last_state, -1], reversed(path)
# 隐含状态
states = ('sunny', 'cloudy', 'rainy')

# 观察状态
observations = ('dry', 'dryish', 'damp', 'soggy')

# 隐含状态的开始概率
start_probability = {'sunny': 0.6, 'cloudy': 0.2, 'rainy': 0.2}

# 隐含状态的转移概率
transition_probability = {
    'sunny': {'sunny': 0.5, 'cloudy': 0.375, 'rainy': 0.125},
    'cloudy': {'sunny': 0.25, 'cloudy': 0.125, 'rainy': 0.625},
    'rainy': {'sunny': 0.25, 'cloudy': 0.375, 'rainy': 0.375}
}

# 隐含状态到观测状态 ———发射概率
emission_probability ={
    'sunny': {'dry': 0.6, 'dryish': 0.2, 'damp': 0.15, 'soggy': 0.05},
    'cloudy': {'dry': 0.25, 'dryish': 0.25, 'damp': 0.25, 'soggy': 0.25},
    'rainy': {'dry': 0.05, 'dryish': 0.1, 'damp': 0.35, 'soggy': 0.50}
}


def generate_index_map(lables):
    index_label = {}
    label_index = {}
    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i
        i += 1
    return label_index, index_label
states_label_index, states_index_label = generate_index_map(states)
observations_label_index, observations_index_label = generate_index_map(observations)

print "states_label_index", states_label_index, "states_index_label", states_index_label
print "observations_label_index", observations_label_index, "observations_index_label", observations_index_label


def convert_observations_to_index(observations, label_index):
    list = []
    for o in observations:
        list.append(label_index[o])
    return list


def convert_map_to_vector(map, label_index):
    v = np.empty(len(map), dtype=float)
    for e in map:
        v[label_index[e]] = map[e]
    return v


def convert_map_to_matrix(map, label_index1, label_index2):
    m = np.empty((len(label_index1), len(label_index2)), dtype=float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m

# 隐状态转移概率
A = convert_map_to_matrix(transition_probability, states_label_index, states_label_index)
print "tp:\n", A

B = convert_map_to_matrix(emission_probability, states_label_index, observations_label_index)
observations_index = convert_observations_to_index(observations, observations_label_index)
print "ep:\n", B

print "observations_index:\n", observations_index

pi = convert_map_to_vector(start_probability, states_label_index)
print "pi:\n", pi

h = HMM(A, B, pi)

observations_data, states_data = h.simulate(10)
print u"随机生成观测序列和隐含状态"
print "观测结果", [observations_index_label[i] for i in observations_data]
print "隐含状态", [states_index_label[i] for i in states_data]

print ("forward: P(O|lambda) = %f" % sum(h._forward(observations_data)[:, -1]))
print ("backward: P(O|lambda) = %f" % sum(h._backward(observations_data)[:, 0]*pi*B[:, 0]))

print '-'*50
p, ss = h.state_path(observations_data)
path = []
for s in ss:
    path.append(states_index_label[s])
print "预测状态", path
print "viterbi: P(I|O) =%f" % p

print '-'*50

test_obj_index = [0, 2, 3]  # ['dry','damp','soggy']
p, ss = h.state_path(test_obj_index)
path = []
for s in ss:
    path.append(states_index_label[s])
print "预测状态", path
print "viterbi: P(I|O) =%f" % p

print '-'*50