#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08/08/2017 4:29 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : viterbi.py
# @Software: PyCharm
import sys
import numpy as np
from pprint import pprint

if __name__ == '__main__':

    a = [[0, 4, 2, 3, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 10, 9, 0, 0, 0, 0],
         [0, 0, 0, 0, 6, 7, 10, 0, 0, 0],
         [0, 0, 0, 0, 0, 3, 8, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 4, 8, 0],
         [0, 0, 0, 0, 0, 0, 0, 9, 6, 0],
         [0, 0, 0, 0, 0, 0, 0, 5, 4, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    pprint(a)

    dist = np.zeros(len(a))
    path = np.zeros(len(a))

    max = sys.maxint
    for i in range(1, len(a)):
        k = max
        for j in range(i):
            if a[j][i] != 0:
                if dist[j] + a[j][i] < k:
                    k = dist[j]+a[j][i]
        dist[i] = k

    cur = 1
    path[0] = 10
    for i in range(len(a), 0, -1):
        for j in range(i, 0, -1):
            if a[j-1][i-1] != 0:
                b = dist[i-1] - a[j-1][i-1]
                if b == dist[j-1]:
                    path[cur] = j
                    i = j
                    cur += 1
                    break
    print u"起始点到各个节点的最短路径:"
    pprint(dist)
    print u"最短路径:"
    pprint(path)
