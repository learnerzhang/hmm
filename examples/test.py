#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/08/2017 9:36 AM
# @Author  : zhangzhen
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import numpy as np

x1 = [1, 2, 3, 4, 5]
x2 = [1, 2, 3, 4, 5]
x = [x1, x2]
X = np.array(x)

x11 = [1,2]
x22 = [1,2]
XX = np.array([x11,x22])
print X
print XX

print np.concatenate((X, XX), axis=1)