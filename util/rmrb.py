#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/08/2017 2:49 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : rmrb.py
# @Software: PyCharm
import re
from pprint import pprint
filePath = u"/Users/zhangzhen/Downloads/rmrb.txt"
outputPath = u"train.txt"

# 判断一个unicode是否是汉字
def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fff':
        return True
    else:
        return False


# 判断一个unicode是否是数字
def is_number(uchar):
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


# 判断一个unicode是否是英文字母
def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


# 判断是否非汉字，数字和英文字符
def is_other(uchar):
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def filter_pos(str):
    reList = list()
    regex = re.compile("/(\w)*\s")
    line, num = re.subn(regex, " ", str)
    for word in line.decode("utf-8").split():
        tmp = list()
        for ch in word:
            if is_chinese(ch):
                tmp.append(ch)
        if len(tmp) > 0:
            reList.append("".join(tmp))
    return reList


def readfile2word_lists(f_path):
    rrs = list()
    with open(f_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        rs = filter_pos(line)
        if len(rs) > 0:
            rrs.append(rs)
    return rrs


def out_train_data(rrs, out_path):
    with open(out_path, "w") as out:
        for rs in rrs:
            tmp = "\t".join(rs)
            out.write(tmp.encode("utf-8")+"\n")
        out.close()


def load_train_data(train_path):
    rrs = list()
    with open(train_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            rs = line.split("\t")
            rrs.append(rs)
    return rrs


def load_train_data_to_tuple(train_path):
    rrs = list()
    with open(train_path, 'r') as f:
        lines = f.readlines()
        tmp = list()
        for line in lines:
            line = line.strip("\n")
            rs = line.split("\t")
            for r in rs:
                r = r.decode("utf-8")
                if len(r) == 1:
                    tmp.append((r, 'S'))
                elif len(r) == 2:
                    tmp.append((r[0], 'B'))
                    tmp.append((r[1], 'E'))
                else:
                    tmp.append((r[0], 'B'))
                    for ch in r[1:-1]:
                        tmp.append((ch, 'M'))
                    tmp.append((r[-1], 'E'))
        rrs.append(tmp)
    return rrs

if __name__ == '__main__':
    # rrs = readfile2word_lists(filePath)
    # out_train_data(rrs, outputPath)
    # load_train_data(outputPath)
    load_train_data_to_tuple(outputPath)

