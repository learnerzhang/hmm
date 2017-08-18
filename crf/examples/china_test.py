#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/08/2017 5:42 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : china_test.py
# @Software: PyCharm
from china import *
from pprint import pprint

def testCase(crf, s):
    test = []
    for w in s:
        test.append((w, check_ch(w), ""))
    rs =crf.predict_marginals_single(sent2features(test))
    print rs
    for i, r in enumerate(rs):
        dictList = sorted(r.iteritems(), key=lambda d: d[1], reverse=True)
        label = dictList[0][0]
        if label =='S' or label == 'E':
            print s[i]
        else:
            print s[i],

if __name__ == '__main__':
    crf = joblib.load("crf.m")
    print crf
    s = u"１９９７年/t  ，/w  是/v  中国/ns  发展/vn  历史/n  上/f  非常/d  重要/a  的/u  很/d  不/d  平凡/a  的/u  一/m  年/q  。/w  中国/ns  人民/n  决心/d  继承/v  邓/nr  小平/nr  同志/n  的/u  遗志/n  ，/w  继续/v  把/p  建设/v  有/v  中国/ns  特色/n  社会主义/n  事业/n  推向/v  前进/v  。/w  [中国/ns  政府/n]nt  顺利/ad  恢复/v  对/p  香港/ns  行使/v  主权/n  ，/w  并/c  按照/p  “/w  一国两制/j  ”/w  、/w  “/w  港人治港/l  ”/w  、/w  高度/d  自治/v  的/u  方针/n  保持/v  香港/ns  的/u  繁荣/an  稳定/an  。/w  [中国/ns  共产党/n]nt  成功/a  地/u  召开/v  了/u  第十五/m  次/q  全国/n  代表大会/n  ，/w  高举/v  邓小平理论/n  伟大/a  旗帜/n  ，/w  总结/v  百年/m  历史/n  ，/w  展望/v  新/a  的/u  世纪/n  ，/w  制定/v  了/u  中国/ns  跨/v  世纪/n  发展/v  的/u  行动/vn  纲领/n  。/w  "
    testCase(crf, s)

