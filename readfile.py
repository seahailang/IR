#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'zsy'
__mtime__ = '2016/10/13'"""
import csv
import math
import os
import pickle
from datetime import datetime
from queue import Empty

import jieba
import numpy as np
import pynlpir
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing.data import normalize

from filepath import *
# BasePath = os.path.abspath(os.path.dirname(__file__))

# TRAINSETFILE = os.path.join(BasePath, 'data/user_tag_query.10W.TRAIN')
# TESTSETFILE = os.path.join(BasePath, 'data/user_tag_query.10W.TEST')
# TEMPFILE = os.path.join(BasePath, 'temp/dataset')
# RESULTFILE = os.path.join(BasePath, 'data/result.csv')


def readFile(filename=TRAINSETFILE, IsTraining=True):
    with open(filename, encoding='GB18030') as file:
        filereader = csv.reader(file, dialect='excel-tab', quoting=csv.QUOTE_NONE)
        if IsTraining:
            infoflag = 4
        else:
            infoflag = 1
        # count_test =0
        querylist = []
        userinfolist = []
        for userquery in filereader:
            userinfolist.append(userquery[:infoflag])
            querylist.append(userquery[infoflag:])
    return querylist, userinfolist


def dump(filename, obj):
    # 把用户信息和词频向量保存在temp中，以dump的方式
    with open(TEMPFILE+'/'+filename, 'wb') as file:
        pickle.dump(obj, file)


def jiebaTokenizer(document):
    return jieba.cut(document, cut_all=False)  # 精确模式
    # return pynlpir.segment(document, pos_tagging=False)


def addComma(querylist):
    return '，'.join(querylist)


def printTime():
    print(datetime.now())


if __name__ == '__main__':
    pynlpir.open()
    printTime()
    trainquery, traininfo = readFile()
    testquery, traininfo = readFile(TESTSETFILE, IsTraining=False)
    trainquery.extend(testquery)
    dump('traininfo', traininfo)
    vectorizer = TfidfVectorizer(norm=None, preprocessor=addComma, tokenizer=jiebaTokenizer, sublinear_tf=True,max_df=0.7,min_df=0.00001)
    csrtrain = vectorizer.fit_transform(trainquery)
    print(csrtrain.shape)
    dump('csrtrain_pickle', csrtrain[:100000,:])
    printTime()
    dump('testinfo', traininfo)
    # csrtest = vectorizer.transform(testquery)
    dump('csrtest_pickle', csrtrain[100000:,:])
    printTime()
    pynlpir.close()
    # with open('csrtrain_sk_pickle', 'rb') as f:
    #     csrtrain = pickle.load(f)
    # print()

    # printTime()
    # trainquery = readFile()
    #
    # L = []
    # for doc in trainquery:
    #     doclen = 0
    #     for item in doc:
    #         doclen += len(item)
    #     L.append(doclen)
    # Larr = np.array(L)
    # Lave = np.mean(Larr)
    # LL = Larr/Lave
    # print()


