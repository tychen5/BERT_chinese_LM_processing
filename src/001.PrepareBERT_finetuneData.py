import numpy as np
import pandas as pd
import sys
import os
from collections import Counter
import gensim
# import lightgbm as lgb
import xlrd
import importlib
import datetime
import shutil

import string
import re

import jieba
import jieba.analyse

# from sklearn import preprocessing

from collections import Counter

from jieba_split_beta import zh_split
from text_mining_beta import text_mining

DATA_PATH = "../Data/SinoPac/v0.93_20180718_Jerry.xlsx"

data_df = pd.read_excel(DATA_PATH, sheet_name=2)
data_df.loc[:, 'LABEL'] = 'Inside'
data_df = data_df.reset_index(drop=True)
data_df = data_df.drop_duplicates()
print(data_df.columns)
print("內規法規數量：", len(np.unique(data_df.法規名稱)))  # big category
print("內規條文數量：", data_df.shape[0])  # small regulation details
print('內規列表')
for c, value in enumerate(np.unique(data_df.法規名稱), 1):
    print(c, value)  # big category


def re_func(o_s, r_s):
    """
    meta function
    o_s: original string
    :param r_s: re string to be applied
    :return: after removal string
    """
    if len(r_s) < 2:
        re_punctuation = "[{}]+".format(r_s)
        line = re.sub(re_punctuation, "", o_s)
        return line
    else:
        re_punctuation = "{}+".format(r_s)
        line = re.sub(re_punctuation, "", o_s)
        return line


def step1(s):
    """
    filter out words before ":"
    :param s: string type of 條文內容
    :return: needed words
    """
    return s.split(":")[-1]


def step2(s):
    """
    filter out symbols
    :param s: string of Series
    :return: preprocessed string
    """
    punctuation_zh = """＂＃＄％＆＇（）＊＋－／；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
    punctuation_en = string.punctuation
    re_punctuation = "[{}]+".format(punctuation_zh + punctuation_en)
    line = re.sub(re_punctuation, "", s)
    return line


def step3(s):
    """
    filter out chinese and english number
    :param s: Series string
    :return: preprocessed string
    """
    ch1 = "壹貳參肆伍陸柒捌玖拾佰仟萬億圓角零"
    ch2 = "壹贰叁肆伍陆柒捌玖拾佰仟万亿元角零"
    ch3 = "一二兩三四五六七八九十"
    en1 = "0123456789"
    re_punctuation = "[{}]+".format(ch1 + ch2 + ch3 + en1)
    line = re.sub(re_punctuation, "", s)
    return line


def step4(s):
    """
    filter out specific words
    :param s: Series of string
    :return: preprocessed string
    """
    s = re_func(s, "註")
    s = re_func(s, "表格")
    s = re_func(s, "圖表")
    s = re_func(s, "附件")
    s = re_func(s, "圖")
    s = re_func(s, "第節")
    s = re_func(s, "第類")
    s = re_func(s, "第條")
    s = re_func(s, "第款")
    line = re_func(s, "第項")
    return line


def step5(s):
    """
    filter out alphabet only one character
    :param s: Series string
    :return: preprcoesses string
    """
    re_s_li = re.findall(r'[a-zA-Z]+', s)
    for r_s in re_s_li:
        if len(r_s) == 1:
            s = re_func(s, r_s)
    return s


def step6(s):
    """
    filter out sentence begin with :
    :param s:
    :return:
    """
    if s[0] == "：":
        s = s[1:]
    if s[0] == ":":
        s = s[1:]
    line = re_func(s, '\n')
    return line


data_df.條文內容 = data_df.條文內容.map(step1)
data_df.條文內容 = data_df.條文內容.map(step2)
data_df.條文內容 = data_df.條文內容.map(step3)
data_df.條文內容 = data_df.條文內容.map(step4)
data_df.條文內容 = data_df.條文內容.map(step5)
data_df.條文內容 = data_df.條文內容.map(step6)


def step_1(s):
    s = re_func(s, "篇")
    s = re_func(s, "修正")
    s = re_func(s, "條文")
    s = re_func(s, "手冊")
    s = re_func(s, "核定")
    s = re_func(s, "說明")
    s = re_func(s, "附件")
    s = re_func(s, "準則")
    s = re_func(s, "共通")
    line = re_func(s, "事項")
    return line


def step_2(s):
    """
    remove english and numbers
    :param s:
    :return:
    """
    return re.sub(r'[a-zA-Z0-9]+', "", s)


def step_3(s):
    """
    remove all punctuations of chinese and english
    :param s:
    :return:
    """
    punctuation_zh = """！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.，"""
    punctuation_en = string.punctuation
    re_punctuation = "[{}]+".format(punctuation_zh + punctuation_en)
    line = re.sub(re_punctuation, "", s)
    return line


def step_4(s):
    """
    remove number of chinese
    :param s:
    :return:
    """
    ch1 = "壹貳參肆伍陸柒捌玖拾佰仟萬億圓角零"
    ch2 = "壹贰叁肆伍陆柒捌玖拾佰仟万亿元角零"
    ch3 = "一二兩三四五六七八九十"
    en1 = "0123456789"
    re_punctuation = "[{}]+".format(ch1 + ch2 + ch3 + en1)
    line = re.sub(re_punctuation, "", s)
    return line


data_df.法規名稱 = data_df.法規名稱.map(step_1)
data_df.法規名稱 = data_df.法規名稱.map(step_2)
data_df.法規名稱 = data_df.法規名稱.map(step_3)
data_df.法規名稱 = data_df.法規名稱.map(step_4)