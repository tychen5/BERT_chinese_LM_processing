# import snownlp
# import bixin
from textblob import TextBlob
import pandas as pd
import time
import numpy as np
from snownlp import SnowNLP
from googletrans import Translator

pre = "D:\python\LEO_TM\BERT_chinese_LM_processing\src\\"  # for pycharm only
# define_dict_path = pre + "../data/TextTokenize_zh/company_dict.txt"
ori_news_path = pre + "../data/TextTokenize_zh/Foxconn_News_2018.csv"  # POC_NEWS.csv
tok_news_path = pre + "../Results/tokenize_Foxconn2018_news_Leo.xlsx"


def snow_nlp(doc):
    """
    SnowNLP sentiment analysis (unicode?)
    :param doc: with a punctuation's document (zh)
    :return: snow nlp sentiment score (1 positive, 0 negative)
    """
    sn = SnowNLP(doc)
    score = []
    for sentence in sn.sentences:
        temp = SnowNLP(sentence)
        score.append(temp.sentiments * 2 - 1)  # -1~1
    return np.mean(score)


translator_en = Translator()


def text_blob(doc):
    """
    TextBlob Sentiment (by segment)
    :param doc: zh doc
    :return: (-1~1)
    """
    try:
        en_text = translator_en.translate(doc).text
        blob = TextBlob(en_text)
    except:
        blob = TextBlob(doc)
        blob = blob.translate()
    score = []
    for sentence in blob.sentences:
        emotion = sentence.sentiment.polarity * sentence.sentiment.subjectivity
        if 0.00001 > emotion > -0.00001:
            continue
        score.append(emotion)
    return np.mean(score)

def combine_tok(s):
    """
    for pandas to combine tokens as a doc
    :param s: pandas series
    :return:  string
    """
    return "".join(s.split())


tok_news_pd = pd.read_excel(tok_news_path)
tok_news_pd['cleaned_news'] = tok_news_pd.tok_title_news.map(combine_tok)
tok_news_pd['tb_score'] = tok_news_pd.cleaned_news.map(text_blob)
tok_news_pd['sn_score'] = tok_news_pd.cleaned_news.map(snow_nlp)

# clean_news = tok_news_pd.cleaned_news.tolist()
# score_li = []
# for news in clean_news:
#     score = text_blob(news)
#     score_li.append(score)

