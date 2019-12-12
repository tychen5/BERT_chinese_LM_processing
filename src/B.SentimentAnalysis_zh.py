# import snownlp
from bixin import predict
from textblob import TextBlob
import pandas as pd
import time
import numpy as np
from snownlp import SnowNLP
from googletrans import Translator
from translate import Translator as tT
from tqdm import tqdm
from translation import iciba
from opencc import OpenCC
import goslate

pre = "D:\python\LEO_TM\BERT_chinese_LM_processing\src\\"  # for pycharm only
# define_dict_path = pre + "../data/TextTokenize_zh/company_dict.txt"
ori_news_path = pre + "../data/TextTokenize_zh/Foxconn_News_2018.csv"  # POC_NEWS.csv
tok_news_path = pre + "../Results/tokenize_Foxconn2018_news_Leo.xlsx"
senta_path = pre + "../data/TextTokenize_zh/Foxconn_Senta.txt"


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
translator_cn = Translator()
cc1 = OpenCC('s2t')
cc2 = OpenCC('t2s') #trad2sim

def text_blob(doc):
    """
    TextBlob Sentiment (by segment)
    :param doc: zh doc
    :return: (-1~1)
    """
    try:
        translator_en = Translator()
        en_text = translator_en.translate(doc).text
        blob = TextBlob(en_text)
    except:
        try:
            blob = TextBlob(doc)
            blob = blob.translate()
        except:
            try:
                en_text = iciba(doc, dst='en')
                blob = TextBlob(en_text)
            except:
                try:
                    gs = goslate.Goslate()
                    en_text = gs.translate(doc, 'en')
                    blob = TextBlob(en_text)
                except:
                    translator = tT(to_lang="en")
                    en_text = translator.translate(doc)
                    blob = TextBlob(en_text)
    score = []
    for sentence in blob.sentences:
        emotion = sentence.sentiment.polarity * sentence.sentiment.subjectivity
        if 0.00001 > emotion > -0.00001:
            continue
        score.append(emotion)
    return np.mean(score)


def bi_xin(doc):
    """
    customize pkg bixin for whole sentence
    :param doc: trad. chninise to simplified chinese
    :return: whole doc score of sentiment
    """
    try:
        translator_cn = Translator()
        cn_text = translator_cn.translate(doc, dest='zh-cn').text
    except:
        blob = TextBlob(doc)
        cn_text = str(blob.translate(to="zh-CN"))
    score = predict(cn_text)
    return score


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
tok_news_pd['bx_score'] = tok_news_pd.cleaned_news.map(bi_xin)

# do in first time for senta only
with open(senta_path, 'w', encoding='utf-8') as f:
    senta_li = tok_news_pd.tok_title_news.tolist()
    for news in tqdm(senta_li):
        news_tok = news.split()
        news_ = "╱".join(news_tok) #need to be the same
        # new_news = []
        # for news_ in news_tok:
        try:
            translator_cn = Translator()
            news_cn = translator_cn.translate(news_, dest='zh-cn').text
        except:
            try:
                blob = TextBlob(news_)
                news_cn = str(blob.translate(to="zh-CN"))
            except:
                news_cn = cc2.convert(news_)
            # new_news.append(news_cn)
        # news_cn = " ".join(new_news)
        news_cn = news_cn.split("╱") #need to be the same
        news_cn = " ".join(news_cn)
        f.write("0" + '\t' + news_cn + '\n')

# clean_news = tok_news_pd.cleaned_news.tolist()
# score_li = []
# for news in clean_news:
#     score = text_blob(news)
#     score_li.append(score)
