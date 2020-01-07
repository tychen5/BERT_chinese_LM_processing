from bixin_2 import predict  # revised version of bixin package by Leo
from textblob import TextBlob
import pandas as pd
import pickle, time
import numpy as np
from snownlp import SnowNLP
from googletrans import Translator
from translate import Translator as tT
from tqdm import tqdm
from translation import iciba
from opencc import OpenCC
import goslate
import subprocess


def sentimize(tok_news_pd):
    senta_path = "./TokSentLeo/Senta/senta.txt"
    fourth_algo_checkpoint = "./TokSentLeo/Senta/SentaDL_output.pkl"
    import nltk
    nltk.download('all')

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
    cc2 = OpenCC('t2s')  # trad2sim

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
            if 0.000001 > emotion > -0.000001:
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
            try:
                blob = TextBlob(doc)
                cn_text = str(blob.translate(to="zh-CN"))
            except:
                cn_text = cc2.convert(doc)
        score = predict(cn_text)
        return score

    def combine_tok(s):
        """
        for pandas to combine tokens as a doc
        :param s: pandas series
        :return:  string
        """
        return "".join(s.split())

    tok_news_pd['cleaned_news'] = tok_news_pd.tok_title_news.map(combine_tok)
    print("running first algorithm...")
    first_start = time.time()
    tok_news_pd['tb_score'] = tok_news_pd.cleaned_news.map(text_blob)
    first_end = time.time() - first_start
    second_start = time.time()
    print("First algo. Time:",
          '{:02f}:{:02f}:{:02f}'.format(first_end // 3600, (first_end % 3600 // 60), first_end % 60))
    print("running second algorithm...")
    tok_news_pd['sn_score'] = tok_news_pd.cleaned_news.map(snow_nlp)
    second_end = time.time() - second_start
    third_start = time.time()
    print("Second algo. Time:",
          '{:02f}:{:02f}:{:02f}'.format(second_end // 3600, (second_end % 3600 // 60), second_end % 60))
    print("running third algorithm...")
    tok_news_pd['bx_score'] = tok_news_pd.cleaned_news.map(bi_xin)
    third_end = time.time() - third_start
    print("Third algo. Time:",
          '{:02f}:{:02f}:{:02f}'.format(third_end // 3600, (third_end % 3600 // 60), third_end % 60))
    # do in first time for senta only
    with open(senta_path, 'w', encoding='utf-8') as f:
        senta_li = tok_news_pd.tok_title_news.tolist()
        for news in tqdm(senta_li):
            news_tok = news.split()
            news_ = "╱".join(news_tok)  # need to be the same
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
            news_cn = news_cn.split("╱")  # need to be the same
            news_cn = " ".join(news_cn)
            f.write("0" + '\t' + news_cn + '\n')

    print("running forth algorithm...")
    subprocess.call("python ./TokSentLeo/Senta/sentiment_classify.py  --test_data_path " + senta_path
                    + "--word_dict_path ./TokSentLeo/Senta/config/train.vocab --mode infer --model_path " +
                      "./TokSentLeo/Senta/config/Senta/")
    fourth_algo = pickle.load(open(fourth_algo_checkpoint, 'rb'))
    four_algos = pd.merge(tok_news_pd, fourth_algo, left_on='Unnamed: 0', right_on='input_id', how='outer')
    four_algos = four_algos.filter(['Date', 'input_id', 'tb_score', 'sn_score', 'bx_score', 'senta_score'])  # df's cols
    return four_algos
