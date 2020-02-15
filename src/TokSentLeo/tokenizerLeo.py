from ckiptagger import construct_dictionary, WS, POS, NER
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import *
from summa import keywords
import time


def tokenize(news_df):
    """
    To tokenize & extract key word
    :param news_df: (title,content,date) 依照時間日期遞增的方式(1月=>12月)排序好之df
    :return: df: ('ori_title', 'ori_news', 'tok_title_news', 'keyWord_algorithm')
    """
    load_start = time.time()
    define_dict_path = "./TokSentLeo/user_dict/company_dict.txt"
    model_path = './TokSentLeo/CKIP_model/'
    ws = WS(model_path)
    pos = POS(model_path)
    ner = NER(model_path)

    word_to_weight = {}
    with open(define_dict_path, "r", encoding='utf8') as file:
        for line in file:
            key, value = line.split()
            word_to_weight[str(key)] = 2
    dictionary = construct_dictionary(word_to_weight)
    all_date_li = news_df.Date.tolist()
    all_news_li = news_df.Content.tolist()
    all_title_li = news_df.Title.tolist()
    all_news_li2 = []
    for title, news in zip(all_title_li, all_news_li):
        if type(news) == float:  # news is nan, only title
            all_news_li2.append(title)
        elif type(title) == float:
            all_news_li2.append(news)
        else:
            all_news_li2.append(title + "：" + news)

    load_end = time.time() - load_start
    tokenize_start = time.time()
    print("Model Load Time:", '{:02f}:{:02f}:{:02f}'.format(load_end // 3600, (load_end % 3600 // 60), load_end % 60))

    word_sentence_list = ws(all_news_li2, recommend_dictionary=dictionary,
                            segment_delimiter_set={",", "。", ":", "?", "!", ";", "，", "：", "？", "！", "；"})
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    temp = []
    temp1 = []
    bad_list = ["<br>", "br", "BR", "<BR>", "，", "【", "】", "╱", "▲", "▼", "&amp;amp;amp;amp;amp;lt;br",
                "&amp;amp;amp;amp;amp;gt", "amp", "lt", "br&", "gt", "&amp", "[", "]"]
    for w_s_l, e_s_l in zip(word_sentence_list, entity_sentence_list):
        # t = []
        # t1 = []
        # for i, x in enumerate(w_s_l):
        #     if x not in bad_list:
        #         t.append(x)
        #         t1.append(e_s_l[i])
        t = [x for x in w_s_l if x not in bad_list]
        temp.append(t)
        # temp1.append(t1)
    word_sentence_list = temp
    # entity_sentence_list = temp1

    tokenize_end = time.time() - tokenize_start
    print("DL Tokenize Time:",
          '{:02f}:{:02f}:{:02f}'.format(tokenize_end // 3600, (tokenize_end % 3600 // 60), tokenize_end % 60))
    algo_start = time.time()

    # by news TF
    TF_news_li = []
    for i, news_toks in enumerate(word_sentence_list):
        temp = dict(Counter(news_toks))
        df = pd.DataFrame(list(temp.items()), columns=['Word', 'TF_norm_score'])
        df.TF_norm_score = (df.TF_norm_score - df.TF_norm_score.min()) / (
                df.TF_norm_score.max() - df.TF_norm_score.min())
        # for item in temp.items():
        #     df.append(pd.Series(list(item)))
        all_words = df.Word.tolist()
        for i, w in enumerate(all_words):
            if len(w) < 2:
                df.iloc[i, 1] = df.iloc[i, 1] - (df.TF_norm_score.mean() + 3 * df.TF_norm_score.std())
        TF_news_li.append(df)

    # by news NER
    NER_news_li = []
    for i, en_sentence in enumerate(entity_sentence_list):
        df = pd.DataFrame(columns=['Word', 'NER'])
        word = []
        for entity in en_sentence:
            word_ = entity[-1]
            if word_ in word:
                continue
            word.append(word_)
            temp = [word_, entity[-2]]
            temp = pd.Series(temp, index=df.columns)
            df = df.append(temp, ignore_index=True)
        NER_news_li.append(df)

    # by news TFIDF
    TFIDF_df_li = []
    all_corpus = []
    for sentence in word_sentence_list:
        all_corpus.append(" ".join(sentence))
    # print(all_corpus[0])
    vectoerizer = CountVectorizer(min_df=3, max_df=0.9, token_pattern='\\b\\w+\\b')
    vectoerizer.fit(all_corpus)
    X = vectoerizer.transform(all_corpus)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(X.toarray())
    word = vectoerizer.get_feature_names()
    weight = tfidf.toarray()
    for i in range(len(weight)):
        # print("text:",i)
        df = pd.DataFrame(columns=['Word', 'Tfidf'])
        for j in range(len(word)):
            if weight[i][j] <= 0:
                continue
            temp = [word[j], weight[i][j]]
            temp = pd.Series(temp, index=df.columns)
            df = df.append(temp, ignore_index=True)
            # print(word[j],weight[i][j])
        TFIDF_df_li.append(df)

    # by NEWS TR
    TR_df_li = []
    for sentence in all_corpus:
        text_rank_words = keywords.keywords(sentence, split=True)
        all_length = len(text_rank_words)
        df = pd.DataFrame(columns=['Word', 'TR_normScore'])
        for i, words in enumerate(text_rank_words):
            word_li = words.split()
            for word in word_li:
                score = (all_length - i) / all_length
                temp = [word, score]
                temp = pd.Series(temp, index=df.columns)
                df = df.append(temp, ignore_index=True)
        all_words = df.Word.tolist()
        for i, w in enumerate(all_words):
            if len(w) < 2:
                df.iloc[i, 1] = df.iloc[i, 1] - (df.TR_normScore.mean() + 3 * df.TR_normScore.std())
        TR_df_li.append(df)

    # combine all
    COM2_df_li = []
    for tf_df, ner_df, tfidf_df, tr_df in zip(TF_news_li, NER_news_li, TFIDF_df_li, TR_df_li):
        com_df = tf_df.merge(ner_df, how='outer', on='Word')
        com_df = com_df.fillna(0)

        def transform(s):
            if s != 0:
                return (tfidf_df.Tfidf.median() + tr_df.TR_normScore.median())  # /2
            else:
                return 0.0

        com_df.NER = com_df.NER.map(transform)
        com_df['score'] = com_df.TF_norm_score + com_df.NER
        com2_df = com_df.merge(tfidf_df, on='Word', how='outer')
        com2_df = com2_df.merge(tr_df, on='Word', how='outer')
        com2_df = com2_df.fillna(0)
        com2_df['score'] = com2_df.score + com2_df.TR_normScore + com2_df.Tfidf
        COM2_df_li.append(com2_df)

    # write result
    df = pd.DataFrame(columns=['Date', 'ori_title', 'ori_news', 'tok_title_news', 'keyWord_algorithm'])  # df's columns
    month = []
    word_month = []
    score_month = []
    for day, title_str, news_str, news_tok_li, com_df in zip(all_date_li, all_title_li, all_news_li, word_sentence_list,
                                                             COM2_df_li):
        key_words = com_df[com_df.score > com_df.score.mean() + 1.65 * com_df.score.std()]  # 2*
        key_words = key_words.Word.tolist()
        key_words_month = com_df[com_df.score > com_df.score.mean() + 2 * com_df.score.std()]
        words_score_month = key_words_month.score.tolist()
        key_words_month = key_words_month.Word.tolist()

        temp = [str(day), title_str, news_str, " ".join(news_tok_li), "、".join(key_words)]
        temp = pd.Series(temp, index=df.columns)
        for word, score in zip(key_words_month, words_score_month):
            month.append(str(day).split('/')[1])
            word_month.append(word)
            score_month.append(score)
        df = df.append(temp, ignore_index=True)

    current_month = month[0]
    need_dict = {}
    dict_order = []
    month_order = []
    for i, (m, w, s) in enumerate(zip(month, word_month, score_month)):
        if m != current_month:
            month_order.append(current_month)
            current_month = m
            for k, v in need_dict.items():
                if len(v) < 3:  # DF<3 do not take
                    need_dict[k] = 0
                else:
                    need_dict[k] = np.mean(v)
            dict_order.append(need_dict)
            need_dict = {}
            need_dict[w] = list([float(s)])
            if i == len(month) - 1:
                dict_order.append(need_dict)

        else:
            if w not in list(need_dict.keys()):
                need_dict[w] = list([float(s)])
            else:
                temp = need_dict[w]
                temp.append(float(s))
                need_dict[w] = temp
            if i == len(month) - 1:
                month_order.append(m)
                for k, v in need_dict.items():
                    if len(v) < 3:  # DF<3 do not take
                        need_dict[k] = 0
                    else:
                        need_dict[k] = np.mean(v)
                dict_order.append(need_dict)
    df_month_key = pd.DataFrame(columns=['Month', 'key_word', 'score'])  # overall month key word with score
    for mo, dict_mo in zip(month_order, dict_order):
        for k, v in dict_mo.items():
            if v < 0.1:  # DF<3 do not take
                continue
            temp = [int(mo), str(k), v]
            temp = pd.Series(temp, index=df_month_key.columns)
            df_month_key = df_month_key.append(temp, ignore_index=True)

    algo_end = time.time() - algo_start
    print("KeyWord Algorithm Time:",
          '{:02f}:{:02f}:{:02f}'.format(algo_end // 3600, (algo_end % 3600 // 60), algo_end % 60))
    return df, df_month_key
