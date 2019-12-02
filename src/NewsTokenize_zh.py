from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from tqdm import tqdm
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import *
from summa import keywords

# data_utils.download_data_gdown("../Model/") # only first time
pre = "D:\python\LEO_TM\BERT_chinese_LM_processing\src\\"  # for pycharm only
define_dict_path = pre + "../data/TextTokenize_zh/company_dict.txt"
news_path = pre + "../data/TextTokenize_zh/POC_NEWS.csv"

ws = WS(pre + "../Model/data")
pos = POS(pre + "../Model/data")
ner = NER(pre + "../Model/data")

word_to_weight = {}
with open(define_dict_path, "r", encoding='utf8') as file:
    for line in tqdm(file):
        key, value = line.split()
        word_to_weight[str(key)] = 2
dictionary = construct_dictionary(word_to_weight)
print(dictionary)  # dict

news_df = pd.read_csv(news_path, encoding='cp950')
all_news_li = news_df.內容.tolist()

word_sentence_list = ws(all_news_li, recommend_dictionary=dictionary)
print(len(word_sentence_list))
print(word_sentence_list)
pos_sentence_list = pos(word_sentence_list)
entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
# TF (1~XXXX)=>min max normalize, NER(1/0 or 1~XX)=>set 1, TF-IDF (0~1)

# by news TF
TF_news_li = []
for i, news_toks in enumerate(word_sentence_list):
    temp = dict(Counter(news_toks))
    df = pd.DataFrame(list(temp.items()), columns=['Word', 'TF_norm_score'])
    df.TF_norm_score = (df.TF_norm_score - df.TF_norm_score.min()) / (df.TF_norm_score.max() - df.TF_norm_score.min())
    # for item in temp.items():
    #     df.append(pd.Series(list(item)))
    all_words = df.Word.tolist()
    for i, w in enumerate(all_words):
        if len(w) < 2:
            df.iloc[i, 1] = df.iloc[i, 1] - df.TF_norm_score.mean()
    TF_news_li.append(df)
print(TF_news_li[0])

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
print(NER_news_li[0])


def transform(s):
    if s != 0:
        return 1
    else:
        return 0.0


COM_df_li = []
for TF_news_df, NER_news_df in zip(TF_news_li, NER_news_li):
    com_df = TF_news_df.merge(NER_news_df, how='outer', on='Word')
    com_df = com_df.fillna(0)
    com_df.NER = com_df.NER.map(transform)
    com_df['score'] = com_df.TF_norm_score + com_df.NER
    COM_df_li.append(com_df)
print(COM_df_li[0])  # 1st doc's each term's score

# by news TFIDF
TFIDF_df_li = []
all_corpus = []
for sentence in word_sentence_list:
    all_corpus.append(" ".join(sentence))
print(all_corpus[0])
vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
vectoerizer.fit(all_corpus)
X = vectoerizer.transform(all_corpus)
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(X.toarray())
word = vectoerizer.get_feature_names()
weight = tfidf.toarray()
for i in range(len(weight)):
    # print("text:",i)
    df = pd.DataFrame(columns=['Word', 'NER'])
    for j in range(len(word)):
        if weight[i][j] <= 0:
            continue
        temp = [word[j], weight[i][j]]
        temp = pd.Series(temp, index=df.columns)
        df = df.append(temp, ignore_index=True)
        # print(word[j],weight[i][j])
    TFIDF_df_li.append(df)
print(TFIDF_df_li[0])

#by NEWS TR
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
            df.iloc[i, 1] = df.iloc[i, 1] - df.TR_normScore.median()
    TR_df_li.append(df)
print(TR_df_li[0])

# for ALL NER
for i, en_sentence in enumerate(entity_sentence_list):  # all
    for entity in en_sentence:
        print(entity)