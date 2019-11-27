# coding: utf-8
"""""""""""""""""""""""""""
@author: hedho
@email: hedho@deloitte.com
"""""""""""""""""""""""""""

### Import Packages ###
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.getcwd())+r'\packages')
from jieba_split_beta import zh_split
from text_mining_beta import text_mining

### Load data ###
path = os.path.dirname(os.getcwd())+r'\data\\'+os.listdir(os.path.dirname(os.getcwd())+r'\data')[0]
內規 = pd.read_excel(path,sheet_name='內部法規')
外規 = pd.read_excel(path,sheet_name='外部法規')
# 合併內外規資料
data = pd.concat([內規['條文內容'],外規['條文內容']]).reset_index().drop('index',axis=1)
# 確保資料格式都為string
data['條文內容'] = data['條文內容'].apply(lambda x:str(x))

### Split Words ###
# 載入繁體版Jieba
analyzer = zh_split(os.path.dirname(os.getcwd())+r'\dictionary\dict.txt.big')
# 載入自定義字典
analyzer.add_dictionary(os.path.dirname(os.getcwd())+r'\dictionary\edu_dict.txt')
analyzer.add_dictionary(os.path.dirname(os.getcwd())+r'\dictionary\company_dict.txt')
analyzer.add_dictionary(os.path.dirname(os.getcwd())+r'\dictionary\geo_dict.txt')
analyzer.add_dictionary(os.path.dirname(os.getcwd())+r'\dictionary\law_dict.txt')
analyzer.add_dictionary(os.path.dirname(os.getcwd())+r'\dictionary\FCS_dict.txt')
# 斷詞
analyzer.split(data['條文內容'])
analyzer.get_dictionary()
# 斷詞結果
analyzer.split_list
analyzer.word_filter(w_len=0, # 過濾單詞字數 <= N的單詞
                     path_word=os.path.dirname(os.getcwd())+r'\dictionary\stop_words.txt' # 停止詞字典
                    )
# 停止詞過濾結果
analyzer.split_list
#找出潛在關鍵字
analyzer.find_keyword(n=2)
#潛在關鍵字結果
analyzer.add_word #會按機率跳出潛在的組合字詞，可以透過人工方式加入字典

### Vectorize ###
# 載入 text monong 分析器
tm = text_mining(analyzer.split_list)
# get dictionary
tm.get_dictionary()
# 計算TF
tm.CounterVector()
#print('TF_VECTOR : ', tm.TF_Vector)
# 計算TFIDF
tm.TfidfVector()
#print('TFIDF_VECTOR : ', tm.TFIDF_Vector)
# LSI
lsi_result, lsi_model = tm.LSI(tm.TFIDF_Vector, 
                               n_dimension = 10 #維度
                              )
# Doc2vec
doc_result, doc_model = tm.doc2vec(vector_size = 10, #維度
                                   epochs = 50, dbow_words= 1, dm=0, iter=1, window=5
                                  )
doc_result = [list(enumerate(row)) for row in doc_result]

### Similarity ###
# lsi similarity
tm.sim(lsi_result, lsi_result)
# doc2vec similarity
tm.sim(doc_result, doc_result)
# ensemble
(tm.sim(lsi_result, lsi_result) + tm.sim(doc_result, doc_result)) / 2
