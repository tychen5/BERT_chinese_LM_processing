
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import sys
sys.path.append('D:/python/TextMining/FUNCTION')
import os
from collections import Counter
import gensim
import lightgbm as lgb
import xlrd
import importlib
import datetime
import shutil


# In[17]:


## For Word Cloud
# import pandas as pd
# import numpy as np
# import os
import re

import jieba
import jieba.analyse

from sklearn import preprocessing 

from collections import Counter


# In[18]:


from jieba_split import zh_split
from text_mining import text_mining


# # Data Prepare

# In[19]:


DATA_PATH = 'D:/python/TextMining/ComplianceBot/POC/Sino/Data/[永豐]法規自動對應分析POC(銀行)_v0.93_20180718_Jerry.xlsx'


# In[20]:


data = pd.read_excel(DATA_PATH, sheetname = 2)
data.loc[:,'LABEL'] = '內規'


# In[21]:


# 調整data columns & index
data = data.reset_index(drop=True)


# In[22]:


data.columns


# In[23]:


# 資料範圍
data = data.drop_duplicates()
print("內規法規數量：", len(np.unique(data.法規名稱)))
print("內規條文數量：", data.shape[0])
print('內規列表')
for c, value in enumerate(np.unique(data.法規名稱), 1):
    print(c, value)


# ### Target Data

# In[24]:


target = pd.read_excel(DATA_PATH, sheetname = 3)
target.loc[:,'LABEL'] = '外規' 


# In[25]:


# 調整data columns & index
target.columns = ['法規名稱', '條文編號', '條文內容','LABEL']
target = target.reset_index(drop=True)


# #### 確定對應範圍(整份對應及單條對應)

# ###### 整份

# In[27]:


rule = pd.read_excel(DATA_PATH, sheetname = 1, nrows = 22, header = None)
rule = rule.drop(rule.columns[[0]], axis=1)


# In[28]:


# 對應規則
WL = []
for i in range(1,rule.shape[1]+1,1):
    for index, content in rule[[i]][1:].iterrows():
        WL.append(content.values[0])
WL = [x for x in list(set(WL)) if pd.notna(x)]


# In[29]:


W_INDEX = target[target.法規名稱.isin(WL)].index.tolist()


# ###### 單條

# In[32]:


in_rule = pd.read_excel(DATA_PATH, sheetname = 2)
in_rule = in_rule[in_rule['對應外部法規1'].notna()]


# In[33]:


l = []
for i in in_rule.index:m
    for j in range(1,15):
        if (pd.isnull(in_rule.loc[i]['對應外部法規'+str(j)]) == False):
            if (pd.isnull(in_rule.loc[i]['對應外部法規條文編號'+str(j)]) == False) & (type(in_rule.loc[i]['對應外部法規條文編號'+str(j)]) != int):
                for e in in_rule.loc[i]['對應外部法規條文編號'+str(j)].split('|'):
                    l.append((in_rule.loc[i]['對應外部法規'+str(j)],e))
            elif (pd.isnull(in_rule.loc[i]['對應外部法規條文編號'+str(j)]) == False) & (type(in_rule.loc[i]['對應外部法規條文編號'+str(j)]) == int):
                l.append((in_rule.loc[i]['對應外部法規'+str(j)],in_rule.loc[i]['對應外部法規條文編號'+str(j)]))


# In[34]:


I_INDEX =[x for i in [target[(target.法規名稱 == t[0]) & (target.條文編號 == t[1])].index.tolist() for t in l] for x in i]


# In[35]:


TARGET_INDEX = list(set(W_INDEX + I_INDEX))
target = target.loc[TARGET_INDEX]
target = target.drop_duplicates()


# In[36]:


# 資料範圍
target = target.drop_duplicates()
print("外規法規數量：", len(np.unique(target.法規名稱)))
print("外規條文數量：", target.shape[0])
print('外規列表')
for c, value in enumerate(np.unique(target.法規名稱), 1):
    print(c, value)


# In[37]:


df = pd.concat([data,target], axis = 0).reset_index(drop=True)
df['條文編號'] = df['條文編號'].astype(str)
df['條文內容'] = df['條文內容'].astype(str)
print("內外規總法規數量：", len(np.unique(df.法規名稱)))
print("內外規總條文數量：", df.shape[0])


# # ZH split

# In[38]:


analyzer = zh_split(jieba_zh_path="D:/python/TextMining/Dictionary/dict.txt.big")


# In[39]:


# 加載字典
DICT_PATH = 'D:/python/TextMining/Dictionary/'
analyzer.add_dictionary(DICT_PATH + 'edu_dict.txt')
analyzer.add_dictionary(DICT_PATH + 'law_dict.txt')
analyzer.add_dictionary(DICT_PATH + 'reg_dict.txt')


# In[40]:


# 斷詞
analyzer.split(df['條文內容'])
df['split_list'] = analyzer.split_list


# In[41]:


# 過濾停止詞
analyzer.word_filter(path_word='D:/python/TextMining/Dictionary/stop_words.txt', pos = 'all')


# In[42]:


df['split_list_filter'] = analyzer.split_list
# 過濾內容過短之條文
print("過濾前資料筆數:", df.shape)
df = df[df.split_list.apply(lambda x: len(x) > 10)]
# df = df.reset_index(drop = True)
print("過濾後資料筆數:", df.shape)


# # Vectorize

# In[43]:


tm = text_mining(df.split_list_filter)


# In[44]:


tm.get_dictionary()


# In[45]:


tm.CounterVector()
tm.TfidfVector()


# # Cluster

# In[46]:


CLUSTER = tm.LDA(tf_vector = tm.TF_Vector,
                          K_range = [10], #range(6,14,2), # 15,20
                          PASS_range = [5,10,20], #range(5,20,5), [5]
                          ITER_range = range(100,401,100), # [200]  
                          alpha = "auto", eta = "auto", seed = 7571)
# 最佳參數-主題數:10 PASS:5 Iter:200 Eval:0.44168113732176606


# ## Cluster Result Output

# In[47]:


topic_doc, topic_word = tm.LDA_output(CLUSTER, n_keyword = 10)


# In[48]:


topic_doc = pd.concat([df[['法規名稱','條文編號','條文內容','LABEL']],topic_doc[['分群類別','機率']]], axis = 1)
topic_doc.head()


# In[49]:


# 各群內外規數量
RESULT = pd.DataFrame()
for i in range(len(topic_doc.分群類別.unique())):
    IN = len(topic_doc[(topic_doc.分群類別 == i) & (topic_doc.LABEL == '內規')])
    OUT = len(topic_doc[(topic_doc.分群類別 == i) & (topic_doc.LABEL == '外規')])
    RESULT = pd.concat([RESULT,pd.DataFrame([[i,IN,OUT]])], axis = 0)
RESULT = RESULT.reset_index(drop = True).rename(columns = {0:'分群類別',1:'內規條文數量',2:'外規條文數量'})
RESULT


# In[50]:


topic_word = topic_word.merge(RESULT, how = 'left', on = '分群類別')
topic_word.head()


# # Word Cloud

# In[51]:


tm.LDA_TopicWord()


# In[52]:


print('關鍵字個數：',len(tm.LDA_TopicWord_Vector))
display(tm.LDA_TopicWord_Vector[0:10])


# In[53]:


wordcloud_LDA = tm.Word_Cloud(word_vector = tm.LDA_TopicWord_Vector)
wordcloud_LDA.head()


# # Rule Mapping(Doc2Vec)

# In[54]:


# Doc2Vec, LSI
# #內規:前66條 / 外規:67-209條
starttime = datetime.datetime.now()

# N = df.shape[0] - df.shape[0] % 100 #(取到百分位)
N = 200
doc_result, doc_model = tm.doc2vec(vector_size = N, epochs = 50, dbow_words= 1, dm=0, iter=1, window=5)
lsi_result, lsi_model = tm.LSI(tm.TFIDF_Vector, n_dimension = N)

endtime = datetime.datetime.now()
print(endtime - starttime)


# ## - By Cluster

# In[61]:


SIM_RESULT = pd.DataFrame()
ID = topic_word.loc[(topic_word.內規條文數量 != 0) & (topic_word.外規條文數量 != 0),'分群類別'].tolist()
for cluster in ID:
    IN_INDEX = topic_doc[(topic_doc.分群類別 == cluster) & (topic_doc.LABEL == '內規')].index.tolist()
    IN_INDEX = [topic_doc.index.get_loc(i) for i in IN_INDEX]
    EX_INDEX = topic_doc[(topic_doc.分群類別 == cluster) & (topic_doc.LABEL == '外規')].index.tolist()
    EX_INDEX = [topic_doc.index.get_loc(i) for i in EX_INDEX]
    # print('內規數量:',len(IN_INDEX),'外規數量:',len(EX_INDEX))
    
    lsi_sim = tm.sim([lsi_result[i] for i in IN_INDEX],[lsi_result[i] for i in EX_INDEX])
    doc2vec_sim = tm.sim([list(enumerate(row)) for row in [doc_result[i] for i in IN_INDEX]],
                     [list(enumerate(row)) for row in [doc_result[i] for i in EX_INDEX]])
    sim_matrix = (lsi_sim + doc2vec_sim)/2
    
    result = [[cluster,
               topic_doc[(topic_doc.分群類別 == cluster) & (topic_doc.LABEL == '內規')].法規名稱.iloc[r],
               topic_doc[(topic_doc.分群類別 == cluster) & (topic_doc.LABEL == '內規')].條文編號.iloc[r],
               topic_doc[(topic_doc.分群類別 == cluster) & (topic_doc.LABEL == '內規')].條文內容.iloc[r],
               topic_doc[(topic_doc.分群類別 == cluster) & (topic_doc.LABEL == '外規')].法規名稱.iloc[c],
               topic_doc[(topic_doc.分群類別 == cluster) & (topic_doc.LABEL == '外規')].條文編號.iloc[c],
               topic_doc[(topic_doc.分群類別 == cluster) & (topic_doc.LABEL == '外規')].條文內容.iloc[c],
               lsi_sim[r,c], 
               doc2vec_sim[r,c], 
               sim_matrix[r,c]]
              for r, cs in enumerate(np.argsort(-sim_matrix, axis=1)[:,:])for c in cs ]
    result = pd.DataFrame(result)
    result.columns = ["分群類別","內規名稱", "內規編號", "內規內容","外規名稱", "外規編號", "外規內容", "對應比例_LSI", "對應比例_Doc2vec", "對應比例"]
    
    SIM_RESULT = pd.concat([SIM_RESULT,result], axis = 0)
SIM_RESULT_CLUSTER = SIM_RESULT.reset_index(drop = True)


# In[62]:


# 對應排名
SIM_RESULT_CLUSTER.loc[:,'對應排名'] = SIM_RESULT_CLUSTER.groupby(['內規名稱','內規編號','內規內容'])["對應比例"].rank(ascending = False)


# ## Output Result

# In[63]:


# 將所有分群編號 + 1(避免有0)
topic_doc.loc[:,'分群類別'] = topic_doc.loc[:,'分群類別'] + 1
topic_word.loc[:,'分群類別'] = topic_word.loc[:,'分群類別'] + 1
wordcloud_LDA.loc[:,'分群類別'] = wordcloud_LDA.loc[:,'分群類別'] + 1
SIM_RESULT_CLUSTER.loc[:,'分群類別'] = SIM_RESULT_CLUSTER.loc[:,'分群類別'] + 1


# In[64]:


OUTPUT_PATH = 'D:/python/TextMining/ComplianceBot/POC/永豐/Result/銀行/'
OUTPUT_CLUSTER = 10
OUTPUT_DATE = 20180719
# 法規相似性分群結果
topic_doc.to_excel(OUTPUT_PATH+'[永豐]法規相似性分群結果(銀行)('+str(OUTPUT_CLUSTER)+'群)_'+str(OUTPUT_DATE)+'.xlsx', index = False)
# 法規相似性分群關鍵字
topic_word.to_excel(OUTPUT_PATH+'[永豐]法規相似性分群關鍵字(銀行)('+str(OUTPUT_CLUSTER)+'群)_'+str(OUTPUT_DATE)+'.xlsx', index = False)
# 法規相似性分群關鍵字(for Tableau)
wordcloud_LDA.to_excel(OUTPUT_PATH+'[永豐]法規相似性分群關鍵字(for Tableau)(銀行)('+str(OUTPUT_CLUSTER)+'群)_'+str(OUTPUT_DATE)+'.xlsx', index = False)
# 法規相似性比對結果(依分群)
SIM_RESULT_CLUSTER.to_excel(OUTPUT_PATH+'[永豐]法規相似性比對結果(依分群)(銀行)('+str(OUTPUT_CLUSTER)+'群)_'+str(OUTPUT_DATE)+'.xlsx', index = False)

