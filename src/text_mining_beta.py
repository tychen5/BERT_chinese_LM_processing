from __future__ import absolute_import, unicode_literals
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import re
from gensim import corpora, models, similarities 
from jieba import analyse
import sys
from operator import itemgetter
from tqdm._tqdm_notebook import tqdm_notebook as tqdm


class text_mining(object):
    
    def __init__(self, split_list, pos = None):
        if pos:
            self.split_list = [[word[0] for word in row if word[1] in pos] for row in split_list]
        else:
            self.split_list = [[word for word in row] for row in split_list]
    
    ## vectorize  
    def get_dictionary(self, below_filter = None, above_filter = None):
        # 建立字典
        self.dic = corpora.Dictionary(self.split_list)
        print("字詞數量 (未處理)：{}".format(len(self.dic.dfs)))
        
        if below_filter != None:
            self.dic.filter_extremes(no_below=below_filter, no_above = 1.0)
            print("字詞數量 (過濾低頻詞)：{}".format(len(self.dic.dfs)))
        
        if above_filter != None:
            self.dic.filter_extremes(no_above=above_filter, no_below = 0)
            print("字詞數量 (過濾高、低頻詞)：{}".format(len(self.dic.dfs)))
    
    #### Vectorize ####        
    def CounterVector(self):
        # TF Vector
        self.TF_Vector = [self.dic.doc2bow(doc) for doc in [[word for word in row] for row in self.split_list]]


    def TfidfVector(self):
        model = models.TfidfModel(self.TF_Vector)
        self.TFIDF_Vector = list(model[self.TF_Vector])
        self.TFIDF_Model = model

     #not using    
#     def tfidf_jieba(self, topK=False, withWeight=True, allowPOS=[]):
#        if not allowPOS:
#            allowPOS = self.allpos
#        self.TFIDF_Vector = \
#        [[[self.dic.token2id[word[0]], word[1]] for word in \
#         analyse.extract_tags(row, topK=topK, withWeight=withWeight, allowPOS=allowPOS) \
#         if word[0] in self.dic.values()] for row in self.raw_list]

    # not using    
#     def textrank_jieba(self, topK=False, withWeight=True, allowPOS=[]):
#         if not allowPOS:
#             allowPOS = self.allpos
#         row_list = []
#         for row in self.raw_list:
#             word_list = []
#             for word in analyse.textrank(row, topK=topK, withWeight=withWeight, allowPOS=allowPOS):
#                 if word[0] in self.dic.values():
#                     word_list.append([self.dic.token2id[word[0]], word[1]])
#             row_list.append(word_list)
#             print("進度:{}%".format(round(len(row_list) / len(self.raw_list) * 100)), end = "\r")
#         self.TEXTRANK_Vector = row_list
#        self.TEXTRANK_Vector = \
#        [[[self.dic.token2id[word[0]], word[1]] for word in \
#         analyse.textrank(row, topK=topK, withWeight=withWeight, allowPOS=allowPOS) \
#         if word[0] in self.dic.values()] for row in self.raw_list]

    #### Keyword ####
    def get_textrank(self, words, span = 5):
        g = UndirectWeightedGraph()
        cm = defaultdict(int)
        for i, wp in enumerate(words):
            for j in range(i + 1, i + span):
                if j >= len(words):
                    break
                cm[(wp, words[j])] += 1

        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()
        tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        return tags
    
    def textrank(self):
        tqdm.pandas(desc = 'get textrank>>>')
        row_list = pd.Series(self.split_list).progress_apply(lambda x: [[self.dic.token2id[w[0]], w[1]] for w in self.get_textrank(x)])
#        row_list = row_list.progress_apply(lambda x: [[self.dic.token2id[w[0]], w[1]] for w in x])
        row_list = row_list.tolist()
        self.TEXTRANK_Vector = row_list


    def key_phrase(self, addword, result):
        sub_list = []
        for w in tqdm(addword, desc = 'add key phrase>>>'):
            result['filter'] = result.關鍵字.isin(w)
            id_ = result.groupby('index')['filter'].sum()
            id_ = id_[id_==2].index
            sub = pd.DataFrame({'關鍵字':''.join(w), 'Value':1, 'index':id_})
            sub_list.extend(sub.values.tolist())
            result = result[(result['index'].isin(id_) == False) & (result.關鍵字.isin(w) == False)]
        result = result.drop(columns = 'filter')
        sub_list = pd.DataFrame(sub_list)
        sub_list.columns = ['關鍵字', 'Value', 'index']
        result = pd.concat([result, sub_list])
        return result
    
    def Word_Cloud(self, word_vector, n_key, data = None, dictionary = None):
        if dictionary is not None: # TFIDF_Vector/ TEXTRANK
            result = pd.Series(word_vector).apply(lambda x:[[dictionary[w[0]], w[1]] for w in x[:n_key]])
            
            result_dt = []
            id_ = 0
            for row in tqdm(result, desc = 'transform to wordcloud>>>'):
                sub = pd.DataFrame(row)
                sub['index'] = id_
                result_dt.extend(sub.values.tolist())
                id_ += 1
            result_dt = pd.DataFrame(result_dt)
            result_dt.columns = ['關鍵字','Value', 'index']
            result = result_dt
            if data is not None:
                result = pd.concat([data,result], axis = 1).reset_index(drop = True)
        else: # LDA
            result = pd.DataFrame(word_vector, columns=['分群類別','關鍵字','Value'])
        return result
    
    def LDA_TopicWord(self, n_keyword = 10):
        num_word = n_keyword
        topic_word_value = []
        for t in range(self.LDA_param[0]):
            topic_word_value.extend([(t, ) + x for x in self.BEST_MODEL.show_topic(t, topn = num_word)])
        self.LDA_TopicWord_Vector = topic_word_value
        #topic_word_value = pd.DataFrame(topic_word_value, columns=['分群類別','關鍵字','Value'])


    #### Dimension Reduce ####
    def LSI(self, corpus, n_dimension = 200, iters = 50, decay = 1.0):
        model = models.LsiModel(corpus
                                 , id2word=self.dic
                                 , num_topics=n_dimension
                                 , power_iters = iters 
                                 , decay = decay)
        result = model[corpus]
        return (result, model)

    def doc2vec_transform(self, split_list, tag):
        alldocs=[]
        for i in range(len(split_list)):
            alldocs.append(models.doc2vec.TaggedDocument(split_list[i]
                                                         ,tag[i]
                                                        )
                          )
        self.alldocs = alldocs
    
    def doc2vec(self, split_list = None, tag = None, vector_size=200, epochs = 50, dbow_words= 1, dm=0, iter=1, window=5 ,alpha=0.025, min_alpha=0.025, seed = 7571):
        if split_list is None:
            split_list = self.split_list
        if tag is None:
            tag = [[str(i)] for i in range(len(split_list))]
        self.doc2vec_transform(split_list, tag)
        model = models.Doc2Vec(self.alldocs, vector_size=vector_size
                               , dbow_words = dbow_words, dm = dm, iter = iter, window=window 
                               , alpha = alpha, min_alpha = min_alpha, seed = seed)
        model.random.seed(seed)
        for epoch in range(epochs):
            model.train(self.alldocs, total_examples=model.corpus_count, epochs = 1)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
            
        result = [model.docvecs[row] for row in range(len(self.alldocs))]
        return (result, model)
    
    #### topic modeling ####
    def LDA(self, tf_vector, K_range, PASS_range, ITER_range, alpha = "auto", eta = "auto", seed = 7571):
        self.LDA_EVAL_LIST = []
        self.LDA_MODEL_LIST = []
        for K in  K_range:
            for PASSES in PASS_range:
                for ITER in ITER_range:
                    print("建模參數測試-主題數:{} PASS:{} Iter:{}".format(K, PASSES, ITER))
                    np.random.seed(seed)
                    MODEL = models.LdaModel(corpus=tf_vector
                                            , id2word=self.dic
                                            , alpha=alpha
                                            , eta=eta
                                            , num_topics=K
                                            , passes = PASSES
                                            , iterations=ITER)
                    self.LDA_MODEL_LIST.append(MODEL)
                    EVAL = models.CoherenceModel(model=MODEL
                                                 , texts=self.split_list
                                                 , dictionary=self.dic
                                                 , coherence='c_v')
                    self.LDA_EVAL_LIST.append((K, PASSES, ITER, EVAL.get_coherence()))
                    
                    # 模型比較
                    self.LDA_index, self.LDA_param = max(enumerate(self.LDA_EVAL_LIST), key=lambda x:x[1][3])
                    self.BEST_MODEL = self.LDA_MODEL_LIST[self.LDA_index]
                    print("最佳參數-主題數:{} PASS:{} Iter:{} Eval:{}".format(self.LDA_param[0]
                                                                            , self.LDA_param[1]
                                                                            , self.LDA_param[2]
                                                                            , self.LDA_param[3]))
                    
    def LDA_output(self, content, path = None, n_keyword = 10):
        # 輸出結果
        doc2topic = list(self.BEST_MODEL.get_document_topics(self.TF_Vector))
        topic_doc = pd.DataFrame(list(map(lambda x:sorted(x, key=lambda y:y[1])[-1], doc2topic)))
        topic_doc.columns = ['分群類別', '機率']
        topic_doc['文本內容'] = self.split_list
        
        
        num_word = n_keyword
        num_topics = self.LDA_param[0]
        topic_word = pd.DataFrame([[topic] + re.findall(r'"(.+?)"', word) for topic, word in self.BEST_MODEL.print_topics(num_topics=num_topics, num_words=num_word)])
        topic_word = pd.merge(topic_word, 
                              pd.DataFrame(list(Counter(topic_doc.分群類別).items())), 
                              on=0, 
                              how='inner')
        
        col = ['分群類別']
        for i in range(1, num_word+1): 
            col.append("關鍵字%s"%i)
        col.append('文本數量')
        topic_word.columns = col
              
        
        # 結果輸出為檔案
        #topic_doc.to_excel("{}/topic_doc.xlsx".format(path), index=False)
        #topic_word.to_excel("{}/topic_word.xlsx".format(path), index=False)
        
        return (topic_doc, topic_word)

    #### similarity ####
    def sim(self, target, index):
        sim_index = similarities.MatrixSimilarity(index)
        result = sim_index[target]
        return(result)
    
    


#### for textrank ####
class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for x in range(10):  # 10 iters
            for n in sorted_keys:
                s = 0
                for e in self.graph[n]:
                    s += e[2] / outSum[e[1]] * ws[e[1]]
                ws[n] = (1 - self.d) + self.d * s

        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])

        for w in ws.values():
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws
    
####
