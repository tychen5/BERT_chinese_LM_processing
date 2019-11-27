import pandas as pd
import numpy as np
import re
import nltk
from collections import Counter
import jieba
import time
from gensim import corpora
from tqdm import tqdm
tqdm.pandas()

class zh_split(object):

    def __init__(self, jieba_zh_path = "dict.txt.big"):
        jieba.set_dictionary(jieba_zh_path)
    
    def add_dictionary(self, path, remix = False):
        dict_df = pd.read_table(path, header=None, encoding = 'utf8')
        dict_df['word'] = dict_df.iloc[: ,0].apply(lambda x:x.split()[0])
        dict_df['weight'] = dict_df.iloc[: ,0].apply(lambda x:x.split()[1])
        if remix:
            dict_df = dict_df.groupby(['word'], as_index=False).weight.sum().values
        else:
            dict_df = dict_df[['word', 'weight']].values
        
        ## 複寫字典
        with open(path, "w", encoding = 'utf8') as f:
            for row in dict_df:
                jieba.add_word(row[0], row[1])
                text = row[0] + " " + row[1] + "\n"
                f.write(text)
        
        print("add user dictionary from {}".format(path))
        
    def split(self, text):
        if type(text) == pd.Series or type(text) == np.array:
            text = text.tolist()
        elif type(text) != list:
            print("text should be Series, array or list")
            return
        
        split_list = []
        print('斷詞......')
        for row in tqdm(text):
            split_list.append(jieba.lcut(row))
        self.split_list = split_list
        
        
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
        
    def find_keyword(self, n = 2, path = None):
        ## TF
        word_count = {self.dic[k]:i for k, i in self.dic.dfs.items()}
        ## bigram
        ngram_list = []
        
        print('Ngram......')
        for raw in  tqdm(self.split_list):
            ngram_list.extend(list(nltk.ngrams(raw, n)))

        ngram_freq = nltk.FreqDist(ngram_list)
        ngram_result = ngram_freq.most_common(1000)
        
        ## weight bigram
        add_word = sorted([(w[0] + w[1], c / (word_count[w[0]]+ word_count[w[1]]), w) for w, c in ngram_result]
                  , reverse=True
                  , key=lambda x:x[1])
        
        self.add_word = add_word
        
        ## output潛在詞彙列表
        if path != None:
            with open(path, "w", encoding = 'utf8') as f:
                for w, p, _ in add_word:
                    text = w + " %s\n"%round(p*1000)
                    f.write(text)
                    
    def word_filter(self, pos = 'all', path_char = None, path_word = None, w_len = 1):
        if pos == None:
            pos = ['n', 'nt', 'nz', 'v', 'a']
        
        if path_char == None:
            stop_re_filter = re.compile("@")
        else:
            stop_char = pd.read_table(path_char, header = None)
            stop_char = "[{}]".format("".join(stop_char.iloc[:, 0].tolist()))
            stop_re = r'.*{}.*'.format(stop_char)
            stop_re_filter = re.compile(stop_re)
        
        if path_word == None:
            stop_word = [""]
        else:            
            stop_word = pd.read_table(path_word, header = None)
            #stop_word = "|".join(stop_word.iloc[:, 0].tolist())
            stop_word = stop_word.iloc[:, 0].tolist()
            


        self.split_list = pd.Series(self.split_list).progress_apply(lambda x:self.word_filter_(x, pos, stop_re_filter, stop_word, w_len)).tolist()
    
    def word_filter_(self, raw, pos, stop_re_filter, stop_word, w_len):
        if pos == "all":
            result = [w for w in raw if re.findall(r"\W+", w) == [] 
                      and re.findall(r"\d+", w) == [] 
                      and not stop_re_filter.match(w)
                      and not w in stop_word
                      and len(w) > w_len
                     ]
        else:
            result = [w for w in raw if re.findall(r"\W+", w) == [] 
                      and re.findall(r"\d+", w) == [] 
                      and not stop_re_filter.match(w)
                      and not w in stop_word
                      and len(w) > w_len
                      and p in pos
                     ]
            
        return result