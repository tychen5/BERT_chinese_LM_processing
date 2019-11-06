import os
import re

import numpy as np
import pandas as pd
import tokenization
from tqdm import tqdm

vocab_file = "../Model/chinese_L-12_H-768_A-12/vocab.txt"

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)


def basic_statistics(all_length):
    '''
    input: length list of elements e.g.[1,1,1,3,5,9,4,2,1,3,54,78,5...]
    output1: mean、std、mode、min、q1、median(q2)、q3、max、iqr、outlier、far out
    output2: statistics graph、10%~90% form
    '''
    stat_dict = {}
    stat_dict['mean'] = np.mean(all_length)
    stat_dict['std'] = np.std(all_length)
    stat_dict['mode'] = np.argmax(np.bincount(all_length))
    stat_dict['min'] = np.min(all_length)
    stat_dict['q1'] = np.quantile(all_length, 0.25)
    stat_dict['median'] = np.quantile(all_length, 0.5)
    stat_dict['q3'] = np.quantile(all_length, 0.75)
    stat_dict['max'] = np.max(all_length)
    stat_dict['iqr'] = stat_dict['q3'] - stat_dict['q1']
    stat_dict['outlier'] = stat_dict['q3'] + 1.5 * stat_dict['iqr']
    stat_dict['far_out'] = stat_dict['q3'] + 3 * stat_dict['iqr']
    for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        stat_dict[str(i) + '%'] = np.percentile(all_length, i)
    return pd.DataFrame.from_dict(stat_dict, orient='index', columns=['length'])


max_length = 512  # BERT need to be lower than 512
sentences = []
root_dir = "../Data/THUCNews_trad/"
trad_cat = next(os.walk(root_dir))[1]
for cat in tqdm(trad_cat):
    trad_files = next(os.walk(root_dir+cat+'/'))[2]
    for file in trad_files:
        in_path = root_dir + cat + '/' + file
        r = open(in_path, 'r', encoding='utf-8')
        text = r.read()
        text = re.sub(r'\n', "", text)
        text = re.sub(r'\u3000', "", text)
        length = len(text)
        iters = int(length / max_length) + 1
        for i in range(iters):
            sentences.append([text[i * max_length:(i + 1) * max_length]]) #一句話放到一個list裡面
        sentences.append([''])
print(sentences[:-10])

length_li = []

## for look into statistics and decide max_length
# for file in tqdm(trad_files):
#     r = open(root_dir+file, 'r', encoding='utf-8')
#     text = r.read()
#     # print(text)
#     text = tokenizer.tokenize(text)
#     length_li.append(len(text))
#     # print(text)
#     # break
# print(basic_statistics(length_li))
'''
mean       911.341323
std        630.633878
mode       104.000000
min         26.000000
q1         574.000000
median     824.000000
q3        1093.250000
max      14425.000000
iqr        519.250000
outlier   1872.125000
far_out   2651.000000
10%        294.000000
20%        506.000000
30%        629.000000
40%        731.000000
50%        824.000000
60%        917.000000
70%       1026.000000
80%       1184.000000
90%       1537.000000
100%     14425.000000

'''
# The input is a plain text file, with one sentence per line.
# (It is important that these be actual sentences for the "next sentence prediction" task).
# Documents are delimited by empty lines.

'''
max_length = 512  # BERT need to be lower than 512
sentences = []
for file in tqdm(trad_files):
    r = open(root_dir + file, 'r', encoding='utf-8')
    text = r.read()
    text = re.sub(r'\n', "", text)
    text = re.sub(r'\u3000', "", text)
    # text_tok = tokenizer.tokenize(text)
    length = len(text)
    # if length <=512: # debug
    #     continue
    # print(length)
    iters = int(length / 512) + 1

    for i in range(iters):
        # print(i)
        sentences.append([text[i * max_length:(i + 1) * max_length]]) #一句話放到一個list裡面
    sentences.append([''])
    # print(sentences[0]) #debug
    # print(sentences[1])
    # print(text[:10])
print(sentences[:-10])
'''
# sentences = [['123'],[''],['kkycc']]
'''
sentences = [['我覺得很怪'],[''],['真的超級奇怪'],['好像有人在騙人']]
'''

with open("../Data/pretrained_sent_doc.txt", 'w', encoding='utf8') as f:
    for item in sentences:
        f.write(item[0] + '\n')

# pickle.dump(obj=sentences, file=open("../Data/pretrained3.txt", 'w'))
