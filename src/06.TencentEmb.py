import gensim
from gensim.models.keyedvectors import KeyedVectors
import ckiptagger
import tensorflow

# file = '../Data/Tencent_AILab_ChineseEmbedding.txt'
file = '../Data/Tencent_AILab_ChineseEmbedding.bin'
# wv = KeyedVectors.load_word2vec_format(file, binary=False)
wv = KeyedVectors.load(file) #about 2 min
try:
    kk = wv.get_vector("whykgfliqehflaihf;ajf;oiawjfe;oihf;oihd")
except KeyError:
    pass
print(kk)
# wv.save('../Data/Tencent_AILab_ChineseEmbedding.bin')



