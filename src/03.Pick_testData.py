import os
import random
import re

from tqdm import tqdm
from googletrans import Translator

root_dir = "../Data/THUCNews_trad/"
cat_dir = next(os.walk(root_dir))[1]
# print(cat_dir)
'''
Goal: random pick 10% file for testing and seperate too long by |||
'''
translator = Translator()


def write_file(output, sent, text_id):
    if not os.path.isdir(output):
        os.makedirs(output, exist_ok=True)
    out_path = output + '/'+text_id
    with open(out_path, 'w', encoding='utf8') as f:
        for item in sent:
            f.write(item[0] + '\n')


max_length = 512  # BERT need to be lower than 512

for cat in tqdm(cat_dir):
    cat_en = translator.translate(cat).text
    in_dir = root_dir + cat + '/'
    files = next(os.walk(in_dir))[2]
    test_files_num = int(len(files) * 0.1)  # take 10% data to test
    test_files = random.choices(files, k=test_files_num)
    sentences = []
    for text_id in test_files:
        in_file_path = in_dir + text_id
        r = open(in_file_path, 'r', encoding='utf-8')
        text = r.read()
        text = re.sub(r'\n', "", text)
        text = re.sub(r'\u3000', "", text)
        length = len(text)
        iters = int(length / max_length) + 1
        for i in range(iters):
            if i % 2 == 1:  # if it's even number (end)
                sentences.append([sent + ' ||| ' + text[i * max_length:(i + 1) * max_length]])
            elif i == iters - 1:  # if it's odd number end
                sentences.append([text[i * max_length:(i + 1) * max_length]])
            else:  # if it's odd number
                sent = text[i * max_length:(i + 1) * max_length]
        out_dir = '../Data/Test_rev/' + cat_en + '/'   # one category many files
        write_file(out_dir, sentences, text_id)
        sentences = []
        # sentences.append([''])  # many test documents seperate by \n

    # out_dir = '../Data/Test/' + cat_en + '/'  # one category one file
    # write_file(out_dir, sentences)
