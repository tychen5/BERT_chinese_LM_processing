import os
from tqdm import tqdm
# voc_file =
root_dir = "../Data/Test_rev/"
cat_dirs = next(os.walk(root_dir))[1]

for cat in tqdm(cat_dirs):
    in_dir = next(os.walk(root_dir+cat))[2]
    for file in in_dir:
        in_file = root_dir + cat +'/'+file
        out_file = root_dir + cat +'/'+file.split(".")[0] + ".jsonl"
        os.system("python extract_features.py --input_file="+in_file+" --output_file="+out_file+" --vocab_file=../Model/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=../Model/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=../Model/model.ckpt-7000 --layers=-1,-2,-3,-4 --max_seq_length=512 --batch_size=64")
