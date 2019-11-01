import os

from opencc import OpenCC
from tqdm import tqdm


def translate(src, dest):
    """
    goal: convert simplified chinese to traditional
    input: source file path(src), target file path(dest)
    output: write converted file to target file path
    """
    cc = OpenCC('s2t')
    source = open(src, 'r', encoding='utf-8')
    result = open(dest, 'w', encoding='utf-8')
    count = 0
    while True:
        line = source.readline()
        line = cc.convert(line)
        if not line:  # readline會一直讀下去，這邊做的break
            break
        # print(line) ##debug
        count = count + 1
        result.write(line)
        # print('===已處理' + str(count) + '行===') ##debug
    source.close()
    result.close()


root_dir = "../Data/THUCNews/"  # simplifies news dir
dest_dir = "../Data/THUCNews_trad/"  # converted(traditional) news dir
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)
files_simple = next(os.walk(root_dir))[2]
for ori_file in tqdm(files_simple):
    src_file = root_dir + ori_file
    dest_file = dest_dir + ori_file
    translate(src_file, dest_file)

# print(files_simple)
