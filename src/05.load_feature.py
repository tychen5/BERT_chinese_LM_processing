import json

in_file = '../Data/Test/ent/output.jsonl'
jf_li = []
i = 0
for line in open(in_file, 'r', encoding='utf8'):
    jf_li.append(json.loads(line))
    i += 1
    if i > 4:
        break
print(jf_li[1])
print(jf_li[3])
# print(jf['data']['hi_info']['login_ip'])
