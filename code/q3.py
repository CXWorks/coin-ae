import re
import os
import subprocess
import sys
import pickle
from collections import defaultdict


def get_tars():
    ans = []
    for f in os.listdir('.'):
        if f.endswith('.pkl') and re.fullmatch(r'\d+test_eval.pkl', f):
            num = int(f.replace('test_eval.pkl', ''))
            if not os.path.exists(f'{num}test.txt') and num > 10:
                ans.append(num)
    ans.sort()
    return ans


def check():
    fs = defaultdict(list)
    for split in ['train', 'valid']:
        with open(f'/mnt/sdb/xiang/coin3/coin_{split}.pkl', 'rb') as fp:
            data = pickle.load(fp)
            for k, vv in data.items():
                for f, text, ls, window in vv:
                    if '/rust183/library/' in f:
                        fs[f].append(ls)
    for k, v in fs.items():
        print(k, v)



if __name__ == '__main__':
    check()
    print('done')