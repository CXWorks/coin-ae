import bisect
import multiprocessing
import os
import json
import multiprocessing as mp
import shlex
import pickle
import subprocess
import sys
import pandas as pd
from typing import List
from transformers import AutoTokenizer
import random


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")


def merge(intervals: List[List[int]]) -> List[List[int]]:
    # intervals = list(intervals_set)
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(list(interval))
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged



def filter_file(f:str):
    new_unsafe = {}
    ct = 0
    with open(f,'rb') as fp:
        unsafe = pickle.load(fp)
        for k,v in unsafe.items():
            remain = []
            if os.path.exists(k):
                with open(k,'r') as fp2:
                    ls = fp2.readlines()
                    for st, ed in v:
                        if st != ed and 'unsafe' in ls[st-1] and 'fn bits(' not in ls[st-1] and 'fn steal(' not in ls[st-1]:
                            remain.append([st, ed])
            if len(remain) > 0:
                new_unsafe[k] = remain
    with open('std_filtered_unsafe.pkl', 'wb') as fp:
        pickle.dump(new_unsafe, fp)


def check_file(f:str):
    new_unsafe = {}
    with open(f,'rb') as fp:
        unsafe = pickle.load(fp)
        ct = 0
        missing = 0
        for k,v in unsafe.items():
            if os.path.exists(k):

                vv = merge(v)
                ct += len(vv)
                new_unsafe[k] = vv
            else:
                k = k.replace('/mnt/md0/xiang/tmp/', '/mnt/sdc1/xiang/unsafe_fn_new/')
                if os.path.exists(k):

                    vv = merge(v)
                    ct += len(vv)
                    new_unsafe[k] = vv
                else:
                    missing += len(v)
    print(len(new_unsafe))
    print(ct, missing)
    with open('std_new_unsafe.pkl','wb') as fp:
        pickle.dump(new_unsafe,fp)


def parse_log(f:str, unsafe_info:dict):
    with open(f, 'r') as fp:
        ls = fp.readlines()
        file = ''
        count = 0
        inside = False
        lines = []
        for idx in range(len(ls)):
            l = ls[idx]
            if inside:
                count -= 1
                l = l.split('\t')[1]
                lw = l.split(' ')
                line = int(lw[0]) + 1
                label = int(lw[3])
                lines.append(line)
                if file not in unsafe_info:
                    unsafe_info[file] = set()
                if count == 0:
                    inside = False
                    unsafe_info[file].add((lines[0], line))
                    lines = []

            if l.startswith('find_unused_unsafe_at'):
                lw = l.split(' ')
                file = lw[1]
                count = int(lw[2])
                inside = True
                lines = []


def count(fd:str):
    unsafe_info = {}
    for f in os.listdir(fd):
        if os.path.exists(fd+'/'+f+'/compile_out.txt'):
            parse_log(fd+'/'+f+'/compile_out.txt', unsafe_info)
    with open('unsafe.pkl', 'wb') as fp:
        pickle.dump(unsafe_info, fp)
    print(len(unsafe_info))


def combine(f1, f2):
    with open(f1, 'rb') as fp:
        unsafe = pickle.load(fp)
        with open(f2, 'rb') as fp2:
            std = pickle.load(fp2)
            for k,v in std.items():
                if k not in unsafe:
                    unsafe[k] = v
                else:
                    unsafe[k].extend(v)
                    unsafe[k] = merge(unsafe[k])
    with open(f1, 'wb') as fp:
        pickle.dump(unsafe, fp)


def recur_scan(fd:str, ans: set):
    for f in os.listdir(fd):
        if os.path.isfile(fd+'/'+f) and f.endswith('.rs') and '/build/' not in os.path.abspath(os.path.join(fd, f)):
            ans.add(os.path.abspath(os.path.join(fd, f)))
        elif os.path.isdir(fd+'/'+f) and not os.path.islink(fd+'/'+f):
            recur_scan(fd+'/'+f, ans)


def run_quick(f:str):
    ret = subprocess.run(shlex.split('/mnt/md0/xiang/unsafe_data/quick-safe/target/release/quick '+f), capture_output=True)
    ans = []
    if ret.returncode == 0:
        for line in ret.stdout.decode('utf-8').splitlines():
            ls = line.split(' ')
            st = int(ls[0].split(':')[0])
            ed = int(ls[1].split(':')[0])
            ans.append((f, st, ed))
    return ans


def generate_prompt(args):
    f= args[0]
    st=args[1]
    ed=args[2]
    ct = 0
    with open(f, 'r') as fp:
        text = fp.readlines()

        for i in range(st-1, ed):
            text[i] = '>\t'+text[i]
        tokenized = tokenizer(''.join(text))['input_ids']
        if len(tokenized) > 128000:
            ct = 1
            return '', '', 0,0
    return ''.join(text), f, st, ed


def generate_prompt_unsafe(args):
    f= args[0]
    st=args[1]
    ed=args[2]
    ct = 0
    with open(f, 'r') as fp:
        text = fp.readlines()
        text[st-1] = text[st-1].replace('unsafe ','')
        for i in range(st-1, ed):
            text[i] = '>\t'+text[i]
        tokenized = tokenizer(''.join(text))['input_ids']
        if len(tokenized) > 128000:
            ct = 1
            return '', '', 0,0
    return ''.join(text), f, st, ed



def read_check(args):
    f=args[0]
    vals = args[1]
    with open(f, 'r') as fp:
        text = fp.readlines()
        tokenized = tokenizer(''.join(text))['input_ids']
        if len(tokenized) > 128000:
            return f, '', vals
        else:
            return f, ''.join(text), vals


def identify_crate(f:str):
    total = set()
    df = pd.read_csv('../unsafe_meta.csv')
    # recur_scan('/mnt/md0/xiang/rust183/', total)
    with open(f, 'rb') as fp:
        unsafe = pickle.load(fp)
        for k,v in unsafe.items():
            ks = k.split('/')
            fd = ''
            if '/index.crates.io' in k:
                for i in range(len(ks)):
                    if 'index.crates.io' in ks[i]:
                        fd = k[:k.find(ks[i+1]) + len(ks[i+1]) +1]
            if 'unsafe_fn_new' in k :
                for i in range(len(ks)):
                    if 'unsafe_fn_new' in ks[i]:
                        fd = k[:k.find(ks[i+1]) + len(ks[i+1]) +1]
            if '/mnt/md0/xiang/rust183' in k:
                pass
            if len(fd) > 0:
                # print(k, fd)
                # recur_scan(fd, total)
                total.add(fd)
        last = ''
        last_fd = None
        new_fds = ['/mnt/md0/xiang/rust183/']
        compete = []
        for fd in sorted(list(total)):
            if last in fd:
                last = fd[:fd.rfind('-')]
            else:
                new_fds.append(last_fd)

                rows = df.loc[df['name'] == last[last.rfind('/')+1:]]
                if len(rows) > 0:
                    for idx, row in rows.iterrows():
                        all = set()
                        recur_scan(last_fd, all)
                        # print(last_fd, row['name'], row['downloads'], len(all))
                        compete.append((last_fd, str(row['name']), int(row['downloads']), len(all)))
                        break
                last = fd[:fd.rfind('-')]
            last_fd = fd
        rows = df.loc[df['name'] == last[last.rfind('/') + 1:]]
        if len(rows) > 0:
            for idx, row in rows.iterrows():
                all = set()
                recur_scan(last_fd, all)
                # print(last_fd, row['name'], row['downloads'], len(all))
                compete.append((last_fd, str(row['name']), int(row['downloads']), len(all)))
                break
        compete.sort(key=lambda x: x[2], reverse=True)
        tt = 0
        total = set()
        recur_scan('/mnt/md0/xiang/rust183/', total)
        for idx, ctnt in enumerate(compete):
            tt += ctnt[3]
            if ctnt[3] <= 200:
                recur_scan(ctnt[0], total)
        return total, unsafe





if __name__ == '__main__':
    total, unsafe = identify_crate(sys.argv[1])
    unsafe_ct = 0
    safe_ct = 0
    wl = []
    data = {'safe':[], 'unsafe': []}
    for k,v in unsafe.items():
        if k in total:
            unsafe_ct += len(v)
            wl.append(k)
            for vv in v:
                data['unsafe'].append((k,vv[0],vv[1]))
    print(len(wl))
    wl = list(total)
    with mp.Pool(108) as pool:
        anss = pool.map(run_quick, wl)
        for ans in anss:
            data['safe'].extend(ans)
    print(safe_ct, unsafe_ct)
    print(len(data['safe']), len(data['unsafe']))
    # total = set()
    # final_data = {}
    #
    with open('dataset.pkl', 'wb') as fp:
        pickle.dump(data, fp)
    #     train = {}
    #     test = {}
    #     valid = {}
    #     for k, argss in data.items():
    #         files = {}
    #         random.shuffle(argss)
    #         total = len(argss)
    #         ct1 = int(total * 0.6)
    #         ct2 = int(total * 0.8)
    #         train[k] = argss[:ct1]
    #         test[k] = argss[ct1:ct2]
    #         valid[k] = argss[ct2:]
    #
    #     with open('coin_train.pkl', 'wb') as fp2:
    #         pickle.dump(train, fp2)
    #     with open('coin_test.pkl', 'wb') as fp2:
    #         pickle.dump(test, fp2)
    #     with open('coin_valid.pkl', 'wb') as fp2:
    #         pickle.dump(valid, fp2)
    # print('done')
    # combine(sys.argv[1], sys.argv[2])
    # unsafe = {}
    # parse_log(sys.argv[1], unsafe)
    # with open('std_unsafe.pkl', 'wb') as fp:
    #     pickle.dump(unsafe, fp)
    # check_file('std_unsafe.pkl')
    # filter_file('std_new_unsafe.pkl')

