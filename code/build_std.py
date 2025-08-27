import os
import sys
import subprocess
from copyreg import pickle

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')


def recur_scan(folder:str, ans: list):
    if not os.path.exists(folder):
        return
    for f in os.listdir(folder):
        path = folder +'/'+f
        if os.path.isfile(path) and path.endswith('.rs'):
            ans.append(path)
        if os.path.isdir(path) and not os.path.islink(path):
            recur_scan(path, ans)


def parse_f(f:str):
    ans = set()
    with open(f, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            ans.add(line)
    return ans


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


def has_overlap(a,b,c,d):
    if b<c or a > d:
        return False
    return True


def find_window(text, st, ed):
    ls = text.splitlines(keepends=True)
    ws = st
    we = ed
    le = len(tokenizer(''.join(ls[st-1:ed]))['input_ids'])
    while le < 8100:
        if ws>0:
            ws-=1
        if we < len(ls):
            we+=1
        nle = len(tokenizer(''.join(ls[ws-1:we]))['input_ids'])
        if nle >= 8100:
            return ws+1, we-1
        else:
            if le == nle:
                return ws, we
            le = nle


if __name__ == '__main__':
    fd = sys.argv[1]
    ans = []
    ss = parse_f(sys.argv[2])
    unsafe_info = {}
    parse_log(sys.argv[3], unsafe_info)
    # print(unsafe_info)
    recur_scan(fd, ans)
    ct = 0
    safe = 0
    unsafe = 0
    lo_unsafe = 0
    data = {'safe': [], 'unsafe': []}
    for an in ans:
        if an not in ss and '/tests/' not in an and '/benches/' not in an and '/core_arch/' not in an:
            ret = subprocess.run('/mnt/md0/xiang/func_collector/target/debug/func_collector {}'.format(an), shell=True, capture_output=True)
            if ret.returncode != 0:
                continue
            text = open(an).read()
            length = len(tokenizer(text)['input_ids'])
            safe_lss = []
            unsafe_lss = []
            ls = ret.stdout.decode('utf-8').splitlines(keepends=False)
            for l in ls:
                ll = l.split(':')
                fn = ll[0]
                st = int(ll[2])
                ed = int(ll[3])
                if 'test_' in fn:
                    continue
                if ':safe:' in l:
                    print('safe', an, l.strip())
                    safe+=1
                    safe_lss.append((st, ed))
                elif ':unsafe:' in l:
                    tt = False
                    if an in unsafe_info:
                        tt=True
                        for c,d in unsafe_info[an]:
                            if has_overlap(st,ed,c,d):
                                tt = False
                    if tt:
                        lo_unsafe += 1
                        print('unsafe',an, l.strip())
                        unsafe_lss.append((st, ed))

                    unsafe+=1
            # done
            if len(tokenizer(text)['input_ids']) < 8130:
                data['safe'].append((an, text, safe_lss, safe_lss))
                if len(unsafe_lss) > 0:
                    data['unsafe'].append((an, text, unsafe_lss, unsafe_lss))
            else:
                windows = []
                for a,b in safe_lss:
                    windows.append(find_window(text, a, b))
                data['safe'].append((an, text, safe_lss, windows))
                windows = []
                for a,b in unsafe_lss:
                    windows.append(find_window(text, a, b))
                if len(unsafe_lss) > 0:
                    data['unsafe'].append((an, text, unsafe_lss, windows))
            #print(an, len(ls))
            ct += len(ls)
    import pickle
    with open('std_bench.pkl', 'wb') as fp:
        pickle.dump(data, fp)
