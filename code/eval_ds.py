import multiprocessing
import os
import subprocess
import pickle
import sys
import random


def one_process(fd:str):
    unsafe = []
    real_unsafe = []
    safe = []
    if os.path.exists(fd+'/res.pickle') and os.path.exists(fd+'/ft_dylint_out.txt'):
        data = parse_log(fd, 'ft_dylint_out.txt')
        with open(fd+'/res.pickle', 'rb') as f:
            res = pickle.load(f)
            for k,v in res.items():
                if os.path.exists(k) and k in data:
                    for fn_name, st, _, ed, _ in v:
                        # try to match back
                        datas = data[k]
                        to_del = []
                        for kd, vd in datas.items():
                            if (st <= kd[0] and ed>=kd[1]) or (kd[0] <= st and kd[1]>=ed):
                                #  found
                                ctxt = vd[0]
                                comments= get_comments(k, st)
                                fn_body = vd[1].replace(' unsafe ', ' ', 1)
                                total_str = ctxt + comments + fn_body
                                unsafe.append(total_str.strip())
                                to_del.append(kd)
                        for kd in to_del:
                            del datas[kd]
            flat = []
            for fname, datas in data.items():
                for kd, vd in datas.items():
                    flat.append((fname, kd, vd))
            ct = 3*len(unsafe)
            safe_set = [x for x in filter(lambda x: 'unsafe fn' not in x[2][1] and os.path.exists(x[0]), flat)]
            unsafe_set = [x for x in filter(lambda x: 'unsafe fn' in x[2][1] and os.path.exists(x[0]), flat)]
            random.shuffle(safe_set)
            selected = safe_set[:min(ct, len(safe_set))]
            for fname, kd, vd in selected:
                ctxt = vd[0]
                comments = get_comments(fname, kd[0])
                fn_body = vd[1]
                total_str = ctxt + comments + fn_body
                safe.append(total_str.strip())
            for fname, kd, vd in unsafe_set:
                ctxt = vd[0]
                comments = get_comments(fname, kd[0])
                fn_body = vd[1].replace(' unsafe ', ' ', 1)
                total_str = ctxt + comments + fn_body
                real_unsafe.append(total_str.strip())
        with open(fd+'/ft.pickle', 'wb') as f:
            pickle.dump({'real_unsafe': real_unsafe, 'unsafe':unsafe, 'safe': safe}, f)
    return len(safe), len(real_unsafe), len(unsafe)


def one_process_eval(fd:str):
    unsafe = []
    real_unsafe = []
    safe = []
    if os.path.exists(fd+'/ft_dylint_out.txt'):
        data = parse_log(fd, 'ft_dylint_out.txt')
        flat = []
        for fname, datas in data.items():
            for kd, vd in datas.items():
                flat.append((fname, kd, vd))
        ct = 3*len(unsafe)
        safe_set = [x for x in filter(lambda x: 'unsafe fn' not in x[2][1] and os.path.exists(x[0]), flat)]
        selected = safe_set
        for fname, kd, vd in selected:
            ctxt = vd[0]
            comments = get_comments(fname, kd[0])
            fn_body = vd[1]
            total_str = ctxt + comments + fn_body
            safe.append((total_str.strip(), fname, kd))
        with open(fd+'/safe_eval2.pickle', 'wb') as f:
            pickle.dump(safe, f)
    return len(safe)


def get_comments(f:str, st:int):
    comments = []
    with open(f, 'r') as f:
        lines = f.readlines()
        st-=2
        while st > -1:
            line = lines[st]
            if line.strip().startswith('//') or line.strip().startswith('#'):
                comments.append(line)
            else:
                break
            st -=1
    return ''.join(reversed(comments))


def parse_log(fd:str, f:str):
    data = {}
    with open(fd+'/'+f, 'r') as fp:
        lines = fp.readlines()
        header= []
        tail = []
        src = None
        st = 0
        ed = 0
        flag = 0
        for line in lines:
            if flag == 0:
                if line.startswith('-------------'):
                    flag = 1
            elif flag == 1:
                if line.startswith('@@@@@@@'):
                    flag = 2
                else:
                    header.append(line)
            elif flag == 2:
                if line.startswith('$$$$$$$$$$$$$$'):
                    flag = 0
                    if '{' in ''.join(tail):
                        data[src][(st, ed)] = (''.join(header).strip(), ''.join(tail))
                    header = []
                    tail = []
                    src = None
                else:
                    if src is None:
                        srcs = line.split(':')
                        try:
                            src = srcs[0]
                            if not src.startswith('/'):
                                src = os.path.abspath(fd+'/'+srcs[0])
                            st = int(srcs[1].strip())
                            ed = int(srcs[3].strip())
                        except Exception as e:
                            print('err', fd, f, line)
                        if src not in data:
                            data[src] = {}
                    else:
                        tail.append(line)
    return data


def split_data(wl):
    unsafe = []
    real_unsafe = []
    safe = []
    for f in wl:
        if os.path.exists(f+'/ft.pickle'):
            with open(f+'/ft.pickle', 'rb') as fp:
                data = pickle.load(fp)
                unsafe.extend(data['unsafe'])
                real_unsafe.extend(data['real_unsafe'])
                safe.extend(data['safe'])
    random.shuffle(unsafe)
    random.shuffle(real_unsafe)
    random.shuffle(safe)
    train = []
    valid = []
    test = []
    l1= len(safe)
    l2 = len(real_unsafe)
    l3 = len(unsafe)
    train.extend([(x, 0) for x in safe[:int(l1*0.7)]])
    train.extend([(x, 1) for x in real_unsafe[:int(l2 * 0.7)]])
    train.extend([(x, 1) for x in unsafe[:int(l3 * 0.7)]])
    valid.extend([(x, 0) for x in safe[int(l1 * 0.7): int(l1*0.8)]])
    valid.extend([(x, 1) for x in real_unsafe[int(l2 * 0.7):int(l2*0.8)]])
    valid.extend([(x, 1) for x in unsafe[int(l3 * 0.7): int(l3*0.8)]])
    test.extend([(x, 0) for x in safe[int(l1 * 0.8):]])
    test.extend([(x, 1) for x in real_unsafe[int(l2 * 0.8):]])
    test.extend([(x, 1) for x in unsafe[int(l3 * 0.8):]])
    with open('ft_train.pickle', 'wb') as fp:
        pickle.dump(train, fp)
    with open('ft_valid.pickle', 'wb') as fp:
        pickle.dump(valid, fp)
    with open('ft_test.pickle', 'wb') as fp:
        pickle.dump(test, fp)





if __name__ == '__main__':
    wl = []
    for fd in os.listdir('/mnt/sdc1/xiang/unsafe_fn/'):
        path = os.path.join('/mnt/sdc1/xiang/unsafe_fn/',fd)
        if os.path.isdir(path):
            wl.append(path)
    print(len(wl))
    # split_data(wl)
    # with multiprocessing.Pool(108) as pool:
    #     anss = pool.map(one_process, wl)
    #     total = {'safe': 0, 'unsafe': 0, 'real_unsafe':0}
    #     for safe, real_unsafe, unsafe in anss:
    #         total['safe'] += safe
    #         total['unsafe'] += unsafe
    #         total['real_unsafe'] += real_unsafe
    total = 0
    with multiprocessing.Pool(108) as pool:
        anss = pool.map(one_process_eval, wl)
        total = sum(anss)
    print('done', total)

