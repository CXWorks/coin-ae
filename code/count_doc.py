import re
import pickle
import os
import subprocess
import multiprocessing


def generate_prompt(text, st, ed, wst, wed):
    ct = 0
    text = text.splitlines(keepends=True)
    for i in range(st-1, ed):
        text[i] = '>\t'+text[i]
    if wst == st and wed == ed:
        return ''.join(text), st, ed
    else:
        return ''.join(text[wst - 1: wed]), st, ed


def collect(vv):
    total = 0
    doc = 0
    comment = 0
    f=vv[0]
    text = vv[1]
    ls = vv[2]
    window = vv[3]


        # Patterns
    rustdoc_line_re = re.compile(r'^\s*(///|//!)\s?(.*)')
    normal_line_re = re.compile(r'^\s*//(?!/|!).*')  # matches // but not /// or //!

    block_comment_re = re.compile(r'/\*(.*?)\*/', re.DOTALL)


    text = text.splitlines(keepends=True)
    for idx, (st, ed) in enumerate(ls):
        total += 1
        has_doc = False
        has_comment = False
        for line in text[st-1:ed]:
            if rustdoc_line_re.match(line):
                has_doc = True
                break
        if not has_doc:
            for line in text[st - 1:ed]:
                if normal_line_re.match(line):
                    has_comment = True
                    break
                if block_comment_re.match(line):
                    has_comment = True
                    break
        if has_doc:
            doc += 1
        if has_comment:
            comment += 1
    return total, doc, comment





def count():
    with open('/mnt/sdb/xiang/coin3/coin_train.pkl', 'rb') as fp:
        data = pickle.load(fp)
        ttotal = 0
        tdoc = 0
        ''.count()
        tcomment = 0
        for k, vv in data.items():

            with multiprocessing.Pool(64) as pool:
                anss = pool.map(collect, vv)

                for total, doc, comment in anss:
                    ttotal += total
                    tdoc += doc
                    tcomment += comment
                print(ttotal, tdoc, tcomment)
        print(ttotal, tdoc, tcomment)

if __name__ == '__main__':
    count()
    print('done')
