import pickle


def generate_prompt(text, st, ed, wst, wed):
    ct = 0
    text = text.splitlines(keepends=True)
    is_test = False
    is_lock = False
    for i in range(st-1, ed):
        text[i] = '>\t'+text[i]
        if '#[test]' in text[i]:
            is_test = True
        if 'unlock(' in text[i]:
            is_lock = True and ed-st>0
    if wst == st and wed == ed:
        return ''.join(text), st, ed, is_test, is_lock
    else:
        return ''.join(text[wst - 1: wed]), st, ed, is_test, is_lock


def process():
    with open(f'/mnt/ssd1/xiang/coin2/coin_eval.pkl', 'rb') as fp:
        data = pickle.load(fp)
        # safe_count = len(data['safe'])
        # unsafe_count = len(data['unsafe'])
        # print(safe_count, unsafe_count)
        for f, text, ls, window in data:
            for idx, (st, ed) in enumerate(ls):
                prompt, _, _, is_test, is_lock = generate_prompt(text, st, ed, window[idx][0], window[idx][1])
                if not is_test and is_lock:
                    print('='*20)
                    print(f)
                    print(prompt)
                    print(f)



if __name__ == '__main__':
    process()
