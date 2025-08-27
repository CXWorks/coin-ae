import pickle
import os
import multiprocessing
import sys
from transformers import AutoTokenizer
import random


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")


def one_task(args):
    f=args[0]
    text=args[1]
    locs = args[2]
    refined_locs = []
    tokenized = tokenizer(''.join(text))['input_ids']
    if len(tokenized) > 8000:
        lines = text.splitlines(keepends=True)
        tokenized_lines = [tokenizer(line)['input_ids'] for line in lines]
        for st, ed in locs:
            mid = (st + ed) // 2  # Midpoint between start and end lines

            # Determine how many tokens are needed on either side of the midpoint
            half_window_tokens = 8000 // 2

            # Initialize counters
            total_tokens = sum([ len(x) for x in tokenized_lines[st:ed]])
            start_line = st
            end_line = ed

            # Extend the window outwards from the midpoint
            while total_tokens < 8000 and (start_line > 0 or end_line < len(tokenized_lines)):
                if total_tokens < 8000 and start_line > 0:
                    start_line -= 1
                    total_tokens += len(tokenized_lines[start_line])
                if total_tokens < 8000 and end_line < len(tokenized_lines):
                    end_line += 1
                    total_tokens += len(tokenized_lines[end_line - 1])

            # Adjust refined locations to fit within the selected lines
            refined_locs.append((start_line, end_line))
        return (f, text, locs, refined_locs)

    else:
        return (f, text, locs, locs)



if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        tasks = pickle.load(f)
        new_tasks = []
        with multiprocessing.Pool(processes=120) as pool:
            anss = pool.map(one_task, tasks)
            new_tasks.extend(anss)
                # new_tasks.append()
        with open("tokenized_coin.pkl", 'wb') as fp:
            pickle.dump(new_tasks, fp)