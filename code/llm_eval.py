import os
import sys
import pickle


def cal_few(model:str):
    ns = [0,1,3,5]
    for n in ns:
        f = f'llm_eval.{model}.{n}.pkl'
        if n == 0:
            f = f'llm_eval.{model}.pkl'
        with open(f,'rb') as f:
            data = pickle.load(f)[model]
            tp = 0  # true positives (predicted unsafe, actually unsafe)
            fp = 0  # false positives (predicted unsafe, actually safe)
            fn = 0  # false negatives (predicted safe, actually unsafe)
            total = len(data)
            for gt, preds in data:
                pred = preds[0]
                # Update TP, FP, FN
                if pred == 1:
                    if gt == 1:
                        tp += 1
                    else:
                        fp += 1
                else:  # predicted_label == 0
                    if gt == 1:
                        fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            print(f"Model {model} results on {total} N={n} examples:")
            print(f"  True Positives (unsafe & predicted unsafe): {tp}")
            print(f"  False Positives (safe & predicted unsafe): {fp}")
            print(f"  False Negatives (unsafe & predicted safe): {fn}")
            print(f"  Precision (unsafe): {precision:.4f}")
            print(f"  Recall    (unsafe): {recall:.4f}\n")


def cal_best(model:str):
    K = [1,3,5]
    f = f'llm_eval.{model}.pkl'
    with open(f, 'rb') as f:
        data = pickle.load(f)[model]
        for k in K:
            tp = 0  # true positives (predicted unsafe, actually unsafe)
            fp = 0  # false positives (predicted unsafe, actually safe)
            fn = 0  # false negatives (predicted safe, actually unsafe)
            total = len(data)
            for gt, preds in data:
                pred = preds[:k]
                # Update TP, FP, FN
                if gt == 1:
                    if gt in pred:
                        tp += 1
                    else:
                        fn +=1
                elif gt == 0:
                    if gt not in pred:
                        fp+=1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            print(f"Model {model} results on {total} Best@K={k} examples:")
            print(f"  True Positives (unsafe & predicted unsafe): {tp}")
            print(f"  False Positives (safe & predicted unsafe): {fp}")
            print(f"  False Negatives (unsafe & predicted safe): {fn}")
            print(f"  Precision (unsafe): {precision:.4f}")
            print(f"  Recall    (unsafe): {recall:.4f}\n")


if __name__ == '__main__':
    model = sys.argv[1]
    cal_few(model)
    cal_best(model)