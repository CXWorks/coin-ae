import pickle
import os
import sys


def combine():
    all = []

    for f in os.listdir('.'):
        if os.path.isfile(f) and f.endswith('.pkl') and f.startswith('data_'):
            with open(f, 'rb') as fp:
                all.extend(pickle.load(fp))
    print(len(all))
    labels = [0]*len(all)
    with open('all_data.pkl', 'wb') as fp:
        pickle.dump(all, fp)
    with open('label_data.pkl', 'wb') as fp:
        pickle.dump(labels, fp)


def mark():
    with open('all_data.pkl', 'rb') as fp:
        data = pickle.load(fp)
    with open('label_data.pkl', 'rb') as fp:
        labels = pickle.load(fp)
    for i in range(len(data)):
        if labels[i] == 0:
            print(i, '='*20)
            print(data[i][1])
            print(data[i][2])
            ans = input('Safe or unsafe? [y/n]')
            if ans == 'y' or ans == 'Y':
                labels[i] = 1
            elif ans == 'n' or ans == 'N':
                labels[i] = -1
            else:
                with open('label_data.pkl', 'wb') as fp:
                    pickle.dump(labels, fp)
                break
    print('exit')

if __name__ == '__main__':
    mark()


