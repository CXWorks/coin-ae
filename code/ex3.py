import os
import subprocess
import multiprocessing
import sys


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def run_one(args):
    cuda = args[0]
    slice_start = args[1]
    slice_end = args[2]
    count = int((slice_end - slice_start) / 15000)
    for i in range(count):
        start = slice_start + i * 15000
        end = min(start + 15000, slice_end)
        print(i, count, f'CUDA_VISIBLE_DEVICES={cuda} INFER_BATCH_START={start} INFER_BATCH_END={end} python3.10 -u eval_batch.py')
        ret = subprocess.run(f'CUDA_VISIBLE_DEVICES={cuda} INFER_BATCH_START={start} INFER_BATCH_END={end} python3.10 -u eval_batch.py | tee batch_{cuda}_{i}.out', shell=True)


if __name__ == '__main__':
    cuda_idx = int(sys.argv[1])
    slice_start = int(sys.argv[2])
    slice_end = int(sys.argv[3])
    wl=[cuda_idx, slice_start, slice_end]
    print(wl)
    # with multiprocessing.Pool(8) as pool:
    #    pool.map(run_one, wl)
    run_one(wl)
    print('done')
