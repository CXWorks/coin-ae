import os
import subprocess
import multiprocessing


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def main():
    tasks = [x for x in range(400, 800)]
    wl = []
    for idx, tsk in enumerate(chunks(tasks, 4)):
        wl.append((idx+4, tsk))
    return wl


def run_one(args):
    cuda = args[0]
    tsks = args[1]
    with open(f'batch_{cuda}.out', 'a') as fp1:
        with open(f'batch_{cuda}.err', 'a') as fp2:
            for ct in tsks:
                ret = subprocess.run(f'CUDA_VISIBLE_DEVICES={cuda} INFER_BATCH={ct} python3.12 -u infer_batch.py', shell=True,
                                     capture_output=True)
                fp1.write(ret.stdout.decode('utf-8'))
                fp1.flush()
                fp2.write(ret.stderr.decode('utf-8'))
                fp2.flush()



if __name__ == '__main__':
    wl = main()
    print(wl)
    with multiprocessing.Pool(4) as pool:
        pool.map(run_one, wl)
    print('done')
