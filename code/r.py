import subprocess
import os
from multiprocessing import Pool

TOTAL_BATCHES = 50
CUDA_DEVICE_GROUPS = ["0,1", "2,3"]

def run_batch(args):
    batch_id, tasks = args
    cuda_devices = CUDA_DEVICE_GROUPS[batch_id]
    for tsk in tasks:
        cmd = ["python3", "-u", "infer.py", "--pi", str(tsk)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices}
        print(f"Starting batch {tsk} on GPUs {cuda_devices}")
        ret = subprocess.run(cmd, env=env, capture_output=True)
        print(f"Finished batch {tsk} {ret.returncode}")


if __name__ == "__main__":
    # Alternate devices for each batch: 0 → 0,1; 1 → 2,3; 2 → 0,1; etc.
    b1 = [x for x in range(24, 62, 2)]
    b2 = [x for x in range(25, 62, 2)]
    args = [(0, b1), (1, b2)]
    with Pool(processes=2) as pool:
        pool.map(run_batch, args)
    print('done')