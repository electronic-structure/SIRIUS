#!/usr/bin/python
import json
import sys
import os
import copy
import shutil
import glob
import subprocess
import time

def main():
    
    if len(sys.argv) != 2:
        print("Usage: run_job_py34.py [mpi_grid]")
        sys.exit(0)

    grid = list(map(int, sys.argv[1].split()))
    
    num_ranks = 1
    for i in grid: num_ranks *= i

    print(os.environ)

    new_env = copy.deepcopy(os.environ)
    new_env["OMP_NUM_THREADS"] = "8"
    new_env["OMP_NESTED"] = "false"
    new_env["MKL_NUM_THREADS"] = "8"
    new_env["MKL_DYNAMIC"] = "false"
    new_env["CRAY_CUDA_MPS"] = "0"
    new_env["MPICH_RDMA_ENABLED_CUDA"] = "1"
    new_env["MPICH_NO_GPU_DIRECT"] = "1"
    new_env["CRAY_LIBSCI_ACC_MODE"] = "1"

    aprun_command = "aprun -n " + str(num_ranks) + " -N1 -d8 -cc none"
    job_command = "./test_zgemm --M=4000 --N=4000 --K=4000"

    command = aprun_command + " " + job_command

    job_duration = -time.time()
    path = "./"
    proc = subprocess.Popen(command.split(), cwd=path, env=new_env)
    proc.wait()
    job_duration += time.time()

    print("job duration: " + str(job_duration))

if __name__ == "__main__":
    main()


