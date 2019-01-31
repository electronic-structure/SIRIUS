import sys
import os
import tarfile
import subprocess
import json

#def print_red(string):
#    print("\033[0;31;40m%s\033[0;37;40m"%string)
#
#def print_green(string):
#    print("\033[0;32;40m%s\033[0;37;40m"%string)
#
#def print_blue(string):
#    print("\033[0;34;40m%s\033[0;37;40m"%string)

max_num_threads = 12

class sirius_scf_launcher:

    def __init__(self, num_nodes, num_threads_per_rank):
        self.env = os.environ.copy()
        self.num_nodes = num_nodes
        self.num_threads_per_rank = num_threads_per_rank
        self.num_ranks_per_node = max_num_threads // num_threads_per_rank
        self.num_ranks = self.num_nodes * self.num_ranks_per_node
        self.env['MPICH_MAX_THREAD_SAFETY'] = 'multiple'
        self.env['OMP_NUM_THREADS'] = str(num_threads_per_rank)
        self.env['MKL_NUM_THREADS'] = str(num_threads_per_rank)
        self.env['CRAY_CUDA_MPS'] = '0'

    def launch_job(self, count, **kwargs):
    
        output_json = "out_%iR_%i.json"%(self.num_ranks, count)
        params = ''
        # basic command
        cmd = ['srun', '-C', 'gpu', '-N', str(self.num_nodes), '-n', str(self.num_ranks), '-c',
               str(self.num_threads_per_rank), '--unbuffered', '--hint=nomultithread', './sirius.scf', '--output=' + output_json]
        if 'processing_unit' in kwargs:
            cmd.append("--processing_unit=%s"%kwargs['processing_unit'])
            params = params + ', ' + kwargs['processing_unit']
        if 'ev_solver' in kwargs:
            cmd.append("--std_evp_solver_name=%s"%kwargs['ev_solver'])
            cmd.append("--gen_evp_solver_name=%s"%kwargs['ev_solver'])
            params = params + ', ' + kwargs['ev_solver']
        if 'mpi_grid' in kwargs:
            cmd.append("--mpi_grid=%i:%i"%(kwargs['mpi_grid'][0], kwargs['mpi_grid'][1]))
            params = params + ', ' + str(kwargs['mpi_grid'])
    
        if count == 0:
            print("Executing reference calculation on %i MPI rank(s)"%(self.num_ranks), end = '')
        else:
            print("Executing calculation on %i MPI rank(s)"%(self.num_ranks), end = '')
    
        print(params)
        print("Full command line: %s"%(' '.join(cmd)))
    
        fstdout = open("stdout_%iR_%i.txt"%(self.num_ranks, count), 'w')
        fstderr = open("stderr_%iR_%i.txt"%(self.num_ranks, count), 'w')
    
        # execut a run
        p = subprocess.Popen(cmd, cwd = './', env = self.env, stdout = fstdout, stderr = fstderr, shell = False)
        p.wait()
        errcode = p.returncode
        fstdout.close()
        fstderr.close()
    
        if errcode == 0:
            print('Success: calculation finished correctly')
            with open(output_json, 'r') as f:
                result = json.load(f)
                tot_en = result["ground_state"]["energy"]["total"]
                print("Total energy: %f"%tot_en)
            if count > 0:
                # open a reference file
                with open("out_%iR_%i.json"%(self.num_ranks, 0), 'r') as f:
                    result = json.load(f)
                    tot_en_ref = result["ground_state"]["energy"]["total"]
                if abs(tot_en - tot_en_ref) > 1e-7:
                    print('Error: total energy is different')
                else:
                    print('Success: total energy is correct')
        else:
            print("Error: calculation failed with error code %i"%errcode)
    
    def launch_jobs(self):
    
        #cwdlibs = os.getcwd() + "/libs/"
    
        #if not os.path.exists(cwdlibs):
        #    os.makedirs(cwdlibs)
    
        mpi_grids = []
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                for k in range(1, self.num_ranks + 1):
                    if i * j * k == self.num_ranks:
                        mpi_grids.append([i, j, k])
    
        count = 0
        self.launch_job(count)
    
        list_pu = {'cpu', 'gpu'}
        list_ev = {'lapack', 'magma', 'scalapack'}
        for pu in list_pu:
            for evs in list_ev:
                for g in mpi_grids:
                    count = count + 1
                    self.launch_job(count, processing_unit=pu, ev_solver=evs, mpi_grid=g)

def main():
    launcher = sirius_scf_launcher(2, 12)
    launcher.launch_jobs()

if __name__ == "__main__":
    main()
