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
        if self.num_ranks_per_node > 1:
            self.env['CRAY_CUDA_MPS'] = '1'

    def launch_job(self, count, **kwargs):

        output_json = "out_%iR_%i.json"%(self.num_ranks, count)
        params = ''
        # basic command
        cmd = ['srun', '-C', 'gpu', '-N', str(self.num_nodes), '-n', str(self.num_ranks), '-c',
               str(self.num_threads_per_rank), '--unbuffered', '--hint=nomultithread', './sirius.scf', '--output=' + output_json]
        # extra parameters
        if 'processing_unit' in kwargs:
            cmd.append("--control.processing_unit=%s"%kwargs['processing_unit'])
            params = params + ', ' + kwargs['processing_unit']
        if 'ev_solver' in kwargs:
            cmd.append("--control.std_evp_solver_name=%s"%kwargs['ev_solver'])
            cmd.append("--control.gen_evp_solver_name=%s"%kwargs['ev_solver'])
            params = params + ', ' + kwargs['ev_solver']
        if 'mpi_grid' in kwargs:
            cmd.append("--control.mpi_grid_dims=%i:%i"%(kwargs['mpi_grid'][0], kwargs['mpi_grid'][1]))
            params = params + ', ' + str(kwargs['mpi_grid'])
        if 'mem_usage' in kwargs:
            cmd.append("--control.memory_usage=%s"%(kwargs['mem_usage']))
            params = params + ', ' + str(kwargs['mem_usage'])
        if 'gamma' in kwargs:
            cmd.append("--parameters.gamma_point=%i"%kwargs['gamma'])
            params = params + ', gamma=' + str(kwargs['gamma'])
    
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
        
        print('')
        return errcode
    
    def launch_jobs(self):
        with open('sirius.json', 'r') as f:
            jin = json.load(f)
    
        ngridk = jin['parameters']['ngridk']
        if ngridk[0] * ngridk[1] * ngridk[2] == 1:
            check_for_gamma = True
        else:
            check_for_gamma = False

        #cwdlibs = os.getcwd() + "/libs/"
    
        #if not os.path.exists(cwdlibs):
        #    os.makedirs(cwdlibs)
    
        mpi_grids = []
        for i in range(1, self.num_ranks + 1):
            for j in range(1, self.num_ranks + 1):
                if check_for_gamma:
                    k_range = 1
                else:
                    k_range = self.num_ranks
                for k in range(1, k_range + 1):
                    if i * j * k == self.num_ranks:
                        mpi_grids.append([i, j, k])
    
        list_pu = {'cpu', 'gpu'}
        list_ev = {'lapack', 'magma', 'cusolver', 'scalapack', 'elpa1'}
        list_mem = {'high', 'low'}
        if check_for_gamma:
            list_gamma = {0, 1}
        else:
            list_gamma = {0}

        count = 0
        self.launch_job(count)

        #print("testing %i calculations\n"%(len(mpi_grids) * len(list_pu) * len(list_ev) * len(list_gamma) * len(list_mem)))

        num_correct = 0
        num_wrong = 0
    
        for pu in list_pu:
            for evs in list_ev:
                if evs == 'cusolver' and pu == 'cpu':
                    continue
                for gp in list_gamma:
                    for mem in list_mem:
                        for g in mpi_grids:
                            count = count + 1
                            errcode = self.launch_job(count, processing_unit=pu, ev_solver=evs, mpi_grid=g, mem_usage=mem, gamma=gp)
                            if errcode == 0:
                                num_correct = num_correct + 1
                            else:
                                num_wrong = num_wrong + 1
        
        print("number of correct calculations : %i"%num_correct)
        print("number of wrong calculations : %i"%num_wrong)

def main():
    launcher = sirius_scf_launcher(4, 12)
    launcher.launch_jobs()

if __name__ == "__main__":
    main()
