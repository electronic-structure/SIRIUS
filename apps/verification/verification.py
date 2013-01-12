import json
import sys
import os
import copy
import shutil
import glob
import subprocess

def create_input_file(inp, conf):
    
    atoms = inp["atoms"]

    k = 0
    for iat in range(len(atoms)):
        for ia in range(len(atoms[iat][1])):
            atoms[iat][1][ia].extend(conf["vector_field"][k])
            k = k + 1
    
    elk_in = "tasks\n2000\n\n"
    
    elk_in += "avec\n" + \
              "%f %f %f\n"%(inp["avec"][0][0], inp["avec"][0][1], inp["avec"][0][2]) + \
              "%f %f %f\n"%(inp["avec"][1][0], inp["avec"][1][1], inp["avec"][1][2]) +  \
              "%f %f %f\n"%(inp["avec"][2][0], inp["avec"][2][1], inp["avec"][2][2]) + "\n" 

    if "scale" in inp:
        elk_in += "scale\n" + "%f\n"%inp["scale"] + "\n"

    if "nempty" in inp:
        elk_in += "nempty\n" + "%i\n"%inp["nempty"] + "\n"
    
    if "spinpol" in inp:
        if inp["spinpol"] == 0:
            elk_in += "spinpol\n.false.\n\n"
        else:
            elk_in += "spinpol\n.true.\n\n"

    elk_in += "atoms\n" + "%i\n"%len(inp["atoms"])
    
    for iat in range(len(inp["atoms"])):
        elk_in += inp["atoms"][iat][0] + "\n"
        elk_in += "%i\n"%len(inp["atoms"][iat][1])
        for ia in range(len(inp["atoms"][iat][1])):
            elk_in += "%f %f %f %f %f %f\n"%(inp["atoms"][iat][1][ia][0], 
                                             inp["atoms"][iat][1][ia][1],
                                             inp["atoms"][iat][1][ia][2],
                                             inp["atoms"][iat][1][ia][3],
                                             inp["atoms"][iat][1][ia][4],
                                             inp["atoms"][iat][1][ia][5])
    elk_in += "\n"
    
    if "ngridk" in inp:
        elk_in += "ngridk\n" + "%i %i %i\n"%(inp["ngridk"][0], inp["ngridk"][1], inp["ngridk"][2]) + "\n"
    else:
        elk_in += "ngridk\n" + "2 2 2\n" + "\n"

    if "ldapu" in inp:
        elk_in += "lda+u\n1 1\n"
        for i in range(len(inp["ldapu"])):
            elk_in += "%i %i %f %f\n"%(inp["ldapu"][i][0], 
                                       inp["ldapu"][i][1], 
                                       inp["ldapu"][i][2], 
                                       inp["ldapu"][i][3])
        elk_in += "\n"

    if "aw_cutoff" in conf:
        elk_in += "rgkmax\n"
        elk_in += "%f\n"%conf["aw_cutoff"]
        elk_in += "\n"

    if "lmaxapw" in conf:
        elk_in += "lmaxapw\n"
        elk_in += "%i\n"%conf["lmaxapw"]
        elk_in += "\n"
    
    return elk_in

def launch_task(irun, inp, conf):
    path = "./" + str(irun)
    shutil.rmtree(path, True)
    os.mkdir(path)

    fout = open(path + "/elk.in", "w")
    fout.write(create_input_file(inp, conf))
    fout.close()

    for i in glob.glob("[A-Z]*"):
        shutil.copy(i, path + "/" + i)

    fout = open(path + "/sirius.json", "w")
    fout.write("{\n")
    fout.write("    \"mpi_grid_dims\": " + str(conf["mpi_grid"]) + ",\n")
    fout.write("    \"cyclic_block_size\" : " + str(conf["cyclic_block_size"]) + "\n")
    fout.write("}\n")
    fout.close()

    np = 1
    for i in range(len(conf["mpi_grid"])): np = np * conf["mpi_grid"][i]

    # simple execution
    new_env = copy.deepcopy(os.environ)
    new_env["OMP_NUM_THREADS"] = str(conf["num_threads"])
    fstdout = open(path + "/output.txt", "w");
    p = subprocess.Popen(["mpirun", "-np", str(np), "../elk"], cwd=path, env=new_env, stdout=fstdout)
    p.wait()
    fstdout.close()

    # batch job
    #fout = open(path + "/batch_job.slrm", "w")
    #fout.write("#!/bin/bash\n")
    #fout.write("#SBATCH --time=00:05:00\n")
    #fout.write("#SBATCH --ntasks=" + str(np) + "\n")
    #fout.write("#SBATCH --ntasks-per-node=1\n")
    #fout.write("#SBATCH --cpus-per-task=" + str(conf["num_threads"]) + "\n")
    #fout.write("#SBATCH --job-name=test-sirius\n")
    #fout.write("#SBATCH --output=output-%j.txt\n")
    #fout.write("#SBATCH --error=output-%j.txt\n")
    #fout.write("#SBATCH --account=s299\n")
    #fout.write("export NCPUS=" + str(conf["num_threads"]) + "\n")
    #fout.write("export GOTO_NUM_THREADS=" + str(conf["num_threads"]) + "\n")
    #fout.write("export OMP_NUM_THREADS=" + str(conf["num_threads"]) + "\n")
    #fout.write("date\n")
    #fout.write("aprun -n " + str(np) + " -N 1 -d " + str(conf["num_threads"]) + " ../elk\n")
    #fout.write("date\n")
    #fout.close()
    #p = subprocess.Popen(["sbatch", "batch_job.slrm"], cwd=path)
    #p.wait()


def main():
    
    fin = open(sys.argv[1], "r")
    jin = json.load(fin)
    fin.close()

    for irun in range(len(jin["run"])):
        
        inp = copy.deepcopy(jin["input"])

        launch_task(irun, inp, jin["run"][irun])







### run tests one by one on one or multiple nodes;
### test is described by: elk input + mpi_grid conf + num_theads
### at this point we want to test: 
###   LSDA collinear vs LSDA at arbitrary direction
###   LSDA+U collinear vs LSDA+U at arbitrary direction
###   LSDA+SO for [100] [010] and [001] directions
###  test is a series of runs
##    create_input_file(avec=[[1, 1, -1], [1, -1, 1], [-1, 1, 1]], 
##                      scale=2.708,
##                      nempty=20,
##                      spinpol=1,
##                      atoms=[["'Fe.in'", [[0, 0, 0, 0.1, 0.1, 0.1] ] ] ]) 
##

if __name__ == "__main__":
    main()


