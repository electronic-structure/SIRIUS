import json
import sys
import os
import copy
import shutil
import glob
import subprocess

def create_input_file(common, private):
    
    k = 0
    for atom_type in common["atoms"]:
        for atom in atom_type[1]:
            if "vector_field" in private: atom.extend(private["vector_field"][k])
            else: atom.extend([0, 0, 0])
            k = k + 1
    
    elk_in = "tasks\n2000\n\n"
    
    elk_in += "avec\n" + \
              "%16.10f %16.10f %16.10f\n"%(common["avec"][0][0], common["avec"][0][1], common["avec"][0][2]) + \
              "%16.10f %16.10f %16.10f\n"%(common["avec"][1][0], common["avec"][1][1], common["avec"][1][2]) +  \
              "%16.10f %16.10f %16.10f\n"%(common["avec"][2][0], common["avec"][2][1], common["avec"][2][2]) + "\n" 
    
    if "scale" in private:
        elk_in += "scale\n" + "%16.10f\n"%private["scale"] + "\n"
    elif "scale" in common:
        elk_in += "scale\n" + "%16.10f\n"%common["scale"] + "\n"

    if "nempty" in common:
        elk_in += "nempty\n" + "%i\n"%common["nempty"] + "\n"
    
    if "spinpol" in common:
        if common["spinpol"] == 0: elk_in += "spinpol\n.false.\n\n"
        else: elk_in += "spinpol\n.true.\n\n"

    elk_in += "atoms\n" + "%i\n"%len(common["atoms"])
    
    for atom_type in common["atoms"]:
        elk_in += atom_type[0] + "\n"
        elk_in += "%i\n"%len(atom_type[1])
        for atom in atom_type[1]:
            elk_in += "%16.10f %16.10f %16.10f %16.10f %16.10f %16.10f\n"%(atom[0], atom[1], atom[2],
                                                                           atom[3], atom[4], atom[5])
    elk_in += "\n"
    
    ngridk = [2, 2, 2]
    if "ngridk" in private: 
        ngridk = private["ngridk"]
    elif "ngridk" in common: ngridk = common["ngridk"]

    elk_in += "ngridk\n" + "%i %i %i\n"%(ngridk[0], ngridk[1], ngridk[2]) + "\n"

    if "ldapu" in common:
        elk_in += "lda+u\n1 1\n"
        for i in range(len(common["ldapu"])):
            elk_in += "%i %i %f %f\n"%(common["ldapu"][i][0], common["ldapu"][i][1], common["ldapu"][i][2], common["ldapu"][i][3])
        elk_in += "\n"

    
    if "aw_cutoff" in private:
        elk_in += "rgkmax\n" + "%f\n"%private["aw_cutoff"] + "\n"
    elif "aw_cutoff" in common:
        elk_in += "rgkmax\n" + "%f\n"%common["aw_cutoff"] + "\n"

    if "lmaxapw" in private:
        elk_in += "lmaxapw\n" + "%i\n"%private["lmaxapw"] + "\n"
    elif "lmaxapw" in common:
        elk_in += "lmaxapw\n" + "%i\n"%common["lmaxapw"] + "\n"
    
    if "lmaxvr" in private:
        elk_in += "lmaxvr\n" + "%i\n"%private["lmaxvr"] + "\n"
    elif "lmaxvr" in common:
        elk_in += "lmaxvr\n" + "%i\n"%common["lmaxvr"] + "\n"

    elk_in += "autormt\n.true.\n"
    
    return elk_in

def launch_task(irun, common, private):
    path = "./" + str(irun)
    shutil.rmtree(path, True)
    os.mkdir(path)

    fout = open(path + "/elk.in", "w")
    fout.write(create_input_file(common, private))
    fout.close()

    for i in glob.glob("[A-Z]*"):
        shutil.copy(i, path + "/" + i)

    if "mpi_grid" in private: 
        mpi_grid = private["mpi_grid"]
    else: 
        mpi_grid = common["mpi_grid"]
    
    cyclic_block_size = 32
    if "cyclic_block_size" in private: 
        cyclic_block_size = private["cyclic_block_size"]
    elif "cyclic_block_size" in common: 
        cyclic_block_size = common["cyclic_block_size"]

    if "num_threads" in private: 
        num_threads = private["num_threads"]
    else: 
        num_threads = common["num_threads"]

    fout = open(path + "/sirius.json", "w")
    fout.write("{\n")
    fout.write("    \"mpi_grid_dims\": " + str(mpi_grid) + ",\n")
    fout.write("    \"cyclic_block_size\" : " + str(cyclic_block_size) + "\n")
    fout.write("}\n")
    fout.close()

    num_proc = 1
    for i in range(len(mpi_grid)): num_proc *= mpi_grid[i]

    # simple execution
    new_env = copy.deepcopy(os.environ)
    new_env["OMP_NUM_THREADS"] = str(num_threads)
    fstdout = open(path + "/output.txt", "w");
    p = subprocess.Popen(["mpirun", "-np", str(num_proc), "../elk"], cwd=path, env=new_env, stdout=fstdout)
    p.wait()
    fstdout.close()

    # batch job
    #fout = open(path + "/batch_job.slrm", "w")
    #fout.write("#!/bin/bash\n")
    #fout.write("#SBATCH --time=00:05:00\n")
    #fout.write("#SBATCH --ntasks=" + str(np) + "\n")
    #fout.write("#SBATCH --ntasks-per-node=1\n")
    #fout.write("#SBATCH --cpus-per-task=" + str(private["num_threads"]) + "\n")
    #fout.write("#SBATCH --job-name=test-sirius\n")
    #fout.write("#SBATCH --output=output-%j.txt\n")
    #fout.write("#SBATCH --error=output-%j.txt\n")
    #fout.write("#SBATCH --account=s299\n")
    #fout.write("export NCPUS=" + str(private["num_threads"]) + "\n")
    #fout.write("export GOTO_NUM_THREADS=" + str(private["num_threads"]) + "\n")
    #fout.write("export OMP_NUM_THREADS=" + str(private["num_threads"]) + "\n")
    #fout.write("date\n")
    #fout.write("aprun -n " + str(np) + " -N 1 -d " + str(private["num_threads"]) + " ../elk\n")
    #fout.write("date\n")
    #fout.close()
    #p = subprocess.Popen(["sbatch", "batch_job.slrm"], cwd=path)
    #p.wait()


def main():
    
    fin = open(sys.argv[1], "r")
    jin = json.load(fin)
    fin.close()

    for irun in range(len(jin["private"])):
        
        common = copy.deepcopy(jin["common"])

        launch_task(irun, common, jin["private"][irun])


if __name__ == "__main__":
    main()


