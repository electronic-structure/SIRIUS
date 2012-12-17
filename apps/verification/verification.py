import json
import sys
import os
import copy
import shutil
import glob
import subprocess

def create_input_file(kwargs):
    elk_in = "tasks\n2000\n\n"
    
    elk_in += "avec\n" + \
              "%f %f %f\n"%(kwargs["avec"][0][0], kwargs["avec"][0][1], kwargs["avec"][0][2]) + \
              "%f %f %f\n"%(kwargs["avec"][1][0], kwargs["avec"][1][1], kwargs["avec"][1][2]) +  \
              "%f %f %f\n"%(kwargs["avec"][2][0], kwargs["avec"][2][1], kwargs["avec"][2][2]) + "\n" 

    elk_in += "scale\n" + "%f\n"%kwargs["scale"] + "\n"

    elk_in += "nempty\n" + "%i\n"%kwargs["nempty"] + "\n"
    
    if kwargs["spinpol"] == 0:
        elk_in += "spinpol\n.false.\n\n"
    else:
        elk_in += "spinpol\n.true.\n\n"

    elk_in += "atoms\n" + "%i\n"%len(kwargs["atoms"])
    
    for iat in range(len(kwargs["atoms"])):
        elk_in += kwargs["atoms"][iat][0] + "\n"
        elk_in += "%i\n"%len(kwargs["atoms"][iat][1])
        for ia in range(len(kwargs["atoms"][iat][1])):
            elk_in += "%f %f %f %f %f %f\n"%(kwargs["atoms"][iat][1][ia][0], 
                                             kwargs["atoms"][iat][1][ia][1],
                                             kwargs["atoms"][iat][1][ia][2],
                                             kwargs["atoms"][iat][1][ia][3],
                                             kwargs["atoms"][iat][1][ia][4],
                                             kwargs["atoms"][iat][1][ia][5])
    elk_in += "\n"
    
    if "ngridk" in kwargs:
        elk_in += "ngridk\n" + "%i %i %i\n"%(kwargs["ngridk"][0], kwargs["ngridk"][1], kwargs["ngridk"][2]) + "\n"
    else:
        elk_in += "ngridk\n" + "2 2 2\n" + "\n"

    if "ldapu" in kwargs:
        elk_in += "lda+u\n1 1\n"
        for i in range(len(kwargs["ldapu"])):
            elk_in += "%i %i %f %f\n"%(kwargs["ldapu"][i][0], 
                                       kwargs["ldapu"][i][1], 
                                       kwargs["ldapu"][i][2], 
                                       kwargs["ldapu"][i][3])
        elk_in += "\n"

    return elk_in

def launch_task(irun, inp, conf):
    path = "./" + str(irun)
    shutil.rmtree(path, True)
    os.mkdir(path)

    atoms = inp["atoms"]

    k = 0
    for iat in range(len(atoms)):
        for ia in range(len(atoms[iat][1])):
            atoms[iat][1][ia].extend(conf["vector_field"][k])
            k = k + 1
    
    fout = open(path + "/elk.in", "w")
    fout.write(create_input_file(inp))
    fout.close()

    for i in glob.glob("[A-Z]*"):
        shutil.copy(i, path + "/" + i)

    fout = open(path + "/sirius.json", "w")
    fout.write("{\n")
    fout.write("    \"mpi_grid_dims\": " + str(conf["mpi_grid"]) + "\n")
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


