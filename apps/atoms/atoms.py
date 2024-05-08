import json
import os, stat, re
import sys

fin = open("atoms.in", "r")

fout = open("run.x", "w")


atoms = []

while True:
    line = fin.readline()
    if not line: break
    if line.find("atom") == 0:
        atom = {}
        
        line = fin.readline()
        s1 = line.split()
        zn = int(s1[0])
        line = fin.readline() # symbol and name
        line = line.replace("'", " ")
        s1 = line.split()
        symbol = s1[0]
        name = s1[1]
        line = fin.readline()
        s1 = line.split()
        mass = float(s1[0]) # mass
        
        t = sys.argv[1] if len(sys.argv) >= 2 else " --type=lo11+lo22 --nrmt=1500"
        fout.write("./atom --symbol=" + symbol + t + "\n")
        
        atom["symbol"] = symbol
        atom["name"] = name
        atom["zn"] = zn
        atom["mass"] = mass
        
        line = fin.readline() # Rmt
        line = fin.readline() # number of states
        s1 = line.split()
        nst = int(s1[0])
        
        levels = []
        for i in range(nst):
            line = fin.readline() # n l k occ
            s1 = line.split()
            n = int(s1[0])
            l = int(s1[1])
            k = int(s1[2])
            occ = float(s1[3])
            levels.append([n, l, k, occ])
        atom["levels"] = levels
        
        line = fin.readline() # NIST LDA Etot
        line = line.strip()
        if line != "":
            s1 = line.split()
            atom["NIST_LDA_Etot"] = float(s1[0])
        
            line = fin.readline() # NIST ScRLDA Etot
            line = line.strip()
            if  (line != ""):
                s1 = line.split()
                atom["NIST_ScRLDA_Etot"] = float(s1[0])
        else:
            atom['NIST_LDA_Etot'] = 0

        # add atom to a list
        atoms.append(atom)
fout.close()
fin.close()

os.chmod("run.x", os.stat("run.x").st_mode | stat.S_IEXEC)

atoms2 = {}
for atom in atoms:
    atoms2[atom['symbol']] = atom

fout = open("atoms.json", "w")
fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ", json.dumps(atoms2, indent=2)))
fout.close()

atomic_data_file = open("atomic_data.hpp", "w")
atomic_data_file.write("// Warning! This is an automatically generated header file!\n")
atomic_data_file.write("/** \\file atomic_data.hpp\n")
atomic_data_file.write(" *  \\brief Basic atomic data information.\n")
atomic_data_file.write(" */\n")
atomic_data_file.write("\n")
atomic_data_file.write("#ifndef __ATOMIC_DATA_HPP__\n")
atomic_data_file.write("#define __ATOMIC_DATA_HPP__\n")
atomic_data_file.write("\n")
atomic_data_file.write("#include <string>\n")
atomic_data_file.write("#include <vector>\n")
atomic_data_file.write("#include <map>\n")
atomic_data_file.write("""
/// Describes single atomic level.
struct atomic_level_descriptor
{
    /// Principal quantum number.
    int n;

    /// Angular momentum quantum number.
    int l;

    /// Quantum number k.
    int k;

    /// Level occupancy.
    double occupancy;

    /// True if this is a core level.
    bool core;
};""")
atomic_data_file.write("\n\n")

atomic_data_file.write("const std::vector<std::string> atomic_symb = {\n")
atomic_data_file.write("    %s\n"%(", ".join(['"' + a['symbol'] + '"' for a in atoms])))
atomic_data_file.write("};\n")
atomic_data_file.write("\n")

atomic_data_file.write("const std::vector<std::string> atomic_name = {\n")
atomic_data_file.write("    %s\n"%(", ".join(['"' + a['name'] + '"' for a in atoms])))
atomic_data_file.write("};\n")
atomic_data_file.write("\n")

atomic_data_file.write("const std::map<std::string, int> atomic_zn = {\n")
atomic_data_file.write("    %s\n"%(", ".join(['{"' + a['symbol'] + '", ' + str(a['zn']) + '}' for a in atoms])))
atomic_data_file.write("};\n")
atomic_data_file.write("\n")

atomic_data_file.write("const std::vector<std::vector<atomic_level_descriptor>> atomic_conf = {\n")
for atom in atoms:
    atomic_data_file.write("    { // %s, z = %i\n"%(atom['symbol'], atom['zn']))
    for level in atom['levels']:
        atomic_data_file.write("        {%i, %i, %i, %f}, \n"%(level[0], level[1], level[2], level[3]))
    atomic_data_file.write("    },\n")
atomic_data_file.write("};\n")
atomic_data_file.write("\n")

atomic_data_file.write("const std::vector<double> atomic_energy_NIST_LDA = {\n")
atomic_data_file.write("    %s\n"%(", ".join(["%12.7f"%(a['NIST_LDA_Etot']) for a in atoms])))
atomic_data_file.write("};\n")
atomic_data_file.write("\n")

atomic_data_file.write("#endif\n")

atomic_data_file.close()


