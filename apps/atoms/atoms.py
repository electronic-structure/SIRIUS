import json
import os, stat, re
import sys

fin = open("atoms.in", "r")

fout = open("run.x", "w")

header_file = open("atomic_conf.h", "w")

atoms = {}

while True:
    line = fin.readline()
    if not line: break
    if line.find("atom") == 0:
        
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
        
        t = sys.argv[1] if len(sys.argv) >= 2 else " --type=lo1"
        fout.write("./atom --symbol=" + symbol + t + "\n");
        
        atoms[symbol] = {}
        atoms[symbol]["name"] = name
        atoms[symbol]["zn"] = zn
        atoms[symbol]["mass"] = mass
        
        line = fin.readline() # Rmt
        line = fin.readline() # number of states
        s1 = line.split()
        nst = int(s1[0])
        
        header_file.write("{ // %s, z = %i\n"%(symbol, zn))
        levels = []
        for i in range(nst):
            if (i != 0):
                header_file.write(",\n")
            line = fin.readline() # n l k occ
            s1 = line.split()
            n = int(s1[0]);
            l = int(s1[1]);
            k = int(s1[2]);
            occ = float(s1[3]);
            levels.append([n, l, k, occ])
            header_file.write("    {%i, %i, %i, %f}"%(n, l, k, occ))
        atoms[symbol]["levels"] = levels
        header_file.write("\n},\n")
        
        line = fin.readline() # NIST LDA Etot
        line = line.strip()
        if  (line != ""):
            s1 = line.split()
            atoms[symbol]["NIST_LDA_Etot"] = float(s1[0])
        
            line = fin.readline() # NIST ScRLDA Etot
            line = line.strip()
            if  (line != ""):
                s1 = line.split()
                atoms[symbol]["NIST_ScRLDA_Etot"] = float(s1[0])

fout.close()
fin.close()

os.chmod("run.x", os.stat("run.x").st_mode | stat.S_IEXEC)

fout = open("atoms.json", "w")
fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ", json.dumps(atoms, indent=2)))
fout.close()

header_file.close()





