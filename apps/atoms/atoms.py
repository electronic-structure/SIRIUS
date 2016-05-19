import json
import os, stat, re

fin = open("atoms.in", "r")

fout = open("run.x", "w")

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

        fout.write("./atom --order=2 --type=lo+LO --symbol=" + symbol + "\n");
        
        atoms[symbol] = {}
        atoms[symbol]["name"] = name
        atoms[symbol]["zn"] = zn
        atoms[symbol]["mass"] = mass
        
        line = fin.readline() # Rmt
        line = fin.readline() # number of states
        s1 = line.split()
        nst = int(s1[0])
        
        levels = []
        for i in range(nst):
            line = fin.readline() # n l k occ
            s1 = line.split()
            n = int(s1[0]);
            l = int(s1[1]);
            k = int(s1[2]);
            occ = int(s1[3]);
            levels.append([n, l, k, occ])
        atoms[symbol]["levels"] = levels
        
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





