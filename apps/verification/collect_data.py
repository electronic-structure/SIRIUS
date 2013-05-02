import json
import sys
import os
import copy
import shutil
import glob

E_V = []

def main():
    
    for fname in glob.glob("./*/output*.json"):
        fin = open(fname, "r")
        jin = json.load(fin)
        fin.close()
        E_V.append([jin["omega"], jin["total_energy"]])
    
    for p in E_V:
        sys.stdout.write("%12.6f  %12.6f\n"%(p[0], p[1]))

if __name__ == "__main__":
    main()


