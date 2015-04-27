import sys
sys.dont_write_bytecode = True

sys.path.append("./PyCifRW")
import CifFile

import re
import math
import json

def parse_symmetry_op(op):
    # search for fractional translation
    result = re.search(r"(-?|\+?)[0-9]{1,10}/[0-9]{1,10}", op)
    trans = 0.0
    if result:
        s = result.group(0)
        i = s.find("/")
        nom_str = s[ : i].strip()
        denom_str = s[i + 1 :].strip()
        trans = float(nom_str) / float(denom_str)

    
    coord_trans = []
    for r in re.findall(r"[\+|\-]*[xXyYzZ]", op):
        sign = 1
        c = ''
        if r[0] == '-': 
            sign = -1
            c = r[1]
        elif r[0] == '+':
            c = r[1]
        else:
            c = r[0]
        coord_trans.append([sign, c])
            
    return [coord_trans, trans]


def parse_symmetry(sym):
    op = sym.split(",")
    # return transformation of each coordinate
    return ([parse_symmetry_op(op[0]), parse_symmetry_op(op[1]), parse_symmetry_op(op[2])])

def apply_symmetry(sym_ops_list, initial_atoms_list):
    full_atoms_list = []

    for atom in initial_atoms_list:
        for sym in sym_ops_list:
            # initialize with fractional translation
            x = [sym[0][1], sym[1][1], sym[2][1]]
            for i in range(3): 
                # apply all permutations of coordinates, such as +z-y
                for coord_trans in sym[i][0]: x[i] += coord_trans[0] * atom[coord_trans[1]]
                if x[i] < 0: x[i] += 1
                if x[i] >= 1: x[i] -= 1
                # roundoff
                x[i] = float("%.8f"%x[i])
            # search for existing atoms
            found = False
            for y in full_atoms_list:
                if (abs(y[0] - x[0]) + abs(y[1] - x[1]) + abs(y[2] - x[2])) < 1e-8: found = True
            if not found: full_atoms_list.append(x)

    return full_atoms_list

def remove_ending_braces(string):
    return re.sub(r"\([0-9]+\)", "", string)

def main():
    
    if len(sys.argv) != 2:
        print("Usage: python ./cif_input.py file.cif")
        sys.exit(0)

    cf = CifFile.ReadCif(sys.argv[1])

    cb = cf.first_block()

    atoms = cb.GetLoop("_atom_site_label")
    
    initial_atoms_list = {}

    for atom in atoms:

        label = atom._atom_site_label
        label = re.sub("[0-9]+", " ", label).strip()

        if not label in initial_atoms_list: initial_atoms_list[label] = []

        initial_atoms_list[label].append({"x" : float(remove_ending_braces(atom._atom_site_fract_x)), \
                                          "y" : float(remove_ending_braces(atom._atom_site_fract_y)), \
                                          "z" : float(remove_ending_braces(atom._atom_site_fract_z))})
    print("Initial list of atoms")
    for label in initial_atoms_list:
        for atom in initial_atoms_list[label]:
            print(label + " at " + "%f %f %f"%(atom["x"], atom["y"], atom["z"]))
        

    sym_ops = cb.GetLoop("_symmetry_equiv_pos_as_xyz")
    
    sym_ops_list = []
    
    for sym in sym_ops:
        sym_ops_list.append(parse_symmetry(sym._symmetry_equiv_pos_as_xyz))

    
    au2ang = 0.52917721092
    a = float(remove_ending_braces(cb["_cell_length_a"]))
    b = float(remove_ending_braces(cb["_cell_length_b"]))
    c = float(remove_ending_braces(cb["_cell_length_c"]))
    alpha = float(cb["_cell_angle_alpha"]) * math.pi / 180
    beta = float(cb["_cell_angle_beta"]) * math.pi / 180
    gamma = float(cb["_cell_angle_gamma"]) * math.pi / 180

  
    x2 = b * math.cos(gamma)
    x2 = 0 if (abs(x2)) < 1e-8 else x2

    y2 = b * math.sin(gamma)
    y2 = 0 if (abs(y2)) < 1e-8 else y2

    x3 = c * math.cos(beta)
    x3 = 0 if abs(x3) < 1e-8 else x3

    y3 = (a * b * math.cos(alpha) - x2 * x3) / y2
    y3 = 0 if abs(y3) < 1e-8 else y3

    z3 = math.sqrt(c * c - x3 * x3 - y3 * y3)
    z3 = 0 if abs(z3) < 1e-8 else z3

    avec = [[a, 0, 0], 
            [x2, y2, 0], 
            [x3, y3, z3]] 

    
    ## fout = open("elk.in", "w")
    ## fout.write("avec\n")
    ## for i in range(3):
    ##     fout.write("%18.10f %18.10f %18.10f\n"%(avec[i][0] / au2ang, avec[i][1] / au2ang, avec[i][2] / au2ang))
    ## fout.write("\n")
    ## fout.write("atoms\n")
    ## fout.write("%i\n"%len(initial_atoms_list.keys()))
    ## 

    atom_dict = {}
     
    natoms = 0
    for key in initial_atoms_list.keys():
        atom_list = apply_symmetry(sym_ops_list, initial_atoms_list[key])
        atom_dict[key] = atom_list
        ## fout.write("'%s.in'\n"%key)
        ## fout.write("%i\n"%len(atom_list))
        for a in atom_list:
            natoms += 1
            ## fout.write("%18.10f %18.10f %18.10f\n"%(a[0], a[1], a[2]))

    ## fout.close()
   
    atom_files = {}
    for label in initial_atoms_list:
        atom_files[label] = label + ".json"

    unit_cell = {}
    unit_cell["lattice_vectors"] = avec
    unit_cell["lattice_vectors_scale"] = 1 / au2ang
    unit_cell["atoms"] = atom_dict
    unit_cell["atom_types"] = initial_atoms_list.keys()
    unit_cell["atom_files"] = atom_files

    fout = open("sirius.json", "w")
    fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", ", ", \
        json.dumps({"unit_cell" : unit_cell}, indent = 4)))
    fout.close()


##    fout = open("pw.in", "w")
##    fout.write("""
##&control
##  calculation='scf',
##  restart_mode='from_scratch',
##  pseudo_dir = './',
##  outdir='./',
##  prefix = 'scf_'
##/""")
##    fout.write("""
##&system
##  ibrav=0, celldm(1)=1, ecutwfc=40, ecutrho = 300,
##  occupations = 'smearing', smearing = 'gauss', degauss = 0.002, nosym=.false.,\n""")
##    fout.write("  nat=" + str(natoms) + ", ntyp=" + str(len(initial_atoms_list.keys())))
##    fout.write("\n/")
##    fout.write("""
##&electrons
##  conv_thr =  1.0d-8,
##  mixing_beta = 0.7,
##  electron_maxstep = 10
##/""")
##    fout.write("\n")
##    fout.write("ATOMIC_SPECIES\n")
##    for key in initial_atoms_list.keys():
##        fout.write("  %s 0.0 %s.UPF\n"%(key, key))
##
##    fout.write("\n")
##    fout.write("CELL_PARAMETERS\n")
##    for i in range(3):
##        fout.write("%18.10f %18.10f %18.10f\n"%(avec[i][0] / au2ang, avec[i][1] / au2ang, avec[i][2] / au2ang))
##    
##    fout.write("\n")
##    fout.write("ATOMIC_POSITIONS (crystal)\n")
##    for key in initial_atoms_list.keys():
##        atom_list = apply_symmetry(sym_ops_list, initial_atoms_list[key])
##        for a in atom_list:
##            fout.write("%s  %18.10f %18.10f %18.10f\n"%(key, a[0], a[1], a[2]))
##
##    fout.write("\n")
##    fout.write("K_POINTS (automatic)\n")
##    fout.write("2 2 2  0 0 0\n")
##
##    fout.close()


    print("Total number of atoms in the unit cell : %i\n"%(natoms))

    return

if __name__ == "__main__":
    main()

