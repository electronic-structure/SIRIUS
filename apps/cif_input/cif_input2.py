import sys
sys.dont_write_bytecode = True

import CifFile
import re
import math

def parse_symmetry_op(op):
    result = re.search("(-?|\+?)[0-9]{1,10}/[0-9]{1,10}", op)
    trans = 0.0
    if result:
        s = result.group(0)
        i = s.find("/")
        nom_str = s[ : i].strip()
        denom_str = s[i + 1 :].strip()
        trans = float(nom_str) / float(denom_str)

    result = re.search("(-?|\+?)[xXyYzZ]", op)
    s = result.group(0)
    sign = 1
    if s[0] == '-': sign = -1
    result = re.search("[xXyYzZ]", s)
    coord = result.group(0)

    return [sign, coord, trans]


def parse_symmetry(sym):
    op = sym.split(",")
    return ([parse_symmetry_op(op[0]), parse_symmetry_op(op[1]), parse_symmetry_op(op[2])])

def apply_symmetry(sym_ops_list, initial_atoms_list):
    full_atoms_list = []

    for atom in initial_atoms_list:
        for sym in sym_ops_list:
            coord = [0, 0, 0]
            for i in range(3): 
                coord[i] = sym[i][0] * atom[sym[i][1]] + sym[i][2]
                if coord[i] < 0: coord[i] += 1
                if abs(coord[i] - 1) < 1e-10: coord[i] = 0
            
            found = False
            for c in full_atoms_list:
                if (abs(c[0] - coord[0]) + abs(c[1] - coord[1]) + abs(c[2] - coord[2])) < 1e-10: found = True
            if not found: full_atoms_list.append(coord)

    return full_atoms_list


def main():
    
    if len(sys.argv) != 2:
        print "Usage: python ./cif_input2.py file.cif" 
        sys.exit(0)

    cf = CifFile.ReadCif(sys.argv[1])

    cb = cf.first_block()


    atoms = cb.GetLoop("_atom_site_label")
    
    initial_atoms_list = {}

    for atom in atoms:
        
        if not atom._atom_site_type_symbol in initial_atoms_list: initial_atoms_list[atom._atom_site_type_symbol] = []
        
        initial_atoms_list[atom._atom_site_type_symbol].append({"x" : float(atom._atom_site_fract_x), \
                                                                "y" : float(atom._atom_site_fract_y), \
                                                                "z" : float(atom._atom_site_fract_z)})


    sym_ops = cb.GetLoop("_symmetry_equiv_pos_as_xyz")
    
    sym_ops_list = []
    
    for sym in sym_ops:
        
        sym_ops_list.append(parse_symmetry(sym._symmetry_equiv_pos_as_xyz))

    
    au2ang = 0.52917721092
    a = float(cb["_cell_length_a"])
    b = float(cb["_cell_length_b"])
    c = float(cb["_cell_length_c"])
    alpha = float(cb["_cell_angle_alpha"]) * math.pi / 180
    beta = float(cb["_cell_angle_beta"]) * math.pi / 180
    gamma = float(cb["_cell_angle_gamma"]) * math.pi / 180

  
    x2 = b * math.cos(gamma)
    y2 = b * math.sin(gamma)
    x3 = c * math.cos(beta)
    y3 = (a * b * math.cos(alpha) - x2 * x3) / y2
    z3 = math.sqrt(c * c - x3 * x3 - y3 * y3)

    avec = [[a, 0, 0], 
            [x2, y2, 0], 
            [x3, y3, z3]] 

    
    fout = open("elk.in", "w")
    fout.write("avec\n")
    for i in range(3):
        fout.write("%18.10f %18.10f %18.10f\n"%(avec[i][0] / au2ang, avec[i][1] / au2ang, avec[i][2] / au2ang))
    fout.write("\n")
    fout.write("atoms\n")
    fout.write("%i\n"%len(initial_atoms_list.keys()))

    for key in initial_atoms_list.keys():
        atom_list = apply_symmetry(sym_ops_list, initial_atoms_list[key])
        fout.write("'%s.in'\n"%key)
        fout.write("%i\n"%len(atom_list))
        for a in atom_list:
            fout.write("%18.10f %18.10f %18.10f\n"%(a[0], a[1], a[2]))

    fout.close()

    return

if __name__ == "__main__":
    main()

