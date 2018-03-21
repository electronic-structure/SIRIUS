import simulation_context as sc
import Gvec
import matrix3d_bind as m3d
import json

Gvec.initialize()


param = {
    "parameters" : {
        "electronic_structure_method" : "pseudopotential"
    }
}


M = [[5,0,0],[0,5,0],[0,0,5]]
Mcpp = m3d.matrix3d(M)
print(Mcpp)
print("Level 1 finished")

ctx = sc.Simulation_context(json.dumps(param))
print("Level 2 finished")
print(ctx.unit_cell().get_symmetry())
ctx.unit_cell().set_lattice_vectors(Mcpp)
print("Level 3 finished")
ctx.unit_cell().add_atom_type("H", "o_lda_v1.2.uspp.F.UPF.json")
print("Level 4 finished")

#atype = ctx.unit_cell().atom_type(0)
v=[0,0,0]
pos = m3d.vector3d(v)
print(pos)
ctx.unit_cell().add_atom("H", pos)
print("Level 5 finished")
ctx.initialize()
print("Level 6 finished")

Gvec.finalize()
print("Level 7 finished")
