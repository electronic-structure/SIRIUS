import sirius
import json

sirius.initialize()


param = {
    "parameters" : {
        "electronic_structure_method" : "pseudopotential"
    }
}


M = [[5,0,0],[0,5,0],[0,0,5]]
Mcpp = sirius.matrix3d(M)
print(Mcpp)
print("Checkpoint 1 reached")

ctx = sirius.Simulation_context(json.dumps(param))
print("Checkpoint 2 reached")
print(ctx.unit_cell().get_symmetry())
ctx.unit_cell().set_lattice_vectors(Mcpp)
print("Checkpoint 3 reached")
ctx.unit_cell().add_atom_type("H", "o_lda_v1.2.uspp.F.UPF.json")
print("Checkpoint 4 reached")

ctx.unit_cell().atom_type(0).zn(1)
v=[0,0,0]
pos = sirius.vector3d(v)
print(pos)
ctx.unit_cell().add_atom("H", pos)
print("Checkpoint 5 reached")
ctx.initialize()
print("Checkpoint 6 reached")

sirius.finalize()
print("Checkpoint 7 reached") #error: Attempting to use an MPI routine after finalizing MPICH --> vanishes if we don't use this (destructor?).
