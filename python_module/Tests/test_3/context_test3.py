import sirius
import json

sirius.initialize()

param = {
    "control" : {
        "processing_unit" : "cpu",
        "std_evp_solver_type" : "lapack",
        "gen_evp_solver_type" : "lapack",
        "verbosity" : 0,
        "print_forces" : True,
        "print_stress" : True
    },

    "parameters" : {
        "electronic_structure_method" : "pseudopotential",

        "num_fv_states" : 40,

        "xc_functionals" : ["XC_LDA_X", "XC_LDA_C_PZ"],

        "smearing_width" : 0.025,

        "use_symmetry" : True,

        "num_mag_dims" : 0,

        "gk_cutoff" : 6.0,
        "pw_cutoff" : 20.00,

        "energy_tol" : 1e-8,
        "potential_tol" : 1e-8,

        "num_dft_iter" : 4,

        "ngridk" : [2,2,2]
    },


    "iterative_solver" : {
        "type" : "davidson",
        "min_occupancy" : 1e-5
    },


    "unit_cell" : {

        "lattice_vectors" : [ [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                            ],
        "lattice_vectors_scale" : 7.260327248,

        "atom_types" : ["Sr", "V", "O"],

        "atom_files" : {
            "Sr" : "sr_lda_v1.uspp.F.UPF.json",
            "V"  : "v_lda_v1.4.uspp.F.UPF.json",
            "O"  : "o_lda_v1.2.uspp.F.UPF.json"
        },

        "atoms" : {
            "Sr" : [
                [0.5, 0.5, 0.5]
            ],
            "V" : [
                [0, 0, 0, 0, 0, 4]
            ],
            "O" : [
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5]
            ]
        }
    },

    "mixer" : {
        "beta" : 0.95,
        "type" : "broyden1",
        "max_history" : 8
    }

}
print("Checkpoint 1 reached")

ctx = sirius.Simulation_context(json.dumps(param))
ctx.initialize()
print("Checkpoint 2 reached")

ctx.parameters_input() #not needed acutally, just for debugging.
print("Checkpoint 3 reached")

dft = sirius.DFT_ground_state(ctx)
print("Checkpoint 4 reached")

#potential = sirius.Potential(dft.potential())
print("Checkpoint 5 reached")

#density = sirius.Density(dft.density())
print("Checkpoint 6 reached")

dft.initial_state()
print("Checkpoint 7 reached")

result = dft.find(ctx.parameters_input().potential_tol_, ctx.parameters_input().energy_tol_, ctx.parameters_input().num_dft_iter_, True)
dft.print_magnetic_moment()
print("Checkpoint 8 reached")

ks = sirius.K_point_set(ctx)
ks.add_kpoint([0,0,0])
ks.add_kpoint([0.1,0.1,0.1])
ks.add_kpoint([0.5,0.5,0.5])
ks.initialize()

band = dft.band()
band.solve(ks, dft.Hamiltonian(), True)

e = ks.get_energies()

plot(e)

dft = None
ctx = None
#dict = {}
#dict["ground_state"] = dft.serialize()

#print(dict)


sirius.finalize()
print("Checkpoint 9 reached")
