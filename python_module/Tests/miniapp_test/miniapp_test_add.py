import sys
sys.path.insert(0, '../../')

import sirius
import json
from bands import plotter


sirius.initialize()

parameters1 = {
    "control" : {
        "cyclic_block_size" : 16,
        "processing_unit" : "cpu",
        "std_evp_solver_type" : "lapack",
        "gen_evp_solver_type" : "lapack",
        "verbosity" : 2,
        "verification" : 0,
        "print_memory_usage" : False,
        "print_checksum" : False,
        "print_forces" : True,
        "print_stress" : True
    },

    "!settings" : {
      "always_update_wf" : False
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

        "num_dft_iter" : 100,

        "ngridk" : [2,2,2],
        "gamma_point" : False
    },


    "iterative_solver" : {
        "!energy_tolerance" : 1e-2,
        "!residual_tolerance" : 1e-6,
        "num_steps" : 20,
        "subspace_size" : 4,
        "type" : "davidson",
        "converge_by_energy" : 1,
        "!orthogonalize" : False,
        "!init_subspace" : "lcao",
        "init_eval_old" : False
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
    },

    "kpoints_rel": {
    "K": [
      0.375,
      0.375,
      0.75
    ],
    "L": [
      0.5,
      0.5,
      0.5
    ],
    "U": [
      0.625,
      0.25,
      0.625
    ],
    "W": [
      0.5,
      0.25,
      0.75
    ],
    "X": [
      0.5,
      0.0,
      0.5
    ],
    "GAMMA": [
      0.0,
      0.0,
      0.0
    ],
    "W_2": [
      0.75,
      0.25,
      0.5
    ]
  },

  "kpoints_path" : ["GAMMA", "K", "L"]

}

param2 = {
    "control" : {
        "processing_unit" : "cpu",
        "std_evp_solver_type" : "lapack",
        "gen_evp_solver_type" : "lapack",
        "verbosity" : 2,
        "print_forces" : True,
        "print_stress" : True
    },

    "parameters" : {
        "electronic_structure_method" : "full_potential_lapwlo",

        "num_fv_states" : 56,

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


   "unit_cell" : {

        "lattice_vectors" : [ [1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]
                            ],
        "lattice_vectors_scale" : 7.260327248,

        "atom_types" : ["Sr", "V", "O"],

        "atom_files" : {
            "Sr" : "/Users/colinkalin/my_SIRIUS/examples/old/2.SrVO3_fp/Sr.json",
            "V"  : "/Users/colinkalin/my_SIRIUS/examples/old/2.SrVO3_fp/V.json",
            "O"  : "/Users/colinkalin/my_SIRIUS/examples/old/2.SrVO3_fp/O.json"
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
    },

"kpoints_rel": {
    "K": [
      0.375,
      0.375,
      0.75
    ],
    "L": [
      0.5,
      0.5,
      0.5
    ],
    "U": [
      0.625,
      0.25,
      0.625
    ],
    "W": [
      0.5,
      0.25,
      0.75
    ],
    "X": [
      0.5,
      0.0,
      0.5
    ],
    "GAMMA": [
      0.0,
      0.0,
      0.0
    ],
    "W_2": [
      0.75,
      0.25,
      0.5
    ]
  },

  "kpoints_path" : ["GAMMA", "K", "L"]
}




print("Checkpoint 1 reached")

ctx = sirius.Simulation_context(json.dumps(parameters1))
ctx.initialize()
print("Checkpoint 2 reached")

ctx.parameters_input() #not needed acutally, just for debugging.
print("Checkpoint 3 reached")

dft = sirius.DFT_ground_state(ctx)
dft.initial_state()
result = dft.find(ctx.parameters_input().potential_tol_, ctx.parameters_input().energy_tol_, ctx.parameters_input().num_dft_iter_, True)
dft.print_magnetic_moment()

#s = sirius.Stress(ctx, dft.k_point_set(), dft.density(), dft.potential())
print("Checkpoint 4 reached")
#s.calc_stress_total()
print("Checkpoint 5 reached")
#s.print_info()
print("Checkpoint 6 reached")

#f = sirius.Force(ctx, dft.density(), dft.potential(), dft.hamiltonian(), dft.k_point_set())
print("Checkpoint 7 reached")
#f.calc_forces_total()
#f.print_info()

sirius.finalize()
