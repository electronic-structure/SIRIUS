import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../miniapp_test')

import sirius
import json
import copy
from bands import plotter
from bands import get_kpoint_path

baseparameters = {
  "control" : {
      "cyclic_block_size" : 16,
      "processing_unit" : "cpu",
      "std_evp_solver_type" : "lapack",
      "gen_evp_solver_type" : "lapack",
      "verbosity": 2
  },

  "parameters" : {

    "xc_functionals" : ["XC_LDA_X", "XC_LDA_C_PZ"],

    "smearing_width" : 0.02,

    "use_symmetry" : True,

    "num_mag_dims" : 0,

    "gk_cutoff" : 6.0,
    "pw_cutoff" : 25.00,

    "energy_tol" : 1e-8,

    "num_dft_iter" : 2,

    "ngridk" : [8,8,8],

    "reduce_gvec": 0
  },



    "unit_cell" : {

        "lattice_vectors" : [ [0, 0.5, 0.5],
                              [0.5, 0, 0.5],
                              [0.5, 0.5, 0]
                            ],
        "lattice_vectors_scale" : 6.73,

        "atom_types" : ["Cu"],

        "atoms" : {
            "Cu" : [
                [0.0, 0.0, 0.0]
            ]
        }
    },

    "mixer" : {
        "beta" : 0.9,
        "type" : "broyden1",
        "max_history" : 8
    },

    "kpoints_rel": {
    "W'": [
      -0.5,
      -0.25,
      -0.75
    ],
    "L'": [
      -0.5,
      -0.5,
      -0.5
    ],
    "K'": [
      -0.375,
      -0.375,
      -0.75
    ],
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
    "X'": [
      -0.5,
      -0.0,
      -0.5
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
    "W_2'": [
      -0.75,
      -0.25,
      -0.5
    ],
    "X": [
      0.5,
      0.0,
      0.5
    ],
    "U'": [
      -0.625,
      -0.25,
      -0.625
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
    "kpoints_path" : ["L", "GAMMA", "X"]
}


def calculate_bands(param):
    print("Checkpoint 1 reached")
    ctx = sirius.Simulation_context(json.dumps(param))
    ctx.set_gamma_point(False)
    ctx.initialize()

    print("Checkpoint 2 reached")
    dft = sirius.DFT_ground_state(ctx)
    dft.initial_state()

    print("Checkpoint 3 reached")
    result = dft.find(1e-6, 1e-6, 100, False) #enter the tolerances directly.
    #print("result_dict=", result)
    dft.print_magnetic_moment()

    if param["parameters"]["electronic_structure_method"] == "pseudopotential":
        ctx.set_iterative_solver_tolerance(1e-12)

    print("Checkpoint 4 reached")
    potential = dft.potential() #sirius.Potential(ctx)
    H = dft.hamiltonian() #sirius.Hamiltonian(ctx, potential)

    density = dft.density()

    print("Checkpoint 5 reached")
    #print("Total Energy = ", dft.total_energy())


    print("Checkpoint 6 reached")
    t = 0

    band = dft.band()

    print("Checkpoint 7 reached")

    print("Checkpoint 8 reached")
    k_point_list = param["kpoints_path"]
    rec_vec = param["kpoints_rel"]

    k_points, x_ticks, x_axis = get_kpoint_path(k_point_list, rec_vec, ctx)
    ks = sirius.K_point_set(ctx, k_points) # create and initalize k-set

    if param["parameters"]["electronic_structure_method"] == "pseudopotential":
        band.initialize_subspace(ks, H)

    print("Checkpoint 9 reached")
    # after creating new k-point set along the symmetry lines
    band = dft.band()
    H = dft.hamiltonian()
    print("Checkpoint 10 reached")

    band.solve(ks, H, True)

    print("Calculation finished.")

    return ctx, ks, x_ticks, x_axis




def make_dict(ctx, ks, x_ticks, x_axis):
    dict = {}
    dict["header"] = {}
    dict["header"]["x_axis"] = x_axis
    dict["header"]["x_ticks"]=[]
    dict["header"]["num_bands"]=ctx.num_bands()
    dict["header"]["num_mag_dims"] = ctx.num_mag_dims()

    for e in enumerate(x_ticks):
        j = {}
        j["x"] = e[1][0]
        j["label"] = e[1][1]
        dict["header"]["x_ticks"].append(j)

    dict["bands"] = []

    for ik in range(ks.num_kpoints()):
        bnd_k = {}
        bnd_k["kpoint"] = [0.0,0.0,0.0]
        for x in range(3):
            bnd_k["kpoint"][x] = ks(ik).vk()(x)
            #if ik == 32:
                #print(bnd_k["kpoint"][x])
        bnd_e = []

        # TODO: simplify to bnd_e = new_ks.get_energies(ctx, ik)
        bnd_e = ks.get_band_energies(ik, 0)
        print(bnd_e)

        bnd_k["values"] = bnd_e
        dict["bands"].append(bnd_k)
    return dict

param_pp = copy.deepcopy(baseparameters)
param_pp["parameters"]["electronic_structure_method"] = "pseudopotential"
param_pp["iterative_solver"] = {
    "energy_tolerance" : 1e-2,
    "residual_tolerance" : 1e-6,
    "num_steps" : 20,
    "subspace_size" : 4,
    "type" : "davidson",
    "converge_by_energy" : 1
}
param_pp["unit_cell"]["atom_files"] = {
    "Cu" : "Cu.pz-dn-rrkjus_psl.0.2.UPF.json"
}


param_fp = copy.deepcopy(baseparameters)
param_fp["parameters"]["electronic_structure_method"] = "full_potential_lapwlo"
param_fp["unit_cell"]["atom_files"] = {
    "Cu" : "Cu.json",
}

sirius.initialize()
ctx, ks, x_ticks, x_axis = calculate_bands(param_pp)
ctx2, ks2, x_ticks2, x_axis2 = calculate_bands(param_fp)

dict1 = make_dict(ctx, ks, x_ticks, x_axis)
dict2 = make_dict(ctx2, ks2, x_ticks2, x_axis2)

plotter(dict1, "pseudopotential", dict2, "full_potential", True)
#plotter(dict2, "full_potential")
dft = None
ctx = None
ctx2 = None


sirius.finalize()
