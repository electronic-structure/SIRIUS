import sirius
import json
import numpy

def make_new_ctx(pw_cutoff, gk_cutoff):
# lattice vectors
    a0 = 5.31148326730872
    lat = numpy.array(
        [[0.0, a0, a0],
         [a0, 0.0, a0],
         [a0, a0, 0.0]
    ])
# basic input parameters
    inp={
        "parameters" : {
            "xc_functionals" : ["XC_LDA_X", "XC_LDA_C_PZ"],
            "electronic_structure_method" : "pseudopotential",
            "pw_cutoff" : pw_cutoff,
            "gk_cutoff" : gk_cutoff
        }
    }
# create simulation context
    ctx = sirius.Simulation_context(json.dumps(inp))
# set lattice vectors
    ctx.unit_cell().set_lattice_vectors(*lat)
# add atom type
    ctx.unit_cell().add_atom_type('Si','Si.json')
# add atoms
    ctx.unit_cell().add_atom('Si', [0.0,0.0,0.0])
    ctx.unit_cell().add_atom('Si', [0.25,0.25,0.25])
# intialize and return simulation context
    ctx.initialize()
    return ctx

def main():
    pw_cutoff = 20 # in a.u.^-1
    gk_cutoff = 7 # in a.u.^-1
    ctx = make_new_ctx(pw_cutoff, gk_cutoff)
    k = 2
    kgrid = sirius.K_point_set(ctx, [k,k,k], [0,0,0], True)
    dft = sirius.DFT_ground_state(kgrid)
    dft.initial_state()
    result = dft.find(1e-6, 1e-6, 1e-2, 100, False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
