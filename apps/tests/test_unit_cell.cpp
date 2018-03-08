#include <sirius.h>

using namespace sirius;

void test1()
{
    Simulation_context ctx(mpi_comm_world(), "pseudopotential");
    ctx.set_processing_unit("cpu");
    
    int N = 7;
    double a{4};
    ctx.unit_cell().set_lattice_vectors({{N*a,0,0}, {0,N*a,0}, {0,0,N*a}});

    ctx.unit_cell().add_atom_type("A");
    
    auto& atype = ctx.unit_cell().atom_type(0);

    ctx.unit_cell().atom_type(0).zn(1);
    ctx.unit_cell().atom_type(0).set_radial_grid(radial_grid_t::lin_exp_grid, 1000, 0, 2);

    std::vector<double> beta(ctx.unit_cell().atom_type(0).num_mt_points());
    for (int i = 0; i < atype.num_mt_points(); i++) {
        double x = atype.radial_grid(i);
        beta[i] = std::exp(-x) * (4 - x * x);
    }
    ctx.unit_cell().atom_type(0).add_beta_radial_function(0, beta);
    ctx.unit_cell().atom_type(0).add_beta_radial_function(1, beta);
    ctx.unit_cell().atom_type(0).add_beta_radial_function(2, beta);

    for (int i1 = 0; i1 < N; i1++) {
        for (int i2 = 0; i2 < N; i2++) {
            for (int i3 = 0; i3 < N; i3++) {
                ctx.unit_cell().add_atom("A", {1.0 * i1 / N, 1.0 * i2 / N, 1.0 * i3 / N});
            }
        }
    }
    
    ctx.initialize();

    ctx.print_info();
    
    double vk[] = {0, 0, 0};
    K_point kp(ctx, vk, 1.0);

    kp.initialize();

    printf("num_gkvec: %i\n", kp.num_gkvec());

    for (int k = 0; k < 10; k++) {
        kp.beta_projectors().prepare();
        for (int ichunk = 0; ichunk < kp.beta_projectors().num_chunks(); ichunk++) {
            kp.beta_projectors().generate(ichunk);
        }
        kp.beta_projectors().dismiss();
    }
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize();

    test1();
    
    if (mpi_comm_world().rank() == 0) {
        sddk::timer::print();
    }

    sirius::finalize();
}
