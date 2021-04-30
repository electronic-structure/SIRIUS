#include <sirius.hpp>

using namespace sirius;

void test_davidson(cmd_args const& args__)
{
    auto pw_cutoff    = args__.value<double>("pw_cutoff", 30);
    auto gk_cutoff    = args__.value<double>("gk_cutoff", 10);
    auto N            = args__.value<int>("N", 1);
    auto mpi_grid     = args__.value("mpi_grid", std::vector<int>({1, 1}));
    auto solver       = args__.value<std::string>("solver", "lapack");
    auto xc_name      = args__.value<std::string>("xc_name", "XC_LDA_X");
    auto num_mag_dims = args__.value<int>("num_mag_dims", 0);

    bool add_dion{false};
    bool add_vloc{false};

    PROFILE_START("test_davidson|setup")

    /* create simulation context */
    Simulation_context ctx(
        "{"
        "   \"parameters\" : {"
        "        \"electronic_structure_method\" : \"pseudopotential\""
        "    },"
        "   \"control\" : {"
        "       \"verification\" : 0"
        "    }"
        "}");

    /* add a new atom type to the unit cell */
    auto& atype = ctx.unit_cell().add_atom_type("Cu");
    /* set pseudo charge */
    atype.zn(11);
    /* set radial grid */
    atype.set_radial_grid(radial_grid_t::lin_exp, 1000, 0.0, 100.0, 6);
    /* cutoff at ~1 a.u. */
    int icut = atype.radial_grid().index_of(1.0);
    double rcut = atype.radial_grid(icut);
    /* create beta radial function */
    std::vector<double> beta(icut + 1);
    std::vector<double> beta1(icut + 1);
    for (int l = 0; l <= 2; l++) {
        for (int i = 0; i <= icut; i++) {
            double x = atype.radial_grid(i);
            beta[i] = utils::confined_polynomial(x, rcut, l, l + 1, 0);
            beta1[i] = utils::confined_polynomial(x, rcut, l, l + 2, 0);
        }
        /* add radial function for l */
        atype.add_beta_radial_function(l, beta);
        atype.add_beta_radial_function(l, beta1);
    }

    std::vector<double> ps_wf(atype.radial_grid().num_points());
    for (int l = 0; l <= 2; l++) {
        for (int i = 0; i < atype.radial_grid().num_points(); i++) {
            double x = atype.radial_grid(i);
            ps_wf[i] = std::exp(-x) * std::pow(x, l);
        }
        /* add radial function for l */
        atype.add_ps_atomic_wf(3, sirius::experimental::angular_momentum(l), ps_wf);
    }

    /* set local part of potential */
    std::vector<double> vloc(atype.radial_grid().num_points(), 0);
    if (add_vloc) {
        for (int i = 0; i < atype.radial_grid().num_points(); i++) {
            double x = atype.radial_grid(i);
            vloc[i] = -atype.zn() / (std::exp(-x * (x + 1)) + x);
        }
    }
    atype.local_potential(vloc);
    /* set Dion matrix */
    int nbf = atype.num_beta_radial_functions();
    matrix<double> dion(nbf, nbf);
    dion.zero();
    if (add_dion) {
        for (int i = 0; i < nbf; i++) {
            dion(i, i) = -10.0;
        }
    }
    atype.d_mtrx_ion(dion);
    /* set atomic density */
    std::vector<double> arho(atype.radial_grid().num_points());
    for (int i = 0; i < atype.radial_grid().num_points(); i++) {
        double x = atype.radial_grid(i);
        arho[i] = 2 * atype.zn() * std::exp(-x * x) * x;
    }
    atype.ps_total_charge_density(arho);

    /* lattice constant */
    double a{5};
    /* set lattice vectors */
    ctx.unit_cell().set_lattice_vectors({{a * N, 0, 0},
                                         {0, a * N, 0},
                                         {0, 0, a * N}});
    /* add atoms */
    double p = 1.0 / N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                ctx.unit_cell().add_atom("Cu", {i * p, j * p, k * p}, {0, 0, 1});
            }
        }
    }

    /* initialize the context */
    ctx.verbosity(2);
    ctx.pw_cutoff(pw_cutoff);
    ctx.gk_cutoff(gk_cutoff);
    ctx.processing_unit(args__.value<std::string>("device", "CPU"));
    ctx.mpi_grid_dims(mpi_grid);
    ctx.gen_evp_solver_name(solver);
    ctx.std_evp_solver_name(solver);
    ctx.set_num_mag_dims(num_mag_dims);
    ctx.add_xc_functional(xc_name);

    PROFILE_STOP("test_davidson|setup")

    /* initialize simulation context */
    ctx.initialize();

    Density rho(ctx);
    rho.initial_density();
    rho.print_info();

    check_xc_potential(rho);
}

int main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"device=", "(string) CPU or GPU"},
                               {"pw_cutoff=", "(double) plane-wave cutoff for density and potential"},
                               {"gk_cutoff=", "(double) plane-wave cutoff for wave-functions"},
                               {"N=", "(int) cell multiplicity"},
                               {"mpi_grid=", "(int[2]) dimensions of the MPI grid for band diagonalization"},
                               {"solver=", "eigen-value solver"},
                               {"xc_name=", "name of XC potential"},
                               {"num_mag_dims=", "number of magnetic dimensions"}
                              });

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test_davidson(args);
    //int rank = Communicator::world().rank();
    sirius::finalize();
    //if (rank == 0)  {
    //    const auto timing_result = ::utils::global_rtgraph_timer.process();
    //    std::cout<< timing_result.print();
    //}
}
