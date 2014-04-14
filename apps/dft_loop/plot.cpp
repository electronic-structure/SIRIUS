#include <sirius.h>

int main(int argn, char** argv)
{
    using namespace sirius;

    cmd_args args;
    args.register_key("--origin=", "{vector3d} plot origin");
    args.register_key("--b1=", "{vector3d} 1st boundary vector");
    args.register_key("--b2=", "{vector3d} 2nd boundary vector");
    args.register_key("--N1=", "{int} number of 1st boundary vector divisions");
    args.register_key("--N2=", "{int} number of 2nd boundary vector divisions");
    args.register_key("--coordinates=", "{cart | frac} Cartesian or fractional coordinates");
    args.parse_args(argn, argv);
    
    if (argn == 1)
    {
        printf("Usage: ./plot [options] \n");
        args.print_help();
        exit(0);
    }

    vector3d<double> origin = args.value< vector3d<double> >("origin");
    vector3d<double> b1 = args.value< vector3d<double> >("b1");
    vector3d<double> b2 = args.value< vector3d<double> >("b2");

    int N1 = args.value<int>("N1");
    int N2 = args.value<int>("N2");


    std::string coords = args["coordinates"];
    bool cart;
    if (coords == "cart")
    {
        cart = true;
    }
    else if (coords == "frac")
    {
        cart = false;
    }
    else
    {
        terminate(__FILE__, __LINE__, "wrong type of coordinates");
        cart = false; // make compiler happy
    }

    
    Platform::initialize(1);

    {
        Global parameters;

        JSON_tree parser("sirius.json");

        parameters.read_unit_cell_input();

        parameters.set_lmax_apw(parser["lmax_apw"].get(10));
        parameters.set_lmax_pot(parser["lmax_pot"].get(10));
        parameters.set_lmax_rho(parser["lmax_rho"].get(10));
        parameters.set_pw_cutoff(parser["pw_cutoff"].get(20.0));
        parameters.set_aw_cutoff(parser["aw_cutoff"].get(7.0));
        parameters.set_gk_cutoff(parser["gk_cutoff"].get(7.0));
        
        parameters.unit_cell()->set_auto_rmt(parser["auto_rmt"].get(0));
        int num_mag_dims = parser["num_mag_dims"].get(0);
        int num_spins = (num_mag_dims == 0) ? 1 : 2;
        
        parameters.set_num_mag_dims(num_mag_dims);
        parameters.set_num_spins(num_spins);

        parameters.initialize();
        
        Potential* potential = new Potential(parameters);
        potential->allocate();

        Density* density = new Density(parameters);
        density->allocate();
        
        density->load();
        potential->load();

        density->generate_pw_coefs();

        splindex<block> spl_N2(N2, Platform::num_mpi_ranks(), Platform::mpi_rank());

        mdarray<double, 2> rho(N1, N2);
        
        Timer t1("compute_density");
        for (int j2 = 0; j2 < spl_N2.local_size(); j2++)
        {
            int i2 = spl_N2[j2];

            std::cout << "column " << i2 << " out of " << N2 << std::endl;

            #pragma omp parallel for
            for (int i1 = 0; i1 < N1; i1++)
            {
                vector3d<double> v;
                for (int x = 0; x < 3; x++) v[x] = origin[x] + double(i1) * b1[x] / (N1 - 1) + 
                                                               double(i2) * b2[x] / (N2 - 1);

                if (!cart) v = parameters.unit_cell()->get_cartesian_coordinates(v);

                rho(i1, i2) = density->rho()->value(parameters, v);
            }
        }
        t1.stop();

        Platform::allgather(&rho(0, 0), spl_N2.global_offset() * N1, spl_N2.local_size() * N1, MPI_COMM_WORLD);

        if (Platform::mpi_rank() == 0)
        {
            HDF5_tree h5out("rho.h5", true);
            h5out.write("rho", rho);
        }

        delete density;
        delete potential;
        
        parameters.clear();

        Timer::print();
    }

    Platform::barrier();
    Platform::finalize();
}
