#include <sirius.h>

int main(int argn, char** argv)
{
    using namespace sirius;
    
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



        vector3d<double> p0(0.0, 0.0, 0);
        vector3d<double> p1(5, 0, 0);
        vector3d<double> p2(0, 5, 0);

        int N1 = 200;
        int N2 = 200;

        mdarray<double, 2> rho(N1, N2);
        rho.zero();

        for (int i1 = 0; i1 < N1 - 1; i1++)
        {
            for (int i2 = 0; i2 < N2 - 1; i2++)
            {
                vector3d<double> v;
                for (int x = 0; x < 3; x++) v[x] = p0[x] + double(i1) * p1[x] / N1 + double(i2) * p2[x] / N2;
                //auto vc = parameters.unit_cell()->get_cartesian_coordinates(v);
                rho(i1, i2) = density->rho()->value(parameters, v);
            }
        }

        HDF5_tree h5out("rho.h5", true);
        h5out.write("rho", rho);







        delete density;
        delete potential;
        
        parameters.clear();

        Timer::print();
    }

    Platform::barrier();
    Platform::finalize();
}
