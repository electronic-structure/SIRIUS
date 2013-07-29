#include <sirius.h>

int main(int argn, char** argv)
{
    using namespace sirius;
    
    Platform::initialize(1);
    
    Global parameters;

    parameters.set_lmax_apw(8);
    parameters.set_lmax_pot(8);
    parameters.set_lmax_rho(8);
    parameters.set_pw_cutoff(20.0);
    parameters.set_aw_cutoff(7.0);

    parameters.initialize(1, 1);
    
    parameters.print_info();
    
    Potential* potential = new Potential(parameters);
    potential->allocate();
        
    int ngridk[] = {1, 1, 1};
    int numkp = ngridk[0] * ngridk[1] * ngridk[2];
    int ik = 0;
    mdarray<double, 2> kpoints(3, numkp);
    std::vector<double> kpoint_weights(numkp);

    for (int i0 = 0; i0 < ngridk[0]; i0++) 
    {
        for (int i1 = 0; i1 < ngridk[1]; i1++) 
        {
            for (int i2 = 0; i2 < ngridk[2]; i2++)
            {
                kpoints(0, ik) = double(i0) / ngridk[0];
                kpoints(1, ik) = double(i1) / ngridk[1];
                kpoints(2, ik) = double(i2) / ngridk[2];
                kpoint_weights[ik] = 1.0 / numkp;
                ik++;
            }
        }
    }

    kset ks(parameters);
    ks.add_kpoints(kpoints, &kpoint_weights[0]);
    ks.initialize();
    ks.print_info();

    Density* density = new Density(parameters, potential);
    density->allocate();
    
    if (Utils::file_exists(storage_file_name))
    {
        density->load();
        potential->load();
    }
    else
    {
        density->initial_density(0);
        potential->generate_effective_potential(density->rho(), density->magnetization());
    }

    DFT_ground_state dft(&parameters, potential, density, &ks);
    dft.scf_loop();
    dft.forces();

    parameters.write_json_output();

    delete density;
    delete potential;
    
    parameters.clear();

    Timer::print();
}
