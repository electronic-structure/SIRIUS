#include <sirius.h>

int main(int argn, char** argv)
{
    using namespace sirius;
    
    Platform::initialize(1);
    
    Global parameters;

    JSON_tree parser("sirius.json");
    std::vector<double> a0 = parser["lattice_vectors"][0].get(std::vector<double>(3, 0)); 
    std::vector<double> a1 = parser["lattice_vectors"][1].get(std::vector<double>(3, 0)); 
    std::vector<double> a2 = parser["lattice_vectors"][2].get(std::vector<double>(3, 0));

    double scale =  parser["lattice_vectors_scale"].get(1.0);
    for (int x = 0; x < 3; x++)
    {
        a0[x] *= scale;
        a1[x] *= scale;
        a2[x] *= scale;
    }
    parameters.set_lattice_vectors(&a0[0], &a1[0], &a2[0]);

    parameters.set_lmax_apw(parser["lmax_apw"].get(10));
    parameters.set_lmax_pot(parser["lmax_pot"].get(10));
    parameters.set_lmax_rho(parser["lmax_rho"].get(10));
    parameters.set_pw_cutoff(parser["pw_cutoff"].get(20.0));
    parameters.set_aw_cutoff(parser["aw_cutoff"].get(7.0));
    
    for (int iat = 0; iat < parser["atoms"].size(); iat++)
    {
        std::string label;
        parser["atoms"][iat][0] >> label;
        parameters.add_atom_type(iat, label);
        for (int ia = 0; ia < parser["atoms"][iat][1].size(); ia++)
        {
            std::vector<double> v;
            parser["atoms"][iat][1][ia] >> v;

            if (!(v.size() == 3 || v.size() == 6)) error_global(__FILE__, __LINE__, "wrong coordinates size");
            if (v.size() == 3) v.resize(6, 0.0);
            
            parameters.add_atom(iat, &v[0], &v[3]);
        }
    }

    parameters.set_auto_rmt(parser["auto_rmt"].get(0));
    int num_mag_dims = parser["num_mag_dims"].get(0);
    int num_spins = (num_mag_dims == 0) ? 1 : 2;
    
    parameters.set_num_mag_dims(num_mag_dims);
    parameters.set_num_spins(num_spins);

    parameters.initialize();
    
    Potential* potential = new Potential(parameters);
    potential->allocate();

    std::vector<int> ngridk = parser["ngridk"].get(std::vector<int>(3, 1));
        
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

    K_set ks(parameters);
    ks.add_kpoints(kpoints, &kpoint_weights[0]);
    ks.initialize();

    Density* density = new Density(parameters);
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

    DFT_ground_state dft(parameters, potential, density, &ks);
    double charge_tol = parser["charge_tol"].get(1e-4);
    double energy_tol = parser["energy_tol"].get(1e-4);

    dft.scf_loop(charge_tol, energy_tol);

    //dft.relax_atom_positions();

    //parameters.write_json_output();

    delete density;
    delete potential;
    
    parameters.clear();

    Timer::print();

    Platform::barrier();
    Platform::finalize();
}
