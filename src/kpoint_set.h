
namespace sirius 
{

// TODO: this is something temporary; think how to handle k- point set

class kpoint_set
{
    private:
    
        std::vector<kpoint*> kpoints_;

        MPIGrid& mpi_grid_;
        
    public:
        kpoint_set(MPIGrid& mpi_grid__) : mpi_grid_(mpi_grid__)
        {
        }
        
        void add_kpoint(double* vk, double weight, Global& parameters)
        {
            //for (int ik = 0; ik < (int)kpoints_.size(); ik++)
            //{
            //    double t[3];
            //    for (int x = 0; x < 3; x++) 
            //        t[x] = vk[x] - kpoints_[ik]->vk()[x];

            //    if (vector_length(t) < 1e-10)
            //        error(__FILE__, __LINE__, "kpoint is already in list");
            //}

            // TODO: kill mpi subgrid here (in the same file where it was allocated!!)
            kpoints_.push_back(new kpoint(parameters, vk, weight));

            std::vector<double> initial_occupancies(parameters.num_bands(), 0.0);

            // in case of non-magnetic, or magnetic non-collinear case occupy first N bands
            if (parameters.num_mag_dims() == 0 || parameters.num_mag_dims() == 3)
            {
                int m = parameters.num_valence_electrons() / parameters.max_occupancy();
                for (int i = 0; i < m; i++)
                    initial_occupancies[i] = double(parameters.max_occupancy());
                initial_occupancies[m] = double(parameters.num_valence_electrons() % parameters.max_occupancy());
            }
            else // otherwise occupy up and down bands
            {
                int m = parameters.num_valence_electrons() / 2;
                for (int i = 0; i < m; i++)
                    initial_occupancies[i] = initial_occupancies[i + parameters.num_fv_states()] = 1.0;
                initial_occupancies[m] = initial_occupancies[m + parameters.num_fv_states()] = 
                    0.5 * parameters.num_valence_electrons() - double(m);
            }

            kpoints_.back()->set_band_occupancies(&initial_occupancies[0]);
        }

   
        inline kpoint* operator[](int i)
        {
            assert(i >= 0 && i < (int)kpoints_.size());
            
            return kpoints_[i];
        }

        void clear()
        {
            for (int ik = 0; ik < (int)kpoints_.size(); ik++)
                delete kpoints_[ik];
            
            kpoints_.clear();
        }
        
        inline int num_kpoints()
        {
            return (int)kpoints_.size();
        }

        void sync_band_energies(int num_bands, splindex<block>& spl_num_kpoints)
        {
            mdarray<double, 2> band_energies(num_bands, num_kpoints());
            band_energies.zero();
            
            // assume that side processors store the full e(k) array
            if (mpi_grid_.side(1 << 0))
                for (int ikloc = 0; ikloc < spl_num_kpoints.local_size(); ikloc++)
                {
                    int ik = spl_num_kpoints[ikloc];
                    kpoints_[ik]->get_band_energies(&band_energies(0, ik));
                }

            //mpi_grid.reduce(&band_energies(0, 0), num_bands * num_kpoints(), 1 << 0, true);

            //Platform::bcast(&band_energies(0, 0), num_bands * num_kpoints(), mpi_grid.world_root());

            Platform::allreduce(&band_energies(0, 0), num_bands * num_kpoints());

            for (int ik = 0; ik < num_kpoints(); ik++)
                kpoints_[ik]->set_band_energies(&band_energies(0, ik));
        }


};

};

