
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
            kpoints_.push_back(new kpoint(parameters, vk, weight));
        }
   
        inline kpoint* operator[](int i)
        {
            assert(i >= 0 && i < (int)kpoints_.size());
            
            return kpoints_[i];
        }

        void clear()
        {
            for (int ik = 0; ik < (int)kpoints_.size(); ik++) delete kpoints_[ik];
            
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
            {
                for (int ikloc = 0; ikloc < spl_num_kpoints.local_size(); ikloc++)
                {
                    int ik = spl_num_kpoints[ikloc];
                    kpoints_[ik]->get_band_energies(&band_energies(0, ik));
                }
            }

            Platform::allreduce(&band_energies(0, 0), num_bands * num_kpoints());

            for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_energies(&band_energies(0, ik));
        }
};

};

