
namespace sirius 
{

// TODO: this is something temporary; think how to handle k- point set

class kpoint_set
{
    private:
    
        Global& parameters_;

        std::vector<kpoint*> kpoints_;

        splindex<block> spl_num_kpoints_;

        Band* band_;

    public:

        kpoint_set(Global& parameters__) : parameters_(parameters__)
        {
            band_ = new Band(parameters_);
        }

        ~kpoint_set()
        {
            clear();
            delete band_;
        }
        
        void add_kpoint(double* vk__, double weight__)
        {
            kpoints_.push_back(new kpoint(parameters_, vk__, weight__));
        }

        void add_kpoints(mdarray<double, 2>& kpoints__, double* weights__)
        {
            for (int ik = 0; ik < kpoints__.size(1); ik++) add_kpoint(&kpoints__(0, ik), weights__[ik]);
        }

        void initialize()
        {
            // ============================================================
            // distribute k-points along the 1-st dimension of the MPI grid
            // ============================================================
            spl_num_kpoints_.split(num_kpoints(), parameters_.mpi_grid().dimension_size(0), 
                                   parameters_.mpi_grid().coordinate(0));

            for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
                kpoints_[spl_num_kpoints_[ikloc]]->initialize(band_);
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

        void sync_band_energies()
        {
            mdarray<double, 2> band_energies(parameters_.num_bands(), num_kpoints());
            band_energies.zero();
            
            // assume that side processors store the full e(k) array
            if (parameters_.mpi_grid().side(1 << 0))
            {
                for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
                {
                    int ik = spl_num_kpoints_[ikloc];
                    kpoints_[ik]->get_band_energies(&band_energies(0, ik));
                }
            }

            Platform::allreduce(&band_energies(0, 0), parameters_.num_bands() * num_kpoints());

            for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_energies(&band_energies(0, ik));
        }

        Band* band()
        {
            return band_;
        }
        
        inline splindex<block>& spl_num_kpoints()
        {
            return spl_num_kpoints_;
        }
        
        inline int spl_num_kpoints(int ikloc)
        {
            return spl_num_kpoints_[ikloc];
        }

        void save_wave_functions()
        {
            if (Platform::mpi_rank() == 0)
            {
                hdf5_tree fout("sirius.h5", false);
                fout["parameters"].write("num_kpoints", num_kpoints());
                fout["parameters"].write("num_bands", parameters_.num_bands());
                fout["parameters"].write("num_spins", parameters_.num_spins());
            }

            if (parameters_.mpi_grid().side(1 << 0 | 1 << band_->dim_col()))
            {
                for (int ik = 0; ik < num_kpoints(); ik++)
                {
                    int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
                    
                    if (parameters_.mpi_grid().coordinate(0) == rank) kpoints_[ik]->save_wave_functions(ik, band_);
                    
                    parameters_.mpi_grid().barrier(1 << 0 | 1 << band_->dim_col());
                }
            }
        }

        void load_wave_functions()
        {
            hdf5_tree fin("sirius.h5", false);
            int num_spins;
            fin["parameters"].read("num_spins", &num_spins);
            if (num_spins != parameters_.num_spins()) error(__FILE__, __LINE__, "wrong number of spins");

            int num_bands;
            fin["parameters"].read("num_bands", &num_bands);
            if (num_bands != parameters_.num_bands()) error(__FILE__, __LINE__, "wrong number of bands");
            
            int num_kpoints_in;
            fin["parameters"].read("num_kpoints", &num_kpoints_in);

            // ==================================================================
            // index of current k-points in the hdf5 file, which (in general) may 
            // contain a different set of k-points 
            // ==================================================================
            std::vector<int> ikidx(num_kpoints(), -1); 
            // read available k-points
            double vk_in[3];
            for (int jk = 0; jk < num_kpoints_in; jk++)
            {
                fin["kpoints"][jk].read("coordinates", vk_in, 3);
                for (int ik = 0; ik < num_kpoints(); ik++)
                {
                    double dvk[3]; 
                    for (int x = 0; x < 3; x++) dvk[x] = vk_in[x] - kpoints_[ik]->vk(x);
                    if (Utils::vector_length(dvk) < 1e-12)
                    {
                        ikidx[ik] = jk;
                        break;
                    }
                }
            }

            for (int ik = 0; ik < num_kpoints(); ik++)
            {
                int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
                
                if (parameters_.mpi_grid().coordinate(0) == rank) kpoints_[ik]->load_wave_functions(ikidx[ik], band_);
            }
        }
};

};

