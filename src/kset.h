
namespace sirius 
{

// TODO: this is something temporary; think how to handle k- point set

class kset
{
    private:
    
        Global& parameters_;

        //Band* band_;

        std::vector<kpoint*> kpoints_;

        splindex<block> spl_num_kpoints_;

    public:

        kset(Global& parameters__) : parameters_(parameters__)
        {
        }

        ~kset()
        {
            clear();
        }
        
        /// Initialize the k-point set
        void initialize(Band* band);
        
        /// Find eigen-states
        /** Radial functions, radial integrals and plane-wave coefficients of V(r)*Theta(r) must be already 
            calculated. */
        void find_eigen_states(Band* band, Potential* potential);

        /// Return sum of valence eigen-values
        double valence_eval_sum();

        void add_kpoint(double* vk__, double weight__)
        {
            kpoints_.push_back(new kpoint(parameters_, vk__, weight__));
        }

        void add_kpoints(mdarray<double, 2>& kpoints__, double* weights__)
        {
            for (int ik = 0; ik < kpoints__.size(1); ik++) add_kpoint(&kpoints__(0, ik), weights__[ik]);
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

        inline splindex<block>& spl_num_kpoints()
        {
            return spl_num_kpoints_;
        }
        
        inline int spl_num_kpoints(int ikloc)
        {
            return spl_num_kpoints_[ikloc];
        }

        void save_wave_functions(Band* band)
        {
            if (Platform::mpi_rank() == 0)
            {
                hdf5_tree fout("sirius.h5", false);
                fout["parameters"].write("num_kpoints", num_kpoints());
                fout["parameters"].write("num_bands", parameters_.num_bands());
                fout["parameters"].write("num_spins", parameters_.num_spins());
            }

            if (parameters_.mpi_grid().side(1 << 0 | 1 << band->dim_col()))
            {
                for (int ik = 0; ik < num_kpoints(); ik++)
                {
                    int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
                    
                    if (parameters_.mpi_grid().coordinate(0) == rank) kpoints_[ik]->save_wave_functions(ik, band);
                    
                    parameters_.mpi_grid().barrier(1 << 0 | 1 << band->dim_col());
                }
            }
        }

        void load_wave_functions(Band* band)
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
                
                if (parameters_.mpi_grid().coordinate(0) == rank) kpoints_[ik]->load_wave_functions(ikidx[ik], band);
            }
        }
};

void kset::initialize(Band* band)
{
    // ============================================================
    // distribute k-points along the 1-st dimension of the MPI grid
    // ============================================================
    spl_num_kpoints_.split(num_kpoints(), parameters_.mpi_grid().dimension_size(0), 
                           parameters_.mpi_grid().coordinate(0));

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
        kpoints_[spl_num_kpoints_[ikloc]]->initialize(band);
}

void kset::find_eigen_states(Band* band, Potential* potential)
{
    Timer t("sirius::kset::find_eigen_states");
    
    // solve secular equation and generate wave functions
    for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = spl_num_kpoints(ikloc);
        kpoints_[ik]->find_eigen_states(band, potential->effective_potential(), potential->effective_magnetic_field());
    }

    // synchronize eigen-values
    sync_band_energies();

    if (Platform::mpi_rank() == 0)
    {
        printf("Lowest band energies\n");
        for (int ik = 0; ik < num_kpoints(); ik++)
        {
            printf("ik : %2i, ", ik); 
            for (int j = 0; j < std::min(10, parameters_.num_bands()); j++) 
                printf("%12.6f", kpoints_[ik]->band_energy(j));
            printf("\n");
        }
    }
    
    // compute eigen-value sums
    double eval_sum = 0.0;
    for (int ik = 0; ik < num_kpoints(); ik++)
    {
        double wk = kpoints_[ik]->weight();
        for (int j = 0; j < parameters_.num_bands(); j++)
            eval_sum += wk * kpoints_[ik]->band_energy(j) * kpoints_[ik]->band_occupancy(j);
    }
    
    parameters_.rti().valence_eval_sum = eval_sum;
}

double kset::valence_eval_sum()
{
    double eval_sum = 0.0;

    for (int ik = 0; ik < num_kpoints(); ik++)
    {
        double wk = kpoints_[ik]->weight();
        for (int j = 0; j < parameters_.num_bands(); j++)
            eval_sum += wk * kpoints_[ik]->band_energy(j) * kpoints_[ik]->band_occupancy(j);
    }

    return eval_sum;
}

};

