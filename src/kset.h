
namespace sirius 
{

// TODO: this is something temporary; think how to handle k- point set

class kset
{
    private:
    
        Global& parameters_;

        Band* band_;

        std::vector<kpoint*> kpoints_;

        splindex<block> spl_num_kpoints_;

    public:

        kset(Global& parameters__) : parameters_(parameters__)
        {
            band_ = new Band(parameters_);
        }

        ~kset()
        {
            clear();
            delete band_;
        }
        
        /// Initialize the k-point set
        void initialize();

        void update();
        
        /// Find eigen-states
        void find_eigen_states(Potential* potential, bool precompute);

        /// Find Fermi energy and band occupation numbers
        void find_band_occupancies();

        /// Return sum of valence eigen-values
        double valence_eval_sum();

        void print_info();

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

        Band* band()
        {
            return band_;
        }
        
        void set_band_occupancies(int ik, double* band_occupancies)
        {
            kpoints_[ik]->set_band_occupancies(band_occupancies);
        }
        
        void get_band_energies(int ik, double* band_energies)
        {
            kpoints_[ik]->get_band_energies(band_energies);
        }
        
        void get_band_occupancies(int ik, double* band_occupancies)
        {
            kpoints_[ik]->get_band_occupancies(band_occupancies);
        }

        int max_num_gkvec()
        {
            int max_num_gkvec_ = 0;
            for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
            {
                int ik = spl_num_kpoints_[ikloc];
                max_num_gkvec_ = std::max(max_num_gkvec_, kpoints_[ik]->num_gkvec());
            }
            Platform::allreduce<op_max>(&max_num_gkvec_, 1);
            return max_num_gkvec_;
        }

        void force(mdarray<double, 2>& forcek)
        {
            mdarray<double, 2> ffac(parameters_.num_gvec_shells(), parameters_.num_atom_types());
            parameters_.get_step_function_form_factors(ffac);

            forcek.zero();

            for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
            {
                 kpoints_[spl_num_kpoints_[ikloc]]->ibs_force<cpu, apwlo>(band_, ffac, forcek);
            }
            Platform::allreduce(&forcek(0, 0), (int)forcek.size());
        }

            
};

void kset::initialize()
{
    // ============================================================
    // distribute k-points along the 1-st dimension of the MPI grid
    // ============================================================
    spl_num_kpoints_.split(num_kpoints(), parameters_.mpi_grid().dimension_size(0), 
                           parameters_.mpi_grid().coordinate(0));

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
        kpoints_[spl_num_kpoints_[ikloc]]->initialize(band_);
}

void kset::update()
{
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++)
        kpoints_[spl_num_kpoints_[ikloc]]->init_gkvec_phase_factors();
}

void kset::find_eigen_states(Potential* potential, bool precompute)
{
    Timer t("sirius::kset::find_eigen_states");
    
    if (precompute)
    {
        potential->generate_pw_coefs();
        potential->update_atomic_potential();
        parameters_.generate_radial_functions();
        parameters_.generate_radial_integrals();
    }
    
    // solve secular equation and generate wave functions
    for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = spl_num_kpoints(ikloc);
        kpoints_[ik]->find_eigen_states(band_, potential->effective_potential(), potential->effective_magnetic_field());
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

    parameters_.rti().valence_eval_sum = eval_sum;
    return eval_sum;
}

void kset::find_band_occupancies()
{
    Timer t("sirius::Density::find_band_occupancies");

    double ef = 0.15;

    double de = 0.1;

    int s = 1;
    int sp;

    double ne = 0.0;

    mdarray<double, 2> bnd_occ(parameters_.num_bands(), num_kpoints());
    
    // TODO: safe way not to get stuck here
    while (true)
    {
        ne = 0.0;
        for (int ik = 0; ik < num_kpoints(); ik++)
        {
            for (int j = 0; j < parameters_.num_bands(); j++)
            {
                bnd_occ(j, ik) = Utils::gaussian_smearing(kpoints_[ik]->band_energy(j) - ef) * 
                                 parameters_.max_occupancy();
                ne += bnd_occ(j, ik) * kpoints_[ik]->weight();
            }
        }

        if (fabs(ne - parameters_.num_valence_electrons()) < 1e-11) break;

        sp = s;
        s = (ne > parameters_.num_valence_electrons()) ? -1 : 1;

        de = s * fabs(de);

        (s != sp) ? de *= 0.5 : de *= 1.25; 
        
        ef += de;
    } 

    parameters_.rti().energy_fermi = ef;
    
    for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_occupancies(&bnd_occ(0, ik));

    double gap = 0.0;
    
    int nve = int(parameters_.num_valence_electrons() + 1e-12);
    if ((parameters_.num_spins() == 2) || 
        ((fabs(nve - parameters_.num_valence_electrons()) < 1e-12) && nve % 2 == 0))
    {
        // find band gap
        std::vector< std::pair<double, double> > eband;
        std::pair<double, double> eminmax;

        for (int j = 0; j < parameters_.num_bands(); j++)
        {
            eminmax.first = 1e10;
            eminmax.second = -1e10;

            for (int ik = 0; ik < num_kpoints(); ik++)
            {
                eminmax.first = std::min(eminmax.first, kpoints_[ik]->band_energy(j));
                eminmax.second = std::max(eminmax.second, kpoints_[ik]->band_energy(j));
            }

            eband.push_back(eminmax);
        }
        
        std::sort(eband.begin(), eband.end());

        int ist = nve;
        if (parameters_.num_spins() == 1) ist /= 2; 

        if (eband[ist].first > eband[ist - 1].second) gap = eband[ist].first - eband[ist - 1].second;

        parameters_.rti().band_gap = gap;
    }
}

void kset::print_info()
{
    pstdout pout(100 * num_kpoints());

    if (parameters_.mpi_grid().side(1 << 0))
    {
        for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++)
        {
            int ik = spl_num_kpoints(ikloc);
            pout.printf("%4i   %8.4f %8.4f %8.4f   %12.6f     %6i            %6i\n", 
                        ik, kpoints_[ik]->vk()[0], kpoints_[ik]->vk()[1], kpoints_[ik]->vk()[2], 
                        kpoints_[ik]->weight(), kpoints_[ik]->num_gkvec(), kpoints_[ik]->apwlo_basis_size());
        }
    }
    if (Platform::mpi_rank() == 0)
    {
        printf("\n");
        printf("total number of k-points : %i\n", num_kpoints());
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n");
        printf("  ik                vk                    weight  num_gkvec  apwlo_basis_size\n");
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n");
    }
    pout.flush(0);
}


};

