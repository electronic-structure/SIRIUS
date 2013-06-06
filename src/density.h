namespace sirius
{

// TODO: introduce external kpoint_set for density

class Density
{
    private:
        
        /// Global set of parameters
        Global& parameters_;

        // TODO: potential must be an external object. Density class not always needs a potential.
        /// Pointer to effective potential 
        Potential* potential_;

        int allocate_f_;
        
        PeriodicFunction<double>* rho_;
        
        PeriodicFunction<double>* magnetization_[3];
        
        std::vector< std::pair<int, int> > dmat_spins_;

        // TODO: clean Gaunt arrays (should be one)
        mdarray< std::vector< std::vector< std::pair<int, complex16> > >, 2> complex_gaunt_;

        GauntCoefficients gaunt12_;

        std::vector<int> l_by_lm_;

        void init();
        
        /// Get the local list of occupied bands
        /** Initially bands are distributed over k-points and columns of the MPI grid used for the diagonalization.
            Additionaly bands are sub split over rows of the 2D MPI grid, so each MPI rank in the total MPI grid gets
            it's local fraction of the bands.
        */
        void get_occupied_bands_list(Band* band, kpoint* kp, std::vector< std::pair<int, double> >& bands);

        /// Reduce complex density matrix over magnetic quantum numbers
        template <int num_mag_dims> 
        void reduce_zdens(int ia, int ialoc, mdarray<complex16, 4>& zdens, mdarray<double, 4>& mt_density_matrix);
        
        /// Add k-point contribution to the auxiliary muffin-tin density matrix
        /** In case of LDA+U the occupation matrix is also computed. It has the following expression:
            \f[
                n_{\ell,mm'}^{\sigma \sigma'} = \sum_{i {\bf k}}^{occ} \int_{0}^{R_{MT}} r^2 dr 
                          \Psi_{\ell m}^{i{\bf k}\sigma *}({\bf r}) \Psi_{\ell m'}^{i{\bf k}\sigma'}({\bf r})
            \f] 
        */
        void add_kpoint_contribution_mt(Band* band, kpoint* kp, mdarray<complex16, 4>& mt_complex_density_matrix);
        
        /// Add k-point contribution to the interstitial density and magnetization
        void add_kpoint_contribution_it(Band* band, kpoint* kp);
        
        /// Generate valence density in the muffin-tins 
        void generate_valence_density_mt(kset& ks);
        
        /// Generate valence density in the muffin-tins using straightforward (slow) approach
        template <processing_unit_t pu> 
        void generate_valence_density_mt_directly(kset& ks);
        
        void generate_valence_density_mt_sht(kset& ks);
        
        /// Generate valence density in the interstitial
        void generate_valence_density_it(kset& ks);
       
        /// Add band contribution to the muffin-tin density
        void add_band_contribution_mt(Band* band, double weight, mdarray<complex16, 3>& fylm, 
                                      std::vector<PeriodicFunction<double, radial_angular>*>& dens);

    public:

        /// Constructor
        Density(Global& parameters__, Potential* potential__, int allocate_f__);
        
        /// Destructor
        ~Density();
       
        /// Set pointers to muffin-tin and interstitial charge density arrays
        void set_charge_density_ptr(double* rhomt, double* rhoir);
        
        /// Set pointers to muffin-tin and interstitial magnetization arrays
        void set_magnetization_ptr(double* magmt, double* magir);
        
        /// Zero density and magnetization
        void zero();
        
        /// Generate initial charge density and magnetization
        void initial_density(int type);

        /// Generate charge density and magnetization from the wave functions
        void generate(kset& ks);
        
        /// Integrtae charge density to get total and partial charges
        void integrate();

        /// Check density at MT boundary
        void check_density_continuity_at_mt();
         
        PeriodicFunction<double>* rho()
        {
            return rho_;
        }
        
        PeriodicFunction<double>** magnetization()
        {
            return magnetization_;
        }

        PeriodicFunction<double>* magnetization(int i)
        {
            return magnetization_[i];
        }
};

Density::Density(Global& parameters__, Potential* potential__, int allocate_f__ = pw_component) : 
    parameters_(parameters__), potential_(potential__), allocate_f_(allocate_f__)
{
    init();
}

Density::~Density()
{
    delete rho_;
    for (int j = 0; j < parameters_.num_mag_dims(); j++) delete magnetization_[j];
}

void Density::init()
{
    rho_ = new PeriodicFunction<double>(parameters_, parameters_.lmax_rho());
    rho_->allocate(allocate_f_);

    for (int i = 0; i < parameters_.num_mag_dims(); i++)
    {
        magnetization_[i] = new PeriodicFunction<double>(parameters_, parameters_.lmax_rho());
        magnetization_[i]->allocate(allocate_f_);
    }

    dmat_spins_.clear();
    dmat_spins_.push_back(std::pair<int, int>(0, 0));
    dmat_spins_.push_back(std::pair<int, int>(1, 1));
    dmat_spins_.push_back(std::pair<int, int>(0, 1));
    
    complex_gaunt_.set_dimensions(parameters_.lmmax_apw(), parameters_.lmmax_rho());
    complex_gaunt_.allocate();

    for (int l2 = 0; l2 <= parameters_.lmax_apw(); l2++)
    {
    for (int m2 = -l2; m2 <= l2; m2++)
    {
        int lm2 = Utils::lm_by_l_m(l2, m2);
        for (int l3 = 0; l3 <= parameters_.lmax_rho(); l3++)
        {
        for (int m3 = -l3; m3 <= l3; m3++)
        {
            int lm3 = Utils::lm_by_l_m(l3, m3);
            complex_gaunt_(lm2, lm3).resize(parameters_.lmax_apw() + 1);
            for (int l1 = 0; l1 <= parameters_.lmax_apw(); l1++) 
            {
                for (int m1 = -l1; m1 <= l1; m1++)
                {
                    complex16 gc = SHT::complex_gaunt(l1, l3, l2, m1, m3, m2);
                    if (abs(gc) > 1e-16)
                        complex_gaunt_(lm2, lm3)[l1].push_back(std::pair<int, complex16>(m1 + l1, gc));
                }
            }
        }
        }
    }
    }

    switch (basis_type)
    {
        case apwlo:
        {
            gaunt12_.set_lmax(parameters_.lmax_apw(), parameters_.lmax_apw(), parameters_.lmax_rho());
            break;
        }
        case pwlo:
        {
            gaunt12_.set_lmax(parameters_.lmax_pw(), parameters_.lmax_pw(), parameters_.lmax_rho());
            break;
        }
    }

    l_by_lm_.resize(parameters_.lmmax_rho());
    for (int l = 0, lm = 0; l <= parameters_.lmax_rho(); l++)
    {
        for (int m = -l; m <= l; m++, lm++) l_by_lm_[lm] = l;
    }

}

void Density::set_charge_density_ptr(double* rhomt, double* rhoir)
{
    rho_->set_rlm_ptr(rhomt);
    rho_->set_it_ptr(rhoir);
}

void Density::set_magnetization_ptr(double* magmt, double* magir)
{
    assert(parameters_.num_spins() == 2);

    // set temporary array wrapper
    mdarray<double, 4> magmt_tmp(magmt, parameters_.lmmax_rho(), parameters_.max_num_mt_points(), 
                                 parameters_.num_atoms(), parameters_.num_mag_dims());
    mdarray<double, 2> magir_tmp(magir, parameters_.fft().size(), parameters_.num_mag_dims());
    
    if (parameters_.num_mag_dims() == 1)
    {
        // z component is the first and only one
        magnetization_[0]->set_rlm_ptr(&magmt_tmp(0, 0, 0, 0));
        magnetization_[0]->set_it_ptr(&magir_tmp(0, 0));
    }

    if (parameters_.num_mag_dims() == 3)
    {
        // z component is the first
        magnetization_[0]->set_rlm_ptr(&magmt_tmp(0, 0, 0, 2));
        magnetization_[0]->set_it_ptr(&magir_tmp(0, 2));
        // x component is the second
        magnetization_[1]->set_rlm_ptr(&magmt_tmp(0, 0, 0, 0));
        magnetization_[1]->set_it_ptr(&magir_tmp(0, 0));
        // y component is the third
        magnetization_[2]->set_rlm_ptr(&magmt_tmp(0, 0, 0, 1));
        magnetization_[2]->set_it_ptr(&magir_tmp(0, 1));
    }
}
    
void Density::zero()
{
    rho_->zero();
    for (int i = 0; i < parameters_.num_mag_dims(); i++) magnetization_[i]->zero();
}

void Density::initial_density(int type = 0)
{
    zero();
    
    if (type == 0)
    {
        parameters_.solve_free_atoms();
        
        double mt_charge = 0.0;
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            double* v = parameters_.atom(ia)->vector_field();
            double len = Utils::vector_length(v);

            int nmtp = parameters_.atom(ia)->type()->num_mt_points();
            Spline<double> rho(nmtp, parameters_.atom(ia)->type()->radial_grid());
            for (int ir = 0; ir < nmtp; ir++)
            {
                rho[ir] = parameters_.atom(ia)->type()->free_atom_density(ir);
                rho_->f_rlm(0, ir, ia) = rho[ir] / y00;
            }
            rho.interpolate();

            // add charge of the MT sphere
            mt_charge += fourpi * rho.integrate(nmtp - 1, 2);

            if (len > 1e-8)
            {
                if (parameters_.num_mag_dims())
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        magnetization_[0]->f_rlm(0, ir, ia) = 0.2 * rho_->f_rlm(0, ir, ia) * v[2] / len;
                }

                if (parameters_.num_mag_dims() == 3)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        magnetization_[1]->f_rlm(0, ir, ia) = 0.2 * rho_->f_rlm(0, ir, ia) * v[0] / len;
                    for (int ir = 0; ir < nmtp; ir++)
                        magnetization_[2]->f_rlm(0, ir, ia) = 0.2 * rho_->f_rlm(0, ir, ia) * v[1] / len;
                }
            }
        }
        
        // distribute remaining charge
        for (int i = 0; i < parameters_.fft().size(); i++)
            rho_->f_it(i) = (parameters_.num_electrons() - mt_charge) / parameters_.volume_it();
    }

    if (type == 1)
    {
        double rho_avg = parameters_.num_electrons() / parameters_.omega();
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            int nmtp = parameters_.atom(ia)->num_mt_points();
            for (int ir = 0; ir < nmtp; ir++) rho_->f_rlm(0, ir, ia) = rho_avg / y00;
        }
        for (int i = 0; i < parameters_.fft().size(); i++) rho_->f_it(i) = rho_avg;
    }

    if (type == 2)
    {
        rho_->f_pw(0) = parameters_.num_electrons() / parameters_.omega();
        for (int ig = 1; ig < parameters_.num_gvec(); ig++)
        {
            rho_->f_pw(ig) = (ig < 20) ? 1.0 / double(ig + 1) / parameters_.omega() : 0.0;
        }

        parameters_.fft().input(parameters_.num_gvec(), parameters_.fft_index(), rho_->f_pw());
        parameters_.fft().transform(1);
        parameters_.fft().output(rho_->f_it());

        rho_->allocate(ylm_component);
        rho_->zero(ylm_component);
        
        sbessel_pw<double> jl(parameters_, parameters_.lmax_rho());
        
        for (int igloc = 0; igloc < parameters_.spl_num_gvec().local_size(); igloc++)
        {
            int ig = parameters_.spl_num_gvec(igloc);
            if (ig < 20)
            {
                std::cout << "ig = " << ig << std::endl;
                jl.load(parameters_.gvec_len(ig));
                complex16 z1 = rho_->f_pw(ig) * fourpi;

                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                {
                    int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
                    complex16 z2 = z1 * parameters_.gvec_phase_factor<local>(igloc, ia);
                    
                    #pragma omp parallel for default(shared)
                    for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
                    {
                        int l = l_by_lm_[lm];
                        complex16 z3 = z2 * pow(std::complex<double>(0, 1), l) * conj(parameters_.gvec_ylm(lm, igloc)); 
                        for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
                            rho_->f_ylm(lm, ir, ia) += z3 * jl(ir, l, iat);
                    }
                }
            }
        }
        SHT sht_;
        sht_.set_lmax(parameters_.lmax_rho());
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
                sht_.convert_to_rlm(parameters_.lmax_rho(), &rho_->f_ylm(0, ir, ia), &rho_->f_rlm(0, ir, ia));
        }
        rho_->deallocate(ylm_component);

        std::cout << "pseudo-random number of electrons : " << rho_->integrate(rlm_component | it_component);
    }
            
}

template <int num_mag_dims> 
void Density::reduce_zdens(int ia, int ialoc, mdarray<complex16, 4>& zdens, mdarray<double, 4>& mt_density_matrix)
{
    AtomType* type = parameters_.atom(ia)->type();
    int mt_basis_size = type->mt_basis_size();

    #pragma omp parallel for default(shared)
    for (int lm3 = 0; lm3 < parameters_.lmmax_rho(); lm3++)
    {
        int l3 = l_by_lm_[lm3];
        
        for (int j2 = 0; j2 < mt_basis_size; j2++)
        {
            int l2 = type->indexb(j2).l;
            int lm2 = type->indexb(j2).lm;
            int idxrf2 = type->indexb(j2).idxrf;
            int offs = idxrf2 * (idxrf2 + 1) / 2;

            int j1 = 0;

            // compute only upper triangular block and later use the symmetry properties of the density matrix
            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
            {
                int l1 = type->indexr(idxrf1).l;
                if ((l1 + l2 + l3) % 2 == 0)
                {
                    for (int k = 0; k < (int)complex_gaunt_(lm2, lm3)[l1].size(); k++)
                    {
                        int m1 = complex_gaunt_(lm2, lm3)[l1][k].first;
                        complex16 gc = complex_gaunt_(lm2, lm3)[l1][k].second;

                        switch (num_mag_dims)
                        {
                            case 3:
                            {
                                mt_density_matrix(offs + idxrf1, lm3, 2, ialoc) += 2.0 * real(zdens(j1 + m1, j2, 2, ialoc) * gc); 
                                mt_density_matrix(offs + idxrf1, lm3, 3, ialoc) -= 2.0 * imag(zdens(j1 + m1, j2, 2, ialoc) * gc);
                            }
                            case 1:
                            {
                                mt_density_matrix(offs + idxrf1, lm3, 1, ialoc) += real(zdens(j1 + m1, j2, 1, ialoc) * gc);
                            }
                            case 0:
                            {
                                mt_density_matrix(offs + idxrf1, lm3, 0, ialoc) += real(zdens(j1 + m1, j2, 0, ialoc) * gc);
                            }
                        }
                    }
                } 
                j1 += (2 * l1 + 1);
            }
        } // j2
    } // lm3
}

void Density::get_occupied_bands_list(Band* band, kpoint* kp, std::vector< std::pair<int, double> >& bands)
{
    bands.clear();
    for (int jsub = 0; jsub < band->num_sub_bands(); jsub++)
    {
        int j = band->idxbandglob(jsub);
        int jloc = band->idxbandloc(jsub);
        double wo = kp->band_occupancy(j) * kp->weight();
        if (wo > 1e-14) bands.push_back(std::pair<int, double>(jloc, wo));
    }
}

void Density::add_kpoint_contribution_mt(Band* band, kpoint* kp, mdarray<complex16, 4>& mt_complex_density_matrix)
{
    Timer t("sirius::Density::add_kpoint_contribution_mt");
    
    std::vector< std::pair<int, double> > bands;
    get_occupied_bands_list(band, kp, bands);
    if (bands.size() == 0) return;
   
    mdarray<complex16, 3> wf1(parameters_.max_mt_basis_size(), (int)bands.size(), parameters_.num_spins());
    mdarray<complex16, 3> wf2(parameters_.max_mt_basis_size(), (int)bands.size(), parameters_.num_spins());

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        int offset_wf = parameters_.atom(ia)->offset_wf();
        int mt_basis_size = parameters_.atom(ia)->type()->mt_basis_size();
        
        for (int i = 0; i < (int)bands.size(); i++)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                memcpy(&wf1(0, i, ispn), &kp->spinor_wave_function(offset_wf, ispn, bands[i].first), 
                       mt_basis_size * sizeof(complex16));
                for (int j = 0; j < mt_basis_size; j++) wf2(j, i, ispn) = wf1(j, i, ispn) * bands[i].second;
            }
        }

        for (int j = 0; j < mt_complex_density_matrix.size(2); j++)
        {
            blas<cpu>::gemm(0, 2, mt_basis_size, mt_basis_size, (int)bands.size(), complex16(1, 0), 
                            &wf1(0, 0, dmat_spins_[j].first), wf1.ld(), 
                            &wf2(0, 0, dmat_spins_[j].second), wf2.ld(), complex16(1, 0), 
                            &mt_complex_density_matrix(0, 0, j, ia), mt_complex_density_matrix.ld());
        }
    }
}

void Density::add_kpoint_contribution_it(Band* band, kpoint* kp)
{
    Timer t("sirius::Density::add_kpoint_contribution_it");
    
    std::vector< std::pair<int, double> > bands;
    get_occupied_bands_list(band, kp, bands);
    if (bands.size() == 0) return;
    mdarray<complex16, 3> wf1(parameters_.max_mt_basis_size(), (int)bands.size(), parameters_.num_spins());
    mdarray<complex16, 3> wf2(parameters_.max_mt_basis_size(), (int)bands.size(), parameters_.num_spins());

    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {
        int thread_id = omp_get_thread_num();

        mdarray<double, 2> it_density_matrix(parameters_.fft().size(), parameters_.num_mag_dims() + 1);
        it_density_matrix.zero();
        
        mdarray<complex16, 2> wfit(parameters_.fft().size(), parameters_.num_spins());

        #pragma omp for
        for (int i = 0; i < (int)bands.size(); i++)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                parameters_.fft().input(kp->num_gkvec(), kp->fft_index(), 
                                        &kp->spinor_wave_function(parameters_.mt_basis_size(), ispn, 
                                                                  bands[i].first), thread_id);
                parameters_.fft().transform(1, thread_id);
                parameters_.fft().output(&wfit(0, ispn), thread_id);
            }
            
            double w = bands[i].second / parameters_.omega();
            
            switch (parameters_.num_mag_dims())
            {
                case 3:
                {
                    for (int ir = 0; ir < parameters_.fft().size(); ir++)
                    {
                        complex16 z = wfit(ir, 0) * conj(wfit(ir, 1)) * w;
                        it_density_matrix(ir, 2) += 2.0 * real(z);
                        it_density_matrix(ir, 3) -= 2.0 * imag(z);
                    }
                }
                case 1:
                {
                    for (int ir = 0; ir < parameters_.fft().size(); ir++)
                        it_density_matrix(ir, 1) += real(wfit(ir, 1) * conj(wfit(ir, 1))) * w;
                }
                case 0:
                {
                    for (int ir = 0; ir < parameters_.fft().size(); ir++)
                        it_density_matrix(ir, 0) += real(wfit(ir, 0) * conj(wfit(ir, 0))) * w;
                }
            }
        }

        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                #pragma omp critical
                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                {
                    magnetization_[1]->f_it(ir) += it_density_matrix(ir, 2);
                    magnetization_[2]->f_it(ir) += it_density_matrix(ir, 3);
                }
            }
            case 1:
            {
                #pragma omp critical
                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                {
                    rho_->f_it(ir) += (it_density_matrix(ir, 0) + it_density_matrix(ir, 1));
                    magnetization_[0]->f_it(ir) += (it_density_matrix(ir, 0) - it_density_matrix(ir, 1));
                }
                break;
            }
            case 0:
            {
                #pragma omp critical
                for (int ir = 0; ir < parameters_.fft().size(); ir++) 
                    rho_->f_it(ir) += it_density_matrix(ir, 0);
            }
        }
    }
}

void Density::generate_valence_density_mt(kset& ks)
{
    Timer t("sirius::Density::generate_valence_density_mt");

    // =======================================================================================
    // if we have ud and du spin blocks, don't compute one of them (du in this implementation)
    // because density matrix is symmetric
    // =======================================================================================
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    // complex density matrix
    mdarray<complex16, 4> mt_complex_density_matrix(parameters_.max_mt_basis_size(), parameters_.max_mt_basis_size(),
                                                    num_zdmat, parameters_.num_atoms());
    mt_complex_density_matrix.zero();
    
    // ========================
    // add k-point contribution
    // ========================
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        add_kpoint_contribution_mt(ks.band(), ks[ik], mt_complex_density_matrix);
    }
    
    mdarray<complex16, 4> mt_complex_density_matrix_loc(parameters_.max_mt_basis_size(), 
                                                        parameters_.max_mt_basis_size(),
                                                        num_zdmat, parameters_.spl_num_atoms().local_size(0));
   
    for (int j = 0; j < num_zdmat; j++)
    {
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            int ialoc = parameters_.spl_num_atoms().location(_splindex_offs_, ia);
            int rank = parameters_.spl_num_atoms().location(_splindex_rank_, ia);

            Platform::reduce(&mt_complex_density_matrix(0, 0, j, ia), &mt_complex_density_matrix_loc(0, 0, j, ialoc),
                             parameters_.max_mt_basis_size() * parameters_.max_mt_basis_size(),
                             parameters_.mpi_grid().communicator(), rank);
        }
    }
   
    // compute occupation matrix
    if (parameters_.uj_correction())
    {
        Timer* t3 = new Timer("sirius::Density::generate:om");
        
        mdarray<complex16, 4> occupation_matrix(16, 16, 2, 2); 
        
        for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.spl_num_atoms(ialoc);
            AtomType* type = parameters_.atom(ia)->type();
            
            occupation_matrix.zero();
            for (int l = 0; l <= 3; l++)
            {
                int num_rf = type->indexr().num_rf(l);

                for (int j = 0; j < num_zdmat; j++)
                {
                    for (int order2 = 0; order2 < num_rf; order2++)
                    {
                    for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
                    {
                        for (int order1 = 0; order1 < num_rf; order1++)
                        {
                        for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
                        {
                            occupation_matrix(lm1, lm2, dmat_spins_[j].first, dmat_spins_[j].second) +=
                                mt_complex_density_matrix_loc(type->indexb_by_lm_order(lm1, order1),
                                                              type->indexb_by_lm_order(lm2, order2), j, ialoc) *
                                parameters_.atom(ia)->symmetry_class()->o_radial_integral(l, order1, order2);
                        }
                        }
                    }
                    }
                }
            }
        
            // restore the du block
            for (int lm1 = 0; lm1 < 16; lm1++)
            {
                for (int lm2 = 0; lm2 < 16; lm2++)
                    occupation_matrix(lm2, lm1, 1, 0) = conj(occupation_matrix(lm1, lm2, 0, 1));
            }

            parameters_.atom(ia)->set_occupation_matrix(&occupation_matrix(0, 0, 0, 0));
        }

        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            int rank = parameters_.spl_num_atoms().location(_splindex_rank_, ia);
            parameters_.atom(ia)->sync_occupation_matrix(rank);
        }

        delete t3;
    }

    int max_num_rf_pairs = parameters_.max_mt_radial_basis_size() * (parameters_.max_mt_radial_basis_size() + 1) / 2;
    
    // real density matrix
    mdarray<double, 4> mt_density_matrix(max_num_rf_pairs, parameters_.lmmax_rho(), parameters_.num_mag_dims() + 1, 
                                         parameters_.spl_num_atoms().local_size());
    mt_density_matrix.zero();
    
    Timer t1("sirius::Density::generate:sum_zdens", false);
    Timer t2("sirius::Density::generate:expand_lm", false);
    mdarray<double, 2> rf_pairs(max_num_rf_pairs, parameters_.max_num_mt_points());
    mdarray<double, 3> dlm(parameters_.lmmax_rho(), parameters_.max_num_mt_points(), 
                           parameters_.num_mag_dims() + 1);
    for (int ialoc = 0; ialoc < parameters_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.spl_num_atoms(ialoc);
        int nmtp = parameters_.atom(ia)->type()->num_mt_points();
        int num_rf_pairs = parameters_.atom(ia)->type()->mt_radial_basis_size() * 
                           (parameters_.atom(ia)->type()->mt_radial_basis_size() + 1) / 2;
        
        t1.start();
        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                reduce_zdens<3>(ia, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
            case 1:
            {
                reduce_zdens<1>(ia, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
            case 0:
            {
                reduce_zdens<0>(ia, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
        }
        t1.stop();

        t2.start();
        // collect radial functions
        for (int idxrf2 = 0; idxrf2 < parameters_.atom(ia)->type()->mt_radial_basis_size(); idxrf2++)
        {
            int offs = idxrf2 * (idxrf2 + 1) / 2;
            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
            {
                int n = (idxrf1 == idxrf2) ? 1 : 2;
                for (int ir = 0; ir < parameters_.atom(ia)->type()->num_mt_points(); ir++)
                {
                    rf_pairs(offs + idxrf1, ir) = n * parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf1) * 
                                                      parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf2); 
                }
            }
        }
        for (int j = 0; j < parameters_.num_mag_dims() + 1; j++)
        {
            blas<cpu>::gemm(1, 0, parameters_.lmmax_rho(), nmtp, num_rf_pairs, 
                            &mt_density_matrix(0, 0, j, ialoc), mt_density_matrix.ld(), 
                            &rf_pairs(0, 0), rf_pairs.ld(), &dlm(0, 0, j), dlm.ld());
        }

        int sz = parameters_.lmmax_rho() * nmtp * (int)sizeof(double);
        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                memcpy(&magnetization_[1]->f_rlm(0, 0, ia), &dlm(0, 0, 2), sz); 
                memcpy(&magnetization_[2]->f_rlm(0, 0, ia), &dlm(0, 0, 3), sz);
            }
            case 1:
            {
                for (int ir = 0; ir < nmtp; ir++)
                {
                    for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
                    {
                        rho_->f_rlm(lm, ir, ia) = dlm(lm, ir, 0) + dlm(lm, ir, 1);
                        magnetization_[0]->f_rlm(lm, ir, ia) = dlm(lm, ir, 0) - dlm(lm, ir, 1);
                    }
                }
                break;
            }
            case 0:
            {
                memcpy(&rho_->f_rlm(0, 0, ia), &dlm(0, 0, 0), sz);
            }
        }
        t2.stop();
    }
    
    rho_->sync(rlm_component);
    for (int j = 0; j < parameters_.num_mag_dims(); j++) magnetization_[j]->sync(rlm_component);
}

void Density::generate_valence_density_it(kset& ks)
{
    Timer t("sirius::Density::generate_valence_density_it");

    // ========================
    // add k-point contribution
    // ========================
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        add_kpoint_contribution_it(ks.band(), ks[ik]);
    }
    
    // ==========================================================================================
    // reduce arrays; assume that each rank (including ranks along second direction) did it's own 
    // fraction of the density
    // ==========================================================================================
    Platform::allreduce(&rho_->f_it(0), parameters_.fft().size()); 
    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        Platform::allreduce(&magnetization_[j]->f_it(0), parameters_.fft().size()); 
}

void Density::add_band_contribution_mt(Band* band, double weight, mdarray<complex16, 3>& fylm, 
                                       std::vector<PeriodicFunction<double, radial_angular>*>& dens)
{
    splindex<block> spl_num_atoms(parameters_.num_atoms(), band->num_ranks_row(), band->rank_row());

    for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++)
    {
        int ia = spl_num_atoms[ialoc];
        #pragma omp parallel for default(shared)
        for (int lm3 = 0; lm3 < parameters_.lmmax_rho(); lm3++)
        {
            for (int k = 0; k < gaunt12_.complex_gaunt_packed_L1_L2_size(lm3); k++)
            {
                int lm1 = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).lm1;
                int lm2 = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).lm2;
                complex16 cg = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).cg;

                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
                {
                    dens[0]->f_rlm(ir, lm3, ia) += weight * real(cg * conj(fylm(ir, lm1, ia)) * fylm(ir, lm2, ia));
                }
            }
        }
    }
}

template<> void Density::generate_valence_density_mt_directly<cpu>(kset& ks)
{
    Timer t("sirius::Density::generate_valence_density_mt_directly");
    
    int lmax = (basis_type == apwlo) ? parameters_.lmax_apw() : parameters_.lmax_pw();
    int lmmax = Utils::lmmax_by_lmax(lmax);
    Band* band = ks.band();
    
    std::vector<PeriodicFunction<double, radial_angular>*> dens(1 + parameters_.num_mag_dims());
    for (int i = 0; i < (int)dens.size(); i++)
    {
        dens[i] = new PeriodicFunction<double, radial_angular>(parameters_, parameters_.lmax_rho());
        dens[i]->allocate(rlm_component);
        dens[i]->zero();
    }
    
    mdarray<complex16, 3> fylm(parameters_.max_num_mt_points(), lmmax, parameters_.num_atoms());

    // add k-point contribution
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        for (int jloc = 0; jloc < band->spl_spinor_wf_col().local_size(); jloc++)
        {
            int j = band->spl_spinor_wf_col(jloc);

            double wo = ks[ik]->band_occupancy(j) * ks[ik]->weight();

            if (wo > 1e-14)
            {
                int ispn = 0;

                ks[ik]->spinor_wave_function_component_mt<radial_angular>(band, lmax, ispn, jloc, fylm);
                
                add_band_contribution_mt(band, wo, fylm, dens);
            }
        }
    }

    for (int i = 0; i < (int)dens.size(); i++)
    {
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            Platform::allreduce(&dens[i]->f_rlm(0, 0, ia), 
                                parameters_.lmmax_rho() * parameters_.max_num_mt_points(), 
                                parameters_.mpi_grid().communicator());
        }
    }
                                                        
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
        {
            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
            {
                rho_->f_rlm(lm, ir, ia) += dens[0]->f_rlm(ir, lm, ia);
            }
        }
    }
    
    for (int i = 0; i < (int)dens.size(); i++) 
    {
        dens[i]->deallocate(rlm_component);
        delete dens[i];
    }
}

void Density::generate_valence_density_mt_sht(kset& ks)
{
    Timer t("sirius::Density::generate_valence_density_mt_sht");
    
    int lmax = (basis_type == apwlo) ? parameters_.lmax_apw() : parameters_.lmax_pw();
    int lmmax = Utils::lmmax_by_lmax(lmax);
    Band* band = ks.band();
    
    //std::vector<PeriodicFunction<double, radial_angular>*> dens(1 + parameters_.num_mag_dims());
    //for (int i = 0; i < (int)dens.size(); i++)
    //{
    //    dens[i] = new PeriodicFunction<double, radial_angular>(parameters_, parameters_.lmax_rho());
    //    dens[i]->allocate(rlm_component);
    //    dens[i]->zero();
    //}

    SHT sht;
    sht.set_lmax(parameters_.lmax_rho());

    
    mdarray<complex16, 3> fylm(parameters_.max_num_mt_points(), lmmax, parameters_.num_atoms());
    mdarray<complex16, 2> gylm(parameters_.lmmax_rho(), parameters_.max_num_mt_points());
    mdarray<complex16, 2> gtp(sht.num_points(), parameters_.max_num_mt_points());
    mdarray<double, 3> rhotp(sht.num_points(), parameters_.max_num_mt_points(), parameters_.num_atoms());
    rhotp.zero();

    mdarray<double, 3> rholm(parameters_.lmmax_rho(), parameters_.max_num_mt_points(), parameters_.num_atoms());
    rholm.zero();

    // add k-point contribution
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        for (int jloc = 0; jloc < band->spl_spinor_wf_col().local_size(); jloc++)
        {
            int j = band->spl_spinor_wf_col(jloc);

            double wo = ks[ik]->band_occupancy(j) * ks[ik]->weight();

            if (wo > 1e-14)
            {
                int ispn = 0;

                ks[ik]->spinor_wave_function_component_mt<radial_angular>(band, lmax, ispn, jloc, fylm);
                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                {
                    for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
                    {
                        for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++) 
                            gylm(lm, ir) = fylm(ir, lm, ia);
                    }
                    sht.ylm_backward_transform(&gylm(0, 0), parameters_.lmmax_rho(), parameters_.atom(ia)->num_mt_points(),
                                               &gtp(0, 0));
                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++) 
                    {
                        for (int itp = 0; itp < sht.num_points(); itp++) 
                            rhotp(itp, ir, ia) += wo * pow(abs(gtp(itp, ir)), 2);
                    }
                }
            }
        }
    }

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        sht.rlm_forward_transform(&rhotp(0, 0, ia), parameters_.lmmax_rho(),parameters_.atom(ia)->num_mt_points(), 
                                  &rholm(0, 0, ia));
    }

    //for (int i = 0; i < (int)dens.size(); i++)
   // {
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            Platform::allreduce(&rholm(0, 0, ia), 
                                parameters_.lmmax_rho() * parameters_.max_num_mt_points(), 
                                parameters_.mpi_grid().communicator());
        }
    //}
                                                        
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
        {
            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
            {
                rho_->f_rlm(lm, ir, ia) = rholm(lm, ir, ia);
            }
        }
    }
}


#ifdef _GPU_
template<> void Density::generate_valence_density_mt_directly<gpu>()
{
    Timer t("sirius::Density::generate_valence_density_mt_directly");
    
    int lmax = (basis_type == apwlo) ? parameters_.lmax_apw() : parameters_.lmax_pw();
    int lmmax = Utils::lmmax_by_lmax(lmax);

    // ==========================
    // prepare Gaunt coefficients
    // ==========================
    int max_num_gaunt = 0;
    for (int lm3 = 0; lm3 < parameters_.lmmax_rho(); lm3++)
        max_num_gaunt = std::max(max_num_gaunt, gaunt12_.complex_gaunt_packed_L1_L2_size(lm3));
   
    mdarray<int, 1> gaunt12_size(parameters_.lmmax_rho());
    mdarray<int, 2> gaunt12_lm1_by_lm3(max_num_gaunt, parameters_.lmmax_rho());
    mdarray<int, 2> gaunt12_lm2_by_lm3(max_num_gaunt, parameters_.lmmax_rho());
    mdarray<complex16, 2> gaunt12_cg(max_num_gaunt, parameters_.lmmax_rho());

    for (int lm3 = 0; lm3 < parameters_.lmmax_rho(); lm3++)
    {
        gaunt12_size(lm3) = gaunt12_.complex_gaunt_packed_L1_L2_size(lm3);
        for (int k = 0; k < gaunt12_.complex_gaunt_packed_L1_L2_size(lm3); k++)
        {
            gaunt12_lm1_by_lm3(k, lm3) = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).lm1;
            gaunt12_lm2_by_lm3(k, lm3) = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).lm2;
            gaunt12_cg(k, lm3) = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).cg;
        }
    }
    gaunt12_size.allocate_on_device();
    gaunt12_size.copy_to_device();
    gaunt12_lm1_by_lm3.allocate_on_device();
    gaunt12_lm1_by_lm3.copy_to_device();
    gaunt12_lm2_by_lm3.allocate_on_device();
    gaunt12_lm2_by_lm3.copy_to_device();
    gaunt12_cg.allocate_on_device();
    gaunt12_cg.copy_to_device();

    mdarray<double, 3> dens_mt(parameters_.max_num_mt_points(), parameters_.lmmax_rho(), parameters_.num_atoms());
    dens_mt.zero();
    dens_mt.allocate_on_device();
    dens_mt.zero_on_device();

    mdarray<complex16, 3> fylm(parameters_.max_num_mt_points(), lmmax, parameters_.num_atoms());
    fylm.pin_memory();
    fylm.allocate_on_device();
    
    splindex<block> spl_num_atoms(parameters_.num_atoms(), band_->num_ranks_row(), band_->rank_row());
    
    mdarray<int, 1> iat_by_ia(parameters_.num_atoms());
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        iat_by_ia(ia) = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
    iat_by_ia.allocate_on_device();
    iat_by_ia.copy_to_device();

    mdarray<int, 1> nmtp_by_iat(parameters_.num_atom_types());
    for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
        nmtp_by_iat(iat) = parameters_.atom_type(iat)->num_mt_points();
    nmtp_by_iat.allocate_on_device();
    nmtp_by_iat.copy_to_device();

    mdarray<int, 1> ia_by_ialoc(spl_num_atoms.local_size());
    for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++)
    {
        int ia = spl_num_atoms[ialoc];
        ia_by_ialoc(ialoc) = ia;
    }
    ia_by_ialoc.allocate_on_device();
    ia_by_ialoc.copy_to_device();
    
    // add k-point contribution
    for (int ikloc = 0; ikloc < kpoint_set_.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = kpoint_set_.spl_num_kpoints(ikloc);
        for (int jloc = 0; jloc < band_->spl_spinor_wf_col().local_size(); jloc++)
        {
            int j = band_->spl_spinor_wf_col(jloc);

            double wo = kpoint_set_[ik]->band_occupancy(j) * kpoint_set_[ik]->weight();

            if (wo > 1e-14)
            {
                int ispn = 0;
                kpoint_set_[ik]->spinor_wave_function_component_mt<radial_angular>(band_, lmax, ispn, jloc, fylm);
                fylm.copy_to_device();
                
                add_band_density_gpu(parameters_.lmmax_rho(), lmmax, parameters_.max_num_mt_points(), 
                                     spl_num_atoms.local_size(), ia_by_ialoc.get_ptr_device(), iat_by_ia.get_ptr_device(),
                                     nmtp_by_iat.get_ptr_device(), max_num_gaunt, 
                                     gaunt12_size.get_ptr_device(), gaunt12_lm1_by_lm3.get_ptr_device(), 
                                     gaunt12_lm2_by_lm3.get_ptr_device(), gaunt12_cg.get_ptr_device(), 
                                     fylm.get_ptr_device(), wo, dens_mt.get_ptr_device());
            }
        }
    }
    dens_mt.copy_to_host();

    //for (int i = 0; i < (int)dens.size(); i++)
    //{
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            Platform::allreduce(&dens_mt(0, 0, ia), parameters_.lmmax_rho() * parameters_.max_num_mt_points(), 
                                parameters_.mpi_grid().communicator());
        }
    //}

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
        {
            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
            {
                rho_->f_rlm(lm, ir, ia) += dens_mt(ir, lm, ia);
            }
        }
    }
}
#endif

void Density::generate(kset& ks)
{
    Timer t("sirius::Density::generate");
    
    double wt = 0.0;
    double ot = 0.0;
    for (int ik = 0; ik < ks.num_kpoints(); ik++)
    {
        wt += ks[ik]->weight();
        for (int j = 0; j < parameters_.num_bands(); j++) ot += ks[ik]->weight() * ks[ik]->band_occupancy(j);
    }

    if (fabs(wt - 1.0) > 1e-12) error(__FILE__, __LINE__, "kpoint weights don't sum to one");

    if (fabs(ot - parameters_.num_valence_electrons()) > 1e-8)
    {
        std::stringstream s;
        s << "wrong occupancies" << std::endl
          << "  computed : " << ot << std::endl
          << "  required : " << parameters_.num_valence_electrons() << std::endl
          << "  difference : " << fabs(ot - parameters_.num_valence_electrons());
        error(__FILE__, __LINE__, s);
    }

    // zero density and magnetization
    zero();

    // interstitial part is independent of basis type
    generate_valence_density_it(ks);
   
    // for muffin-tin part
    switch (basis_type)
    {
        case apwlo:
        {
            generate_valence_density_mt(ks);
            //generate_valence_density_mt_sht();
            break;
        }
        case pwlo:
        {
            switch (parameters_.processing_unit())
            {
                case cpu:
                {
                    generate_valence_density_mt_directly<cpu>(ks);
                    break;
                }
                #ifdef _GPU_
                case gpu:
                {
                    generate_valence_density_mt_directly<gpu>(ks);
                    break;
                }
                #endif
                default:
                {
                    error(__FILE__, __LINE__, "wrong processing unit");
                }
            }
            break;
        }
    }

    // compute core states
    for (int icloc = 0; icloc < parameters_.spl_num_atom_symmetry_classes().local_size(); icloc++)
    {
        int ic = parameters_.spl_num_atom_symmetry_classes(icloc);
        parameters_.atom_symmetry_class(ic)->generate_core_charge_density();
    }

    double eval_sum = 0.0;
    double core_leakage = 0.0;
    for (int ic = 0; ic < parameters_.num_atom_symmetry_classes(); ic++)
    {
        int rank = parameters_.spl_num_atom_symmetry_classes().location(1, ic);
        parameters_.atom_symmetry_class(ic)->sync_core_charge_density(rank);

        eval_sum += parameters_.atom_symmetry_class(ic)->core_eval_sum() *
                    parameters_.atom_symmetry_class(ic)->num_atoms();
        
        core_leakage += parameters_.atom_symmetry_class(ic)->core_leakage() * 
                        parameters_.atom_symmetry_class(ic)->num_atoms();
    }
    assert(eval_sum == eval_sum);
    assert(core_leakage == core_leakage);
    
    // add core contribution
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
            rho_->f_rlm(0, ir, ia) += parameters_.atom(ia)->symmetry_class()->core_charge_density(ir) / y00;
    }

    parameters_.rti().core_eval_sum = eval_sum;
    
    std::vector<double> nel_mt;
    double nel_it;
    double nel = rho_->integrate(rlm_component | it_component, nel_mt, nel_it);
    
    //if (Platform::mpi_rank() == 0)
    //{
    //    printf("\n");
    //    printf("Charges before symmetrization\n");
    //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    //    {
    //        printf("ia : %i  q : %f\n", ia, nel_mt[ia]);
    //    }
    //    printf("interstitial : %f\n", nel_it);
    //}
    
    if (fabs(nel - parameters_.num_electrons()) > 1e-5)
    {
        std::stringstream s;
        s << "wrong charge density after k-point summation" << std::endl
          << "obtained value : " << nel << std::endl 
          << "target value : " << parameters_.num_electrons() << std::endl
          << "difference : " << fabs(nel - parameters_.num_electrons()) << std::endl
          << "core leakage : " << core_leakage;
        warning(__FILE__, __LINE__, s);
    }
    
    if (debug_level > 1) check_density_continuity_at_mt();
}

void Density::integrate()
{
    Timer t("sirius::Density::integrate");

    parameters_.rti().total_charge = rho_->integrate(rlm_component | it_component, 
                                                     parameters_.rti().mt_charge, 
                                                     parameters_.rti().it_charge); 

    for (int j = 0; j < parameters_.num_mag_dims(); j++)
    {
        parameters_.rti().total_magnetization[j] = 
            magnetization_[j]->integrate(rlm_component | it_component, 
                                         parameters_.rti().mt_magnetization[j], 
                                         parameters_.rti().it_magnetization[j]);
    }
}

void Density::check_density_continuity_at_mt()
{
    // generate plane-wave coefficients of the potential in the interstitial region
    parameters_.fft().input(rho_->f_it());
    parameters_.fft().transform(-1);
    parameters_.fft().output(parameters_.num_gvec(), parameters_.fft_index(), rho_->f_pw());
    
    SHT sht;
    sht.set_lmax(parameters_.lmax_rho());

    double diff = 0.0;
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        for (int itp = 0; itp < sht.num_points(); itp++)
        {
            double vc[3];
            for (int x = 0; x < 3; x++) vc[x] = sht.coord(x, itp) * parameters_.atom(ia)->mt_radius();

            double val_it = 0.0;
            for (int ig = 0; ig < parameters_.num_gvec(); ig++) 
            {
                double vgc[3];
                parameters_.get_coordinates<cartesian, reciprocal>(parameters_.gvec(ig), vgc);
                val_it += real(rho_->f_pw(ig) * exp(complex16(0.0, Utils::scalar_product(vc, vgc))));
            }

            double val_mt = 0.0;
            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
                val_mt += rho_->f_rlm(lm, parameters_.atom(ia)->num_mt_points() - 1, ia) * sht.rlm_backward(lm, itp);

            diff += fabs(val_it - val_mt);
        }
    }
    printf("Total and average charge difference at MT boundary : %.12f %.12f\n", diff, diff / parameters_.num_atoms() / sht.num_points());
}

};
