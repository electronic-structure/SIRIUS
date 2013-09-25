void Reciprocal_lattice::init(int lmax)
{
    Timer t("sirius::Reciprocal_lattice::init");
    
    int max_frac_coord[3];
    find_translation_limits<reciprocal>(pw_cutoff(), max_frac_coord);
    
    fft_.init(max_frac_coord);
    
    mdarray<int, 2> gvec_tmp(3, fft_.size());
    std::vector< std::pair<double, int> > gvec_tmp_length;
    
    int ig = 0;
    for (int i0 = fft_.grid_limits(0, 0); i0 <= fft_.grid_limits(0, 1); i0++)
    {
        for (int i1 = fft_.grid_limits(1, 0); i1 <= fft_.grid_limits(1, 1); i1++)
        {
            for (int i2 = fft_.grid_limits(2, 0); i2 <= fft_.grid_limits(2, 1); i2++)
            {
                gvec_tmp(0, ig) = i0;
                gvec_tmp(1, ig) = i1;
                gvec_tmp(2, ig) = i2;

                int fracc[] = {i0, i1, i2};
                double cartc[3];
                get_coordinates<cartesian, reciprocal>(fracc, cartc);

                gvec_tmp_length.push_back(std::pair<double, int>(Utils::vector_length(cartc), ig++));
            }
        }
    }

    Timer t1("sirius::Reciprocal_lattice::init|sort_G");
    // sort G-vectors by length
    std::sort(gvec_tmp_length.begin(), gvec_tmp_length.end());
    t1.stop();

    // create sorted list of G-vectors
    gvec_.set_dimensions(3, fft_.size());
    gvec_.allocate();

    // find number of G-vectors within the cutoff
    num_gvec_ = 0;
    for (int i = 0; i < fft_.size(); i++)
    {
        for (int j = 0; j < 3; j++) gvec_(j, i) = gvec_tmp(j, gvec_tmp_length[i].second);
        
        if (gvec_tmp_length[i].first <= pw_cutoff_) num_gvec_++;
    }
    
    index_by_gvec_.set_dimensions(dimension(fft_.grid_limits(0, 0), fft_.grid_limits(0, 1)),
                                  dimension(fft_.grid_limits(1, 0), fft_.grid_limits(1, 1)),
                                  dimension(fft_.grid_limits(2, 0), fft_.grid_limits(2, 1)));
    index_by_gvec_.allocate();
    
    fft_index_.resize(fft_.size());
    
    gvec_shell_.resize(fft_.size());
    gvec_shell_len_.clear();
    
    for (int ig = 0; ig < fft_.size(); ig++)
    {
        int i0 = gvec_(0, ig);
        int i1 = gvec_(1, ig);
        int i2 = gvec_(2, ig);

        // mapping from G-vector to it's index
        index_by_gvec_(i0, i1, i2) = ig;

        // mapping of FFT buffer linear index
        fft_index_[ig] = fft_.index(i0, i1, i2);

        // find G-shells
        double t = gvec_tmp_length[ig].first;
        if (gvec_shell_len_.empty() || fabs(t - gvec_shell_len_.back()) > 1e-10) gvec_shell_len_.push_back(t);
        gvec_shell_[ig] = (int)gvec_shell_len_.size() - 1;
    }

    //ig_by_igs_.clear();
    //ig_by_igs_.resize(num_gvec_shells());
    //for (int ig = 0; ig < num_gvec_; ig++)
    //{
    //    int igs = gvec_shell_[ig];
    //    ig_by_igs_[igs].push_back(ig);
    //}
    
    // create split index
    spl_num_gvec_.split(num_gvec(), Platform::num_mpi_ranks(), Platform::mpi_rank());

    spl_fft_size_.split(fft().size(), Platform::num_mpi_ranks(), Platform::mpi_rank());
    
    // precompute spherical harmonics of G-vectors 
    gvec_ylm_.set_dimensions(Utils::lmmax(lmax), spl_num_gvec_.local_size());
    gvec_ylm_.allocate();
    
    Timer t2("sirius::Reciprocal_lattice::init|ylm_G");
    for (int igloc = 0; igloc < spl_num_gvec_.local_size(); igloc++)
    {
        int ig = spl_num_gvec_[igloc];
        double xyz[3];
        double rtp[3];
        gvec_cart(ig, xyz);
        SHT::spherical_coordinates(xyz, rtp);
        SHT::spherical_harmonics(lmax, rtp[1], rtp[2], &gvec_ylm_(0, igloc));
    }
    t2.stop();
    
    update();
}

void Reciprocal_lattice::update()
{
    Timer t2("sirius::Reciprocal_lattice::update");
    // precompute G-vector phase factors
    gvec_phase_factors_.set_dimensions(spl_num_gvec_.local_size(), num_atoms());
    gvec_phase_factors_.allocate();
    for (int igloc = 0; igloc < spl_num_gvec_.local_size(); igloc++)
    {
        int ig = spl_num_gvec_[igloc];
        for (int ia = 0; ia < num_atoms(); ia++) gvec_phase_factors_(igloc, ia) = gvec_phase_factor<global>(ig, ia);
    }
}

void Reciprocal_lattice::clear()
{
    fft_.clear();
    gvec_.deallocate();
    gvec_shell_len_.clear();
    gvec_shell_.clear();
    index_by_gvec_.deallocate();
    fft_index_.clear();
}

void Reciprocal_lattice::print_info()
{
    printf("\n");
    printf("plane wave cutoff : %f\n", pw_cutoff());
    printf("number of G-vectors within the cutoff : %i\n", num_gvec());
    printf("number of G-shells : %i\n", num_gvec_shells());
    printf("FFT grid size : %i %i %i   total : %i\n", fft_.size(0), fft_.size(1), fft_.size(2), fft_.size());
    printf("FFT grid limits : %i %i   %i %i   %i %i\n", fft_.grid_limits(0, 0), fft_.grid_limits(0, 1),
                                                        fft_.grid_limits(1, 0), fft_.grid_limits(1, 1),
                                                        fft_.grid_limits(2, 0), fft_.grid_limits(2, 1));
}

template <index_domain_t index_domain>
inline int Reciprocal_lattice::gvec_shell(int ig)
{
    switch (index_domain)
    {
        case global:
        {
            assert(ig >= 0 && ig < (int)gvec_shell_.size());
            return gvec_shell_[ig];
            break;
        }
        case local:
        {
            return gvec_shell_[spl_num_gvec_[ig]];
            break;
        }
    }
}

template <index_domain_t index_domain>
inline complex16 Reciprocal_lattice::gvec_phase_factor(int ig, int ia)
{
    switch (index_domain)
    {
        case global:
        {
            return exp(complex16(0.0, twopi * Utils::scalar_product(gvec(ig), atom(ia)->position())));
            break;
        }
        case local:
        {
            return gvec_phase_factors_(ig, ia);
            break;
        }
    }
}

template <index_domain_t index_domain>
inline void Reciprocal_lattice::gvec_ylm_array(int ig, complex16* ylm, int lmax)
{
    switch (index_domain)
    {
        case local:
        {
            int lmmax = Utils::lmmax(lmax);
            assert(lmmax <= gvec_ylm_.size(0));
            memcpy(ylm, &gvec_ylm_(0, ig), lmmax * sizeof(complex16));
            return;
        }
        case global:
        {
            double vgc[3];
            gvec_cart(ig, vgc);
            double rtp[3];
            SHT::spherical_coordinates(vgc, rtp);
            SHT::spherical_harmonics(lmax, rtp[1], rtp[2], ylm);
            return;
        }
    }
}
