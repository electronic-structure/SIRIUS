void K_point::initialize()
{
    Timer t("sirius::K_point::initialize");
    
    zil_.resize(parameters_.lmax_apw() + 1);
    for (int l = 0; l <= parameters_.lmax_apw(); l++) zil_[l] = pow(double_complex(0, 1), l);
   
    l_by_lm_ = Utils::l_by_lm(parameters_.lmax_apw());

    band_energies_.resize(parameters_.num_bands());

    if (use_second_variation)
    {
        fv_eigen_values_.resize(parameters_.num_fv_states());
        // in case of collinear magnetism store pure up and pure dn components, otherwise store both up and dn components
        int ns = (parameters_.num_mag_dims() == 3) ? 2 : 1;
        sv_eigen_vectors_.set_dimensions(ns * parameters_.spl_fv_states_row().local_size(), parameters_.spl_spinor_wf_col().local_size());
        sv_eigen_vectors_.allocate();
    }
    
    update();
}

void K_point::update()
{
    double gk_cutoff = 0;
    switch (parameters_.esm_type())
    {
        case ultrasoft_pseudopotential:
        {
            gk_cutoff = parameters_.gk_cutoff();
            break;
        }
        default:
        {
            gk_cutoff = parameters_.aw_cutoff() / parameters_.unit_cell()->min_mt_radius();
        }
    }

    generate_gkvec(gk_cutoff);
    
    if (parameters_.unit_cell()->full_potential())
    {
        build_apwlo_basis_descriptors();
        distribute_block_cyclic();
        
        atom_lo_cols_.clear();
        atom_lo_cols_.resize(parameters_.unit_cell()->num_atoms());

        atom_lo_rows_.clear();
        atom_lo_rows_.resize(parameters_.unit_cell()->num_atoms());

        for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
        {
            int ia = apwlo_basis_descriptors_col(icol).ia;
            atom_lo_cols_[ia].push_back(icol);
        }
        
        for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
        {
            int ia = apwlo_basis_descriptors_row(irow).ia;
            atom_lo_rows_[ia].push_back(irow);
        }
    }
    if (parameters_.esm_type() == full_potential_pwlo)
    {
        /** \todo Correct the memory leak */
        stop_here
        sbessel_.resize(num_gkvec_loc()); 
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        {
            sbessel_[igkloc] = new sbessel_pw<double>(parameters_.unit_cell(), parameters_.lmax_pw());
            sbessel_[igkloc]->interpolate(gkvec_len_[igkloc]);
        }
    }
    
    init_gkvec();
    
    // cache "b" vector of linear equations Ax=b for matching coefficients (A will be a matrix of radial derivatives)
    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        alm_b_.set_dimensions(3, num_gkvec_loc(), parameters_.lmax_apw() + 1, parameters_.unit_cell()->num_atom_types());
        alm_b_.allocate();
        alm_b_.zero();

        // compute values and first and second derivatives of the spherical Bessel functions at the MT boundary
        mdarray<double, 2> sbessel_mt(parameters_.lmax_apw() + 2, 3);
        sbessel_mt.zero();

        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        {
            for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
            {
                double R = parameters_.unit_cell()->atom_type(iat)->mt_radius();

                double gkR = gkvec_len_[igkloc] * R;

                gsl_sf_bessel_jl_array(parameters_.lmax_apw() + 1, gkR, &sbessel_mt(0, 0));
                
                // Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                //
                // In[]:= FullSimplify[D[SphericalBesselJ[n,a*x],{x,1}]]
                // Out[]= (n SphericalBesselJ[n,a x])/x-a SphericalBesselJ[1+n,a x]
                //
                // In[]:= FullSimplify[D[SphericalBesselJ[n,a*x],{x,2}]]
                // Out[]= (((-1+n) n-a^2 x^2) SphericalBesselJ[n,a x]+2 a x SphericalBesselJ[1+n,a x])/x^2
                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    sbessel_mt(l, 1) = -sbessel_mt(l + 1, 0) * gkvec_len_[igkloc] + (l / R) * sbessel_mt(l, 0);
                    sbessel_mt(l, 2) = 2 * gkvec_len_[igkloc] * sbessel_mt(l + 1, 0) / R + 
                                       ((l - 1) * l - pow(gkR, 2)) * sbessel_mt(l, 0) / pow(R, 2);
                }
                
                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    double f = fourpi / sqrt(parameters_.unit_cell()->omega());
                    alm_b_(0, igkloc, l, iat) = zil_[l] * f * sbessel_mt(l, 0); 
                    alm_b_(1, igkloc, l, iat) = zil_[l] * f * sbessel_mt(l, 1);
                    alm_b_(2, igkloc, l, iat) = zil_[l] * f * sbessel_mt(l, 2);
                }
            }
        }
    }

    // compute radial integrals of |beta> functions
    if (parameters_.esm_type() == ultrasoft_pseudopotential)
    {
        Timer t1("sirius::K_point::update|beta_pw");

        auto uc = parameters_.unit_cell();

        beta_pw_.set_dimensions(num_gkvec(), uc->num_beta_t()); 
        beta_pw_.allocate();

        #pragma omp parallel
        {
            mdarray<Spline<double>*, 2> splines(uc->max_mt_radial_basis_size(), uc->num_atom_types());
            for (int iat = 0; iat < uc->num_atom_types(); iat++)
            {
                auto atom_type = uc->atom_type(iat);
                for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                {
                    int nr = atom_type->uspp().num_beta_radial_points[idxrf];
                    splines(idxrf, iat) = new Spline<double>(nr, atom_type->radial_grid());
                }
            }

            std::vector<double> beta_radial_integrals_(uc->max_mt_radial_basis_size());
            sbessel_pw<double> jl(uc, parameters_.lmax_beta());
            #pragma omp for
            for (int igk = 0; igk < num_gkvec(); igk++)
            {
                jl.load(gkvec_len_[igk]);

                for (int iat = 0; iat < uc->num_atom_types(); iat++)
                {
                    auto atom_type = uc->atom_type(iat);
                    for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                    {
                        int nr = atom_type->uspp().num_beta_radial_points[idxrf];
                        int l = atom_type->indexr(idxrf).l;
                        for (int ir = 0; ir < nr; ir++) 
                            (*splines(idxrf, iat))[ir] = jl(ir, l, iat) * atom_type->uspp().beta_radial_functions(ir, idxrf);
                        beta_radial_integrals_[idxrf] = splines(idxrf, iat)->interpolate().integrate(1);
                    }

                    for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
                    {
                        int l = atom_type->indexb(xi).l;
                        int lm = atom_type->indexb(xi).lm;
                        int idxrf = atom_type->indexb(xi).idxrf;

                        double_complex z = pow(double_complex(0, -1), l) * fourpi / sqrt(parameters_.unit_cell()->omega());
                        beta_pw_(igk, uc->beta_t_ofs(iat) + xi) = z * gkvec_ylm_(lm, igk) * beta_radial_integrals_[idxrf];
                    }
                }
            }
            for (int iat = 0; iat < uc->num_atom_types(); iat++)
            {
                auto atom_type = uc->atom_type(iat);
                for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                {
                    delete splines(idxrf, iat);
                }
            }
        }
    }

    spinor_wave_functions_.set_dimensions(wf_size(), parameters_.num_spins(), parameters_.spl_spinor_wf_col().local_size());

    if (use_second_variation)
    {
        // allocate memory for first-variational eigen vectors
        if (parameters_.unit_cell()->full_potential())
        {
            fv_eigen_vectors_.set_dimensions(apwlo_basis_size_row(), parameters_.spl_fv_states_col().local_size());
            fv_eigen_vectors_.allocate();
        }
        
        // allocate memory for first-variational states
        fv_states_col_.set_dimensions(wf_size(), parameters_.spl_fv_states_col().local_size());
        fv_states_col_.allocate();

        if (parameters_.esm_type() == ultrasoft_pseudopotential)
        {
            fv_states_col_.zero();
            for (int i = 0; i < parameters_.num_fv_states(); i++)
            {
                int rank = parameters_.spl_fv_states_col().location(_splindex_rank_, i);
                int iloc = parameters_.spl_fv_states_col().location(_splindex_offs_, i);
                
                if (rank == rank_col_) fv_states_col_(i, iloc) = 1.0; 
            }
        }
        
        if (num_ranks() == 1)
        {
            fv_states_row_.set_dimensions(wf_size(), parameters_.num_fv_states());
            fv_states_row_.set_ptr(fv_states_col_.get_ptr());
        }
        else
        {
            fv_states_row_.set_dimensions(wf_size(), parameters_.spl_fv_states_row().local_size());
            fv_states_row_.allocate();
        }
        
        if (parameters_.need_sv())
        {
            spinor_wave_functions_.allocate();
        }
        else
        {
            spinor_wave_functions_.set_ptr(fv_states_col_.get_ptr());
        }
    }
    else
    {
        if (parameters_.unit_cell()->full_potential())
        {
            fd_eigen_vectors_.set_dimensions(apwlo_basis_size_row(), parameters_.spl_spinor_wf_col().local_size());
            fd_eigen_vectors_.allocate();
            spinor_wave_functions_.allocate();
        }
    }
}

/// First order matching coefficients, conjugated
/** It is more convenient to store conjugated coefficients because then the overlap matrix is set with 
    single matrix-matrix multiplication without further conjugation.
    \todo (l,m) -> lm++;
*/
template<> 
void K_point::generate_matching_coefficients_l<1, true>(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                                        mdarray<double, 2>& A, mdarray<double_complex, 2>& alm)
{
    if ((fabs(A(0, 0)) < 1.0 / sqrt(parameters_.unit_cell()->omega())) && (debug_level >= 1))
    {   
        std::stringstream s;
        s << "Ill defined plane wave matching problem for atom " << ia << ", l = " << l << std::endl
          << "  radial function value at the MT boundary : " << A(0, 0); 
        
        warning_local(__FILE__, __LINE__, s);
    }
    
    A(0, 0) = 1.0 / A(0, 0);

    double_complex zt;
    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++)
    {
        zt = gkvec_phase_factors_(igkloc, ia) * alm_b_(0, igkloc, l, iat) * A(0, 0);

        int idxb = type->indexb_by_l_m_order(l, -l, 0);
        for (int m = -l; m <= l; m++) alm(igkloc, idxb++) = gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc) * conj(zt);
    }
}

/// First order matching coefficients, non-conjugated
template<> 
void K_point::generate_matching_coefficients_l<1, false>(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                                         mdarray<double, 2>& A, mdarray<double_complex, 2>& alm)
{
    if ((fabs(A(0, 0)) < 1.0 / sqrt(parameters_.unit_cell()->omega())) && (debug_level >= 1))
    {   
        std::stringstream s;
        s << "Ill defined plane wave matching problem for atom " << ia << ", l = " << l << std::endl
          << "  radial function value at the MT boundary : " << A(0, 0); 
        
        warning_local(__FILE__, __LINE__, s);
    }
    
    A(0, 0) = 1.0 / A(0, 0);

    double_complex zt;
    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++)
    {
        zt = gkvec_phase_factors_(igkloc, ia) * alm_b_(0, igkloc, l, iat) * A(0, 0);

        int idxb = type->indexb_by_l_m_order(l, -l, 0);
        for (int m = -l; m <= l; m++) alm(igkloc, idxb++) = conj(gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc)) * zt;
    }
}

/// Second order matching coefficients, conjugated
/** It is more convenient to store conjugated coefficients because then the overlap matrix is set with 
    single matrix-matrix multiplication without further conjugation.
*/
template<> void K_point::generate_matching_coefficients_l<2, true>(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                                                   mdarray<double, 2>& A, mdarray<double_complex, 2>& alm)
{
    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    
    if ((fabs(det) < 1.0 / sqrt(parameters_.unit_cell()->omega())) && (debug_level >= 1))
    {   
        std::stringstream s;
        s << "Ill defined plane wave matching problem for atom " << ia << ", l = " << l << std::endl
          << "  radial function value at the MT boundary : " << A(0 ,0); 
        
        warning_local(__FILE__, __LINE__, s);
    }
    std::swap(A(0, 0), A(1, 1));
    A(0, 0) /= det;
    A(1, 1) /= det;
    A(0, 1) = -A(0, 1) / det;
    A(1, 0) = -A(1, 0) / det;
    
    double_complex zt[2];
    double_complex zb[2];
    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++)
    {
        zt[0] = gkvec_phase_factors_(igkloc, ia) * alm_b_(0, igkloc, l, iat);
        zt[1] = gkvec_phase_factors_(igkloc, ia) * alm_b_(1, igkloc, l, iat);

        zb[0] = A(0, 0) * zt[0] + A(0, 1) * zt[1];
        zb[1] = A(1, 0) * zt[0] + A(1, 1) * zt[1];
        
        for (int lm = Utils::lm_by_l_m(l, -l); lm <= Utils::lm_by_l_m(l, l); lm++)
        {
            int idxb0 = type->indexb_by_lm_order(lm, 0);
            int idxb1 = type->indexb_by_lm_order(lm, 1);
                        
            alm(igkloc, idxb0) = gkvec_ylm_(lm, igkloc) * conj(zb[0]);
            alm(igkloc, idxb1) = gkvec_ylm_(lm, igkloc) * conj(zb[1]);
        }
    }
}

/// Second order matching coefficients, non-conjugated
template<> void K_point::generate_matching_coefficients_l<2, false>(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                                                    mdarray<double, 2>& A, mdarray<double_complex, 2>& alm)
{
    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    
    if ((fabs(det) < 1.0 / sqrt(parameters_.unit_cell()->omega())) && (debug_level >= 1))
    {   
        std::stringstream s;
        s << "Ill defined plane wave matching problem for atom " << ia << ", l = " << l << std::endl
          << "  radial function value at the MT boundary : " << A(0 ,0); 
        
        warning_local(__FILE__, __LINE__, s);
    }
    std::swap(A(0, 0), A(1, 1));
    A(0, 0) /= det;
    A(1, 1) /= det;
    A(0, 1) = -A(0, 1) / det;
    A(1, 0) = -A(1, 0) / det;
    
    double_complex zt[2];
    double_complex zb[2];
    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++)
    {
        zt[0] = gkvec_phase_factors_(igkloc, ia) * alm_b_(0, igkloc, l, iat);
        zt[1] = gkvec_phase_factors_(igkloc, ia) * alm_b_(1, igkloc, l, iat);

        zb[0] = A(0, 0) * zt[0] + A(0, 1) * zt[1];
        zb[1] = A(1, 0) * zt[0] + A(1, 1) * zt[1];

        for (int m = -l; m <= l; m++)
        {
            int idxb0 = type->indexb_by_l_m_order(l, m, 0);
            int idxb1 = type->indexb_by_l_m_order(l, m, 1);
                        
            alm(igkloc, idxb0) = conj(gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc)) * zb[0];
            alm(igkloc, idxb1) = conj(gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc)) * zb[1];
        }
    }
}

template<> void K_point::generate_matching_coefficients_l<3, true>(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                                                   mdarray<double, 2>& A, mdarray<double_complex, 2>& alm)
{
    linalg<lapack>::invert_ge(&A(0, 0), 3);
    
    double_complex zt[3];
    double_complex zb[3];
    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++)
    {
        zt[0] = gkvec_phase_factors_(igkloc, ia) * alm_b_(0, igkloc, l, iat);
        zt[1] = gkvec_phase_factors_(igkloc, ia) * alm_b_(1, igkloc, l, iat);
        zt[2] = gkvec_phase_factors_(igkloc, ia) * alm_b_(2, igkloc, l, iat);

        zb[0] = A(0, 0) * zt[0] + A(0, 1) * zt[1] + A(0, 2) * zt[2];
        zb[1] = A(1, 0) * zt[0] + A(1, 1) * zt[1] + A(1, 2) * zt[2];
        zb[2] = A(2, 0) * zt[0] + A(2, 1) * zt[1] + A(2, 2) * zt[2];

        for (int m = -l; m <= l; m++)
        {
            int idxb0 = type->indexb_by_l_m_order(l, m, 0);
            int idxb1 = type->indexb_by_l_m_order(l, m, 1);
            int idxb2 = type->indexb_by_l_m_order(l, m, 2);
                        
            alm(igkloc, idxb0) = gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc) * conj(zb[0]);
            alm(igkloc, idxb1) = gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc) * conj(zb[1]);
            alm(igkloc, idxb2) = gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc) * conj(zb[2]);
        }
    }
}

template<> void K_point::generate_matching_coefficients_l<3, false>(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                                                    mdarray<double, 2>& A, mdarray<double_complex, 2>& alm)
{
    linalg<lapack>::invert_ge(&A(0, 0), 3);
    
    double_complex zt[3];
    double_complex zb[3];
    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++)
    {
        zt[0] = gkvec_phase_factors_(igkloc, ia) * alm_b_(0, igkloc, l, iat);
        zt[1] = gkvec_phase_factors_(igkloc, ia) * alm_b_(1, igkloc, l, iat);
        zt[2] = gkvec_phase_factors_(igkloc, ia) * alm_b_(2, igkloc, l, iat);

        zb[0] = A(0, 0) * zt[0] + A(0, 1) * zt[1] + A(0, 2) * zt[2];
        zb[1] = A(1, 0) * zt[0] + A(1, 1) * zt[1] + A(1, 2) * zt[2];
        zb[2] = A(2, 0) * zt[0] + A(2, 1) * zt[1] + A(2, 2) * zt[2];

        for (int m = -l; m <= l; m++)
        {
            int idxb0 = type->indexb_by_l_m_order(l, m, 0);
            int idxb1 = type->indexb_by_l_m_order(l, m, 1);
            int idxb2 = type->indexb_by_l_m_order(l, m, 2);
                        
            alm(igkloc, idxb0) = conj(gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc)) * zb[0];
            alm(igkloc, idxb1) = conj(gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc)) * zb[1];
            alm(igkloc, idxb2) = conj(gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc)) * zb[2];
        }
    }
}

template<bool conjugate>
void K_point::generate_matching_coefficients(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm)
{
    Timer t("sirius::K_point::generate_matching_coefficients");

    Atom* atom = parameters_.unit_cell()->atom(ia);
    Atom_type* type = atom->type();

    assert(type->max_aw_order() <= 3);

    int iat = type->id();

    #pragma omp parallel default(shared)
    {
        mdarray<double, 2> A(3, 3);

        #pragma omp for
        for (int l = 0; l <= parameters_.lmax_apw(); l++)
        {
            int num_aw = (int)type->aw_descriptor(l).size();

            for (int order = 0; order < num_aw; order++)
            {
                for (int dm = 0; dm < num_aw; dm++) A(dm, order) = atom->symmetry_class()->aw_surface_dm(l, order, dm);
            }

            switch (num_aw)
            {
                case 1:
                {
                    generate_matching_coefficients_l<1, conjugate>(ia, iat, type, l, num_gkvec_loc, A, alm);
                    break;
                }
                case 2:
                {
                    generate_matching_coefficients_l<2, conjugate>(ia, iat, type, l, num_gkvec_loc, A, alm);
                    break;
                }
                case 3:
                {
                    generate_matching_coefficients_l<3, conjugate>(ia, iat, type, l, num_gkvec_loc, A, alm);
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong order of augmented wave");
                }
            }
        } //l
    }
    
    // check alm coefficients
    if (debug_level > 1) check_alm(num_gkvec_loc, ia, alm);
}

void K_point::check_alm(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm)
{
    static SHT* sht = NULL;
    if (!sht) sht = new SHT(parameters_.lmax_apw());

    Atom* atom = parameters_.unit_cell()->atom(ia);
    Atom_type* type = atom->type();

    mdarray<double_complex, 2> z1(sht->num_points(), type->mt_aw_basis_size());
    for (int i = 0; i < type->mt_aw_basis_size(); i++)
    {
        int lm = type->indexb(i).lm;
        int idxrf = type->indexb(i).idxrf;
        double rf = atom->symmetry_class()->radial_function(atom->num_mt_points() - 1, idxrf);
        for (int itp = 0; itp < sht->num_points(); itp++)
        {
            z1(itp, i) = sht->ylm_backward(lm, itp) * rf;
        }
    }

    mdarray<double_complex, 2> z2(sht->num_points(), num_gkvec_loc);
    blas<cpu>::gemm(0, 2, sht->num_points(), num_gkvec_loc, type->mt_aw_basis_size(), z1.get_ptr(), z1.ld(),
                    alm.get_ptr(), alm.ld(), z2.get_ptr(), z2.ld());

    vector3d<double> vc = parameters_.unit_cell()->get_cartesian_coordinates(parameters_.unit_cell()->atom(ia)->position());
    
    double tdiff = 0;
    for (int igloc = 0; igloc < num_gkvec_loc; igloc++)
    {
        vector3d<double> gkc = gkvec_cart(igkglob(igloc));
        for (int itp = 0; itp < sht->num_points(); itp++)
        {
            double_complex aw_value = z2(itp, igloc);
            vector3d<double> r;
            for (int x = 0; x < 3; x++) r[x] = vc[x] + sht->coord(x, itp) * type->mt_radius();
            double_complex pw_value = exp(double_complex(0, Utils::scalar_product(r, gkc))) / sqrt(parameters_.unit_cell()->omega());
            tdiff += abs(pw_value - aw_value);
        }
    }

    printf("atom : %i  absolute alm error : %e  average alm error : %e\n", 
           ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
}

inline void K_point::copy_lo_blocks(const double_complex* z, double_complex* vec)
{
    for (int j = num_gkvec_row(); j < apwlo_basis_size_row(); j++)
    {
        int ia = apwlo_basis_descriptors_row(j).ia;
        int lm = apwlo_basis_descriptors_row(j).lm;
        int order = apwlo_basis_descriptors_row(j).order;
        vec[parameters_.unit_cell()->atom(ia)->offset_wf() + parameters_.unit_cell()->atom(ia)->type()->indexb_by_lm_order(lm, order)] = z[j];
    }
}

inline void K_point::copy_pw_block(const double_complex* z, double_complex* vec)
{
    memset(vec, 0, num_gkvec() * sizeof(double_complex));

    for (int j = 0; j < num_gkvec_row(); j++) vec[apwlo_basis_descriptors_row(j).igk] = z[j];
}

void K_point::generate_fv_states()
{
    log_function_enter(__func__);
    Timer t("sirius::K_point::generate_fv_states");

    fv_states_col_.zero();

    mdarray<double_complex, 2> alm(num_gkvec_row(), parameters_.unit_cell()->max_mt_aw_basis_size());
    
    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            Atom* atom = parameters_.unit_cell()->atom(ia);
            Atom_type* type = atom->type();
            
            generate_matching_coefficients<true>(num_gkvec_row(), ia, alm);

            blas<cpu>::gemm(2, 0, type->mt_aw_basis_size(), parameters_.spl_fv_states_col().local_size(),
                            num_gkvec_row(), &alm(0, 0), alm.ld(), &fv_eigen_vectors_(0, 0), 
                            fv_eigen_vectors_.ld(), &fv_states_col_(atom->offset_wf(), 0), 
                            fv_states_col_.ld());
        }
    }

    for (int j = 0; j < parameters_.spl_fv_states_col().local_size(); j++)
    {
        copy_lo_blocks(&fv_eigen_vectors_(0, j), &fv_states_col_(0, j));

        copy_pw_block(&fv_eigen_vectors_(0, j), &fv_states_col_(parameters_.unit_cell()->mt_basis_size(), j));
    }

    for (int j = 0; j < parameters_.spl_fv_states_col().local_size(); j++)
    {
        Platform::allreduce(&fv_states_col_(0, j), wf_size(), parameters_.mpi_grid().communicator(1 << _dim_row_));
    }

    log_function_exit(__func__);
}

void K_point::generate_spinor_wave_functions()
{
    log_function_enter(__func__);
    Timer t("sirius::K_point::generate_spinor_wave_functions");

    int wfld = spinor_wave_functions_.size(0) * spinor_wave_functions_.size(1); // size of each spinor wave-function
    int nrow = parameters_.spl_fv_states_row().local_size();
    int ncol = parameters_.spl_fv_states_col().local_size();

    if (use_second_variation) 
    {
        if (!parameters_.need_sv()) return;

        spinor_wave_functions_.zero();


        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        {
            if (parameters_.num_mag_dims() != 3)
            {
                // multiply up block for first half of the bands, dn block for second half of the bands
                blas<cpu>::gemm(0, 0, wf_size(), ncol, nrow, &fv_states_row_(0, 0), fv_states_row_.ld(), 
                                &sv_eigen_vectors_(0, ispn * ncol), sv_eigen_vectors_.ld(), 
                                &spinor_wave_functions_(0, ispn, ispn * ncol), wfld);
            }
            else
            {
                // multiply up block and then dn block for all bands
                blas<cpu>::gemm(0, 0, wf_size(), parameters_.spl_spinor_wf_col().local_size(), nrow, 
                                &fv_states_row_(0, 0), fv_states_row_.ld(), 
                                &sv_eigen_vectors_(ispn * nrow, 0), sv_eigen_vectors_.ld(), 
                                &spinor_wave_functions_(0, ispn, 0), wfld);
            }
        }
    }
    else
    {
        mdarray<double_complex, 2> alm(num_gkvec_row(), parameters_.unit_cell()->max_mt_aw_basis_size());

        /** \todo generalize for non-collinear case */
        spinor_wave_functions_.zero();
        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        {
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                Atom* atom = parameters_.unit_cell()->atom(ia);
                Atom_type* type = atom->type();
                
                /** \todo generate unconjugated coefficients for better readability */
                generate_matching_coefficients<true>(num_gkvec_row(), ia, alm);

                blas<cpu>::gemm(2, 0, type->mt_aw_basis_size(), ncol, num_gkvec_row(), &alm(0, 0), alm.ld(), 
                                &fd_eigen_vectors_(0, ispn * ncol), fd_eigen_vectors_.ld(), 
                                &spinor_wave_functions_(atom->offset_wf(), ispn, ispn * ncol), wfld); 
            }

            for (int j = 0; j < ncol; j++)
            {
                copy_lo_blocks(&fd_eigen_vectors_(0, j + ispn * ncol), &spinor_wave_functions_(0, ispn, j + ispn * ncol));

                copy_pw_block(&fd_eigen_vectors_(0, j + ispn * ncol), &spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size(), ispn, j + ispn * ncol));
            }
        }
        /** \todo how to distribute states in case of full diagonalziation. num_fv_states will probably be reused. 
                  maybe the 'fv' should be renamed. */
    }
    
    for (int i = 0; i < parameters_.spl_spinor_wf_col().local_size(); i++)
        Platform::allreduce(&spinor_wave_functions_(0, 0, i), wfld, parameters_.mpi_grid().communicator(1 << _dim_row_));
    
    log_function_exit(__func__);
}

void K_point::generate_gkvec(double gk_cutoff)
{
    if ((gk_cutoff * parameters_.unit_cell()->max_mt_radius() > double(parameters_.lmax_apw())) && 
        parameters_.unit_cell()->full_potential())
    {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff << ") is too large for a given lmax (" 
          << parameters_.lmax_apw() << ") and a maximum MT radius (" << parameters_.unit_cell()->max_mt_radius() << ")" << std::endl
          << "minimum value for lmax : " << int(gk_cutoff * parameters_.unit_cell()->max_mt_radius()) + 1;
        error_local(__FILE__, __LINE__, s);
    }

    if (gk_cutoff * 2 > parameters_.pw_cutoff())
    {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << parameters_.pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff * 2;
        error_local(__FILE__, __LINE__, s);
    }

    std::vector< std::pair<double, int> > gkmap;

    // find G-vectors for which |G+k| < cutoff
    for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
    {
        vector3d<double> vgk;
        for (int x = 0; x < 3; x++) vgk[x] = parameters_.reciprocal_lattice()->gvec(ig)[x] + vk_[x];

        vector3d<double> v = parameters_.reciprocal_lattice()->get_cartesian_coordinates(vgk);
        double gklen = v.length();

        if (gklen <= gk_cutoff) gkmap.push_back(std::pair<double, int>(gklen, ig));
    }

    std::sort(gkmap.begin(), gkmap.end());

    gkvec_.set_dimensions(3, (int)gkmap.size());
    gkvec_.allocate();

    gkvec_gpu_.set_dimensions((int)gkmap.size(), 3);
    gkvec_gpu_.allocate();

    gvec_index_.resize(gkmap.size());

    for (int ig = 0; ig < (int)gkmap.size(); ig++)
    {
        gvec_index_[ig] = gkmap[ig].second;
        for (int x = 0; x < 3; x++)
        {
            gkvec_(x, ig) = parameters_.reciprocal_lattice()->gvec(gkmap[ig].second)[x] + vk_[x];
            gkvec_gpu_(ig, x) = gkvec_(x, ig);
        }
    }
    
    fft_index_.resize(num_gkvec());
    for (int igk = 0; igk < num_gkvec(); igk++) fft_index_[igk] = parameters_.reciprocal_lattice()->fft_index(gvec_index_[igk]);

    if (parameters_.esm_type() == ultrasoft_pseudopotential)
    {
        fft_index_coarse_.resize(num_gkvec());
        for (int igk = 0; igk < num_gkvec(); igk++)
        {
            int ig = gvec_index_[igk]; // G-vector index in the fine mesh
            vector3d<int> gvec = parameters_.reciprocal_lattice()->gvec(ig); // G-vector lattice coordinates

            // linear index inside coarse FFT buffer
            fft_index_coarse_[igk] = parameters_.reciprocal_lattice()->fft_coarse()->index(gvec[0], gvec[1], gvec[2]);
        }
    }
}

template <index_domain_t index_domain> 
void K_point::init_gkvec_ylm_and_len(int lmax, int ngk)
{
    gkvec_ylm_.set_dimensions(Utils::lmmax(lmax), ngk);
    gkvec_ylm_.allocate();
    
    gkvec_len_.resize(ngk);

    #pragma omp parallel for default(shared)
    for (int igk = 0; igk < ngk; igk++)
    {
        int igk_glob = (index_domain == global) ? igk : igkglob(igk);

        double vs[3];

        SHT::spherical_coordinates(gkvec_cart(igk_glob), vs); // vs = {r, theta, phi}
        
        SHT::spherical_harmonics(lmax, vs[1], vs[2], &gkvec_ylm_(0, igk));
        
        gkvec_len_[igk] = vs[0];
    }
}

template <index_domain_t index_domain> 
void K_point::init_gkvec_phase_factors(int ngk)
{
    gkvec_phase_factors_.set_dimensions(ngk, parameters_.unit_cell()->num_atoms());
    gkvec_phase_factors_.allocate();

    #pragma omp parallel for default(shared)
    for (int igk = 0; igk < ngk; igk++)
    {
        int igk_glob = (index_domain == global) ? igk : igkglob(igk);

        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            double phase = twopi * Utils::scalar_product(gkvec(igk_glob), parameters_.unit_cell()->atom(ia)->position());

            gkvec_phase_factors_(igk, ia) = exp(double_complex(0.0, phase));
        }
    }
}

void K_point::init_gkvec()
{
    int lmax = -1;
    
    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            lmax = parameters_.lmax_apw();
            break;
        }
        case full_potential_pwlo:
        {
            lmax =  parameters_.lmax_pw();
            break;
        }
        case ultrasoft_pseudopotential:
        {
            lmax = parameters_.lmax_beta();
            break;
        }
    }
    
    // spherical harmonics of G+k vectors
    if (parameters_.unit_cell()->full_potential())
    {
        init_gkvec_ylm_and_len<local>(lmax, num_gkvec_loc());
        init_gkvec_phase_factors<local>(num_gkvec_loc());
    }

    if (parameters_.esm_type() == ultrasoft_pseudopotential)
    {
        if (num_gkvec() != wf_size()) error_local(__FILE__, __LINE__, "wrong size of wave-functions");
        init_gkvec_ylm_and_len<global>(lmax, num_gkvec());
        init_gkvec_phase_factors<global>(num_gkvec());
    }
}

void K_point::build_apwlo_basis_descriptors()
{
    apwlo_basis_descriptors_.clear();

    apwlo_basis_descriptor apwlobd;

    // G+k basis functions
    for (int igk = 0; igk < num_gkvec(); igk++)
    {
        apwlobd.igk = igk;
        apwlobd.ig = gvec_index(igk);
        apwlobd.ia = -1;
        apwlobd.lm = -1;
        apwlobd.l = -1;
        apwlobd.order = -1;
        apwlobd.idxrf = -1;
        apwlobd.idxglob = (int)apwlo_basis_descriptors_.size();

        apwlobd.gkvec = gkvec(igk);
        apwlobd.gkvec_cart = gkvec_cart(igk);

        apwlo_basis_descriptors_.push_back(apwlobd);
    }

    // local orbital basis functions
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        Atom* atom = parameters_.unit_cell()->atom(ia);
        Atom_type* type = atom->type();
    
        int lo_index_offset = type->mt_aw_basis_size();
        
        for (int j = 0; j < type->mt_lo_basis_size(); j++) 
        {
            int l = type->indexb(lo_index_offset + j).l;
            int lm = type->indexb(lo_index_offset + j).lm;
            int order = type->indexb(lo_index_offset + j).order;
            int idxrf = type->indexb(lo_index_offset + j).idxrf;
            apwlobd.igk = -1;
            apwlobd.ig = -1;
            apwlobd.ia = ia;
            apwlobd.lm = lm;
            apwlobd.l = l;
            apwlobd.order = order;
            apwlobd.idxrf = idxrf;
            apwlobd.idxglob = (int)apwlo_basis_descriptors_.size();

            apwlo_basis_descriptors_.push_back(apwlobd);
        }
    }
    
    // ckeck if we count basis functions correctly
    if ((int)apwlo_basis_descriptors_.size() != (num_gkvec() + parameters_.unit_cell()->mt_lo_basis_size()))
    {
        std::stringstream s;
        s << "(L)APW+lo basis descriptors array has a wrong size" << std::endl
          << "size of apwlo_basis_descriptors_ : " << apwlo_basis_descriptors_.size() << std::endl
          << "num_gkvec : " << num_gkvec() << std::endl 
          << "mt_lo_basis_size : " << parameters_.unit_cell()->mt_lo_basis_size();
        error_local(__FILE__, __LINE__, s);
    }
}

/// Block-cyclic distribution of relevant arrays 
void K_point::distribute_block_cyclic()
{
    // distribute APW+lo basis between rows
    splindex<block_cyclic> spl_row(apwlo_basis_size(), num_ranks_row_, rank_row_, parameters_.cyclic_block_size());
    apwlo_basis_descriptors_row_.resize(spl_row.local_size());
    for (int i = 0; i < spl_row.local_size(); i++)
        apwlo_basis_descriptors_row_[i] = apwlo_basis_descriptors_[spl_row[i]];

    // distribute APW+lo basis between columns
    splindex<block_cyclic> spl_col(apwlo_basis_size(), num_ranks_col_, rank_col_, parameters_.cyclic_block_size());
    apwlo_basis_descriptors_col_.resize(spl_col.local_size());
    for (int i = 0; i < spl_col.local_size(); i++)
        apwlo_basis_descriptors_col_[i] = apwlo_basis_descriptors_[spl_col[i]];
    
    #if defined(_SCALAPACK) || defined(_ELPA_)
    if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa)
    {
        int nr = linalg<scalapack>::numroc(apwlo_basis_size(), parameters_.cyclic_block_size(), 
                                           band->rank_row(), 0, band->num_ranks_row());
        
        if (nr != apwlo_basis_size_row()) 
            error_local(__FILE__, __LINE__, "numroc returned a different local row size");

        int nc = linalg<scalapack>::numroc(apwlo_basis_size(), parameters_.cyclic_block_size(), 
                                           band->rank_col(), 0, band->num_ranks_col());
        
        if (nc != apwlo_basis_size_col()) 
            error_local(__FILE__, __LINE__, "numroc returned a different local column size");
    }
    #endif

    // get the number of row- and column- G+k-vectors
    num_gkvec_row_ = 0;
    for (int i = 0; i < apwlo_basis_size_row(); i++)
    {
        if (apwlo_basis_descriptors_row_[i].igk != -1) num_gkvec_row_++;
    }
    
    num_gkvec_col_ = 0;
    for (int i = 0; i < apwlo_basis_size_col(); i++)
    {
        if (apwlo_basis_descriptors_col_[i].igk != -1) num_gkvec_col_++;
    }
}

//Periodic_function<double_complex>* K_point::spinor_wave_function_component(Band* band, int lmax, int ispn, int jloc)
//{
//    Timer t("sirius::K_point::spinor_wave_function_component");
//
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//
//    Periodic_function<double_complex, index_order>* func = 
//        new Periodic_function<double_complex, index_order>(parameters_, lmax);
//    func->allocate(ylm_component | it_component);
//    func->zero();
//    
//    if (basis_type == pwlo)
//    {
//        if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//
//        double fourpi_omega = fourpi / sqrt(parameters_.omega());
//        
//        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//        {
//            int igk = igkglob(igkloc);
//            double_complex z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//
//            // TODO: possilbe optimization with zgemm
//            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//            {
//                int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//                
//                #pragma omp parallel for default(shared)
//                for (int lm = 0; lm < lmmax; lm++)
//                {
//                    int l = l_by_lm_(lm);
//                    double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                        func->f_ylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//                }
//            }
//        }
//
//        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&func->f_ylm(0, 0, ia), lmmax * parameters_.max_num_mt_points(),
//                                parameters_.mpi_grid().communicator(1 << band->dim_row()));
//        }
//    }
//
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
//        {
//            int lm = parameters_.atom(ia)->type()->indexb(i).lm;
//            int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
//            switch (index_order)
//            {
//                case angular_radial:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(lm, ir, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//                case radial_angular:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(ir, lm, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//            }
//        }
//    }
//
//    // in principle, wave function must have an overall e^{ikr} phase factor
//    parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//                            &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, jloc));
//    parameters_.fft().transform(1);
//    parameters_.fft().output(func->f_it());
//
//    for (int i = 0; i < parameters_.fft().size(); i++) func->f_it(i) /= sqrt(parameters_.omega());
//    
//    return func;
//}

//== void K_point::spinor_wave_function_component_mt(int lmax, int ispn, int jloc, mt_functions<double_complex>& psilm)
//== {
//==     Timer t("sirius::K_point::spinor_wave_function_component_mt");
//== 
//==     //int lmmax = Utils::lmmax_by_lmax(lmax);
//== 
//==     psilm.zero();
//==     
//==     //if (basis_type == pwlo)
//==     //{
//==     //    if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//== 
//==     //    double fourpi_omega = fourpi / sqrt(parameters_.omega());
//== 
//==     //    mdarray<double_complex, 2> zm(parameters_.max_num_mt_points(),  num_gkvec_row());
//== 
//==     //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     //    {
//==     //        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//==     //        for (int l = 0; l <= lmax; l++)
//==     //        {
//==     //            #pragma omp parallel for default(shared)
//==     //            for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //            {
//==     //                int igk = igkglob(igkloc);
//==     //                double_complex z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//==     //                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia) * zil_[l];
//==     //                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//==     //                    zm(ir, igkloc) = z2 * (*sbessel_[igkloc])(ir, l, iat);
//==     //            }
//==     //            blas<cpu>::gemm(0, 2, parameters_.atom(ia)->num_mt_points(), (2 * l + 1), num_gkvec_row(),
//==     //                            &zm(0, 0), zm.ld(), &gkvec_ylm_(Utils::lm_by_l_m(l, -l), 0), gkvec_ylm_.ld(), 
//==     //                            &fylm(0, Utils::lm_by_l_m(l, -l), ia), fylm.ld());
//==     //        }
//==     //    }
//==     //    //for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //    //{
//==     //    //    int igk = igkglob(igkloc);
//==     //    //    double_complex z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//== 
//==     //    //    // TODO: possilbe optimization with zgemm
//==     //    //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     //    //    {
//==     //    //        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//==     //    //        double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//==     //    //        
//==     //    //        #pragma omp parallel for default(shared)
//==     //    //        for (int lm = 0; lm < lmmax; lm++)
//==     //    //        {
//==     //    //            int l = l_by_lm_(lm);
//==     //    //            double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//==     //    //            for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//==     //    //                fylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//==     //    //        }
//==     //    //    }
//==     //    //}
//== 
//==     //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     //    {
//==     //        Platform::allreduce(&fylm(0, 0, ia), lmmax * parameters_.max_num_mt_points(),
//==     //                            parameters_.mpi_grid().communicator(1 << band->dim_row()));
//==     //    }
//==     //}
//== 
//==     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     {
//==         for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
//==         {
//==             int lm = parameters_.atom(ia)->type()->indexb(i).lm;
//==             int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
//==             for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//==             {
//==                 psilm(lm, ir, ia) += 
//==                     spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//==                     parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//==             }
//==         }
//==     }
//== }

void K_point::test_fv_states(int use_fft)
{
    std::vector<double_complex> v1;
    std::vector<double_complex> v2;
    
    if (use_fft == 0) 
    {
        v1.resize(num_gkvec());
        v2.resize(fft_->size());
    }
    
    if (use_fft == 1) 
    {
        v1.resize(fft_->size());
        v2.resize(fft_->size());
    }
    
    double maxerr = 0;

    for (int j1 = 0; j1 < parameters_.spl_fv_states_col().local_size(); j1++)
    {
        if (use_fft == 0)
        {
            fft_->input(num_gkvec(), &fft_index_[0], &fv_states_col_(parameters_.unit_cell()->mt_basis_size(), j1));
            fft_->transform(1);
            fft_->output(&v2[0]);

            for (int ir = 0; ir < fft_->size(); ir++) 
                v2[ir] *= parameters_.step_function(ir);
            
            fft_->input(&v2[0]);
            fft_->transform(-1);
            fft_->output(num_gkvec(), &fft_index_[0], &v1[0]); 
        }
        
        if (use_fft == 1)
        {
            fft_->input(num_gkvec(), &fft_index_[0], &fv_states_col_(parameters_.unit_cell()->mt_basis_size(), j1));
            fft_->transform(1);
            fft_->output(&v1[0]);
        }
       
        for (int j2 = 0; j2 < parameters_.spl_fv_states_row().local_size(); j2++)
        {
            double_complex zsum(0, 0);
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
                Atom_type* type = parameters_.unit_cell()->atom(ia)->type();
                Atom_symmetry_class* symmetry_class = parameters_.unit_cell()->atom(ia)->symmetry_class();

                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    int ordmax = type->indexr().num_rf(l);
                    for (int io1 = 0; io1 < ordmax; io1++)
                    {
                        for (int io2 = 0; io2 < ordmax; io2++)
                        {
                            for (int m = -l; m <= l; m++)
                            {
                                zsum += conj(fv_states_col_(offset_wf + type->indexb_by_l_m_order(l, m, io1), j1)) *
                                             fv_states_row_(offset_wf + type->indexb_by_l_m_order(l, m, io2), j2) * 
                                             symmetry_class->o_radial_integral(l, io1, io2);
                            }
                        }
                    }
                }
            }
            
            if (use_fft == 0)
            {
               for (int ig = 0; ig < num_gkvec(); ig++)
                   zsum += conj(v1[ig]) * fv_states_row_(parameters_.unit_cell()->mt_basis_size() + ig, j2);
            }
           
            if (use_fft == 1)
            {
                fft_->input(num_gkvec(), &fft_index_[0], &fv_states_row_(parameters_.unit_cell()->mt_basis_size(), j2));
                fft_->transform(1);
                fft_->output(&v2[0]);

                for (int ir = 0; ir < fft_->size(); ir++)
                    zsum += conj(v1[ir]) * v2[ir] * parameters_.step_function(ir) / double(fft_->size());
            }
            
            if (use_fft == 2) 
            {
                for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
                {
                    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
                    {
                        int ig3 = parameters_.reciprocal_lattice()->index_g12(gvec_index(ig1), gvec_index(ig2));
                        zsum += conj(fv_states_col_(parameters_.unit_cell()->mt_basis_size() + ig1, j1)) * 
                                     fv_states_row_(parameters_.unit_cell()->mt_basis_size() + ig2, j2) * 
                                parameters_.step_function()->theta_pw(ig3);
                    }
               }
            }

            if (parameters_.spl_fv_states_col(j1) == parameters_.spl_fv_states_row(j2)) zsum = zsum - double_complex(1, 0);
           
            maxerr = std::max(maxerr, abs(zsum));
        }
    }

    Platform::allreduce<op_max>(&maxerr, 1, parameters_.mpi_grid().communicator(1 << _dim_row_ | 1 << _dim_col_));

    if (parameters_.mpi_grid().side(1 << _dim_k_)) 
    {
        printf("k-point: %f %f %f, interstitial integration : %i, maximum error : %18.10e\n", 
               vk_[0], vk_[1], vk_[2], use_fft, maxerr);
    }
}

void K_point::test_spinor_wave_functions(int use_fft)
{
    if (num_ranks() > 1) error_local(__FILE__, __LINE__, "test of spinor wave functions on multiple ranks is not implemented");

    std::vector<double_complex> v1[2];
    std::vector<double_complex> v2;

    if (use_fft == 0 || use_fft == 1) v2.resize(fft_->size());
    
    if (use_fft == 0) 
    {
        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++) v1[ispn].resize(num_gkvec());
    }
    
    if (use_fft == 1) 
    {
        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++) v1[ispn].resize(fft_->size());
    }
    
    double maxerr = 0;

    for (int j1 = 0; j1 < parameters_.num_bands(); j1++)
    {
        if (use_fft == 0)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                fft_->input(num_gkvec(), &fft_index_[0], 
                                       &spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size(), ispn, j1));
                fft_->transform(1);
                fft_->output(&v2[0]);

                for (int ir = 0; ir < fft_->size(); ir++) v2[ir] *= parameters_.step_function(ir);
                
                fft_->input(&v2[0]);
                fft_->transform(-1);
                fft_->output(num_gkvec(), &fft_index_[0], &v1[ispn][0]); 
            }
        }
        
        if (use_fft == 1)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                fft_->input(num_gkvec(), &fft_index_[0], 
                                       &spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size(), ispn, j1));
                fft_->transform(1);
                fft_->output(&v1[ispn][0]);
            }
        }
       
        for (int j2 = 0; j2 < parameters_.num_bands(); j2++)
        {
            double_complex zsum(0, 0);
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
                {
                    int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
                    Atom_type* type = parameters_.unit_cell()->atom(ia)->type();
                    Atom_symmetry_class* symmetry_class = parameters_.unit_cell()->atom(ia)->symmetry_class();

                    for (int l = 0; l <= parameters_.lmax_apw(); l++)
                    {
                        int ordmax = type->indexr().num_rf(l);
                        for (int io1 = 0; io1 < ordmax; io1++)
                        {
                            for (int io2 = 0; io2 < ordmax; io2++)
                            {
                                for (int m = -l; m <= l; m++)
                                {
                                    zsum += conj(spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io1), ispn, j1)) *
                                            spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io2), ispn, j2) * 
                                            symmetry_class->o_radial_integral(l, io1, io2);
                                }
                            }
                        }
                    }
                }
            }
            
            if (use_fft == 0)
            {
               for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
               {
                   for (int ig = 0; ig < num_gkvec(); ig++)
                       zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size() + ig, ispn, j2);
               }
            }
           
            if (use_fft == 1)
            {
                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                {
                    fft_->input(num_gkvec(), &fft_index_[0], 
                                           &spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size(), ispn, j2));
                    fft_->transform(1);
                    fft_->output(&v2[0]);

                    for (int ir = 0; ir < fft_->size(); ir++)
                        zsum += conj(v1[ispn][ir]) * v2[ir] * parameters_.step_function(ir) / double(fft_->size());
                }
            }
            
            if (use_fft == 2) 
            {
                for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
                {
                    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
                    {
                        int ig3 = parameters_.reciprocal_lattice()->index_g12(gvec_index(ig1), gvec_index(ig2));
                        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                        {
                            zsum += conj(spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size() + ig1, ispn, j1)) * 
                                    spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size() + ig2, ispn, j2) * 
                                    parameters_.step_function()->theta_pw(ig3);
                        }
                    }
               }
           }

           zsum = (j1 == j2) ? zsum - double_complex(1.0, 0.0) : zsum;
           maxerr = std::max(maxerr, abs(zsum));
        }
    }
    std :: cout << "maximum error = " << maxerr << std::endl;
}

void K_point::save(int id)
{
    if (num_ranks() > 1) error_local(__FILE__, __LINE__, "writing of distributed eigen-vectors is not implemented");

    if (parameters_.mpi_grid().root(1 << _dim_col_))
    {
        HDF5_tree fout(storage_file_name, false);

        fout["K_set"].create_node(id);
        fout["K_set"][id].create_node("spinor_wave_functions");
        fout["K_set"][id].write("coordinates", &vk_[0], 3);
        fout["K_set"][id].write("band_energies", band_energies_);
        fout["K_set"][id].write("band_occupancies", band_occupancies_);
        if (num_ranks() == 1)
        {
            fout["K_set"][id].write_mdarray("fv_eigen_vectors", fv_eigen_vectors_);
            fout["K_set"][id].write_mdarray("sv_eigen_vectors", sv_eigen_vectors_);
        }
    }
    
    Platform::barrier(parameters_.mpi_grid().communicator(1 << _dim_col_));
    
    mdarray<double_complex, 2> wfj(NULL, wf_size(), parameters_.num_spins()); 
    for (int j = 0; j < parameters_.num_bands(); j++)
    {
        int rank = parameters_.spl_spinor_wf_col().location(_splindex_rank_, j);
        int offs = parameters_.spl_spinor_wf_col().location(_splindex_offs_, j);
        if (parameters_.mpi_grid().coordinate(_dim_col_) == rank)
        {
            HDF5_tree fout(storage_file_name, false);
            wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
            fout["K_set"][id]["spinor_wave_functions"].write_mdarray(j, wfj);
        }
        Platform::barrier(parameters_.mpi_grid().communicator(_dim_col_));
    }
}

void K_point::load(HDF5_tree h5in, int id)
{
    band_energies_.resize(parameters_.num_bands());
    h5in[id].read("band_energies", band_energies_);

    band_occupancies_.resize(parameters_.num_bands());
    h5in[id].read("band_occupancies", band_occupancies_);
    
    h5in[id].read_mdarray("fv_eigen_vectors", fv_eigen_vectors_);
    h5in[id].read_mdarray("sv_eigen_vectors", sv_eigen_vectors_);
}

//== void K_point::save_wave_functions(int id)
//== {
//==     if (parameters_.mpi_grid().root(1 << _dim_col_))
//==     {
//==         HDF5_tree fout(storage_file_name, false);
//== 
//==         fout["K_points"].create_node(id);
//==         fout["K_points"][id].write("coordinates", &vk_[0], 3);
//==         fout["K_points"][id].write("mtgk_size", mtgk_size());
//==         fout["K_points"][id].create_node("spinor_wave_functions");
//==         fout["K_points"][id].write("band_energies", &band_energies_[0], parameters_.num_bands());
//==         fout["K_points"][id].write("band_occupancies", &band_occupancies_[0], parameters_.num_bands());
//==     }
//==     
//==     Platform::barrier(parameters_.mpi_grid().communicator(1 << _dim_col_));
//==     
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), parameters_.num_spins()); 
//==     for (int j = 0; j < parameters_.num_bands(); j++)
//==     {
//==         int rank = parameters_.spl_spinor_wf_col().location(_splindex_rank_, j);
//==         int offs = parameters_.spl_spinor_wf_col().location(_splindex_offs_, j);
//==         if (parameters_.mpi_grid().coordinate(_dim_col_) == rank)
//==         {
//==             HDF5_tree fout(storage_file_name, false);
//==             wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
//==             fout["K_points"][id]["spinor_wave_functions"].write_mdarray(j, wfj);
//==         }
//==         Platform::barrier(parameters_.mpi_grid().communicator(_dim_col_));
//==     }
//== }
//== 
//== void K_point::load_wave_functions(int id)
//== {
//==     HDF5_tree fin(storage_file_name, false);
//==     
//==     int mtgk_size_in;
//==     fin["K_points"][id].read("mtgk_size", &mtgk_size_in);
//==     if (mtgk_size_in != mtgk_size()) error_local(__FILE__, __LINE__, "wrong wave-function size");
//== 
//==     band_energies_.resize(parameters_.num_bands());
//==     fin["K_points"][id].read("band_energies", &band_energies_[0], parameters_.num_bands());
//== 
//==     band_occupancies_.resize(parameters_.num_bands());
//==     fin["K_points"][id].read("band_occupancies", &band_occupancies_[0], parameters_.num_bands());
//== 
//==     spinor_wave_functions_.set_dimensions(mtgk_size(), parameters_.num_spins(), 
//==                                           parameters_.spl_spinor_wf_col().local_size());
//==     spinor_wave_functions_.allocate();
//== 
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), parameters_.num_spins()); 
//==     for (int jloc = 0; jloc < parameters_.spl_spinor_wf_col().local_size(); jloc++)
//==     {
//==         int j = parameters_.spl_spinor_wf_col(jloc);
//==         wfj.set_ptr(&spinor_wave_functions_(0, 0, jloc));
//==         fin["K_points"][id]["spinor_wave_functions"].read_mdarray(j, wfj);
//==     }
//== }

void K_point::get_fv_eigen_vectors(mdarray<double_complex, 2>& fv_evec)
{
    assert(fv_evec.size(0) >= apwlo_basis_size());
    assert(fv_evec.size(1) == parameters_.num_fv_states());
    
    fv_evec.zero();

    for (int iloc = 0; iloc < parameters_.spl_fv_states_col().local_size(); iloc++)
    {
        int i = parameters_.spl_fv_states_col(iloc);
        for (int jloc = 0; jloc < apwlo_basis_size_row(); jloc++)
        {
            int j = apwlo_basis_descriptors_row(jloc).idxglob;
            fv_evec(j, i) = fv_eigen_vectors_(jloc, iloc);
        }
    }
    Platform::allreduce(fv_evec.get_ptr(), (int)fv_evec.size(), 
                        parameters_.mpi_grid().communicator((1 << _dim_row_) | (1 << _dim_col_)));
}

void K_point::get_sv_eigen_vectors(mdarray<double_complex, 2>& sv_evec)
{
    assert(sv_evec.size(0) == parameters_.num_bands());
    assert(sv_evec.size(1) == parameters_.num_bands());

    sv_evec.zero();

    if (parameters_.num_mag_dims() == 0)
    {
        for (int iloc = 0; iloc < parameters_.spl_spinor_wf_col().local_size(); iloc++)
        {
            int i = parameters_.spl_spinor_wf_col(iloc);
            for (int jloc = 0; jloc < parameters_.spl_fv_states_row().local_size(); jloc++)
            {
                int j = parameters_.spl_fv_states_row(jloc);
                sv_evec(j, i) = sv_eigen_vectors_(jloc, iloc);
            }
        }
    }
    if (parameters_.num_mag_dims() == 1)
    {
        assert(sv_eigen_vectors_.size(0) == parameters_.num_fv_states());

        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        {
            for (int i = 0; i < parameters_.num_fv_states(); i++)
            {
                memcpy(&sv_evec(ispn * parameters_.num_fv_states(), ispn * parameters_.num_fv_states() + i), 
                       &sv_eigen_vectors_(0, ispn * parameters_.num_fv_states() + i), 
                       sv_eigen_vectors_.size(0) * sizeof(double_complex));
            }
        }
    }
    if (parameters_.num_mag_dims() == 3)
    {
        assert(sv_eigen_vectors_.size(0) == parameters_.num_bands());
        for (int i = 0; i < parameters_.num_bands(); i++)
            memcpy(&sv_evec(0, i), &sv_eigen_vectors_(0, i), sv_eigen_vectors_.size(0) * sizeof(double_complex));
    }
    
    Platform::allreduce(sv_evec.get_ptr(), (int)sv_evec.size(), 
                        parameters_.mpi_grid().communicator((1 << _dim_row_) | (1 << _dim_col_)));
}

void K_point::distribute_fv_states_row()
{
    if (num_ranks_ == 1) return;

    for (int i = 0; i < parameters_.spl_fv_states_row().local_size(); i++)
    {
        int ist = parameters_.spl_fv_states_row(i);
        
        // find local column index of fv state
        int offset_col = parameters_.spl_fv_states_col().location(_splindex_offs_, ist);
        int root_col = parameters_.spl_fv_states_col().location(_splindex_rank_, ist);
        
        // find column MPI rank which stores this fv state and copy fv state if this rank stores it
        if (rank_col() == root_col) memcpy(&fv_states_row_(0, i), &fv_states_col_(0, offset_col), wf_size() * sizeof(double_complex));
        
        // send fv state to all column MPI ranks; communication happens between the columns of the MPI grid
        Platform::bcast(&fv_states_row_(0, i), wf_size(), parameters_.mpi_grid().communicator(1 << _dim_col_), root_col); 
    }
}

void K_point::generate_beta_pw(double_complex* beta_pw__, int ia)
{
    Timer t("sirius::K_point::generate_beta_pw");
    auto atom_type = parameters_.unit_cell()->atom(ia)->type();
    int iat = atom_type->id();
    
    mdarray<double_complex, 2> beta_pw(beta_pw__, num_gkvec(), atom_type->mt_basis_size());
    
    for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
    {
        //== int l = atom_type->indexb(xi).l;
        //== int lm = atom_type->indexb(xi).lm;
        //== int idxrf = atom_type->indexb(xi).idxrf;

        //== double_complex z = pow(double_complex(0, -1), l) * fourpi / sqrt(parameters_.unit_cell()->omega());
        for (int igk = 0; igk < num_gkvec(); igk++)
        {
            //== beta_pw(igk, xi) = z * gkvec_ylm_(lm, igk) * beta_radial_integrals_(igk, idxrf, iat) * 
            //==                    conj(gkvec_phase_factors_(igk, ia));
            beta_pw(igk, xi) = beta_pw_(igk, parameters_.unit_cell()->beta_t_ofs(iat) + xi) * conj(gkvec_phase_factors_(igk, ia));
        }
    }
}

void K_point::generate_beta_pw(double_complex* beta_pw__, Atom_type* atom_type)
{
    Timer t("sirius::K_point::generate_beta_pw");
    int iat = atom_type->id();
    
    mdarray<double_complex, 2> beta_pw(beta_pw__, num_gkvec(), atom_type->mt_basis_size());
    
    for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
    {
        for (int igk = 0; igk < num_gkvec(); igk++)
        {
            beta_pw(igk, xi) = beta_pw_(igk, parameters_.unit_cell()->beta_t_ofs(iat) + xi);
        }
    }
}




