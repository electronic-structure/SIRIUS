template <typename T>
inline void Band::initialize_subspace(K_point* kp__,
                                      Periodic_function<double>* effective_potential__,
                                      Periodic_function<double>* effective_magnetic_field__[3],
                                      int num_ao__,
                                      int lmax__,
                                      std::vector< std::vector< Spline<double> > >& rad_int__) const
{
    PROFILE_WITH_TIMER("sirius::Band::initialize_subspace");

    /* number of basis functions */
    int num_phi = std::max(num_ao__, ctx_.num_fv_states());

    wave_functions phi(ctx_, kp__->comm(), kp__->gkvec(), num_phi);

    #pragma omp parallel
    {
        std::vector<double> gkvec_rlm(Utils::lmmax(lmax__));
        /* fill first N functions with atomic orbitals */
        #pragma omp for
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->gkvec().gvec_offset(kp__->comm().rank()) + igk_loc;
            /* vs = {r, theta, phi} */
            auto vs = SHT::spherical_coordinates(kp__->gkvec().gkvec_cart(igk));
            /* compute real spherical harmonics for G+k vector */
            SHT::spherical_harmonics(lmax__, vs[1], vs[2], &gkvec_rlm[0]);

            int n{0};
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                double phase = twopi * (kp__->gkvec().gkvec(igk) * unit_cell_.atom(ia).position());
                double_complex phase_factor = std::exp(double_complex(0.0, -phase));

                auto& atom_type = unit_cell_.atom(ia).type();
                for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
                    int l = atom_type.pp_desc().atomic_pseudo_wfs_[i].first;
                    for (int m = -l; m <= l; m++) {
                        int lm = Utils::lm_by_l_m(l, m);
                        double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                        phi.pw_coeffs().prime(igk_loc, n++) = z * phase_factor * gkvec_rlm[lm] * rad_int__[atom_type.id()][i](vs[0]);
                    }
                }
            }
        }
    }
    //#pragma omp parallel for
    //for (int i = num_ao__; i < num_phi; i++) {
    //    for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
    //        phi.pw_coeffs().prime(igk_loc, i) = 0;
    //    }
    //}

    //std::vector<vector3d<int>> gkv;
    //for (int igk = 0; igk < kp__->num_gkvec(); igk++) {
    //    gkv.push_back(kp__->gkvec().gvec(igk));
    //}
    //std::sort(gkv.begin(), gkv.end(), [](vector3d<int>& a, vector3d<int>& b) {
    //    int la = a.l1norm();
    //    int lb = b.l1norm();
    //    if (la < lb) {
    //        return true;
    //    }
    //    if (la > lb) {
    //        return false;
    //    }
    //    for (int x: {0, 1, 2}) {
    //        if (a[x] < b[x]) {
    //            return true;
    //        }
    //        if (a[x] > b[x]) {
    //            return false;
    //        }
    //    }
    //    return false;
    //});
    //
    //for (int i = 0; i < num_phi - num_ao__; i++) {
    //    auto v1 = gkv[i];
    //    for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
    //        /* global index of G+k vector */
    //        int igk = kp__->gkvec().gvec_offset(kp__->comm().rank()) + igk_loc;
    //        auto v2 = kp__->gkvec().gvec(igk);
    //        if (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) {
    //            phi.pw_coeffs().prime(igk_loc, num_ao__ + i) = 1.0;
    //        }
    //    }
    //}
    //for (int i = 0; i < num_phi - num_ao__; i++) {
    //    auto v1 = gkv[i];
    //    for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
    //        /* global index of G+k vector */
    //        int igk = kp__->gkvec().gvec_offset(kp__->comm().rank()) + igk_loc;
    //        auto v2 = kp__->gkvec().gvec(igk);
    //        if (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) {
    //            phi.pw_coeffs().prime(igk_loc, num_ao__ + i) = 1.0;
    //        }
    //    }
    //}
    double norm = std::sqrt(1.0 / kp__->num_gkvec()); 
    std::vector<double_complex> v(kp__->num_gkvec());
    for (int i = 0; i < num_phi - num_ao__; i++) {
        std::generate(v.begin(), v.end(), []{return type_wrapper<double_complex>::random();});
        v[0] = 1.0;
        //auto v1 = gkv[i];
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->gkvec().gvec_offset(kp__->comm().rank()) + igk_loc;
            //auto v2 = kp__->gkvec().gvec(igk);
            //if (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]) {
            //    phi.pw_coeffs().prime(igk_loc, num_ao__ + i) = 1.0;
            //}
            phi.pw_coeffs().prime(igk_loc, num_ao__ + i) = v[igk] * norm;
        }
    }

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    Hloc_operator hloc(ctx_.fft_coarse(), kp__->gkvec_vloc(), ctx_.mpi_grid_fft_vloc().communicator(1 << 1), 
                       ctx_.num_mag_dims(), ctx_.gvec_coarse(), effective_potential__, effective_magnetic_field__);

    ctx_.fft_coarse().prepare(kp__->gkvec().partition());
    
    D_operator<T> d_op(ctx_, kp__->beta_projectors());
    Q_operator<T> q_op(ctx_, kp__->beta_projectors());

    /* allocate wave-functions */
    wave_functions hphi(ctx_, kp__->comm(), kp__->gkvec(), num_phi);
    wave_functions ophi(ctx_, kp__->comm(), kp__->gkvec(), num_phi);
    wave_functions wf_tmp(ctx_, kp__->comm(), kp__->gkvec(), num_phi);

    //#ifdef __GPU
    //if (gen_evp_solver_->type() == ev_magma)
    //{
    //    hmlt.pin_memory();
    //    ovlp.pin_memory();
    //}
    //#endif

    int bs = ctx_.cyclic_block_size();
    dmatrix<T> hmlt(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> ovlp(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> hmlt_old;

    std::vector<double> eval(num_bands);
    
    kp__->beta_projectors().prepare();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        phi.allocate_on_device();
        phi.copy_to_device(0, num_phi);
        hphi.allocate_on_device();
        ophi.allocate_on_device();
        wf_tmp.allocate_on_device();
        evec.allocate(memory_t::device);
        ovlp.allocate(memory_t::device);
        hmlt.allocate(memory_t::device);
    }
    #endif
    
    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs = phi.checksum(0, num_phi);
        DUMP("checksum(phi): %18.10f %18.10f", cs.real(), cs.imag());
    }
    #endif

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o<T>(kp__, ispn, 0, num_phi, phi, hphi, ophi, hloc, d_op, q_op);
        
        orthogonalize<T>(0, num_phi, phi, hphi, ophi, ovlp, wf_tmp);

        /* setup eigen-value problem */
        set_subspace_mtrx<T>(0, num_phi, phi, hphi, hmlt, hmlt_old);

        /* solve generalized eigen-value problem with the size N */
        if (std_evp_solver().solve(num_phi,  num_bands, hmlt.template at<CPU>(), hmlt.ld(),
                                   eval.data(), evec.template at<CPU>(), evec.ld(),
                                   hmlt.num_rows_local(), hmlt.num_cols_local())) {
            std::stringstream s;
            s << "error in diagonalziation";
            TERMINATE(s);
        }

        #ifdef __PRINT_OBJECT_CHECKSUM
        {
            auto cs = evec.checksum();
            kp__->comm().allreduce(&cs, 1);
            DUMP("checksum(evec): %18.10f", std::abs(cs));
            double cs1{0};
            for (int i = 0; i < num_bands; i++) {
                cs1 += eval[i];
            }
            DUMP("checksum(eval): %18.10f", cs1);
        }
        #endif

        #if (__VERBOSITY > 2)
        if (kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands; i++) {
                DUMP("eval[%i]=%20.16f", i, eval[i]);
            }
        }
        #endif
        
        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            kp__->spinor_wave_functions(ispn).pw_coeffs().allocate_on_device();
        }
        #endif

        transform<T>(phi, 0, num_phi, evec, 0, 0, kp__->spinor_wave_functions(ispn), 0, num_bands);

        #ifdef __PRINT_OBJECT_CHECKSUM
        {
            auto cs = kp__->spinor_wave_functions(ispn).checksum(0, num_bands);
            DUMP("checksum(spinor_wave_functions): %18.10f %18.10f", cs.real(), cs.imag());
        }
        #endif

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            kp__->spinor_wave_functions(ispn).pw_coeffs().copy_to_host(0, num_bands);
            kp__->spinor_wave_functions(ispn).pw_coeffs().deallocate_on_device();
        }
        #endif

        for (int j = 0; j < ctx_.num_fv_states(); j++) {
            kp__->band_energy(j + ispn * ctx_.num_fv_states()) = eval[j];
        }
    }

    kp__->beta_projectors().dismiss();

    ctx_.fft_coarse().dismiss();
}
