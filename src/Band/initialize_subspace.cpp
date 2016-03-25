#include "band.h"

namespace sirius {

template <typename T>
void Band::initialize_subspace(K_point* kp__,
                               Periodic_function<double>* effective_potential__,
                               Periodic_function<double>* effective_magnetic_field__[3],
                               int num_ao__,
                               int lmax__,
                               std::vector< std::vector< Spline<double> > >& rad_int__) const
{
    auto pu = ctx_.processing_unit();

    /* number of basis functions */
    int num_phi = std::max(num_ao__, ctx_.num_fv_states());

    Wave_functions<false> phi(num_phi, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);

    #pragma omp parallel
    {
        std::vector<double> gkvec_rlm(Utils::lmmax(lmax__));
        /* fill first N functions with atomic orbitals */
        #pragma omp for
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++)
        {
            /* global index of G+k vector */
            int igk = kp__->gkvec().offset_gvec(kp__->comm().rank()) + igk_loc;
            /* vs = {r, theta, phi} */
            auto vs = SHT::spherical_coordinates(kp__->gkvec().cart_shifted(igk));
            /* compute real spherical harmonics for G+k vector */
            SHT::spherical_harmonics(lmax__, vs[1], vs[2], &gkvec_rlm[0]);

            int n = 0;
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
            {
                double phase = twopi * (kp__->gkvec().gvec_shifted(igk) * unit_cell_.atom(ia).position());
                double_complex phase_factor =  std::exp(double_complex(0.0, -phase));

                auto& atom_type = unit_cell_.atom(ia).type();
                for (size_t i = 0; i < atom_type.uspp().atomic_pseudo_wfs_.size(); i++)
                {
                    int l = atom_type.uspp().atomic_pseudo_wfs_[i].first;
                    for (int m = -l; m <= l; m++)
                    {
                        int lm = Utils::lm_by_l_m(l, m);
                        double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                        phi(igk_loc, n++) = z * phase_factor * gkvec_rlm[lm] * rad_int__[atom_type.id()][i](vs[0]);
                    }
                }
            }
        }
    }

    /* fill the remaining basis functions with random numbers */
    #pragma omp parallel for
    for (int i = num_ao__; i < num_phi; i++)
    {
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++)
            phi(igk_loc, i) = 0;

        auto G = kp__->gkvec()[i];

        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++)
        {
            /* global index of G+k vector */
            int igk = kp__->gkvec().offset_gvec(kp__->comm().rank()) + igk_loc;
            auto G1 = kp__->gkvec()[igk];

            if (G[0] == G1[0] && G[1] == G1[1] && G[2] == G1[2]) phi(igk_loc, i) = 1;
            if (G[0] == -G1[0] && G[1] == -G1[1] && G[2] == -G1[2]) phi(igk_loc, i) = 1;
        }
    }

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    ctx_.fft_coarse().prepare();

    Hloc_operator hloc(ctx_.fft_coarse(), ctx_.gvec_coarse(), kp__->gkvec(), ctx_.num_mag_dims(),
                       effective_potential__, effective_magnetic_field__);
    
    D_operator<T> d_op(ctx_, kp__->beta_projectors());
    Q_operator<T> q_op(ctx_, kp__->beta_projectors());

    /* allocate wave-functions */
    Wave_functions<false> hphi(num_phi, num_phi, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions<false> ophi(num_phi, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);

    /* allocate Hamiltonian and overlap */
    matrix<T> hmlt(num_phi, num_phi);
    matrix<T> ovlp(num_phi, num_phi);

    matrix<T> hmlt_old(num_phi, num_phi);
    matrix<T> ovlp_old(num_phi, num_phi);

    #ifdef __GPU
    if (gen_evp_solver_->type() == ev_magma)
    {
        hmlt.pin_memory();
        ovlp.pin_memory();
    }
    #endif

    matrix<T> evec(num_phi, num_phi);

    int bs = ctx_.cyclic_block_size();

    dmatrix<T> hmlt_dist;
    dmatrix<T> ovlp_dist;
    dmatrix<T> evec_dist;
    if (kp__->comm().size() == 1)
    {
        hmlt_dist = dmatrix<T>(&hmlt(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<T>(&ovlp(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
        evec_dist = dmatrix<T>(&evec(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    }
    else
    {
        hmlt_dist = dmatrix<T>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<T>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
        evec_dist = dmatrix<T>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    }

    std::vector<double> eval(num_bands);
    
    kp__->beta_projectors().prepare();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
    {
        phi.allocate_on_device();
        phi.copy_to_device(0, num_phi);
        hphi.allocate_on_device();
        ophi.allocate_on_device();
        evec.allocate_on_device();
        ovlp.allocate_on_device();
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs = mdarray<double_complex, 1>(&phi(0, 0), kp__->num_gkvec_loc() * num_phi).checksum();
        kp__->comm().allreduce(&cs, 1);
        DUMP("checksum(phi): %18.10f %18.10f", cs.real(), cs.imag());
    }
    #endif

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o<T>(kp__, ispn, 0, num_phi, phi, hphi, ophi, hloc, d_op, q_op);
        
        orthogonalize<T>(kp__, 0, num_phi, phi, hphi, ophi, ovlp);

        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_h_o<T>(kp__, 0, num_phi, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old);

        /* solve generalized eigen-value problem with the size N */
        diag_h_o<T>(kp__, num_phi, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);

        #if (__VERBOSITY > 2)
        if (kp__->comm().rank() == 0)
        {
            for (int i = 0; i < num_bands; i++) DUMP("eval[%i]=%20.16f", i, eval[i]);
        }
        #endif
        
        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) kp__->spinor_wave_functions<false>(ispn).allocate_on_device();
        #endif

        kp__->spinor_wave_functions<false>(ispn).transform_from<T>(phi, num_phi, evec, num_bands);

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU)
        {
            kp__->spinor_wave_functions<false>(ispn).copy_to_host(0, num_bands);
            kp__->spinor_wave_functions<false>(ispn).deallocate_on_device();
        }
        #endif

        //for (int j = 0; j < ctx_.num_fv_states(); j++)
        //    //kp__->band_energy(j + ispn * ctx_.num_fv_states()) = eval[j];
        //    kp__->band_energy(j + ispn * ctx_.num_fv_states()) = j;
    }

    kp__->beta_projectors().dismiss();

    ctx_.fft_coarse().dismiss();
}

template void Band::initialize_subspace<double>(K_point* kp__,
                                                Periodic_function<double>* effective_potential__,
                                                Periodic_function<double>* effective_magnetic_field__[3],
                                                int num_ao__,
                                                int lmax__,
                                                std::vector< std::vector< Spline<double> > >& rad_int__) const;

template void Band::initialize_subspace<double_complex>(K_point* kp__,
                                                        Periodic_function<double>* effective_potential__,
                                                        Periodic_function<double>* effective_magnetic_field__[3],
                                                        int num_ao__,
                                                        int lmax__,
                                                        std::vector< std::vector< Spline<double> > >& rad_int__) const;
};
