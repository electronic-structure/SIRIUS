// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file initialize_subspace.hpp
 *
 *  \brief Initialize subspace for iterative diagonalization.
 */

inline void Band::initialize_subspace(K_point_set& kset__, Hamiltonian& H__) const
{
    PROFILE("sirius::Band::initialize_subspace");

    int N{0};

    if (ctx_.iterative_solver_input().init_subspace_ == "lcao") {
        /* get the total number of atomic-centered orbitals */
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            int n{0};
            for (int i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                n += (2 * std::abs(atom_type.ps_atomic_wf(i).first) + 1);
            }
            N += atom_type.num_atoms() * n;
        }

        if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 2) {
            printf("number of atomic orbitals: %i\n", N);
        }
    }

    H__.prepare();
    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__[ik];
        if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
            initialize_subspace<double>(kp, H__, N);
        } else {
            initialize_subspace<double_complex>(kp, H__, N);
        }
    }
    H__.dismiss();

    /* reset the energies for the iterative solver to do at least two steps */
    for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
        for (int ispn = 0; ispn < ctx_.num_spin_dims(); ispn++) {
            for (int i = 0; i < ctx_.num_bands(); i++) {
                kset__[ik]->band_energy(i, ispn, 0);
                kset__[ik]->band_occupancy(i, ispn, ctx_.max_occupancy());
            }
        }
    }
}

template <typename T>
inline void Band::initialize_subspace(K_point* kp__, Hamiltonian& H__, int num_ao__) const
{
    PROFILE("sirius::Band::initialize_subspace|kp");

    /* number of non-zero spin components */
    const int num_sc = (ctx_.num_mag_dims() == 3) ? 2 : 1;

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_bands();

    /* number of basis functions */
    int num_phi = std::max(num_ao__, num_bands / num_sc);

    int num_phi_tot = num_phi * num_sc;

    auto& mp = ctx_.mem_pool(ctx_.host_memory_t());

    ctx_.print_memory_usage(__FILE__, __LINE__);

    /* initial basis functions */
    Wave_functions phi(mp, kp__->gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    for (int ispn = 0; ispn < num_sc; ispn++) {
        phi.pw_coeffs(ispn).prime().zero();
    }

    utils::timer t1("sirius::Band::initialize_subspace|kp|wf");
    /* get proper lmax */
    int lmax{0};
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        lmax = std::max(lmax, atom_type.lmax_ps_atomic_wf());
    }
    lmax = std::max(lmax, unit_cell_.lmax());

    if (num_ao__ > 0) {
        kp__->generate_atomic_wave_functions(num_ao__, phi);
    }

    /* fill remaining wave-functions with pseudo-random guess */
    assert(kp__->num_gkvec() > num_phi + 10);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_phi - num_ao__; i++) {
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->idxgk(igk_loc);
            if (igk == i + 1) {
                phi.pw_coeffs(0).prime(igk_loc, num_ao__ + i) = 1.0;
            }
            if (igk == i + 2) {
                phi.pw_coeffs(0).prime(igk_loc, num_ao__ + i) = 0.5;
            }
            if (igk == i + 3) {
                phi.pw_coeffs(0).prime(igk_loc, num_ao__ + i) = 0.25;
            }
        }
    }

    std::vector<double> tmp(4096);
    for (int i = 0; i < 4096; i++) {
        tmp[i] = utils::random<double>();
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_phi; i++) {
        for (int igk_loc = kp__->gkvec().skip_g0(); igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->idxgk(igk_loc);
            phi.pw_coeffs(0).prime(igk_loc, i) += tmp[igk & 0xFFF] * 1e-5;
        }
    }

    if (ctx_.num_mag_dims() == 3) {
        /* make pure spinor up- and dn- wave functions */
        phi.copy_from(device_t::CPU, num_phi, phi, 0, 0, 1, num_phi);
    }
    t1.stop();

    ctx_.fft_coarse().prepare(kp__->gkvec_partition());
    H__.local_op().prepare(kp__->gkvec_partition());

    /* allocate wave-functions */
    Wave_functions hphi(mp, kp__->gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    Wave_functions ophi(mp, kp__->gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);
    /* temporary wave-functions required as a storage during orthogonalization */
    Wave_functions wf_tmp(mp, kp__->gkvec_partition(), num_phi_tot, ctx_.preferred_memory_t(), num_sc);

    int bs = ctx_.cyclic_block_size();

    auto& gen_solver = ctx_.gen_evp_solver();

    dmatrix<T> hmlt(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> ovlp(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> evec(mp, num_phi_tot, num_phi_tot, ctx_.blacs_grid(), bs, bs);

    std::vector<double> eval(num_bands);

    ctx_.print_memory_usage(__FILE__, __LINE__);

    kp__->beta_projectors().prepare();

    if (is_device_memory(ctx_.preferred_memory_t())) {
        auto& mpd = ctx_.mem_pool(memory_t::device);

        for (int ispn = 0; ispn < num_sc; ispn++) {
            phi.pw_coeffs(ispn).allocate(mpd);
            phi.pw_coeffs(ispn).copy_to(memory_t::device, 0, num_phi_tot);
        }

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__->spinor_wave_functions().pw_coeffs(ispn).allocate(mpd);
        }

        for (int ispn = 0; ispn < num_sc; ispn++) {
            hphi.pw_coeffs(ispn).allocate(mpd);
            ophi.pw_coeffs(ispn).allocate(mpd);
            wf_tmp.pw_coeffs(ispn).allocate(mpd);
        }
        evec.allocate(mpd);
        hmlt.allocate(mpd);
        ovlp.allocate(mpd);
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            auto cs = phi.checksum_pw(get_device_t(ctx_.preferred_memory_t()), ispn, 0, num_phi_tot);
            if (kp__->comm().rank() == 0) {
                std::stringstream s;
                s << "initial_phi" << ispn;
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    for (int ispn_step = 0; ispn_step < ctx_.num_spin_dims(); ispn_step++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        H__.apply_h_s<T>(kp__, (ctx_.num_mag_dims() == 3) ? 2 : ispn_step, 0, num_phi_tot, phi, &hphi, &ophi);

        /* do some checks */
        if (ctx_.control().verification_ >= 1) {

            set_subspace_mtrx<T>(1, num_phi_tot, phi, ophi, hmlt);
            if (ctx_.control().verification_ >= 2) {
                hmlt.serialize("overlap", num_phi_tot);
            }

            double max_diff = check_hermitian(hmlt, num_phi_tot);
            if (max_diff > 1e-12) {
                std::stringstream s;
                s << "overlap matrix is not hermitian, max_err = " << max_diff;
                TERMINATE(s);
            }
            std::vector<double> eo(num_phi_tot);
            auto& std_solver = ctx_.std_evp_solver();
            if (std_solver.solve(num_phi_tot, num_phi_tot, hmlt, eo.data(), evec)) {
                std::stringstream s;
                s << "error in diagonalization";
                TERMINATE(s);
            }
            if (kp__->comm().rank() == 0) {
                printf("[verification] minimum eigen-value of the overlap matrix: %18.12f\n", eo[0]);
            }
            if (eo[0] < 0) {
                TERMINATE("overlap matrix is not positively defined");
            }
        }

        /* setup eigen-value problem */
        set_subspace_mtrx<T>(0, num_phi_tot, phi, hphi, hmlt);
        set_subspace_mtrx<T>(0, num_phi_tot, phi, ophi, ovlp);

        if (ctx_.control().verification_ >= 2) {
            hmlt.serialize("hmlt", num_phi_tot);
            ovlp.serialize("ovlp", num_phi_tot);
        }

        /* solve generalized eigen-value problem with the size N and get lowest num_bands eigen-vectors */
        if (gen_solver.solve(num_phi_tot, num_bands, hmlt, ovlp, eval.data(), evec)) {
            std::stringstream s;
            s << "error in diagonalziation";
            TERMINATE(s);
        }

        if (ctx_.control().print_checksum_) {
            auto cs = evec.checksum();
            evec.blacs_grid().comm().allreduce(&cs, 1);
            double cs1{0};
            for (int i = 0; i < num_bands; i++) {
                cs1 += eval[i];
            }
            if (kp__->comm().rank() == 0) {
                utils::print_checksum("evec", cs);
                utils::print_checksum("eval", cs1);
            }
        }

        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands; i++) {
                printf("eval[%i]=%20.16f\n", i, eval[i]);
            }
        }

        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        transform<T>(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), (ctx_.num_mag_dims() == 3) ? 2 : ispn_step,
                     {&phi}, 0, num_phi_tot, evec, 0, 0, {&kp__->spinor_wave_functions()}, 0, num_bands);

        for (int j = 0; j < num_bands; j++) {
            kp__->band_energy(j, ispn_step, eval[j]);
        }
    }

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            auto cs = kp__->spinor_wave_functions().checksum_pw(get_device_t(ctx_.preferred_memory_t()), ispn, 0, num_bands);
            std::stringstream s;
            s << "initial_spinor_wave_functions_" << ispn;
            if (kp__->comm().rank() == 0) {
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    if (is_device_memory(ctx_.preferred_memory_t())) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__->spinor_wave_functions().pw_coeffs(ispn).copy_to(memory_t::host, 0, num_bands);
            kp__->spinor_wave_functions().pw_coeffs(ispn).deallocate(memory_t::device);
        }
    }

    if (ctx_.control().print_checksum_) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            auto cs = kp__->spinor_wave_functions().checksum_pw(device_t::CPU, ispn, 0, num_bands);
            std::stringstream s;
            s << "initial_spinor_wave_functions_" << ispn;
            if (kp__->comm().rank() == 0) {
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    kp__->beta_projectors().dismiss();
    ctx_.fft_coarse().dismiss();

    ctx_.print_memory_usage(__FILE__, __LINE__);
}
