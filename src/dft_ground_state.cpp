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

/** \file dft_ground_state.cpp
 *
 *  \brief Contains implementation of sirius::DFT_ground_state class.
 */

#include "dft_ground_state.hpp"
#include "utils/profiler.hpp"

namespace sirius {

void DFT_ground_state::initial_state()
{
    density_.initial_density();
    potential_.generate(density_);
    if (!ctx_.full_potential()) {
        Hamiltonian0 H0(potential_);
        Band(ctx_).initialize_subspace(kset_, H0);
    }
}

void DFT_ground_state::update()
{
    PROFILE("sirius::DFT_ground_state::update");

    ctx_.update();
    kset_.update();
    potential_.update();
    density_.update();

    if (!ctx_.full_potential()) {
        ewald_energy_ = sirius::ewald_energy(ctx_, ctx_.gvec(), ctx_.unit_cell());
    }
}

double DFT_ground_state::energy_kin_sum_pw() const
{
    double ekin{0};

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];

        #pragma omp parallel for schedule(static) reduction(+:ekin)
        for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
            auto Gk = kp->gkvec().gkvec_cart<index_domain_t::local>(igloc);

            double d{0};
            for (int ispin = 0; ispin < ctx_.num_spins(); ispin++) {
                for (int i = 0; i < kp->num_occupied_bands(ispin); i++) {
                    double f = kp->band_occupancy(i, ispin);
                    auto z = kp->spinor_wave_functions().pw_coeffs(ispin).prime(igloc, i);
                    d += f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                }
            }
            if (kp->gkvec().reduced()) {
                d *= 2;
            }
            ekin += 0.5 * d * kp->weight() * Gk.length2();
        } // igloc
    } // ikloc
    ctx_.comm().allreduce(&ekin, 1);
    return ekin;
}

double DFT_ground_state::total_energy() const
{
    return sirius::total_energy(ctx_, kset_, density_, potential_, ewald_energy_);
}

json DFT_ground_state::serialize()
{
    json dict;

    dict["mpi_grid"] = ctx_.mpi_grid_dims();

    std::vector<int> fftgrid = {ctx_.spfft().dim_x(),ctx_.spfft().dim_y(), ctx_.spfft().dim_z()};
    dict["fft_grid"] = fftgrid;
    fftgrid = {ctx_.spfft_coarse().dim_x(),ctx_.spfft_coarse().dim_y(), ctx_.spfft_coarse().dim_z()};
    dict["fft_coarse_grid"]         = fftgrid;
    dict["num_fv_states"]           = ctx_.num_fv_states();
    dict["num_bands"]               = ctx_.num_bands();
    dict["aw_cutoff"]               = ctx_.aw_cutoff();
    dict["pw_cutoff"]               = ctx_.pw_cutoff();
    dict["omega"]                   = ctx_.unit_cell().omega();
    dict["chemical_formula"]        = ctx_.unit_cell().chemical_formula();
    dict["num_atoms"]               = ctx_.unit_cell().num_atoms();
    dict["energy"]                  = json::object();
    dict["energy"]["total"]         = total_energy();
    dict["energy"]["enuc"]          = energy_enuc(ctx_, potential_);
    dict["energy"]["core_eval_sum"] = core_eval_sum(ctx_.unit_cell());
    dict["energy"]["vha"]           = energy_vha(potential_);
    dict["energy"]["vxc"]           = energy_vxc(density_, potential_);
    dict["energy"]["exc"]           = energy_exc(density_, potential_);
    dict["energy"]["bxc"]           = energy_bxc(density_, potential_, ctx_.num_mag_dims());
    dict["energy"]["veff"]          = energy_veff(density_, potential_);
    dict["energy"]["eval_sum"]      = eval_sum(ctx_.unit_cell(), kset_);
    dict["energy"]["kin"]           = energy_kin(ctx_, kset_, density_, potential_);
    dict["energy"]["ewald"]         = ewald_energy_;
    if (!ctx_.full_potential()) {
        dict["energy"]["vloc"]      = energy_vloc(density_, potential_);
    }
    dict["efermi"]                  = kset_.energy_fermi();
    dict["band_gap"]                = kset_.band_gap();
    dict["core_leakage"]            = density_.core_leakage();

    return dict;
}

/// A quick check of self-constent density in case of pseudopotential.
json DFT_ground_state::check_scf_density()
{
    if (ctx_.full_potential()) {
        return json();
    }
    std::vector<double_complex> rho_pw(ctx_.gvec().count());
    for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
        rho_pw[ig] = density_.rho().f_pw_local(ig);
    }

    double etot = total_energy();

    /* create new potential */
    Potential pot(ctx_);
    /* generate potential from existing density */
    pot.generate(density_);
    /* create new Hamiltonian */
    Hamiltonian0 H0(pot);
    /* set the high tolerance */
    ctx_.iterative_solver_tolerance(ctx_.settings().itsol_tol_min_);
    /* initialize the subspace */
    Band(ctx_).initialize_subspace(kset_, H0);
    /* find new wave-functions */
    Band(ctx_).solve(kset_, H0, true);
    /* find band occupancies */
    kset_.find_band_occupancies();
    /* generate new density from the occupied wave-functions */
    density_.generate(kset_, true, false);
    /* symmetrize density and magnetization */
    if (ctx_.use_symmetry()) {
        density_.symmetrize();
        if (ctx_.electronic_structure_method() == electronic_structure_method_t::pseudopotential) {
            density_.symmetrize_density_matrix();
        }
    }
    density_.fft_transform(1);
    double rms{0};
    for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
        rms += std::pow(std::abs(density_.rho().f_pw_local(ig) - rho_pw[ig]), 2);
    }
    ctx_.comm().allreduce(&rms, 1);
    json dict;
    dict["rss"]   = rms;
    dict["rms"]   = std::sqrt(rms / ctx_.gvec().num_gvec());
    dict["detot"] = total_energy() - etot;

    if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
        std::printf("[sirius::DFT_ground_state::check_scf_density] RSS: %18.12E\n", dict["rss"].get<double>());
        std::printf("[sirius::DFT_ground_state::check_scf_density] RMS: %18.12E\n", dict["rms"].get<double>());
        std::printf("[sirius::DFT_ground_state::check_scf_density] dEtot: %18.12E\n", dict["detot"].get<double>());
        std::printf("[sirius::DFT_ground_state::check_scf_density] Eold: %18.12E  Enew: %18.12E\n", etot, total_energy());
    }

    return dict;
}

json DFT_ground_state::find(double rms_tol, double energy_tol, double initial_tolerance, int num_dft_iter, bool write_state)
{
    PROFILE("sirius::DFT_ground_state::scf_loop");

    auto tstart = std::chrono::high_resolution_clock::now();

    double eold{0}, rms{0};

    density_.mixer_init(ctx_.mixer_input());

    int num_iter{-1};
    std::vector<double> rms_hist;
    std::vector<double> etot_hist;

    if (ctx_.hubbard_correction()) { // TODO: move to inititialization functions
        potential_.U().hubbard_compute_occupation_numbers(kset_);
        potential_.U().calculate_hubbard_potential_and_energy();
    }

    ctx_.iterative_solver_tolerance(initial_tolerance);

    for (int iter = 0; iter < num_dft_iter; iter++) {
        PROFILE("sirius::DFT_ground_state::scf_loop|iteration");

        if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
            std::printf("\n");
            std::printf("+------------------------------+\n");
            std::printf("| SCF iteration %3i out of %3i |\n", iter, num_dft_iter);
            std::printf("+------------------------------+\n");
        }
        Hamiltonian0 H0(potential_);
        /* find new wave-functions */
        Band(ctx_).solve(kset_, H0, true);
        /* find band occupancies */
        kset_.find_band_occupancies();
        /* generate new density from the occupied wave-functions */
        density_.generate(kset_, ctx_.use_symmetry(), true, true);

        /* mix density */
        rms = density_.mix();

        double old_tol = ctx_.iterative_solver_tolerance();
        /* estimate new tolerance of iterative solver */
        double tol = std::min(ctx_.settings().itsol_tol_scale_[0] * rms, ctx_.settings().itsol_tol_scale_[1] * old_tol);
        tol = std::max(ctx_.settings().itsol_tol_min_, tol);
        /* set new tolerance of iterative solver */
        ctx_.iterative_solver_tolerance(tol);

        /* check number of elctrons */
        density_.check_num_electrons();

        /* compute new potential */
        potential_.generate(density_);

        if (!ctx_.full_potential() && ctx_.control().verification_ >= 2) {
            ctx_.message(1, __function_name__, "%s", "checking functional derivative of Exc\n");
            double eps{0.1};
            for (int i = 0; i < 10; i++) {
                Potential p1(ctx_);
                p1.scale_rho_xc(1 + eps);
                p1.generate(density_);

                double evxc = potential_.energy_vxc(density_) + potential_.energy_vxc_core(density_) + energy_bxc(density_, potential_, ctx_.num_mag_dims());
                double deriv = (p1.energy_exc(density_) - potential_.energy_exc(density_)) / eps;

                std::printf("eps              : %18.12f\n", eps);
                std::printf("Energy Vxc       : %18.12f\n", evxc);
                std::printf("numerical deriv  : %18.12f\n", deriv);
                std::printf("difference       : %18.12f\n", std::abs(evxc - deriv));
                eps /= 10;
            }
        }

        /* symmetrize potential and effective magnetic field */
        if (ctx_.use_symmetry()) {
            potential_.symmetrize();
        }

        /* transform potential to real space after symmetrization */
        potential_.fft_transform(1);

        /* compute new total energy for a new density */
        double etot = total_energy();

        etot_hist.push_back(etot);

        rms_hist.push_back(rms);

        /* write some information */
        print_info();
        if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
            std::printf("iteration : %3i, RMS %18.12E, energy difference : %18.12E\n", iter, rms, etot - eold);
        }
        /* check if the calculation has converged */
        if (std::abs(eold - etot) < energy_tol && rms < rms_tol) {
            if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
                std::printf("\n");
                std::printf("converged after %i SCF iterations!\n", iter + 1);
            }
            num_iter = iter;
            break;
        }

        /* Compute the hubbard correction */
        if (ctx_.hubbard_correction()) {
            potential_.U().hubbard_compute_occupation_numbers(kset_);
            potential_.U().calculate_hubbard_potential_and_energy();
        }

        eold = etot;
    }

    if (write_state) {
        ctx_.create_storage_file();
        if (ctx_.full_potential()) { // TODO: why this is necessary?
            density_.rho().fft_transform(-1);
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                density_.magnetization(j).fft_transform(-1);
            }
        }
        potential_.save();
        density_.save();
        //kset_.save(storage_file_name);
    }

    auto tstop = std::chrono::high_resolution_clock::now();

    json dict = serialize();
    dict["scf_time"] = std::chrono::duration_cast<std::chrono::duration<double>>(tstop - tstart).count();
    dict["etot_history"] = etot_hist;
    if (num_iter >= 0) {
        dict["converged"]          = true;
        dict["num_scf_iterations"] = num_iter;
        dict["rms_history"]        = rms_hist;
    } else {
        dict["converged"] = false;
    }

    //if (ctx_.control().verification_ >= 1) {
    //    check_scf_density();
    //}

    // dict["volume"] = ctx.unit_cell().omega() * std::pow(bohr_radius, 3);
    // dict["volume_units"] = "angstrom^3";
    // dict["energy"] = dft.total_energy() * ha2ev;
    // dict["energy_units"] = "eV";

    return dict;
}

void DFT_ground_state::print_info()
{
    double evalsum1 = kset_.valence_eval_sum();
    double evalsum2 = core_eval_sum(ctx_.unit_cell());
    double ekin     = energy_kin(ctx_, kset_, density_, potential_);
    double evxc     = energy_vxc(density_, potential_);
    double eexc     = energy_exc(density_, potential_);
    double ebxc     = energy_bxc(density_, potential_, ctx_.num_mag_dims());
    double evha     = energy_vha(potential_);
    double etot     = sirius::total_energy(ctx_, kset_, density_, potential_, ewald_energy_);
    double gap      = kset_.band_gap() * ha2ev;
    double ef       = kset_.energy_fermi();
    double enuc     = energy_enuc(ctx_, potential_);

    double one_elec_en = evalsum1 - (evxc + evha);

    if (ctx_.electronic_structure_method() == electronic_structure_method_t::pseudopotential) {
        one_elec_en -= potential_.PAW_one_elec_energy();
    }

    auto result = density_.rho().integrate();

    auto total_charge = std::get<0>(result);
    auto it_charge    = std::get<1>(result);
    auto mt_charge    = std::get<2>(result);

    auto result_mag = density_.get_magnetisation();
    auto total_mag  = std::get<0>(result_mag);
    auto it_mag     = std::get<1>(result_mag);
    auto mt_mag     = std::get<2>(result_mag);

    //double total_mag[3];
    //std::vector<double> mt_mag[3];
    //double it_mag[3];
    //for (int j = 0; j < ctx_.num_mag_dims(); j++) {
    //    auto result = density_.magnetization(j).integrate();

    //    total_mag[j] = std::get<0>(result);
    //    it_mag[j]    = std::get<1>(result);
    //    mt_mag[j]    = std::get<2>(result);
    //}

    if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 1) {
        std::printf("\n");
        std::printf("Charges and magnetic moments\n");
        for (int i = 0; i < 80; i++) {
            std::printf("-");
        }
        std::printf("\n");
        if (ctx_.full_potential()) {
            double total_core_leakage{0.0};
            std::printf("atom      charge    core leakage");
            if (ctx_.num_mag_dims()) {
                std::printf("              moment                |moment|");
            }
            std::printf("\n");
            for (int i = 0; i < 80; i++) {
                std::printf("-");
            }
            std::printf("\n");

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                double core_leakage = unit_cell_.atom(ia).symmetry_class().core_leakage();
                total_core_leakage += core_leakage;
                std::printf("%4i  %10.6f  %10.8e", ia, mt_charge[ia], core_leakage);
                if (ctx_.num_mag_dims()) {
                    vector3d<double> v(mt_mag[ia]);
                    std::printf("  [%8.4f, %8.4f, %8.4f]  %10.6f", v[0], v[1], v[2], v.length());
                }
                std::printf("\n");
            }

            std::printf("\n");
            std::printf("total core leakage    : %10.8e\n", total_core_leakage);
            std::printf("interstitial charge   : %10.6f\n", it_charge);
            if (ctx_.num_mag_dims()) {
                vector3d<double> v(it_mag);
                std::printf("interstitial moment   : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", v[0], v[1], v[2],
                       v.length());
            }
        } else {
            if (ctx_.num_mag_dims()) {
                std::printf("atom              moment                |moment|");
                std::printf("\n");
                for (int i = 0; i < 80; i++) {
                    std::printf("-");
                }
                std::printf("\n");

                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    vector3d<double> v(mt_mag[ia]);
                    std::printf("%4i  [%8.4f, %8.4f, %8.4f]  %10.6f", ia, v[0], v[1], v[2], v.length());
                    std::printf("\n");
                }

                std::printf("\n");
            }
        }
        std::printf("total charge          : %10.6f\n", total_charge);

        if (ctx_.num_mag_dims()) {
            vector3d<double> v(total_mag);
            std::printf("total moment          : [%8.4f, %8.4f, %8.4f], magnitude : %10.6f\n", v[0], v[1], v[2], v.length());
        }

        std::printf("\n");
        std::printf("Energy\n");
        for (int i = 0; i < 80; i++) {
            std::printf("-");
        }
        std::printf("\n");

        std::printf("valence_eval_sum          : %18.8f\n", evalsum1);
        if (ctx_.full_potential()) {
            std::printf("core_eval_sum             : %18.8f\n", evalsum2);
            std::printf("kinetic energy            : %18.8f\n", ekin);
            std::printf("enuc                      : %18.8f\n", enuc);
        }
        std::printf("<rho|V^{XC}>              : %18.8f\n", evxc);
        std::printf("<rho|E^{XC}>              : %18.8f\n", eexc);
        std::printf("<mag|B^{XC}>              : %18.8f\n", ebxc);
        std::printf("<rho|V^{H}>               : %18.8f\n", evha);
        if (!ctx_.full_potential()) {
            std::printf("one-electron contribution : %18.8f (Ha), %18.8f (Ry)\n", one_elec_en,
                   one_elec_en * 2); // eband + deband in QE
            std::printf("hartree contribution      : %18.8f\n", 0.5 * evha);
            std::printf("xc contribution           : %18.8f\n", eexc);
            std::printf("ewald contribution        : %18.8f\n", ewald_energy_);
            std::printf("PAW contribution          : %18.8f\n", potential_.PAW_total_energy());
        }
        if (ctx_.hubbard_correction()) {
            std::printf("Hubbard energy            : %18.8f (Ha), %18.8f (Ry)\n", potential_.U().hubbard_energy(),
                   potential_.U().hubbard_energy() * 2.0);
        }

        std::printf("Total energy              : %18.8f (Ha), %18.8f (Ry)\n", etot, etot * 2);

        std::printf("\n");
        std::printf("band gap (eV) : %18.8f\n", gap);
        std::printf("Efermi        : %18.8f\n", ef);
        std::printf("\n");
        // if (ctx_.control().verbosity_ >= 3 && !ctx_.full_potential()) {
        //    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        //        std::printf("atom: %i\n", ia);
        //        int nbf = unit_cell_.atom(ia).type().mt_basis_size();
        //        for (int j = 0; j < ctx_.num_mag_comp(); j++) {
        //            //printf("component of density matrix: %i\n", j);
        //            //for (int xi1 = 0; xi1 < nbf; xi1++) {
        //            //    for (int xi2 = 0; xi2 < nbf; xi2++) {
        //            //        auto z = density_.density_matrix()(xi1, xi2, j, ia);
        //            //        std::printf("(%f, %f) ", z.real(), z.imag());
        //            //    }
        //            //    std::printf("\n");
        //            //}
        //            std::printf("diagonal components of density matrix: %i\n", j);
        //            for (int xi2 = 0; xi2 < nbf; xi2++) {
        //                auto z = density_.density_matrix()(xi2, xi2, j, ia);
        //                std::printf("(%10.6f, %10.6f) ", z.real(), z.imag());
        //            }
        //            std::printf("\n");
        //        }
        //    }
        //}
    }
}

} // namespace sirius
