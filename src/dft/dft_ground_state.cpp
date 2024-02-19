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

#include <iomanip>
#include "dft_ground_state.hpp"
#include "core/profiler.hpp"
#include "hamiltonian/initialize_subspace.hpp"
#include "hamiltonian/diagonalize.hpp"

namespace sirius {

void
DFT_ground_state::initial_state()
{
    PROFILE("sirius::DFT_ground_state::initial_state");

    density_.initial_density();
    potential_.generate(density_, ctx_.use_symmetry(), true);
    if (!ctx_.full_potential()) {
        if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(SIRIUS_USE_FP32)
            Hamiltonian0<float> H0(potential_, true);
            initialize_subspace(kset_, H0);
#else
            RTE_THROW("not compiled with FP32 support");
#endif

        } else {
            Hamiltonian0<double> H0(potential_, true);
            initialize_subspace(kset_, H0);
        }
    }
}

void
DFT_ground_state::create_H0()
{
    PROFILE("sirius::DFT_ground_state::create_H0");

    H0_ = std::shared_ptr<Hamiltonian0<double>>(new Hamiltonian0<double>(potential_, true));
}

void
DFT_ground_state::update()
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

double
DFT_ground_state::energy_kin_sum_pw() const
{
    double ekin{0};

    for (auto it : kset_.spl_num_kpoints()) {
        auto kp = kset_.get<double>(it.i);

        #pragma omp parallel for schedule(static) reduction(+:ekin)
        for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
            auto Gk = kp->gkvec().gkvec_cart<index_domain_t::local>(igloc);

            double d{0};
            for (int ispin = 0; ispin < ctx_.num_spins(); ispin++) {
                for (int i = 0; i < kp->num_occupied_bands(ispin); i++) {
                    double f = kp->band_occupancy(i, ispin);
                    auto z   = kp->spinor_wave_functions().pw_coeffs(igloc, wf::spin_index(ispin), wf::band_index(i));
                    d += f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                }
            }
            if (kp->gkvec().reduced()) {
                d *= 2;
            }
            ekin += 0.5 * d * kp->weight() * Gk.length2();
        } // igloc
    }     // ikloc
    ctx_.comm().allreduce(&ekin, 1);
    return ekin;
}

double
DFT_ground_state::total_energy() const
{
    return sirius::total_energy(ctx_, kset_, density_, potential_, ewald_energy_) + this->scf_correction_energy_;
}

json
DFT_ground_state::serialize()
{
    return energy_dict(ctx_, kset_, density_, potential_, ewald_energy_, this->scf_correction_energy_);
}

/// A quick check of self-constent density in case of pseudopotential.
json
DFT_ground_state::check_scf_density()
{
    if (ctx_.full_potential()) {
        return json();
    }

    auto gs0 = energy_dict(ctx_, kset_, density_, potential_, ewald_energy_);

    /* create new potential */
    Potential pot(ctx_);
    /* generate potential from existing density */
    bool transform_to_rg{true};
    pot.generate(density_, ctx_.use_symmetry(), transform_to_rg);
    /* create new Hamiltonian */
    bool precompute_lapw{true};
    Hamiltonian0<double> H0(pot, precompute_lapw);
    /* initialize the subspace */
    ::sirius::initialize_subspace(kset_, H0);
    /* find new wave-functions */
    ::sirius::diagonalize<double, double>(H0, kset_, ctx_.cfg().settings().itsol_tol_min(),
                                          ctx_.cfg().iterative_solver().num_steps());
    /* find band occupancies */
    kset_.find_band_occupancies<double>();
    /* generate new density from the occupied wave-functions */
    bool add_core{true};
    /* create new density */
    Density rho(ctx_);
    rho.generate<double>(kset_, ctx_.use_symmetry(), add_core, transform_to_rg);

    auto gs1 = energy_dict(ctx_, kset_, rho, pot, ewald_energy_);

    auto calc_rms = [&](Field4D& a, Field4D& b) -> double {
        double rms{0};
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
                rms += std::pow(std::abs(a.component(j).rg().f_pw_local(ig) - b.component(j).rg().f_pw_local(ig)), 2);
            }
        }
        ctx_.comm().allreduce(&rms, 1);
        return std::sqrt(rms / ctx_.gvec().num_gvec());
    };

    double rms      = calc_rms(density_, rho);
    double rms_veff = calc_rms(potential_, pot);

    json dict;
    dict["rms"]   = rms;
    dict["detot"] = gs0["energy"]["total"].get<double>() - gs1["energy"]["total"].get<double>();

    rms_veff = std::sqrt(rms_veff / ctx_.gvec().num_gvec());

    if (ctx_.verbosity() >= 1) {
        RTE_OUT(ctx_.out()) << "RMS_rho: " << dict["rms"].get<double>() << std::endl
                            << "RMS_veff: " << rms_veff << std::endl
                            << "Eold: " << gs0["energy"]["total"].get<double>()
                            << " Enew: " << gs1["energy"]["total"].get<double>() << std::endl;

        std::vector<std::string> labels({"total", "vha", "vxc", "exc", "bxc", "veff", "eval_sum", "kin", "ewald",
                                         "vloc", "scf_correction", "entropy_sum"});

        for (auto e : labels) {
            RTE_OUT(ctx_.out()) << "energy component: " << e << ", diff: "
                                << std::abs(gs0["energy"][e].get<double>() - gs1["energy"][e].get<double>())
                                << std::endl;
        }
    }

    return dict;
}

json
DFT_ground_state::find(double density_tol__, double energy_tol__, double iter_solver_tol__, int num_dft_iter__,
                       bool write_state__)
{
    PROFILE("sirius::DFT_ground_state::scf_loop");

    auto tstart = std::chrono::high_resolution_clock::now();

    double eold{0}, rms{0};

    density_.mixer_init(ctx_.cfg().mixer());

    int num_iter{-1};
    std::vector<double> rms_hist;
    std::vector<double> etot_hist;

    Density rho1(ctx_);

    std::stringstream s;
    s << "density_tol               : " << density_tol__ << std::endl
      << "energy_tol                : " << energy_tol__ << std::endl
      << "iter_solver_tol (initial) : " << iter_solver_tol__ << std::endl
      << "iter_solver_tol (target)  : " << ctx_.cfg().settings().itsol_tol_min() << std::endl
      << "num_dft_iter              : " << num_dft_iter__;
    ctx_.message(1, __func__, s);

    for (int iter = 0; iter < num_dft_iter__; iter++) {
        PROFILE("sirius::DFT_ground_state::scf_loop|iteration");
        std::stringstream s;
        s << std::endl;
        s << "+------------------------------+" << std::endl
          << "| SCF iteration " << std::setw(3) << iter << " out of " << std::setw(3) << num_dft_iter__ << '|'
          << std::endl
          << "+------------------------------+" << std::endl;
        ctx_.message(2, __func__, s);

        diagonalize_result_t result;

        if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(SIRIUS_USE_FP32)
            Hamiltonian0<float> H0(potential_, true);
            /* find new wave-functions */
            if (ctx_.cfg().parameters().precision_hs() == "fp32") {
                result = sirius::diagonalize<float, float>(H0, kset_, iter_solver_tol__,
                                                           ctx_.cfg().iterative_solver().num_steps());
            } else {
                result = sirius::diagonalize<float, double>(H0, kset_, iter_solver_tol__,
                                                            ctx_.cfg().iterative_solver().num_steps());
            }
            /* find band occupancies */
            kset_.find_band_occupancies<float>();
            /* generate new density from the occupied wave-functions */
            density_.generate<float>(kset_, ctx_.use_symmetry(), true, true);
#else
            RTE_THROW("not compiled with FP32 support");
#endif
        } else {
            Hamiltonian0<double> H0(potential_, true);
            /* find new wave-functions */
            result = sirius::diagonalize<double, double>(H0, kset_, iter_solver_tol__,
                                                         ctx_.cfg().iterative_solver().num_steps());
            /* find band occupancies */
            kset_.find_band_occupancies<double>();
            /* generate new density from the occupied wave-functions */
            density_.generate<double>(kset_, ctx_.use_symmetry(), true, true);
        }

        double e1 = energy_potential(density_, potential_);
        copy(density_, rho1);

        /* mix density */
        rms = density_.mix();

        double eha_res = density_residual_hartree_energy(density_, rho1);

        /* estimate new tolerance of the iterative solver */
        double tol = rms;
        if (ctx_.cfg().mixer().use_hartree()) {
            // tol = rms * rms / std::max(1.0, unit_cell_.num_electrons());
            tol = eha_res / std::max(1.0, unit_cell_.num_electrons());
        }
        tol = std::min(ctx_.cfg().settings().itsol_tol_scale()[0] * tol,
                       ctx_.cfg().settings().itsol_tol_scale()[1] * iter_solver_tol__);
        /* tolerance can't be too small */
        iter_solver_tol__ = std::max(ctx_.cfg().settings().itsol_tol_min(), tol);

        bool iter_solver_converged{true};
        if (ctx_.cfg().iterative_solver().type() != "exact") {
            iter_solver_converged = (tol <= ctx_.cfg().settings().itsol_tol_min());
        }

#if defined(SIRIUS_USE_FP32)
        if (ctx_.cfg().parameters().precision_gs() != "auto") {
            /* if the final precision is not equal to the current precision */
            if (ctx_.cfg().parameters().precision_gs() == "fp64" && ctx_.cfg().parameters().precision_wf() == "fp32") {
                /* if we reached the mimimum tolerance for fp32 */
                if ((ctx_.cfg().settings().fp32_to_fp64_rms() == 0 &&
                     iter_solver_tol__ <= ctx_.cfg().settings().itsol_tol_min()) ||
                    (rms < ctx_.cfg().settings().fp32_to_fp64_rms())) {
                    std::cout << "switching to FP64" << std::endl;
                    ctx_.cfg().unlock();
                    ctx_.cfg().settings().itsol_tol_min(std::numeric_limits<double>::epsilon() * 10);
                    ctx_.cfg().parameters().precision_wf("fp64");
                    ctx_.cfg().parameters().precision_hs("fp64");
                    ctx_.cfg().lock();

                    for (auto it : kset_.spl_num_kpoints()) {
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            wf::copy(memory_t::host, kset_.get<float>(it.i)->spinor_wave_functions(),
                                     wf::spin_index(ispn), wf::band_range(0, ctx_.num_bands()),
                                     kset_.get<double>(it.i)->spinor_wave_functions(), wf::spin_index(ispn),
                                     wf::band_range(0, ctx_.num_bands()));
                        }
                    }
                    for (int ik = 0; ik < kset_.num_kpoints(); ik++) {
                        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                            for (int j = 0; j < ctx_.num_bands(); j++) {
                                kset_.get<double>(ik)->band_energy(j, ispn, kset_.get<float>(ik)->band_energy(j, ispn));
                                kset_.get<double>(ik)->band_occupancy(j, ispn,
                                                                      kset_.get<float>(ik)->band_occupancy(j, ispn));
                            }
                        }
                    }
                }
            }
        }
#endif
        if (ctx_.cfg().control().verification() >= 1) {
            /* check number of electrons */
            density_.check_num_electrons();
        }

        /* compute new potential */
        potential_.generate(density_, ctx_.use_symmetry(), true);

        if (!ctx_.full_potential() && ctx_.cfg().control().verification() >= 2) {
            if (ctx_.verbosity() >= 1) {
                RTE_OUT(ctx_.out()) << "checking functional derivative of Exc\n";
            }
            sirius::check_xc_potential(density_);
        }

        if (ctx_.cfg().parameters().use_scf_correction()) {
            double e2                    = energy_potential(rho1, potential_);
            this->scf_correction_energy_ = e2 - e1;
        }

        /* compute new total energy for a new density */
        double etot = total_energy();

        etot_hist.push_back(etot);

        rms_hist.push_back(rms);

        /* write some information */
        std::stringstream out;
        out << std::endl;
        print_info(out);
        out << std::endl;
        out << "iteration : " << iter << ", RMS : " << std::setprecision(12) << std::scientific << rms
            << ", energy difference : " << std::setprecision(12) << std::scientific << etot - eold;
        if (!ctx_.full_potential()) {
            out << std::endl
                << "Hartree energy of density residual : " << eha_res << std::endl
                << "bands are converged : " << boolstr(result.converged);
        }
        if (ctx_.cfg().iterative_solver().type() != "exact") {
            out << std::endl << "iterative solver converged : " << boolstr(iter_solver_converged);
        }

        ctx_.message(2, __func__, out);
        /* check if the calculation has converged */
        bool converged{true};
        converged = (std::abs(eold - etot) < energy_tol__) && result.converged && iter_solver_converged;
        if (ctx_.cfg().mixer().use_hartree()) {
            converged = converged && (eha_res < density_tol__);
        } else {
            converged = converged && (rms < density_tol__);
        }
        if (converged) {
            std::stringstream out;
            out << std::endl;
            out << "converged after " << iter + 1 << " SCF iterations!" << std::endl;
            ctx_.message(1, __func__, out);
            density_.check_num_electrons();
            num_iter = iter;
            break;
        }

        eold = etot;
    }
    std::stringstream out;
    out << std::endl;
    print_info(out);
    ctx_.message(1, __func__, out);

    if (write_state__) {
        ctx_.create_storage_file(storage_file_name);
        if (ctx_.full_potential()) { // TODO: why this is necessary?
            density_.rho().rg().fft_transform(-1);
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                density_.mag(j).rg().fft_transform(-1);
            }
        }
        potential_.save(storage_file_name);
        density_.save(storage_file_name);
        // kset_.save(storage_file_name);
    }

    auto tstop = std::chrono::high_resolution_clock::now();

    auto dict = serialize();
    if (ctx_.num_mag_dims()) {
        dict["magnetisation"]          = {};
        auto m                         = density_.get_magnetisation();
        dict["magnetisation"]["total"] = std::vector<double>({m[0].total, m[1].total, m[2].total});
        std::vector<std::vector<double>> v;
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            v.push_back({m[0].mt[ia], m[1].mt[ia], m[2].mt[ia]});
        }
        dict["magnetisation"]["atoms"] = v;
    }

    /* check density */
    if (num_iter >= 0) {
        density_.rho().rg().fft_transform(1);
        double rho_min{1e100};
        for (int ir = 0; ir < density_.rho().rg().spfft().local_slice_size(); ir++) {
            rho_min = std::min(rho_min, density_.rho().rg().value(ir));
        }
        dict["rho_min"] = rho_min;
        ctx_.comm().allreduce<double, mpi::op_t::min>(&rho_min, 1);
    }

    dict["scf_time"]     = std::chrono::duration_cast<std::chrono::duration<double>>(tstop - tstart).count();
    dict["etot_history"] = etot_hist;
    if (num_iter >= 0) {
        dict["converged"]          = true;
        dict["num_scf_iterations"] = num_iter;
        dict["rms_history"]        = rms_hist;
    } else {
        dict["converged"] = false;
    }

    if (env::check_scf_density()) {
        check_scf_density();
    }

    // dict["volume"] = ctx.unit_cell().omega() * std::pow(bohr_radius, 3);
    // dict["volume_units"] = "angstrom^3";
    // dict["energy"] = dft.total_energy() * ha2ev;
    // dict["energy_units"] = "eV";

    return dict;
}

void
DFT_ground_state::print_info(std::ostream& out__) const
{
    double evalsum1     = kset_.valence_eval_sum();
    double evalsum2     = core_eval_sum(ctx_.unit_cell());
    double s_sum        = kset_.entropy_sum();
    double ekin         = energy_kin(ctx_, kset_, density_, potential_);
    double evxc         = energy_vxc(density_, potential_);
    double eexc         = energy_exc(density_, potential_);
    double ebxc         = energy_bxc(density_, potential_);
    double evha         = energy_vha(potential_);
    double hub_one_elec = one_electron_energy_hubbard(density_, potential_);
    double etot         = total_energy();
    double gap          = kset_.band_gap() * ha2ev;
    double ef           = kset_.energy_fermi();
    double enuc         = energy_enuc(ctx_, potential_);

    double one_elec_en = evalsum1 - (evxc + evha + ebxc);

    if (ctx_.electronic_structure_method() == electronic_structure_method_t::pseudopotential) {
        one_elec_en -= potential_.PAW_one_elec_energy(density_);
        one_elec_en -= hub_one_elec;
    }

    density_.print_info(out__);

    out__ << std::endl;
    out__ << "Energy" << std::endl << hbar(80, '-') << std::endl;

    auto write_energy = [&](std::string label__, double value__) {
        out__ << std::left << std::setw(30) << label__ << " : " << std::right << std::setw(16) << std::setprecision(8)
              << std::fixed << value__ << std::endl;
    };

    auto write_energy2 = [&](std::string label__, double value__) {
        out__ << std::left << std::setw(30) << label__ << " : " << std::right << std::setw(16) << std::setprecision(8)
              << std::fixed << value__ << " (Ha), " << std::setw(16) << std::setprecision(8) << std::fixed
              << value__ * 2 << " (Ry)" << std::endl;
    };

    write_energy("valence_eval_sum", evalsum1);
    if (ctx_.full_potential()) {
        write_energy("core_eval_sum", evalsum2);
        write_energy("kinetic energy", ekin);
        write_energy("enuc", enuc);
    }
    write_energy("<rho|V^{XC}>", evxc);
    write_energy("<rho|E^{XC}>", eexc);
    write_energy("<mag|B^{XC}>", ebxc);
    write_energy("<rho|V^{H}>", evha);
    if (!ctx_.full_potential()) {
        write_energy2("one-electron contribution", one_elec_en); // eband + deband in QE
        write_energy("hartree contribution", 0.5 * evha);
        write_energy("xc contribution", eexc);
        write_energy("ewald contribution", ewald_energy_);
        write_energy("PAW contribution", potential_.PAW_total_energy(density_));
    }
    write_energy("smearing (-TS)", s_sum);
    write_energy("SCF correction", this->scf_correction_energy_);
    if (ctx_.hubbard_correction()) {
        auto e = ::sirius::energy(density_.occupation_matrix());
        write_energy2("Hubbard energy", e);
        write_energy2("Hubbard one-el contribution", hub_one_elec);
    }
    write_energy2("Total energy", etot);
    out__ << std::endl;
    write_energy("band gap (eV)", gap);
    write_energy("Efermi", ef);
}

} // namespace sirius
