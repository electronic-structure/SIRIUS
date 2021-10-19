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
#include "utils/profiler.hpp"

namespace sirius {

void
DFT_ground_state::initial_state()
{
    density_.initial_density();
    potential_.generate(density_, ctx_.use_symmetry(), true);
    if (!ctx_.full_potential()) {
        if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(USE_FP32)
            Hamiltonian0<float> H0(potential_);
            Band(ctx_).initialize_subspace(kset_, H0);
#else
            RTE_THROW("not compiled with FP32 support");
#endif

        } else {
            Hamiltonian0<double> H0(potential_);
            Band(ctx_).initialize_subspace(kset_, H0);
        }
    }
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

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_.get<double>(ik);

        #pragma omp parallel for schedule(static) reduction(+:ekin)
        for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
            auto Gk = kp->gkvec().gkvec_cart<index_domain_t::local>(igloc);

            double d{0};
            for (int ispin = 0; ispin < ctx_.num_spins(); ispin++) {
                for (int i = 0; i < kp->num_occupied_bands(ispin); i++) {
                    double f = kp->band_occupancy(i, ispin);
                    auto z   = kp->spinor_wave_functions().pw_coeffs(ispin).prime(igloc, i);
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
    return sirius::total_energy(ctx_, kset_, density_, potential_, ewald_energy_) + this->scf_energy_;
}

json
DFT_ground_state::serialize()
{
    nlohmann::json dict;

    dict["energy"]                  = json::object();
    dict["energy"]["total"]         = total_energy();
    dict["energy"]["enuc"]          = energy_enuc(ctx_, potential_);
    dict["energy"]["core_eval_sum"] = core_eval_sum(ctx_.unit_cell());
    dict["energy"]["vha"]           = energy_vha(potential_);
    dict["energy"]["vxc"]           = energy_vxc(density_, potential_);
    dict["energy"]["exc"]           = energy_exc(density_, potential_);
    dict["energy"]["bxc"]           = energy_bxc(density_, potential_);
    dict["energy"]["veff"]          = energy_veff(density_, potential_);
    dict["energy"]["eval_sum"]      = eval_sum(ctx_.unit_cell(), kset_);
    dict["energy"]["kin"]           = energy_kin(ctx_, kset_, density_, potential_);
    dict["energy"]["ewald"]         = ewald_energy_;
    if (!ctx_.full_potential()) {
        dict["energy"]["vloc"] = energy_vloc(density_, potential_);
    }
    dict["energy"]["scf_correction"] = this->scf_energy_;
    dict["energy"]["entropy_sum"]    = kset_.entropy_sum();
    dict["efermi"]                   = kset_.energy_fermi();
    dict["band_gap"]                 = kset_.band_gap();
    dict["core_leakage"]             = density_.core_leakage();

    return dict;
}

/// A quick check of self-constent density in case of pseudopotential.
json
DFT_ground_state::check_scf_density()
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
    pot.generate(density_, ctx_.use_symmetry(), true);
    /* create new Hamiltonian */
    Hamiltonian0<double> H0(pot);
    /* initialize the subspace */
    Band(ctx_).initialize_subspace(kset_, H0);
    /* find new wave-functions */
    Band(ctx_).solve<double, double>(kset_, H0, true, ctx_.cfg().settings().itsol_tol_min());
    /* find band occupancies */
    kset_.find_band_occupancies<double>();
    /* generate new density from the occupied wave-functions */
    density_.generate<double>(kset_, true, true, false);
    double rms{0};
    for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
        rms += std::pow(std::abs(density_.rho().f_pw_local(ig) - rho_pw[ig]), 2);
    }
    ctx_.comm().allreduce(&rms, 1);
    json dict;
    dict["rss"]   = rms;
    dict["rms"]   = std::sqrt(rms / ctx_.gvec().num_gvec());
    dict["detot"] = total_energy() - etot;

    ctx_.message(1, __function_name__, "RSS: %18.12E\n", dict["rss"].get<double>());
    ctx_.message(1, __function_name__, "RMS: %18.12E\n", dict["rms"].get<double>());
    ctx_.message(1, __function_name__, "dEtot: %18.12E\n", dict["detot"].get<double>());
    ctx_.message(1, __function_name__, "Eold: %18.12E  Enew: %18.12E\n", etot, total_energy());

    return dict;
}

json DFT_ground_state::find(double density_tol, double energy_tol, double itsol_tol, int num_dft_iter, bool write_state)
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
    s << "density_tol            : " << density_tol << std::endl
      << "energy_tol             : " << energy_tol << std::endl
      << "itsol_tol (initial)    : " << itsol_tol << std::endl
      << "itsol_tol_min          : " << ctx_.cfg().settings().itsol_tol_min() << std::endl
      << "num_dft_iter           : " << num_dft_iter;
    ctx_.message(1, __func__, s);

    for (int iter = 0; iter < num_dft_iter; iter++) {
        PROFILE("sirius::DFT_ground_state::scf_loop|iteration");
        std::stringstream s;
        s << std::endl;
        s << "+------------------------------+" << std::endl
          << "| SCF iteration " << std::setw(3) << iter << " out of " << std::setw(3) << num_dft_iter << '|' << std::endl
          << "+------------------------------+" << std::endl;
        ctx_.message(2, __func__, s);

        if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(USE_FP32)
            Hamiltonian0<float> H0(potential_);
            /* find new wave-functions */
            if (ctx_.cfg().parameters().precision_hs() == "fp32") {
                Band(ctx_).solve<float, float>(kset_, H0, true, itsol_tol);
            } else {
                Band(ctx_).solve<float, double>(kset_, H0, true, itsol_tol);
            }
            /* find band occupancies */
            kset_.find_band_occupancies<float>();
            /* generate new density from the occupied wave-functions */
            density_.generate<float>(kset_, ctx_.use_symmetry(), true, true);
#else
            RTE_THROW("not compiled with FP32 support");
#endif
        } else {
            Hamiltonian0<double> H0(potential_);
            /* find new wave-functions */
            Band(ctx_).solve<double, double>(kset_, H0, true, itsol_tol);
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
            //tol = rms * rms / std::max(1.0, unit_cell_.num_electrons());
            tol = eha_res / std::max(1.0, unit_cell_.num_electrons());
        }
        tol = std::min(ctx_.cfg().settings().itsol_tol_scale()[0] * tol,
                       ctx_.cfg().settings().itsol_tol_scale()[1] * itsol_tol);
        /* tolerance can't be too small */
        itsol_tol = std::max(ctx_.cfg().settings().itsol_tol_min(), tol);

#if defined(USE_FP32)
        /* if the final precision is not equal to the current precision */
        if (ctx_.cfg().parameters().precision_gs() == "fp64" && ctx_.cfg().parameters().precision_wf() == "fp32") {
            /* if we reached the mimimum tolerance for fp32 */
            if ((ctx_.cfg().settings().fp32_to_fp64_rms() == 0 && itsol_tol <= ctx_.cfg().settings().itsol_tol_min()) ||
                (rms < ctx_.cfg().settings().fp32_to_fp64_rms())) {
                std::cout << "switching to FP64" << std::endl;
                ctx_.cfg().unlock();
                ctx_.cfg().settings().itsol_tol_min(std::numeric_limits<double>::epsilon() * 10);
                ctx_.cfg().parameters().precision_wf("fp64");
                ctx_.cfg().parameters().precision_hs("fp64");
                ctx_.cfg().lock();

                for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
                    int ik = kset_.spl_num_kpoints(ikloc);
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        kset_.get<double>(ik)->spinor_wave_functions().copy_from(device_t::CPU, ctx_.num_bands(),
                                kset_.get<float>(ik)->spinor_wave_functions(), ispn, 0, ispn, 0);
                    }
                }
                for (int ik = 0; ik < kset_.num_kpoints(); ik++) {
                    for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                        for (int j = 0; j < ctx_.num_bands(); j++) {
                            kset_.get<double>(ik)->band_energy(j, ispn, kset_.get<float>(ik)->band_energy(j, ispn));
                            kset_.get<double>(ik)->band_occupancy(j, ispn, kset_.get<float>(ik)->band_occupancy(j, ispn));
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
            ctx_.message(1, __function_name__, "%s", "checking functional derivative of Exc\n");
            sirius::check_xc_potential(density_);
        }

        if (ctx_.cfg().parameters().use_scf_correction()) {
            double e2         = energy_potential(rho1, potential_);
            this->scf_energy_ = e2 - e1;
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
            << ", energy difference : " << std::setprecision(12) << std::scientific << etot - eold << std::endl
            << "residual density Hartree energy : " << eha_res;
        ctx_.message(2, __func__, out);
        /* check if the calculation has converged */
        bool converged{true};
        converged = converged && (std::abs(eold - etot) < energy_tol);
        if (ctx_.cfg().mixer().use_hartree()) {
            converged = converged && (eha_res < density_tol);
        } else {
            converged = converged && (rms < density_tol);
        }
        if (converged) {
            std::stringstream out;
            out << std::endl;
            out << "converged after " << iter + 1 << " SCF iterations!";
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
        // kset_.save(storage_file_name);
    }

    auto tstop = std::chrono::high_resolution_clock::now();

    json dict            = serialize();
    dict["scf_time"]     = std::chrono::duration_cast<std::chrono::duration<double>>(tstop - tstart).count();
    dict["etot_history"] = etot_hist;
    if (num_iter >= 0) {
        dict["converged"]          = true;
        dict["num_scf_iterations"] = num_iter;
        dict["rms_history"]        = rms_hist;
    } else {
        dict["converged"] = false;
    }

    // if (ctx_.control().verification_ >= 1) {
    //    check_scf_density();
    //}

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

    auto result = density_.rho().integrate();

    auto total_charge = std::get<0>(result);
    auto it_charge    = std::get<1>(result);
    auto mt_charge    = std::get<2>(result);

    auto result_mag = density_.get_magnetisation();
    auto total_mag  = std::get<0>(result_mag);
    auto it_mag     = std::get<1>(result_mag);
    auto mt_mag     = std::get<2>(result_mag);

    auto draw_bar = [&](int w) { out__ << std::setfill('-') << std::setw(w) << '-' << std::setfill(' ') << std::endl; };

    auto write_vector = [&](vector3d<double> v__) {
        out__ << "[" << std::setw(9) << std::setprecision(5) << std::fixed << v__[0] << ", " << std::setw(9)
              << std::setprecision(5) << std::fixed << v__[1] << ", " << std::setw(9) << std::setprecision(5)
              << std::fixed << v__[2] << "]";
    };

    out__ << "Charges and magnetic moments" << std::endl;
    draw_bar(80);
    if (ctx_.full_potential()) {
        double total_core_leakage{0.0};
        out__ << "atom      charge    core leakage";
        if (ctx_.num_mag_dims()) {
            out__ << "                 moment                |moment|";
        }
        out__ << std::endl;
        draw_bar(80);

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            double core_leakage = unit_cell_.atom(ia).symmetry_class().core_leakage();
            total_core_leakage += core_leakage;
            out__ << std::setw(4) << ia << std::setw(12) << std::setprecision(6) << std::fixed << mt_charge[ia]
                  << std::setw(16) << std::setprecision(6) << std::scientific << core_leakage;
            if (ctx_.num_mag_dims()) {
                vector3d<double> v(mt_mag[ia]);
                out__ << "  ";
                write_vector(v);
                out__ << std::setw(12) << std::setprecision(6) << std::fixed << v.length();
            }
            out__ << std::endl;
        }
        out__ << std::endl;
        out__ << "total core leakage    : " << std::setprecision(8) << std::scientific << total_core_leakage
              << std::endl
              << "interstitial charge   : " << std::setprecision(6) << std::fixed << it_charge << std::endl;
        if (ctx_.num_mag_dims()) {
            vector3d<double> v(it_mag);
            out__ << "interstitial moment   : ";
            write_vector(v);
            out__ << ", magnitude : " << std::setprecision(6) << std::fixed << v.length() << std::endl;
        }
    } else {
        if (ctx_.num_mag_dims()) {
            out__ << "atom                moment                |moment|" << std::endl;
            draw_bar(80);

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                vector3d<double> v(mt_mag[ia]);
                out__ << std::setw(4) << ia << " ";
                write_vector(v);
                out__ << std::setw(12) << std::setprecision(6) << std::fixed << v.length() << std::endl;
            }
            out__ << std::endl;
        }
    }
    out__ << "total charge          : " << std::setprecision(6) << std::fixed << total_charge << std::endl;

    if (ctx_.num_mag_dims()) {
        vector3d<double> v(total_mag);
        out__ << "total moment          : ";
        write_vector(v);
        out__ << ", magnitude : " << std::setprecision(6) << std::fixed << v.length() << std::endl;
    }

    out__ << std::endl;
    out__ << "Energy" << std::endl;
    draw_bar(80);

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
        write_energy("PAW contribution", potential_.PAW_total_energy());
    }
    write_energy("smearing (-TS)", s_sum);
    write_energy("SCF correction", this->scf_energy_);
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
