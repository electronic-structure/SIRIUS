// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
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

#include <limits>
#include "dft/smearing.hpp"
#include "k_point/k_point.hpp"
#include "k_point/k_point_set.hpp"
#include "symmetry/get_irreducible_reciprocal_mesh.hpp"
#include <iomanip>

#include "hamiltonian/non_local_operator.hpp"
#include "linalg/inverse_sqrt.hpp"



namespace sirius {

template <typename T, sync_band_t what>
void
K_point_set::sync_band()
{
    PROFILE("sirius::K_point_set::sync_band");

    sddk::mdarray<double, 3> data(ctx_.num_bands(), ctx_.num_spinors(), num_kpoints(),
                                  get_memory_pool(sddk::memory_t::host), "K_point_set::sync_band.data");

    int nb = ctx_.num_bands() * ctx_.num_spinors();
#pragma omp parallel
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
        int ik  = spl_num_kpoints_[ikloc];
        auto kp = this->get<T>(ik);
        switch (what) {
            case sync_band_t::energy: {
                std::copy(&kp->band_energies_(0, 0), &kp->band_energies_(0, 0) + nb, &data(0, 0, ik));
                break;
            }
            case sync_band_t::occupancy: {
                std::copy(&kp->band_occupancies_(0, 0), &kp->band_occupancies_(0, 0) + nb, &data(0, 0, ik));
                break;
            }
        }
    }

    comm().allgather(data.at(sddk::memory_t::host), nb * spl_num_kpoints_.local_size(),
                     nb * spl_num_kpoints_.global_offset());

#pragma omp parallel for
    for (int ik = 0; ik < num_kpoints(); ik++) {
        auto kp = this->get<T>(ik);
        switch (what) {
            case sync_band_t::energy: {
                std::copy(&data(0, 0, ik), &data(0, 0, ik) + nb, &kp->band_energies_(0, 0));
                break;
            }
            case sync_band_t::occupancy: {
                std::copy(&data(0, 0, ik), &data(0, 0, ik) + nb, &kp->band_occupancies_(0, 0));
                break;
            }
        }
    }
}

template void K_point_set::sync_band<double, sync_band_t::energy>();

template void K_point_set::sync_band<double, sync_band_t::occupancy>();

#if defined(USE_FP32)
template void K_point_set::sync_band<float, sync_band_t::energy>();

template void K_point_set::sync_band<float, sync_band_t::occupancy>();
#endif

void
K_point_set::create_k_mesh(r3::vector<int> k_grid__, r3::vector<int> k_shift__, int use_symmetry__)
{
    PROFILE("sirius::K_point_set::create_k_mesh");

    int nk;
    sddk::mdarray<double, 2> kp;
    std::vector<double> wk;
    if (use_symmetry__) {
        auto result = get_irreducible_reciprocal_mesh(ctx_.unit_cell().symmetry(), k_grid__, k_shift__);
        nk          = std::get<0>(result);
        wk          = std::get<1>(result);
        auto tmp    = std::get<2>(result);
        kp          = sddk::mdarray<double, 2>(3, nk);
        for (int i = 0; i < nk; i++) {
            for (int x : {0, 1, 2}) {
                kp(x, i) = tmp[i][x];
            }
        }
    } else {
        nk = k_grid__[0] * k_grid__[1] * k_grid__[2];
        wk = std::vector<double>(nk, 1.0 / nk);
        kp = sddk::mdarray<double, 2>(3, nk);

        int ik = 0;
        for (int i0 = 0; i0 < k_grid__[0]; i0++) {
            for (int i1 = 0; i1 < k_grid__[1]; i1++) {
                for (int i2 = 0; i2 < k_grid__[2]; i2++) {
                    kp(0, ik) = double(i0 + k_shift__[0] / 2.0) / k_grid__[0];
                    kp(1, ik) = double(i1 + k_shift__[1] / 2.0) / k_grid__[1];
                    kp(2, ik) = double(i2 + k_shift__[2] / 2.0) / k_grid__[2];
                    ik++;
                }
            }
        }
    }

    for (int ik = 0; ik < nk; ik++) {
        add_kpoint(&kp(0, ik), wk[ik]);
    }

    initialize();
}

void
K_point_set::initialize(std::vector<int> const& counts)
{
    if (this->initialized_) {
        RTE_THROW("K-point set is already initialized");
    }
    PROFILE("sirius::K_point_set::initialize");
    /* distribute k-points along the 1-st dimension of the MPI grid */
    if (counts.empty()) {
        sddk::splindex<sddk::splindex_t::block> spl_tmp(num_kpoints(), comm().size(), comm().rank());
        spl_num_kpoints_ =
            sddk::splindex<sddk::splindex_t::chunk>(num_kpoints(), comm().size(), comm().rank(), spl_tmp.counts());
    } else {
        spl_num_kpoints_ = sddk::splindex<sddk::splindex_t::chunk>(num_kpoints(), comm().size(), comm().rank(), counts);
    }

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
        kpoints_[spl_num_kpoints_[ikloc]]->initialize();
#if defined(USE_FP32)
        kpoints_float_[spl_num_kpoints_[ikloc]]->initialize();
#endif
    }

    if (ctx_.verbosity() > 0) {
        this->print_info();
    }
    print_memory_usage(ctx_.out(), FILE_LINE);
    this->initialized_ = true;
}

template <class F>
double
bisection_search(F&& f, double a, double b, double tol, int maxstep = 1000)
{
    double x  = (a + b) / 2;
    double fi = f(x);
    int step{0};
    /* compute fermy energy */
    while (std::abs(fi) >= tol) {
        /* compute total number of electrons */

        if (fi > 0) {
            b = x;
        } else {
            a = x;
        }

        x  = (a + b) / 2.0;
        fi = f(x);

        if (step > maxstep) {
            std::stringstream s;
            s << "search of band occupancies failed after 10000 steps";
            TERMINATE(s);
        }
        step++;
    }

    return x;
}

/**
 *  Newton minimization to determine the chemical potential.
 *
 *  \param  N       number of electrons as a function of \f$\mu\f$
 *  \param  dN      \f$\partial_\mu N(\mu)\f$
 *  \param  ddN     \f$\partial^2_\mu N(\mu)\f$
 *  \param  mu0     initial guess
 *  \param  ne      target number of electrons
 *  \param  tol     tolerance
 *  \param  maxstep max number of Newton iterations
 */
template <class Nt, class DNt, class D2Nt>
auto
newton_minimization_chemical_potential(Nt&& N, DNt&& dN, D2Nt&& ddN, double mu0, double ne, double tol,
                                       int maxstep = 1000)
{
    struct
    {
        double mu;              // chemical potential
        int iter;               // newton information
        std::vector<double> ys; // newton history
    } res;
    double mu = mu0;
    double alpha{1.0}; // Newton damping
    int iter{0};
    while (true) {
        // compute
        double Nf   = N(mu);
        double dNf  = dN(mu);
        double ddNf = ddN(mu);
        /* minimize (N(mu) - ne)^2  */
        // double F = (Nf - ne) * (Nf - ne);
        double dF  = 2 * (Nf - ne) * dNf;
        double ddF = 2 * dNf * dNf + 2 * (Nf - ne) * ddNf;
        mu         = mu - alpha * dF / std::abs(ddF);

        res.ys.push_back(mu);

        if (std::abs(dF) < tol) {
            res.iter = iter;
            res.mu   = mu;
            return res;
        }

        if (std::abs(ddF) < 1e-10) {
            std::stringstream s;
            s << "Newton minimization (chemical potential) failed because 2nd derivative too close to zero!";
            RTE_THROW(s);
        }

        iter++;
        if (iter > maxstep) {
            std::stringstream s;
            s << "Newton minimization (chemical potential) failed after " << maxstep << " steps!" << std::endl
              << "target number of electrons : " << ne << std::endl
              << "initial guess for chemical potential : " << mu0 << std::endl
              << "current value of chemical potential : " << mu;
            RTE_THROW(s);
        }
    }
}

template <typename T>
void
K_point_set::find_band_occupancies()
{
    PROFILE("sirius::K_point_set::find_band_occupancies");

    double tol{1e-11};

    auto band_occ_callback = ctx_.band_occ_callback();
    if (band_occ_callback) {
        band_occ_callback();
        return;
    }

    /* target number of electrons */
    const double ne_target = ctx_.unit_cell().num_valence_electrons() - ctx_.cfg().parameters().extra_charge();

    /* this is a special case when there are no empty states */
    if (ctx_.num_mag_dims() != 1 && std::abs(ctx_.num_bands() * ctx_.max_occupancy() - ne_target) < 1e-10) {
        /* this is an insulator, skip search for band occupancies */
        this->band_gap_ = 0;

        /* determine fermi energy as max occupied band energy. */
        energy_fermi_ = std::numeric_limits<double>::lowest();
        for (int ik = 0; ik < num_kpoints(); ik++) {
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                for (int j = 0; j < ctx_.num_bands(); j++) {
                    energy_fermi_ = std::max(energy_fermi_, this->get<T>(ik)->band_energy(j, ispn));
                }
            }
        }
        for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
            int ik = spl_num_kpoints_[ikloc];
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
#pragma omp parallel for
                for (int j = 0; j < ctx_.num_bands(); j++) {
                    this->get<T>(ik)->band_occupancy(j, ispn, ctx_.max_occupancy());
                }
            }
        }

        this->sync_band<T, sync_band_t::occupancy>();
        return;
    }

    if (ctx_.smearing_width() == 0) {
        RTE_THROW("zero smearing width");
    }

    /* get minimum and maximum band energies */

    auto emin = std::numeric_limits<double>::max();
    auto emax = std::numeric_limits<double>::lowest();

#pragma omp parallel for reduction(min : emin) reduction(max : emax)
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
        int ik = spl_num_kpoints_[ikloc];
        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
            emin = std::min(emin, this->get<T>(ik)->band_energy(0, ispn));
            emax = std::max(emax, this->get<T>(ik)->band_energy(ctx_.num_bands() - 1, ispn));
        }
    }
    comm().allreduce<double, mpi::op_t::min>(&emin, 1);
    comm().allreduce<double, mpi::op_t::max>(&emax, 1);

    sddk::splindex<sddk::splindex_t::block> splb(ctx_.num_bands(), ctx_.comm_band().size(), ctx_.comm_band().rank());

    /* computes N(ef; f) = \sum_{i,k} f(ef - e_{k,i}) */
    auto compute_ne = [&](double ef, auto&& f) {
        double ne{0};
        for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
            int ik = spl_num_kpoints_[ikloc];
            double tmp{0};
#pragma omp parallel reduction(+ : tmp)
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
#pragma omp for
                for (int j = 0; j < splb.local_size(); j++) {
                    tmp += f(ef - this->get<T>(ik)->band_energy(splb[j], ispn)) * ctx_.max_occupancy();
                }
            }
            ne += tmp * kpoints_[ik]->weight();
        }
        ctx_.comm().allreduce(&ne, 1);
        return ne;
    };

    /* smearing function */
    std::function<double(double)> f;
    if (ctx_.smearing() == smearing::smearing_t::cold || ctx_.smearing() == smearing::smearing_t::methfessel_paxton) {
        // obtain initial guess for non-monotous smearing with Gaussian
        f = [&](double x) { return smearing::gaussian::occupancy(x, ctx_.smearing_width()); };
    } else {
        f = smearing::occupancy(ctx_.smearing(), ctx_.smearing_width());
    }

    try {
        auto F        = [&compute_ne, ne_target, &f](double x) { return compute_ne(x, f) - ne_target; };
        energy_fermi_ = bisection_search(F, emin, emax, 1e-11);

        /* for cold and Methfessel Paxton smearing start newton minimization  */
        if (ctx_.smearing() == smearing::smearing_t::cold ||
            ctx_.smearing() == smearing::smearing_t::methfessel_paxton) {
            f               = smearing::occupancy(ctx_.smearing(), ctx_.smearing_width());
            auto df         = smearing::occupancy_deriv(ctx_.smearing(), ctx_.smearing_width());
            auto ddf        = smearing::occupancy_deriv2(ctx_.smearing(), ctx_.smearing_width());
            auto N          = [&](double mu) { return compute_ne(mu, f); };
            auto dN         = [&](double mu) { return compute_ne(mu, df); };
            auto ddN        = [&](double mu) { return compute_ne(mu, ddf); };
            auto res_newton = newton_minimization_chemical_potential(N, dN, ddN, energy_fermi_, ne_target, tol, 300);
            energy_fermi_   = res_newton.mu;
            if (ctx_.verbosity() >= 2) {
                RTE_OUT(ctx_.out()) << "newton iteration converged after " << res_newton.iter << " steps\n";
            }
        }
    } catch (std::exception const& e) {
        if (ctx_.verbosity() >= 2) {
            RTE_OUT(ctx_.out()) << e.what() << std::endl << "fallback to bisection search" << std::endl;
        }
        f             = smearing::occupancy(ctx_.smearing(), ctx_.smearing_width());
        auto F        = [&compute_ne, ne_target, &f](double x) { return compute_ne(x, f) - ne_target; };
        energy_fermi_ = bisection_search(F, emin, emax, tol);
    }

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
        int ik = spl_num_kpoints_[ikloc];
        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
#pragma omp parallel for
            for (int j = 0; j < ctx_.num_bands(); j++) {
                auto o = f(energy_fermi_ - this->get<T>(ik)->band_energy(j, ispn)) * ctx_.max_occupancy();
                this->get<T>(ik)->band_occupancy(j, ispn, o);
            }
        }
    }

    this->sync_band<T, sync_band_t::occupancy>();

    band_gap_ = 0.0;

    int nve = static_cast<int>(ne_target + 1e-12);
    if (ctx_.num_spins() == 2 || (std::abs(nve - ne_target) < 1e-12 && nve % 2 == 0)) {
        /* find band gap */
        std::vector<std::pair<double, double>> eband(ctx_.num_bands() * ctx_.num_spinors());

        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
#pragma omp for
            for (int j = 0; j < ctx_.num_bands(); j++) {
                std::pair<double, double> eminmax;
                eminmax.first  = std::numeric_limits<double>::max();
                eminmax.second = std::numeric_limits<double>::lowest();

                for (int ik = 0; ik < num_kpoints(); ik++) {
                    eminmax.first  = std::min(eminmax.first, this->get<T>(ik)->band_energy(j, ispn));
                    eminmax.second = std::max(eminmax.second, this->get<T>(ik)->band_energy(j, ispn));
                }

                eband[j + ispn * ctx_.num_bands()] = eminmax;
            }
        }

        std::sort(eband.begin(), eband.end());

        int ist = nve;
        if (ctx_.num_spins() == 1) {
            ist /= 2;
        }

        if (eband[ist].first > eband[ist - 1].second) {
            band_gap_ = eband[ist].first - eband[ist - 1].second;
        }
    }
}

template void K_point_set::find_band_occupancies<double>();
#if defined(USE_FP32)
template void K_point_set::find_band_occupancies<float>();
#endif

template <typename T>
double
K_point_set::valence_eval_sum() const
{
    double eval_sum{0};

    sddk::splindex<sddk::splindex_t::block> splb(ctx_.num_bands(), ctx_.comm_band().size(), ctx_.comm_band().rank());

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
        auto ik        = spl_num_kpoints_[ikloc];
        auto const& kp = this->get<T>(ik);
        double tmp{0};
#pragma omp parallel for reduction(+ : tmp)
        for (int j = 0; j < splb.local_size(); j++) {
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                tmp += kp->band_energy(splb[j], ispn) * kp->band_occupancy(splb[j], ispn);
            }
        }
        eval_sum += kp->weight() * tmp;
    }
    ctx_.comm().allreduce(&eval_sum, 1);

    return eval_sum;
}

double
K_point_set::valence_eval_sum() const
{
    if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(USE_FP32)
        return this->valence_eval_sum<float>();
#else
        RTE_THROW("not compiled with FP32 support");
        return 0; // make compiled happy
#endif
    } else {
        return this->valence_eval_sum<double>();
    }
}

template <typename T>
double
K_point_set::entropy_sum() const
{
    double s_sum{0};

    double ne_target = ctx_.unit_cell().num_valence_electrons() - ctx_.cfg().parameters().extra_charge();

    bool only_occ = (ctx_.num_mag_dims() != 1 && std::abs(ctx_.num_bands() * ctx_.max_occupancy() - ne_target) < 1e-10);

    if (only_occ) {
        return 0;
    }

    auto f = smearing::entropy(ctx_.smearing(), ctx_.smearing_width());

    sddk::splindex<sddk::splindex_t::block> splb(ctx_.num_bands(), ctx_.comm_band().size(), ctx_.comm_band().rank());

    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
        auto ik        = spl_num_kpoints_[ikloc];
        auto const& kp = this->get<T>(ik);
        double tmp{0};
#pragma omp parallel for reduction(+ : tmp)
        for (int j = 0; j < splb.local_size(); j++) {
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                tmp += ctx_.max_occupancy() * f(energy_fermi_ - kp->band_energy(splb[j], ispn));
            }
        }
        s_sum += kp->weight() * tmp;
    }
    ctx_.comm().allreduce(&s_sum, 1);

    return s_sum;
}

double
K_point_set::entropy_sum() const
{
    if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(USE_FP32)
        return this->entropy_sum<float>();
#else
        RTE_THROW("not compiled with FP32 support");
        return 0; // make compiler happy
#endif
    } else {
        return this->entropy_sum<double>();
    }
}

void
K_point_set::print_info()
{
    mpi::pstdout pout(this->comm());

    if (ctx_.comm().rank() == 0) {
        pout << std::endl;
        pout << "total number of k-points : " << num_kpoints() << std::endl;
        pout << utils::hbar(80, '-') << std::endl;
        pout << std::endl;
        pout << "  ik                vk                    weight  num_gkvec";
        if (ctx_.full_potential()) {
            pout << "   gklo_basis_size";
        }
        pout << std::endl << utils::hbar(80, '-') << std::endl;
    }

    for (int ikloc = 0; ikloc < spl_num_kpoints().local_size(); ikloc++) {
        int ik = spl_num_kpoints(ikloc);
        pout << std::setw(4) << ik << utils::ffmt(9, 4) << kpoints_[ik]->vk()[0] << utils::ffmt(9, 4)
             << kpoints_[ik]->vk()[1] << utils::ffmt(9, 4) << kpoints_[ik]->vk()[2] << utils::ffmt(17, 6)
             << kpoints_[ik]->weight() << std::setw(11) << kpoints_[ik]->num_gkvec();

        if (ctx_.full_potential()) {
            pout << std::setw(18) << kpoints_[ik]->gklo_basis_size();
        }
        pout << std::endl;
    }
    RTE_OUT(ctx_.out()) << pout.flush(0);
}

void
K_point_set::save(std::string const& name__) const
{
    if (ctx_.comm().rank() == 0) {
        if (!utils::file_exists(name__)) {
            sddk::HDF5_tree(name__, sddk::hdf5_access_t::truncate);
        }
        sddk::HDF5_tree fout(name__, sddk::hdf5_access_t::read_write);
        fout.create_node("K_point_set");
        fout["K_point_set"].write("num_kpoints", num_kpoints());
    }
    ctx_.comm().barrier();
    for (int ik = 0; ik < num_kpoints(); ik++) {
        /* check if this ranks stores the k-point */
        if (ctx_.comm_k().rank() == spl_num_kpoints_.local_rank(ik)) {
            this->get<double>(ik)->save(name__, ik);
        }
        /* wait for all */
        ctx_.comm().barrier();
    }
}

/// \todo check parameters of saved data in a separate function
void
K_point_set::load()
{
    STOP();

    //== HDF5_tree fin(storage_file_name, false);

    //== int num_kpoints_in;
    //== fin["K_point_set"].read("num_kpoints", &num_kpoints_in);

    //== std::vector<int> ikidx(num_kpoints(), -1);
    //== // read available k-points
    //== double vk_in[3];
    //== for (int jk = 0; jk < num_kpoints_in; jk++)
    //== {
    //==     fin["K_point_set"][jk].read("coordinates", vk_in, 3);
    //==     for (int ik = 0; ik < num_kpoints(); ik++)
    //==     {
    //==         r3::vector<double> dvk;
    //==         for (int x = 0; x < 3; x++) dvk[x] = vk_in[x] - kpoints_[ik]->vk(x);
    //==         if (dvk.length() < 1e-12)
    //==         {
    //==             ikidx[ik] = jk;
    //==             break;
    //==         }
    //==     }
    //== }

    //== for (int ik = 0; ik < num_kpoints(); ik++)
    //== {
    //==     int rank = spl_num_kpoints_.local_rank(ik);
    //==
    //==     if (comm_.rank() == rank) kpoints_[ik]->load(fin["K_point_set"], ikidx[ik]);
    //== }
}

//== void K_point_set::save_wave_functions()
//== {
//==     if (Platform::mpi_rank() == 0)
//==     {
//==         HDF5_tree fout(storage_file_name, false);
//==         fout["parameters"].write("num_kpoints", num_kpoints());
//==         fout["parameters"].write("num_bands", ctx_.num_bands());
//==         fout["parameters"].write("num_spins", ctx_.num_spins());
//==     }
//==
//==     if (ctx_.mpi_grid().side(1 << _dim_k_ | 1 << _dim_col_))
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++)
//==         {
//==             int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
//==
//==             if (ctx_.mpi_grid().coordinate(_dim_k_) == rank) kpoints_[ik]->save_wave_functions(ik);
//==
//==             ctx_.mpi_grid().barrier(1 << _dim_k_ | 1 << _dim_col_);
//==         }
//==     }
//== }
//==
//== void K_point_set::load_wave_functions()
//== {
//==     HDF5_tree fin(storage_file_name, false);
//==     int num_spins;
//==     fin["parameters"].read("num_spins", &num_spins);
//==     if (num_spins != ctx_.num_spins()) s between the Samsung Galaxy A33 and Galaxy A53 are found in the camera
// specs and design features. The Galaxy A53 has a slightly larger phone screen than the Galaxy A33, and is heavier. It
// also has a higher refresh rate than the A33, meaning your phone interactions are smoother.(__FILE__, __LINE__, "wrong
// number of spins");
//==
//==     int num_bands;
//==     fin["parameters"].read("num_bands", &num_bands);
//==     if (num_bands != ctx_.num_bands()) error_local(__FILE__, __LINE__, "wrong number of bands");
//==
//==     int num_kpoints_in;
//==     fin["parameters"].read("num_kpoints", &num_kpoints_in);
//==
//==     // ==================================================================
//==     // index of current k-points in the hdf5 file, which (in general) may
//==     // contain a different set of k-points
//==     // ==================================================================
//==     std::vector<int> ikidx(num_kpoints(), -1);
//==     // read available k-points
//==     double vk_in[3];
//==     for (int jk = 0; jk < num_kpoints_in; jk++)
//==     {
//==         fin["kpoints"][jk].read("coordinates", vk_in, 3);
//==         for (int ik = 0; ik < num_kpoints(); ik++)
//==         {
//==             r3::vector<double> dvk;
//==             for (int x = 0; x < 3; x++) dvk[x] = vk_in[x] - kpoints_[ik]->vk(x);
//==             if (dvk.length() < 1e-12)
//==             {
//==                 ikidx[ik] = jk;
//==                 break;
//==             }
//==         }
//==     }
//==
//==     for (int ik = 0; ik < num_kpoints(); ik++)
//==     {
//==         int rank = spl_num_kpoints_.location(_splindex_rank_, ik);
//==
//==         if (ctx_.mpi_grid().coordinate(0) == rank) kpoints_[ik]->load_wave_functions(ikidx[ik]);
//==     }
//== }

//== void K_point_set::fixed_band_occupancies()
//== {
//==     Timer t("sirius::K_point_set::fixed_band_occupancies");
//==
//==     if (ctx_.num_mag_dims() != 1) error_local(__FILE__, __LINE__, "works only for collinear magnetism");
//==
//==     double n_up = (ctx_.num_valence_electrons() + ctx_.fixed_moment()) / 2.0;
//==     double n_dn = (ctx_.num_valence_electrons() - ctx_.fixed_moment()) / 2.0;
//==
//==     mdarray<double, 2> bnd_occ(ctx_.num_bands(), num_kpoints());
//==     bnd_occ.zero();
//==
//==     int j = 0;
//==     while (n_up > 0)
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++) bnd_occ(j, ik) = std::min(double(ctx_.max_occupancy()), n_up);
//==         j++;
//==         n_up -= ctx_.max_occupancy();
//==     }
//==
//==     j = ctx_.num_fv_states();
//==     while (n_dn > 0)
//==     {
//==         for (int ik = 0; ik < num_kpoints(); ik++) bnd_occ(j, ik) = std::min(double(ctx_.max_occupancy()), n_dn);
//==         j++;
//==         n_dn -= ctx_.max_occupancy();
//==     }
//==
//==     for (int ik = 0; ik < num_kpoints(); ik++) kpoints_[ik]->set_band_occupancies(&bnd_occ(0, ik));
//==
//==     double gap = 0.0;
//==
//==     int nve = int(ctx_.num_valence_electrons() + 1e-12);
//==     if ((ctx_.num_spins() == 2) ||
//==         ((fabs(nve - ctx_.num_valence_electrons()) < 1e-12) && nve % 2 == 0))
//==     {
//==         // find band gap
//==         std::vector< std::pair<double, double> > eband;
//==         std::pair<double, double> eminmax;
//==
//==         for (int j = 0; j < ctx_.num_bands(); j++)
//==         {
//==             eminmax.first = 1e10;
//==             eminmax.second = -1e10;
//==
//==             for (int ik = 0; ik < num_kpoints(); ik++)
//==             {
//==                 eminmax.first = std::min(eminmax.first, kpoints_[ik]->band_energy(j));
//==                 eminmax.second = std::max(eminmax.second, kpoints_[ik]->band_energy(j));
//==             }
//==
//==             eband.push_back(eminmax);
//==         }
//==
//==         std::sort(eband.begin(), eband.end());
//==
//==         int ist = nve;
//==         if (ctx_.num_spins() == 1) ist /= 2;
//==
//==         if (eband[ist].first > eband[ist - 1].second) gap = eband[ist].first - eband[ist - 1].second;
//==
//==         band_gap_ = gap;
//==     }
//== }

extern "C" {
void wannier_setup_(const char*, int32_t*, int32_t*, const double*, const double*, double*,
                    int32_t*, // care! arg (4,5) changed with const
                    int32_t*, char (*)[3], double*, bool*, bool*, int32_t*, int32_t*, int32_t*, int32_t*, int32_t*,
                    double*, int32_t*, int32_t*, int32_t*, double*, double*, double*, int32_t*, int32_t*, double*,
                    size_t, size_t);

void wannier_run_(const char*, int32_t*, int32_t*, double*, double*, double*, int32_t*, int32_t*, int32_t*, int32_t*,
                  char (*)[3], double*, bool*, std::complex<double>*, std::complex<double>*, double*,
                  std::complex<double>*, std::complex<double>*, bool*, double*, double*, double*, size_t, size_t);
}

void
write_Amn(sddk::mdarray<std::complex<double>, 3>& Amn)
{
    std::ofstream writeAmn;
    writeAmn.open("sirius.amn");
    std::string line;
    writeAmn << "#produced in sirius" << std::endl;
    writeAmn << std::setw(10) << Amn.size(0);
    writeAmn << std::setw(10) << Amn.size(2);
    writeAmn << std::setw(10) << Amn.size(1);
    writeAmn << std::endl;

    for (size_t ik = 0; ik < Amn.size(2); ik++) {
        for (size_t n = 0; n < Amn.size(1); n++) {
            for (size_t m = 0; m < Amn.size(0); m++) {
                writeAmn << std::fixed << std::setw(5) << m + 1;
                writeAmn << std::fixed << std::setw(5) << n + 1;
                writeAmn << std::fixed << std::setw(5) << ik + 1;
                writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << Amn(m, n, ik).real();
                writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << Amn(m, n, ik).imag();
                //writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << abs(Amn(m, n, ik));
                writeAmn << std::endl;
            }
        }
    }
}

void
write_Mmn(sddk::mdarray<std::complex<double>, 4>& M, sddk::mdarray<int, 2>& nnlist, sddk::mdarray<int32_t, 3>& nncell)
{
    std::ofstream writeMmn;
    writeMmn.open("sirius.mmn");
    writeMmn << "#produced in sirius" << std::endl;
    writeMmn << std::setw(10) << M.size(0);
    writeMmn << std::setw(10) << M.size(3);
    writeMmn << std::setw(10) << M.size(2);
    writeMmn << std::endl;
    for (size_t ik = 0; ik < M.size(3); ik++) {
        for (size_t ib = 0; ib < M.size(2); ib++) {
            writeMmn << std::setw(5) << ik + 1;
            writeMmn << std::setw(5) << nnlist(ik, ib);
            writeMmn << std::setw(5) << nncell(0, ik, ib);
            writeMmn << std::setw(5) << nncell(1, ik, ib);
            writeMmn << std::setw(5) << nncell(2, ik, ib);
            writeMmn << std::endl;
            for (size_t n = 0; n < M.size(1); n++) {
                for (size_t m = 0; m < M.size(0); m++) {
                    writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << M(m, n, ib, ik).real();
                    writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << M(m, n, ib, ik).imag();
                    //writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << abs(M(m, n, ib, ik));
                    writeMmn << std::endl;
                }
            }
        }
    }
    writeMmn.close();
}


void write_eig(sddk::mdarray<double, 2>& eigval){
    std::ofstream writeEig;
    writeEig.open("sirius.eig");
    for(int ik=0; ik<eigval.size(1); ik++){
        for(int iband=0; iband<eigval.size(0); iband++){
            writeEig << std::setw(5) << iband + 1;
            writeEig << std::setw(5) << ik + 1;
            writeEig << std::fixed << std::setprecision(12) << std::setw(18) << eigval(iband,ik);
            writeEig << std::endl;
        }
    }
    writeEig.close();
} 

/// Generate the necessary data for the W90 input.
/** Wave-functions:
 * \f[
 *  \psi_{n{\bf k}} ({\bf r}) = \sum_{\bf G} e^{i({\bf G+k}){\bf r}} C_{n{\bf k}}({\bf G})
 * \f]
 *
 *  Matrix elements:
 *  \f{eqnarray*}{
 *  M_{nn'} &= \int e^{-i{\bf qr}}  \psi_{n{\bf k}}^{*} ({\bf r})  \psi_{n'{\bf k+q}} ({\bf r}) d{\bf r} =
 *    \sum_{\bf G} e^{-i({\bf G+k}){\bf r}} C_{n{\bf k}}^{*}({\bf G})
 *    \sum_{\bf G'} e^{i({\bf G'+k+q}){\bf r}} C_{n{\bf k+q}}({\bf G'}) e^{-i{\bf qr}} = \\
 *    &= \sum_{\bf GG'} \int e^{i({\bf G'-G}){\bf r}} d{\bf r}  C_{n{\bf k}}^{*}({\bf G}) C_{n{\bf k+q}}({\bf G'}) =
 *    \sum_{\bf G}  C_{n{\bf k}}^{*}({\bf G}) C_{n{\bf k+q}}({\bf G})
 *  \f}
 *
 *  Let's rewrite \f$ {\bf k + q} = {\bf \tilde G} + {\bf \tilde k} \f$. Now, through the property of plane-wave
 *  expansion coefficients \f$ C_{n{\bf k+q}}({\bf G}) = C_{n{\bf \tilde k}}({\bf G + \tilde G}) \f$ it follows that
 *  \f[
 *    M_{nn'} = \sum_{\bf G} C_{n{\bf k}}^{*}({\bf G}) C_{n{\bf \tilde k}}({\bf G + \tilde G})
 *  \f]
 */
void
K_point_set::generate_w90_coeffs(Hamiltonian0<double>& H0) // sirius::K_point_set& k_set__)
{

    std::cout << "I am rank " << ctx().comm().rank() << " ik " << ctx().comm_k().rank() << " band " << ctx().comm_band().rank();
    std::cout << " fft " << ctx().comm_fft().rank() << " ortho_fft " << ctx().comm_ortho_fft().rank();
    std::cout << " fft_coarse " << ctx().comm_fft_coarse().rank() << " ortho_fft_coarse " << ctx().comm_ortho_fft_coarse().rank();
    std::cout << " band_ortho_fft_coarse " << ctx().comm_band_ortho_fft_coarse().rank();
    // phase1: k-point exchange
    // each MPI rank sores the local set of k-points
    // for each k-point we have a list of q vectors to compute k+q. In general we assume that the number
    // of q-points nq(k) is nefferent for each k
    // The easy way to implement send/recieve of k-points is through brute-force broadcast:
    // each MPI rank broadcasts one-by-one each of its local k-points. Everyone listens and recieves the data;
    // only MPI ranks that need the broadcasted point as k+q are storing it in the local array. Yes, there is
    // some overhead in moving data between the MPI ranks, but this can be optimized later.
    //
    // phase1 is not required intially for the sequential code
    //
    // phase2: construnction of the k+q wave-functions and bringin them to the order of G+k G-vectors
    //
    // we are going to compute <psi_{n,k} | S exp{-iqr} | psi_{n',k+q}>
    // where S = 1 + \sum_{\alpha} \sum_{\xi, \xi'} |beta_{\xi}^{\alpha} Q_{\xi,\xi'}^{\alpha} <beta_{\xi'}^{\alpha}|
    //
    // the inner product splits into following contributions:
    // <psi_{n,k} | 1 + |beta>Q<beta|  psi_{n',k+q}> = <psi_{n,k} | exp^{-iqr} | psi_{n',k+q}> +
    // <psi_{n,k} | exp^{-iqr} |beta>Q<beta|  psi_{n',k+q}>
    //
    // we will need: |psi_{n',k+q}> in the order of G+k vectors
    //               <beta_{\xi'}^{\alpha}|  psi_{n',k+q}> computed at k+q
    //
    // we can then apply the Q matrix to <beta_{\xi'}^{\alpha}|  psi_{j,k+q}> and compute 1st and 2nd contributions
    // as two matrix multiplications.
    //
    //
    // For the ultrasoft contribution (2nd term):
    //   construct the matrix of <beta_{\xi'}^{\alpha}| psi_{n',k'}>, where k'+G'=k+q for all local k-points;
    //   exchange information between MPI ranks as is done for the wave-functions
    //
    //
    // 1st step: get a list of q-vectors for each k-point and a G' vector that bring k+q back into 1st Brilloun zone
    // this is the library equivalent step of producing nnkp file from w90
    //
    // 2nd step: compute <beta_{\xi'}^{\alpha}|  psi_{j,k+q}>; check how this is done in the Beta_projector class;
    // Q-operator can be applied here. Look how this is done in Non_local_operator::apply();
    // (look for Beta_projectors_base::inner() function; understand the "chunks" of beta-projectors
    //
    // 3nd step: copy wave-function at k+q (k') into an auxiliary wave-function object of G+k order and see how
    // the G+k+q index can be reshuffled. Check the implementation of G-vector class which handles all the G- and G+k-
    // indice
    //
    // 4th step: allocate resulting matrix M_{nn'}, compute contribution from C*C (1st part) using wf::inner() function;
    // compute contribution from ultrasoft part using a matrix-matrix multiplication
    //
    // 5th step: parallelize over k-points
    //
    // 6ts step: parallelize over G+k vectors and k-points
    PROFILE("sirius::K_point_set::generate_w90_coeffs");
    std::cout << "\n\n\nWannierization!!!!\n\n\n";

    /*
     * TASK 0: Find wavefunctions in the full brillouin zone from the ones in the irreducible wedge
    */


    /*
     * STEP 0.1: Recover (k points) full Brillouin zone from Irreducible wedge when symmetry is on
     * Eq. to use: 
     *                           k_fbz = R.k+G = k_full + G
     *   alias |          explanation             |                code variable 
     * ------------------------------------------------------------------------------------------------------       
     *     R   | point symmetry                   |  kset->ctx().unit_cell().symmetry()[ik2isym[ik]].spg_op.R  
     *     G   | reciprocal lattice vector        |  ik2ig[ik]
     *     k   | k vector in Irreducible Wedge    |  kset->kpoints_[ik2ir[ik]]
     *   k_fbz | k vector in First Brillouin zone |  kset_fbz.kpoints_[ik]
     *  k_full | k vector after the rotation      |   
     *--------------------------------------------------------------------------------------------------------
     * The equation to hold is:
     *   kset_fbz.kpoints_[ik] = kset->ctx().unit_cell().symmetry()[ik2isym[ik]].spg_op.R*kset->kpoints_[ik2ir[ik]]+ik2ig[ik]  
     *           k_fbz         =                          R                              .            k            +   G 
     */
    std::vector<int> ik2ir;      
    std::vector<int> ik2isym;
    std::vector<int> ir2ik[this->num_kpoints()];
    std::vector<r3::vector<int>> ik2ig;

    // Reconstructing FBZ
    K_point_set kset_fbz(this->ctx());

    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::unfold_fbz");

    std::vector<r3::vector<double>> k_temp;
    // Apply symmetry to all points of the IBZ. Save indices of ibz, fbz, sym
    for (int ik = 0; ik < this->num_kpoints(); ik++) {
        for (int isym = 0; isym < ctx().unit_cell().symmetry().size(); isym++) {
            auto& R     = ctx().unit_cell().symmetry()[isym].spg_op.R; // point symmetry rotation in crystal coordinates
            auto k_full = r3::dot(this->kpoints_[ik]->vk(), R);
            r3::vector<double> k_fbz;
            r3::vector<double> G; // it must be integer, but here we need double for modf
            // this loop supposes our BZ is in (-0.5,0.5]
            for (int ix : {0, 1, 2}) {
                k_fbz[ix] = modf(k_full[ix]+0.5, &G[ix]);
	    	    k_fbz[ix] -= 0.5;
                if (k_fbz[ix] <= -0.5) {
                    k_fbz[ix] += 1;
                    G[ix] -= 1;
                }
    	    }
            bool found = false;
            for (int ik_ = 0; ik_ < k_temp.size(); ik_++) {
                if ((k_temp[ik_] - k_fbz).length() < 1.e-05) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                k_temp.push_back(k_fbz);
                ik2ir.push_back(ik);
                ir2ik[ik].push_back(ik2ir.size() - 1);
                ik2isym.push_back(isym);
                ik2ig.push_back(
                    r3::vector<int>(-int(G[0]), -int(G[1]),
                                    -int(G[2]))); // note:: the minus is needed because k_fbz = R.k+G = k_full + G
            }
        } // end isym
    }// end ik


   
    //remove additional G vector from k_temp. After this ik2ig is a set of {0,0,0}. We defined the FBZ uniquely.
    for (int ik = 0; ik < k_temp.size(); ik++) {
        if(ik2ig[ik].length() > 0.01){
	    for(int ix : {0,1,2}){
	        k_temp[ik][ix] -= ik2ig[ik][ix];
		    ik2ig[ik][ix]-= ik2ig[ik][ix];
	   	    }
	    }
    }
    


    for(int ik=0; ik<k_temp.size(); ik++){
        kset_fbz.add_kpoint(k_temp[ik], 1.);
    }
    kset_fbz.initialize();


    if (k_temp.size() != this->ctx().cfg().parameters().ngridk().data()[0] *
                             this->ctx().cfg().parameters().ngridk().data()[1] *
                             this->ctx().cfg().parameters().ngridk().data()[2]) {
        std::cout << "Warning!!!!! I could not recover the FBZ!! The program will break at wannier_setup_\n";
    }
    /*
    if (ctx().comm().rank() == 0) {
            std::ofstream kp;
            kp.open("k.txt");
            for (int ik = 0; ik < k_temp.size(); ik++) {
                kp << std::setw(5) << ik;
                kp << std::setw(15) << std::setprecision(10) << std::fixed << k_temp[ik][0]; 
                kp << std::setw(15) << std::setprecision(10) << std::fixed << k_temp[ik][1];
                kp << std::setw(15) << std::setprecision(10) << std::fixed << k_temp[ik][2];
                kp << std::setw(15) << std::setprecision(10) << this->kpoints_[ik2ir[ik]]->vk();
                kp << std::setw(15) << std::setprecision(10) << ctx().unit_cell().symmetry()[ik2isym[ik]].spg_op.R;
                kp << std::setw(15) << std::setprecision(10) << ctx().unit_cell().symmetry()[ik2isym[ik]].spg_op.t;
	        kp << std::setw(15) << std::setprecision(10) << ik2ig[ik] << std::endl;
            }
            kp.close();
    }*/
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::unfold_fbz");

    
    /*
     *  STEP 0.2: Reconstruct wfs and gvectors in the full bz from the one in Irreducible wedge.
     *  Eq to use:
     *        c_{n, R.k}(G) = e^{-iG.tau} e^{-i(R.k).tau} c_{n,k}(R^{-1}G)
     *  or, in real space:
     *        u_{n, R.k}(r) = u_{n, k} (R^{-1}.(r-tau))
     *  tau is the fractional translation associated with R in the point symmetry group.
     */


    int num_bands_tot = this->ctx().num_bands();


    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::unfold_wfs");
    std::complex<double> imtwopi = std::complex<double>(0.,twopi);
    std::complex<double> exp1, exp2;

    //if(ctx().comm().rank()==0)
    //    for (int ik = 0; ik < kset_fbz.num_kpoints(); ik++)
	//        std::cout << ik << " " << kset_fbz.spl_num_kpoints_.local_rank(ik) << std::endl; 
    
    srand(time(NULL));

    for (int ik = 0; ik < kset_fbz.num_kpoints(); ik++) {
    	int& ik_ = ik2ir[ik];
	    int& isym = ik2isym[ik];
        //send wf from rank that have it in IBZ to the one that have it in FBZ
	    int src_rank  = this->spl_num_kpoints_.local_rank(ik_);
	    int dest_rank = kset_fbz.spl_num_kpoints_.local_rank(ik);
	    bool use_mpi;

	    if(src_rank != kset_fbz.ctx().comm_k().rank() && dest_rank != kset_fbz.ctx().comm_k().rank()){
		    continue;
    	}
	    else if(src_rank == kset_fbz.ctx().comm_k().rank() && dest_rank == kset_fbz.ctx().comm_k().rank()){
		    use_mpi = false;
	    }
	    else{
		    use_mpi = true;
	    }

	    //std::cout << "rank: " << ctx().comm_k().rank() << " ik " << ik << " ik_ " << ik_ ;
	    //std::cout << " src_rank " << src_rank << " dest_rank" << dest_rank <<  " mpi: " << use_mpi << std::endl;
	    auto gkvec_IBZ = use_mpi  ? std::make_shared<fft::Gvec>(static_cast<fft::Gvec>(this->get_gkvec(ik_, dest_rank)))
	                              : this->kpoints_[ik_]->gkvec_;
	    sddk::mdarray<std::complex<double>, 2> temp;
	    if(use_mpi && kset_fbz.ctx().comm_k().rank() == dest_rank){
	        temp = sddk::mdarray<std::complex<double>, 2>(gkvec_IBZ->num_gvec(), num_bands_tot);
	    }
        auto& wf_IBZ = use_mpi  ? temp
                     		    : this->kpoints_[ik_]->spinor_wave_functions_->pw_coeffs(wf::spin_index(0));	
    
        int tag = src_rank + kset_fbz.num_kpoints()*dest_rank;
	    if(kset_fbz.ctx().comm_k().rank() == src_rank){
		    if(use_mpi){
			    ctx().comm_k().send(this->kpoints_[ik_]->spinor_wave_functions().at(
                                    sddk::memory_t::host, 0, wf::spin_index(0), wf::band_index(0)),
                                    this->kpoints_[ik_]->gkvec_->num_gvec() * num_bands_tot,
                                    dest_rank, tag);	
		    }
    	}
	    if(kset_fbz.ctx().comm_k().rank() == dest_rank){
		    if(use_mpi){
                ctx().comm_k().recv(&wf_IBZ(0,0),
                                    gkvec_IBZ->num_gvec()*num_bands_tot, src_rank, tag);	
		    }

            kset_fbz.kpoints_[ik]->spinor_wave_functions_ = std::make_unique<wf::Wave_functions<double>>(
                    kset_fbz.kpoints_[ik]->gkvec_, wf::num_mag_dims(0), wf::num_bands(num_bands_tot), ctx_.host_memory_t());
            kset_fbz.kpoints_[ik]->spinor_wave_functions_->zero(sddk::memory_t::host);
	
    		auto& invR = kset_fbz.ctx().unit_cell().symmetry()[isym].spg_op.invR;
            auto& tau = kset_fbz.ctx().unit_cell().symmetry()[isym].spg_op.t;
            r3::vector<int> invRG;
            std::complex<double> exp1 = exp(-imtwopi*r3::dot(kset_fbz.kpoints_[ik]->vk(), tau));    //this is the exponential with the irreducible point
            for (int ig = 0; ig < kset_fbz.kpoints_[ik]->gkvec_->num_gvec(); ig++) {
                //WARNING!! I suppose always that ik2ig[ik]=0 so i don't have it in the equation.
                invRG = r3::dot(kset_fbz.kpoints_[ik]->gkvec_->gvec<sddk::index_domain_t::local>(ig),invR);
				exp2 = exp(-imtwopi*r3::dot(kset_fbz.kpoints_[ik]->gkvec_->gvec<sddk::index_domain_t::local>(ig), tau)); 
                int ig_ = gkvec_IBZ->index_by_gvec(invRG);
                if (ig_ == -1) {
                    std::cout << "WARNING !!!!! point G=" << invRG << " obtained from "<< kset_fbz.kpoints_[ik]->gkvec_->gvec<sddk::index_domain_t::local>(ig) <<
                    " applying the rotation matrix " << invR << " has not been found!! " << std::endl;
                    continue;
                }
                for (int iband = 0; iband < num_bands_tot; iband++) {
                    kset_fbz.kpoints_[ik]->spinor_wave_functions_->pw_coeffs(ig, wf::spin_index(0),
                                                                             wf::band_index(iband)) =
                        exp1*exp2*wf_IBZ(ig_, iband)
                                +std::complex<double>(rand()%1000,rand()%1000)*1.e-08;//needed to not get stuck on local minima. 
                                                                                      //not working with 1.e-09
                }
            }
	    }
    }//end ik loop

    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::unfold_wfs");

   /*
    * STEP 0.3: Check that the wavefunctions diagonalize the Hamiltonian
    * From H|u> = ES|u>
    *           <u|H|u> = E<u|S|u> 
    * but
    *           <u|S|u> = 1
    */
/*
   std::cout << "calculating Hamiltonian...\n"; 
    for (int ikloc = 0; ikloc < kset_fbz.spl_num_kpoints_.local_size(); ikloc++) {
        int ik = kset_fbz.spl_num_kpoints_[ikloc];
        std::cout << "k=" << ik << "\n";
	    auto Hk = H0(*kset_fbz.kpoints_[ik]);
        auto hpsi = wave_function_factory(kset_fbz.ctx(), *kset_fbz.kpoints_[ik], wf::num_bands(num_bands_tot), wf::num_mag_dims(0), false);
        auto spsi = wave_function_factory(kset_fbz.ctx(), *kset_fbz.kpoints_[ik], wf::num_bands(num_bands_tot), wf::num_mag_dims(0), false);

        //std::cout << kset_fbz.kpoints_[ik]->spinor_wave_functions_->ld() << "   " << hpsi->ld() << "   " << spsi->ld() << std::endl;
        //std::cout << kset_fbz.kpoints_[ik]->beta_projectors().num_gkvec_loc() << std::endl; 
	    Hk.apply_h_s<std::complex<double>>(wf::spin_range(0), wf::band_range(0, num_bands_tot), 
			      *kset_fbz.kpoints_[ik]->spinor_wave_functions_, hpsi.get(), spsi.get());

        la::dmatrix<std::complex<double>> UdaggerHU(num_bands_tot, num_bands_tot);



        wf::inner(ctx_.spla_context(), kset_fbz.ctx_.processing_unit_memory_t(), 
                wf::spin_range(0), *kset_fbz.kpoints_[ik]->spinor_wave_functions_, 
                wf::band_range(0, num_bands_tot), *hpsi, wf::band_range(0, num_bands_tot), 
                UdaggerHU, 0, 0);


        la::dmatrix<std::complex<double>> UdaggerSU(num_bands_tot, num_bands_tot);
        wf::inner(ctx_.spla_context(), kset_fbz.ctx_.processing_unit_memory_t(), 
                wf::spin_range(0), *kset_fbz.kpoints_[ik]->spinor_wave_functions_, 
                //wf::band_range(0, num_bands_tot), *hpsi, wf::band_range(0, num_bands_tot), 
                wf::band_range(0, num_bands_tot), *spsi, wf::band_range(0, num_bands_tot), 
                UdaggerSU, 0, 0);


        bool Hdiag = true;
        for(int i=0; i<num_bands_tot; i++){
            for(int j=0; j<num_bands_tot; j++){
                if(i!=j && abs(UdaggerHU(i,j))>1.e-08){
                    Hdiag = false;
                }
            }
        }


        bool Sdiag = true;
        bool Sunit = true;
        for(int i=0; i<num_bands_tot; i++){
            for(int j=0; j<num_bands_tot; j++){
                if(i!=j && abs(UdaggerSU(i,j))>1.e-08){
                     Sdiag = false;
               }
		        else if(i==j && abs(UdaggerSU(i,j)-1.)>1.e-08){
                    Sunit = false;
                }
            }
        }


        if(!Hdiag){
            std::cout << "H at ik= " << ik << " is not diagonal!!\n";
        }

        if(!Sdiag){
            std::cout << "S at ik= " << ik << " is not diagonal!!\n";
        }

        if(!Sunit){
            std::cout << "S at ik= " << ik << " is not 1 on diagonals!!\n";
        }
    
    //    std::ofstream Hkk;
    //    std::string name;
    //    name = "Hk_"+ std::to_string(ik);
    //    Hkk.open(name.c_str());
    //    for(int i=0; i<num_bands_tot; i++){
    //        for(int j=0; j<num_bands_tot; j++){
    //            //Hkk << std::scientific << std::setw(20) << std::setprecision(9) << UdaggerHU(i,j);
    //            Hkk << std::fixed << std::setw(6) << std::setprecision(1) << log10(abs(UdaggerHU(i,j)));
    //        }
    //        Hkk<< std::endl;
    //    }
    //    Hkk.close();
    //    name = "Sk_"+ std::to_string(ik);
    //    Hkk.open(name.c_str());
    //    for(int i=0; i<num_bands_tot; i++){
    //        for(int j=0; j<num_bands_tot; j++){
    //            //Hkk << std::scientific << std::setw(20) << std::setprecision(9) << UdaggerSU(i,j);
    //            Hkk << std::fixed << std::setw(6) << std::setprecision(1) << log10(abs(UdaggerSU(i,j)));
    //        }
    //        Hkk<< std::endl;
    //    }
    //    Hkk.close();

    }
*/	
	
	
    /*
     * TASK 1: Use wannier_setup_ to initialize all the variables needed to calculate Mmn and Amn
     */


    /*
     * STEP 1.1: Allocate memory for all the variables needed in wannier_setup_. Initialize the input variables
     *           with the values they have in SIRIUS
     */

    //scalar variables definition
    size_t length_seedname = 100;    // aux variable for the length of a string
    int32_t num_kpts;                // input
    //int32_t num_bands_tot;           // input
    int32_t num_atoms;               // input
    size_t length_atomic_symbol = 3; // aux, as expected from wannier90 lib
    bool gamma_only;                 // input
    bool spinors;                    // input
    int32_t num_bands;               // output
    int32_t num_wann;                // output
    int32_t nntot;                   // output
    int32_t num_nnmax = 12;          // aux variable for max number of neighbors
                                     // fixed, as in pw2wannier or in wannier90 docs


    //scalar variables initialization
    num_kpts      = kset_fbz.num_kpoints();
    //num_bands_tot = this->get<double>(spl_num_kpoints_[0])->spinor_wave_functions().num_wf();
    num_atoms     = this->ctx().unit_cell().num_atoms();
    gamma_only    = this->ctx().gamma_point();
    spinors       = false; // right now, generate_wave_functions only works with noncolin!
    // WARNING we need to compare with .win file!!!

    //non-scalar variables definition + space allocation
    char seedname[length_seedname];                          // input
    sddk::mdarray<int32_t ,1> mp_grid(3);                    //input
    sddk::mdarray<double, 2> real_lattice(3, 3);             // input BOHR!
    sddk::mdarray<double, 2> recip_lattice(3, 3);            // input BOHR^{-1}!
    sddk::mdarray<double,2> kpt_lattice(3,num_kpts);         //input
    char atomic_symbol[num_atoms][3];                        // input
    sddk::mdarray<double, 2> atoms_cart(3, num_atoms);       // input
    sddk::mdarray<int,2> nnlist(num_kpts,num_nnmax);         // output
    sddk::mdarray<int32_t ,3> nncell(3,num_kpts,num_nnmax);  // output
    sddk::mdarray<double, 2> proj_site(3, num_bands_tot);    // output
    sddk::mdarray<int32_t, 1> proj_l(num_bands_tot);         // output
    sddk::mdarray<int32_t, 1> proj_m(num_bands_tot);         // output
    sddk::mdarray<int32_t, 1> proj_radial(num_bands_tot);    // output
    sddk::mdarray<double, 2> proj_z(3, num_bands_tot);       // output
    sddk::mdarray<double, 2> proj_x(3, num_bands_tot);       // output
    sddk::mdarray<double, 1> proj_zona(num_bands_tot);       // output
    sddk::mdarray<int32_t, 1> exclude_bands(num_bands_tot);  // output
    sddk::mdarray<int32_t, 1> proj_s(num_bands_tot);         // output - optional
    sddk::mdarray<double, 2> proj_s_qaxis(3, num_bands_tot); // output - optional

    //non-scalar variables initialization
    std::string aux = "silicon";
    strcpy(seedname, aux.c_str());
    length_seedname = aux.length();

    for (int ivec = 0; ivec < 3; ivec++) {
        for (int icoor = 0; icoor < 3; icoor++) {
            real_lattice(ivec, icoor)  = ctx().unit_cell().lattice_vectors()(icoor, ivec) * bohr_radius;
            recip_lattice(ivec, icoor) = ctx().unit_cell().reciprocal_lattice_vectors()(icoor, ivec) / bohr_radius;
        }
    }
    
    for (int ix : {0, 1, 2}) {
        for (int ik = 0; ik < num_kpts; ik++) {
            auto& kk = kset_fbz.kpoints_[ik]->vk_;
            kpt_lattice(ix, ik) = kk[ix];
        }
    }

    for (int iat = 0; iat < num_atoms; iat++) {
        std::fill(atomic_symbol[iat], atomic_symbol[iat] + 3, ' ');
        std::strcpy(atomic_symbol[iat], this->ctx().unit_cell().atom(iat).type().label().c_str());
        // position is saved in fractional coordinates, we need cartesian for wannier_setup_
        auto frac_coord = this->unit_cell().atom(iat).position();
        auto cart_coord = this->ctx().unit_cell().get_cartesian_coordinates(frac_coord);
        for (int icoor = 0; icoor < 3; icoor++) {
            atoms_cart(icoor, iat) = cart_coord[icoor] * bohr_radius;
        }
    }

    /*
     * STEP 1.2: Call wannier_setup_ from wannier library. This calculates two important arrays:
     * nnlist(ik,ib) is the index of the neighbor ib of the vector at index ik
     * nncell(ix,ik,ib) is the ix-th coordinate of the G vector that brings back the vector defined in nnlist(ik,ib)
     * nntot is the total number of neighbors. 
     * to the first Brillouin zone. Eq. to hold:
     *             kpoints_[nnlist(ik,ib)] = kpoints_[ik] + (neighbor b) - nncell(.,ik,ib)
     */
    std::cout << "I am process " << ctx().comm().rank() << " and I go inside the wannier_setup\n";
    
    
    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::wannier_setup");
    if (ctx().comm().rank() == 0) {
        std::cout << "starting wannier_setup_\n";
        wannier_setup_(seedname,
                       this->ctx().cfg().parameters().ngridk().data(), // input
                       &num_kpts,                                      // input
                       real_lattice.at(sddk::memory_t::host),          // input
                       recip_lattice.at(sddk::memory_t::host),         // input
                       kpt_lattice.at(sddk::memory_t::host),           // input
                       &num_bands_tot,                                 // input
                       &num_atoms,                                     // input
                       atomic_symbol,                                  // input
                       atoms_cart.at(sddk::memory_t::host),            // input
                       &gamma_only,                                    // input
                       &spinors,                                       // input
                       &nntot,                                         // output
                       nnlist.at(sddk::memory_t::host),                // output
                       nncell.at(sddk::memory_t::host),                // output
                       &num_bands,                                     // output
                       &num_wann,                                      // output
                       proj_site.at(sddk::memory_t::host),             // output
                       proj_l.at(sddk::memory_t::host),                // output
                       proj_m.at(sddk::memory_t::host),                // output
                       proj_radial.at(sddk::memory_t::host),           // output
                       proj_z.at(sddk::memory_t::host),                // output
                       proj_x.at(sddk::memory_t::host),                // output
                       proj_zona.at(sddk::memory_t::host),             // output
                       exclude_bands.at(sddk::memory_t::host),         // output
                       proj_s.at(sddk::memory_t::host),                // output
                       proj_s_qaxis.at(sddk::memory_t::host),          // output
                       length_seedname,                                // aux-length of a string
                       length_atomic_symbol);                          // aux-length of a string


                       //std::cout << "center_w:: " << proj_site << std::endl;
    }
    ctx().comm_k().bcast(&nntot, 1, 0);
    ctx().comm_k().bcast(nnlist.at(sddk::memory_t::host), num_kpts * num_nnmax, 0);
    ctx().comm_k().bcast(nncell.at(sddk::memory_t::host), 3 * num_kpts * num_nnmax, 0);
    ctx().comm_k().bcast(&num_bands, 1, 0);
    ctx().comm_k().bcast(&num_wann, 1, 0);
    ctx().comm_k().bcast(exclude_bands.at(sddk::memory_t::host), num_bands_tot, 0);

    std::cout << "\n\n\n\n\n\n";
    std::cout << "wannier_setup succeeded. rank " << ctx().comm().rank() << "\n";



    if(kset_fbz.spl_num_kpoints_.local_size()!=0){//this is needed because if not the memory could go out of bound
        if (num_bands != kset_fbz.kpoints_[kset_fbz.spl_num_kpoints_[0]]->spinor_wave_functions().num_wf()) {
            std::cout << "\n\n\n\nBAD!!! num_bands from w90 different than number of wfs in SIRIUS!! The program will "
                         "break!!\n\n\n\n"
                      << std::endl;
        }
    }
    
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::wannier_setup");


    num_wann = ctx_.unit_cell().num_ps_atomic_wf().first;

    /*
     * TASK 2: Calculate the matrix elements
     *                    A_{mn}(k)   = <u_{mk}|w_{nk}>
     *                    M_{mn}(k,b) = <u_{mk}|u_{n,k+b}>
     */


    /*
     * STEP 2.1: Calculate A.
     *    Amn(k) = <psi_{mk} | S | w_{nk}> = conj(<w_{nk} | S | psi_{mk}>)      
     *           Here we calculate the projectors over all the atomic orbitals in the pseudopotential
    */

    sddk::mdarray<std::complex<double>, 3> A(num_bands, num_wann, kset_fbz.num_kpoints()); 
    A.zero();
    la::dmatrix<std::complex<double>> Ak(num_bands, num_wann);     //matrix at the actual k point 
    Ak.zero();

    std::vector<int> atoms(ctx_.unit_cell().num_atoms());
    std::iota(atoms.begin(), atoms.end(), 0); // we need to understand which orbitals to pick up, I am using every here
    int num_atomic_wf  = kset_fbz.ctx().unit_cell().num_ps_atomic_wf().first;

    std::unique_ptr<wf::Wave_functions<double>> Swf_k;
    //sddk::mdarray<std::complex<double>, 3> psidotpsi(num_bands, num_bands, num_kpts); // sirius2wannier
    //sddk::mdarray<std::complex<double>, 3> atdotat(num_wann, num_wann, num_kpts);     // sirius2wannier
    //psidotpsi.zero();
    //atdotat.zero();
    std::cout << "Calculating Amn...\n";
    auto mem = kset_fbz.ctx_.processing_unit_memory_t();
 


    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::calculate_Amn");
 
 
    for (int ikloc = 0; ikloc < kset_fbz.spl_num_kpoints_.local_size(); ikloc++) {
        int ik = kset_fbz.spl_num_kpoints_[ikloc];
        
        //calculate atomic orbitals + orthogonalization
        auto q_op = (kset_fbz.kpoints_[ik]->unit_cell_.augment())
                            ? std::make_unique<Q_operator<double>>(kset_fbz.kpoints_[ik]->ctx_)
                            : nullptr;
        kset_fbz.kpoints_[ik]->beta_projectors().prepare();
        Swf_k = std::make_unique<wf::Wave_functions<double>>(kset_fbz.kpoints_[ik]->gkvec_, wf::num_mag_dims(0),
                                                                 wf::num_bands(num_bands), ctx_.host_memory_t());
        apply_S_operator<double, std::complex<double>>(
                mem, wf::spin_range(0), wf::band_range(0, num_bands), kset_fbz.kpoints_[ik]->beta_projectors(),
                (kset_fbz.kpoints_[ik]->spinor_wave_functions()), q_op.get(), *Swf_k);
        
        // allocate space for atomic wfs, same that for hubbard
        // we should move the following 2 lines in initialize() function of kpoint clas
        kset_fbz.kpoints_[ik]->atomic_wave_functions_ = std::make_unique<wf::Wave_functions<double>>(
                kset_fbz.kpoints_[ik]->gkvec_, wf::num_mag_dims(0), wf::num_bands(num_atomic_wf), ctx_.host_memory_t());
        kset_fbz.kpoints_[ik]->atomic_wave_functions_S_ = std::make_unique<wf::Wave_functions<double>>(
                kset_fbz.kpoints_[ik]->gkvec_, wf::num_mag_dims(0), wf::num_bands(num_atomic_wf), ctx_.host_memory_t());
        //defining pw expansion of atomic wave functions, defined in pseudopotential file
        kset_fbz.kpoints_[ik]->generate_atomic_wave_functions(
                atoms, [&](int iat) { return &kset_fbz.ctx_.unit_cell().atom_type(iat).indexb_wfs(); },
                kset_fbz.ctx_.ps_atomic_wf_ri(), *(kset_fbz.kpoints_[ik]->atomic_wave_functions_));

        /*
         *  Pick up only needed atomic functions, with their proper linear combinations
        //define index in atomic_wave_functions for atom iat
        std::vector<int> offset(kset_fbz.ctx().unit_cell().num_atoms());
        offset[0]=0;
        for(int i=1; i<kset_fbz.ctx().unit_cell().num_atoms(); i++){
            offset[i] = offset[i-1] + kset_fbz.ctx_.unit_cell().atom_type(i-1).indexb_wfs()->size();
        }
       //reconstruct map i-th wann func -> atom, l, m
        std::vector<std::array<int,3>> atoms_info(num_wann);
        
        auto needed_atomic_wf = std::make_unique<wf::Wave_functions<double>>(
                               kset_fbz.kpoints_[ik]->gkvec_, wf::num_mag_dims(0), wf::num_bands(num_wann), ctx_.host_memory_t());

        for(int iw=0; iw<num_wann; iw++)
        {
            int iat__=-1;
            for(int iat=0; iat<kset_fbz.ctx().unit_cell().num_atoms(); iat++){
                //calculate norm of center_w - atomic_position to decide which atom is the correct one
                auto& frac = this->unit_cell().atom(iat).position();
                r3::vector<double> diff = {center_w(0,iw)-frac[0], center_w(1,iw)-frac[1], center_w(2,iw)-frac[2] }
                if(diff.length() < 1.e-08){
                    iat__ = iat;
                    break;
                }
            }
            if(iat__==-1){
                std::cout <<"\n\n\nWARNING!! Could not find center_w: " << center_w(0,iw) << "  " << center_w(1,iw);
                std::cout <<"  " << center_w(2,iw) << std::endl << std::endl;
            }

            atoms_info[iw][0] = offset[iat__];
            atoms_info[iw][1] = proj_l(iw);
            atoms_info[iw][2] = proj_m(iw);
        }//end definition of atoms_info
        */


        // ORTHOGONALIZING -CHECK HUBBARD FUNCTION
        apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, num_wann),
                                                           kset_fbz.kpoints_[ik]->beta_projectors(),
                                                           *(kset_fbz.kpoints_[ik]->atomic_wave_functions_), q_op.get(),
                                                           *(kset_fbz.kpoints_[ik]->atomic_wave_functions_S_));

        int BS = kset_fbz.ctx_.cyclic_block_size();
        la::dmatrix<std::complex<double>> ovlp(num_wann, num_wann, kset_fbz.ctx_.blacs_grid(), BS, BS);
        wf::inner(kset_fbz.ctx_.spla_context(), mem, wf::spin_range(0),
                  *(kset_fbz.kpoints_[ik]->atomic_wave_functions_), wf::band_range(0, num_wann),
                  *(kset_fbz.kpoints_[ik]->atomic_wave_functions_S_), wf::band_range(0, num_wann), ovlp, 0, 0);

        auto B = std::get<0>(inverse_sqrt(ovlp, num_wann));
        wf::transform(kset_fbz.ctx_.spla_context(), mem, *B, 0, 0, 1.0,
                      *(kset_fbz.kpoints_[ik]->atomic_wave_functions_), wf::spin_index(0),
                       wf::band_range(0, num_wann), 0.0, *(kset_fbz.kpoints_[ik]->atomic_wave_functions_S_),
                       wf::spin_index(0), wf::band_range(0, num_wann));
        wf::copy(mem, *(kset_fbz.kpoints_[ik]->atomic_wave_functions_S_), wf::spin_index(0),
                 wf::band_range(0, num_wann), *(kset_fbz.kpoints_[ik]->atomic_wave_functions_), wf::spin_index(0),
                 wf::band_range(0, num_wann));
        apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, num_wann),
                                                       kset_fbz.kpoints_[ik]->beta_projectors(),
                                                       *(kset_fbz.kpoints_[ik]->atomic_wave_functions_), q_op.get(),
                                                       *(kset_fbz.kpoints_[ik]->atomic_wave_functions_S_));
        kset_fbz.kpoints_[ik]->beta_projectors().dismiss();
        //END of the orthogonalization. 
            
            
        wf::inner(ctx_.spla_context(), mem, wf::spin_range(0), kset_fbz.kpoints_[ik]->spinor_wave_functions(),
                  wf::band_range(0, num_bands), *(kset_fbz.kpoints_[ik]->atomic_wave_functions_S_),
                  wf::band_range(0, num_wann), Ak, 0, 0);
        // already in the correct way, we just copy in the bigger array. (alternative:: create dmatrix with an index
        // as multiindex to avoid copies) note!! we need +1 to copy the last element
        std::copy(Ak.at(sddk::memory_t::host, 0, 0), Ak.at(sddk::memory_t::host, num_bands - 1, num_wann - 1) + 1,
                  A.at(sddk::memory_t::host, 0, 0, ik));


        std::cout << "Calculated Amn in rank " << ctx().comm().rank() << " ik: " << ik << std::endl;
    } // end ik loop for Amn
   

    for(int ik=0; ik<num_kpts; ik++){
	int local_rank = kset_fbz.spl_num_kpoints_.local_rank(ik);
        ctx().comm_k().bcast(A.at(sddk::memory_t::host,0,0,ik), num_bands*num_wann, local_rank);
	ctx().comm_k().barrier();
    }
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::calculate_Amn");




    if (ctx().comm().rank() == 0) {
        write_Amn(A);
    }
    

    /*
     * STEP 2.2: Send-receive messages from k+b to k
     *           The "owner" of Mmn(k,b) will be the task containing k, so we send k+b to k
     */

	std::vector<std::shared_ptr<fft::Gvec>> gkpb_mpi;//auto gkvec_kpb = use_mpi  ? std::make_shared<fft::Gvec>(static_cast<fft::Gvec>(kset_fbz.get_gkvec(ikpb, dest_rank)))
	std::vector<sddk::mdarray<std::complex<double>, 2>> wkpb_mpi;
    std::vector<int> ikpb2ik_(num_kpts,-1);
    
    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::send_k+b");
    int index=-1; //to keep track of the index to use
    for (int ik = 0; ik < num_kpts; ik++) {
        for (int ib = 0; ib < nntot; ib++) {
           	int ikpb = nnlist(ik,ib)-1;
            int src_rank = kset_fbz.spl_num_kpoints_.local_rank(ikpb);
            int dest_rank = kset_fbz.spl_num_kpoints_.local_rank(ik);
            //std::cout << "rank="<< ctx().comm().rank() << " ik=" << ik << " ikpb=" << ikpb << " src_rank=" << src_rank << " dest_rank=" << dest_rank << std::endl;
	    	bool use_mpi = ( src_rank==kset_fbz.ctx().comm_k().rank() )^( dest_rank==kset_fbz.ctx().comm_k().rank());//xor operations: it is true if 01 or 10
            if(!use_mpi){
                continue;
            }
            
            bool found;
            int tag = src_rank + kset_fbz.num_kpoints()*kset_fbz.num_kpoints()*dest_rank;
            if(dest_rank == kset_fbz.ctx().comm_k().rank()){
                found= ikpb2ik_[ikpb] != -1;//std::find(ikpb2ik_.begin(), ikpb2ik_.end(), ikpb) != ikpb2ik_.end(); //false if ikpb is not in ikpb2ik_ 
				kset_fbz.ctx().comm_k().send(&found,
                                    		 1,
                                    		src_rank, tag);
            }else{
                kset_fbz.ctx().comm_k().recv(&found,
                                	         1,
                                             dest_rank, tag);	
            }
            if(found){
                continue;
            }
            //std::cout << "rank="<< ctx().comm().rank() << " ik=" << ik << " ikpb=" << ikpb << " src_rank=" << src_rank << " dest_rank=" << dest_rank << "TOBEDONE" << std::endl;
            tag = src_rank + kset_fbz.num_kpoints()*dest_rank;
            if(kset_fbz.ctx().comm_k().rank()==src_rank){
                auto trash = kset_fbz.get_gkvec(ikpb, dest_rank);
                ctx().comm_k().send(kset_fbz.kpoints_[ikpb]->spinor_wave_functions().at(
                                    		    sddk::memory_t::host, 0, wf::spin_index(0), wf::band_index(0)),
                                    		    kset_fbz.kpoints_[ikpb]->gkvec_->num_gvec() * num_bands_tot,
                                    		    dest_rank, tag);
            }
            else{
                index++;
                gkpb_mpi.emplace_back();
                gkpb_mpi[index] = std::make_shared<fft::Gvec>(static_cast<fft::Gvec>(kset_fbz.get_gkvec(ikpb, dest_rank)));

                ikpb2ik_[ikpb]=index;
                wkpb_mpi.push_back(sddk::mdarray<std::complex<double>, 2>(gkpb_mpi[index]->num_gvec(), num_bands_tot));
                ctx().comm_k().recv(& wkpb_mpi[index](0,0),
                                	gkpb_mpi[index]->num_gvec()*num_bands_tot, src_rank, tag);	
            }
            //std::cout << "rank="<< ctx().comm().rank() << " ik=" << ik << " ikpb=" << ikpb << " src_rank=" << src_rank << " dest_rank=" << dest_rank << "DONE"<< std::endl;

        }//end ib
    }//end ik
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::send_k+b");


    /*
     * STEP 2.3: Calculate M. 
     * Strategy: for each couple k,k+b we reshuffle the indices c_{k+b}(G) such that
     * they are aligned with the one at c_{k}(G). 
     * Some care must be taken when nncell(_,ik,ib) != {0,0,0}. 
     * In this case     kpoints[nnlist(ik,ib)] = kpoints[k] + b - nncell(_,ik,ib) 
     * and
     *              c_{n,k+b} = c_{n, nnlist(ik,ib)} (G+nncell(_,ik,ib))
     */
    /***********************************************************************************
     *      Mmn(k,b) = <u_{mk} | S | u_{n,k+b}> = conj(<u_{n,k+b} | S | u_{m,k}>)      *
     ***********************************************************************************/

    sddk::mdarray<std::complex<double>, 4> M(num_bands, num_bands, nntot, kset_fbz.num_kpoints()); 
    M.zero();
    la::dmatrix<std::complex<double>> Mbk(num_bands, num_bands);
    Mbk.zero(); 

    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::calculate_Mmn");
    for (int ikloc = 0; ikloc < kset_fbz.spl_num_kpoints_.local_size(); ikloc++) {
        int ik = kset_fbz.spl_num_kpoints_[ikloc];
       	auto q_op = (kset_fbz.kpoints_[ik]->unit_cell_.augment())
       	            ? std::make_unique<Q_operator<double>>(kset_fbz.kpoints_[ik]->ctx_)
                    : nullptr;
        kset_fbz.kpoints_[ik]->beta_projectors().prepare();
        Swf_k = std::make_unique<wf::Wave_functions<double>>(kset_fbz.kpoints_[ik]->gkvec_, wf::num_mag_dims(0),
                                                             wf::num_bands(num_bands), ctx_.host_memory_t());
        apply_S_operator<double, std::complex<double>>(
                  	mem, wf::spin_range(0), wf::band_range(0, num_bands), kset_fbz.kpoints_[ik]->beta_projectors(),
                	(kset_fbz.kpoints_[ik]->spinor_wave_functions()), q_op.get(), *Swf_k);
	    
        std::cout << "Calculating Mmn. ik = " << ik << std::endl;
        for (int ib = 0; ib < nntot; ib++) {
           	int ikpb = nnlist(ik,ib)-1;
            bool use_mpi = ( kset_fbz.ctx().comm_k().rank() != kset_fbz.spl_num_kpoints_.local_rank(ikpb) );

	    	auto& gkvec_kpb = use_mpi  ? gkpb_mpi[ikpb2ik_[ikpb]]//std::make_shared<fft::Gvec>(static_cast<fft::Gvec>(kset_fbz.get_gkvec(ikpb, dest_rank)))
	                             	  : kset_fbz.kpoints_[ikpb]->gkvec_;
	    	//sddk::mdarray<std::complex<double>, 2> temp;
	    	//if(use_mpi && kset_fbz.ctx().comm_k().rank() == dest_rank){
	        //	temp = sddk::mdarray<std::complex<double>, 2>(gkvec_kpb->num_gvec(), num_bands_tot);
	    	//}
        	auto& wf_kpb = use_mpi  ? wkpb_mpi[ikpb2ik_[ikpb]]//temp
                     		    	: kset_fbz.kpoints_[ikpb]->spinor_wave_functions_->pw_coeffs(wf::spin_index(0));	
    /*
        	int tag = src_rank + kset_fbz.num_kpoints()*dest_rank;
	    	if(kset_fbz.ctx().comm_k().rank() == src_rank){
			    if(use_mpi){
				    ctx().comm_k().send(kset_fbz.kpoints_[ikpb]->spinor_wave_functions().at(
                                    		    sddk::memory_t::host, 0, wf::spin_index(0), wf::band_index(0)),
                                    		    kset_fbz.kpoints_[ikpb]->gkvec_->num_gvec() * num_bands_tot,
                                    		    dest_rank, tag);	
		    	}
    		}
	    	if(kset_fbz.ctx().comm_k().rank() == dest_rank){
			    if(use_mpi){
                		ctx().comm_k().recv(&wf_kpb(0,0),
                                    	gkvec_kpb->num_gvec()*num_bands_tot, src_rank, tag);	
		    	}
*/
                std::unique_ptr<wf::Wave_functions<double>> aux_psi_kpb;
                aux_psi_kpb = std::make_unique<wf::Wave_functions<double>>(
                        kset_fbz.kpoints_[ik]->gkvec_, wf::num_mag_dims(0), wf::num_bands(num_bands), ctx_.host_memory_t());
                		aux_psi_kpb->zero(sddk::memory_t::host);
                r3::vector<int> G;
                for (int ig = 0; ig < kset_fbz.kpoints_[ik]->gkvec_.get()->num_gvec(); ig++) {
                    // compute the total vector to use to get the index in kpb
                    G = kset_fbz.kpoints_[ik]->gkvec_.get()->gvec<sddk::index_domain_t::local>(ig);
                    G += r3::vector<int>(nncell(0, ik, ib), nncell(1, ik, ib),
                                   		nncell(2, ik, ib)); 
                    int ig_ = gkvec_kpb->index_by_gvec(G); // kpoints_[ikpb]->gkvec_->index_by_gvec(G);
                    if (ig_ == -1) {
                    	continue;
                    		}
                    		for (int iband = 0; iband < num_bands; iband++) {
                        		aux_psi_kpb->pw_coeffs(ig, wf::spin_index(0), wf::band_index(iband)) =
                            			wf_kpb(ig_, iband);
                    		}

                	} // end ig
                
			wf::inner(ctx_.spla_context(), mem, wf::spin_range(0), *aux_psi_kpb, wf::band_range(0, num_bands),
                          *Swf_k, wf::band_range(0, num_bands), Mbk, 0, 0);
                	for (int n = 0; n < num_bands; n++) {
                    		for (int m = 0; m < num_bands; m++) {
                        		M(m, n, ib, ik) = std::conj(Mbk(n, m));
                    		}
                	}
		//}
            } // end ib
        }     // end ik

    std::cout << "Mmn calculated.\n";
    std::cout << "starting broadcast...\n";
    for(int ik=0; ik<num_kpts; ik++){
    	int local_rank = kset_fbz.spl_num_kpoints_.local_rank(ik);
		kset_fbz.ctx().comm_k().bcast(M.at(sddk::memory_t::host,0,0,0,ik), num_bands*num_bands*nntot, local_rank);
		ctx().comm_k().barrier();
    }
    
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::calculate_Mmn");

    std::cout << "writing Mmn..\n";
    if (ctx().comm().rank() == 0) {
        write_Mmn(M, nnlist, nncell);
    }



    //Initialize eigval with the value of the energy dispersion
    sddk::mdarray<double,2>               eigval(num_bands, num_kpts);                     //input
    for(int ik = 0; ik < num_kpts; ik++) {
	int ik_ = ik2ir[ik];
	int local_rank = this->spl_num_kpoints_.local_rank(ik_);
        if(kset_fbz.ctx().comm_k().rank()==local_rank){
		for(int iband=0; iband<num_bands; iband++){
                	eigval(iband, ik) = this->kpoints_[ik_]->band_energy(iband, 0)*ha2ev;//sirius saves energy in
					                                                     //Hartree, we need it in eV
			}
		}
		kset_fbz.ctx().comm_k().bcast(eigval.at(sddk::memory_t::host,0,ik), num_bands, local_rank);
    }  
    	
    

    if(kset_fbz.ctx().comm_k().rank() == 0){
        // compute wannier orbitals
    	// define additional arguments
    	//sddk::mdarray<double,2>               eigval(num_bands, num_kpts);                     //input
    	sddk::mdarray<std::complex<double>,3> U_matrix(num_wann, num_wann, num_kpts);          //output
    	sddk::mdarray<std::complex<double>,3> U_dis(num_bands, num_wann, num_kpts);            //output
    	sddk::mdarray<bool,2>                 lwindow(num_bands, num_kpts);                    //output
    	sddk::mdarray<double,2>               wannier_centres(3, num_wann);                    //output
    	sddk::mdarray<double,1>               wannier_spreads(num_wann);                       //output
    	sddk::mdarray<double,1>               spread_loc(3);                                   //output-op


    	write_eig(eigval);
    
    	U_matrix.zero();
    	U_dis.zero();
    	lwindow.zero();
    	wannier_centres.zero();
    	wannier_spreads.zero();
    	spread_loc.zero();

    PROFILE_START("sirius::K_point_set::generate_w90_coeffs::wannier_run");

    	std::cout << "Starting wannier_run..." << std::endl;
    	wannier_run_(seedname,
                     this->ctx().cfg().parameters().ngridk().data(),
                        &num_kpts,
                        real_lattice.at(sddk::memory_t::host),
                        recip_lattice.at(sddk::memory_t::host),
                        kpt_lattice.at(sddk::memory_t::host),
                        &num_bands,
                        &num_wann,
                        &nntot,
                        &num_atoms,
                        atomic_symbol,
                        atoms_cart.at(sddk::memory_t::host),
                        &gamma_only,
                        M.at(sddk::memory_t::host),
                        A.at(sddk::memory_t::host),
                        eigval.at(sddk::memory_t::host),
                        U_matrix.at(sddk::memory_t::host),
                        U_dis.at(sddk::memory_t::host),
                        lwindow.at(sddk::memory_t::host),
                        wannier_centres.at(sddk::memory_t::host),
                        wannier_spreads.at(sddk::memory_t::host),
                        spread_loc.at(sddk::memory_t::host),
                        length_seedname,
                        length_atomic_symbol);
    std::cout << "Wannier_run succeeded. " << std::endl;
    }
    PROFILE_STOP("sirius::K_point_set::generate_w90_coeffs::wannier_run");
}

} // namespace sirius
  // Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
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


