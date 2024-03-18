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

namespace sirius {

template <typename T, sync_band_t what>
void
K_point_set::sync_band()
{
    PROFILE("sirius::K_point_set::sync_band");

    mdarray<double, 3> data({ctx_.num_bands(), ctx_.num_spinors(), num_kpoints()}, get_memory_pool(memory_t::host),
                            mdarray_label("K_point_set::sync_band.data"));
    data.zero();

    int nb = ctx_.num_bands() * ctx_.num_spinors();
    #pragma omp parallel
    for (auto it : spl_num_kpoints_) {
        auto kp = this->get<T>(it.i);
        switch (what) {
            case sync_band_t::energy: {
                std::copy(&kp->band_energies_(0, 0), &kp->band_energies_(0, 0) + nb, &data(0, 0, it.i));
                break;
            }
            case sync_band_t::occupancy: {
                std::copy(&kp->band_occupancies_(0, 0), &kp->band_occupancies_(0, 0) + nb, &data(0, 0, it.i));
                break;
            }
        }
    }

    comm().allreduce(data.at(memory_t::host), static_cast<int>(data.size()));

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

template void
K_point_set::sync_band<double, sync_band_t::energy>();

template void
K_point_set::sync_band<double, sync_band_t::occupancy>();

#if defined(SIRIUS_USE_FP32)
template void
K_point_set::sync_band<float, sync_band_t::energy>();

template void
K_point_set::sync_band<float, sync_band_t::occupancy>();
#endif

void
K_point_set::create_k_mesh(r3::vector<int> k_grid__, r3::vector<int> k_shift__, int use_symmetry__)
{
    PROFILE("sirius::K_point_set::create_k_mesh");

    int nk;
    mdarray<double, 2> kp;
    std::vector<double> wk;
    if (use_symmetry__) {
        auto result = get_irreducible_reciprocal_mesh(ctx_.unit_cell().symmetry(), k_grid__, k_shift__);
        nk          = std::get<0>(result);
        wk          = std::get<1>(result);
        auto tmp    = std::get<2>(result);
        kp          = mdarray<double, 2>({3, nk});
        for (int i = 0; i < nk; i++) {
            for (int x : {0, 1, 2}) {
                kp(x, i) = tmp[i][x];
            }
        }
    } else {
        nk = k_grid__[0] * k_grid__[1] * k_grid__[2];
        wk = std::vector<double>(nk, 1.0 / nk);
        kp = mdarray<double, 2>({3, nk});

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
    if (comm().size() > this->num_kpoints()) {
        std::stringstream s;
        s << "Number of MPI ranks for k-points is larger than the number of k-points; parallelization is not optimal"
          << std::endl
          << "  number of k-points                   : " << this->num_kpoints() << std::endl
          << "  k-point communicator size            : " << comm().size() << std::endl
          << "  optimal size of k-point communicator : ";
        for (int i = 1; i <= this->num_kpoints(); i++) {
            if (this->num_kpoints() % i == 0) {
                s << i << " ";
            }
        }
        s << std::endl;
        s << "Check if you need to set control.mpi_grid_dims for band parallelization";
        if (ctx_.comm().rank() == 0) {
            RTE_WARNING(s);
        }
    }
    /* distribute k-points along the 1-st dimension of the MPI grid */
    if (counts.empty()) {
        splindex_block<> spl_tmp(num_kpoints(), n_blocks(comm().size()), block_id(comm().rank()));
        spl_num_kpoints_ = splindex_chunk<kp_index_t>(num_kpoints(), n_blocks(comm().size()), block_id(comm().rank()),
                                                      spl_tmp.counts());
    } else {
        spl_num_kpoints_ =
                splindex_chunk<kp_index_t>(num_kpoints(), n_blocks(comm().size()), block_id(comm().rank()), counts);
    }

    for (auto it : spl_num_kpoints_) {
        kpoints_[it.i]->initialize();
#if defined(SIRIUS_USE_FP32)
        kpoints_float_[it.i]->initialize();
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
            RTE_THROW(s);
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
    // Newton finds the minimum, not necessarily N(mu) == ne, tolerate up to `tol_ne` difference in number of electrons
    // if |N(mu_0) -ne| > tol_ne an error is thrown.
    const double tol_ne = 1e-2;

    struct
    {
        double mu;              // chemical potential
        int iter{0};            // newton information
        std::vector<double> ys; // newton history
    } res;

    double mu = mu0;
    double alpha{1.0}; // Newton damping
    int iter{0};

    if (std::abs(N(mu) - ne) < tol) {
        res.mu   = mu;
        res.iter = iter;
        res.ys   = {};
        return res;
    }

    while (true) {
        // compute
        double Nf   = N(mu);
        double dNf  = dN(mu);
        double ddNf = ddN(mu);
        /* minimize (N(mu) - ne)^2  */
        // double F = (Nf - ne) * (Nf - ne);
        double dF   = 2 * (Nf - ne) * dNf;
        double ddF  = 2 * dNf * dNf + 2 * (Nf - ne) * ddNf;
        double step = alpha * dF / std::abs(ddF);
        mu          = mu - step;

        res.ys.push_back(mu);

        if (std::abs(ddF) < 1e-30) {
            std::stringstream s;
            s << "Newton minimization (Fermi energy) failed because 2nd derivative too close to zero!"
              << std::setprecision(8) << std::abs(Nf - ne) << "\n";
            RTE_THROW(s);
        }

        if (std::abs(step) < tol || std::abs(Nf - ne) < tol) {
            if (std::abs(Nf - ne) > tol_ne) {
                std::stringstream s;
                s << "Newton minimization (Fermi energy) got stuck in a local minimum. Fallback to bisection search."
                  << "\n";
                RTE_THROW(s);
            }

            res.iter = iter;
            res.mu   = mu;
            return res;
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
        for (auto it : spl_num_kpoints_) {
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                #pragma omp parallel for
                for (int j = 0; j < ctx_.num_bands(); j++) {
                    this->get<T>(it.i)->band_occupancy(j, ispn, ctx_.max_occupancy());
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

    #pragma omp parallel for reduction(min:emin) reduction(max:emax)
    for (auto it : spl_num_kpoints_) {
        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
            emin = std::min(emin, this->get<T>(it.i)->band_energy(0, ispn));
            emax = std::max(emax, this->get<T>(it.i)->band_energy(ctx_.num_bands() - 1, ispn));
        }
    }
    comm().allreduce<double, mpi::op_t::min>(&emin, 1);
    comm().allreduce<double, mpi::op_t::max>(&emax, 1);

    splindex_block<> splb(ctx_.num_bands(), n_blocks(ctx_.comm_band().size()), block_id(ctx_.comm_band().rank()));

    /* computes N(ef; f) = \sum_{i,k} f(ef - e_{k,i}) */
    auto compute_ne = [&](double ef, auto&& f) {
        double ne{0};
        for (auto it : spl_num_kpoints_) {
            double tmp{0};
            #pragma omp parallel reduction(+ : tmp)
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                #pragma omp for
                for (int j = 0; j < splb.local_size(); j++) {
                    tmp += f(ef - this->get<T>(it.i)->band_energy(splb.global_index(j), ispn)) * ctx_.max_occupancy();
                }
            }
            ne += tmp * kpoints_[it.i]->weight();
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
            auto df         = smearing::delta(ctx_.smearing(), ctx_.smearing_width());
            auto ddf        = smearing::dxdelta(ctx_.smearing(), ctx_.smearing_width());
            auto N          = [&](double mu) { return compute_ne(mu, f); };
            auto dN         = [&](double mu) { return compute_ne(mu, df); };
            auto ddN        = [&](double mu) { return compute_ne(mu, ddf); };
            auto res_newton = newton_minimization_chemical_potential(N, dN, ddN, energy_fermi_, ne_target, tol, 1000);
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

    for (auto it : spl_num_kpoints_) {
        for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
            #pragma omp parallel for
            for (int j = 0; j < ctx_.num_bands(); j++) {
                auto o = f(energy_fermi_ - this->get<T>(it.i)->band_energy(j, ispn)) * ctx_.max_occupancy();
                this->get<T>(it.i)->band_occupancy(j, ispn, o);
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

template void
K_point_set::find_band_occupancies<double>();
#if defined(SIRIUS_USE_FP32)
template void
K_point_set::find_band_occupancies<float>();
#endif

template <typename T>
double
K_point_set::valence_eval_sum() const
{
    double eval_sum{0};

    splindex_block<> splb(ctx_.num_bands(), n_blocks(ctx_.comm_band().size()), block_id(ctx_.comm_band().rank()));

    for (auto it : spl_num_kpoints_) {
        auto const& kp = this->get<T>(it.i);
        double tmp{0};
        #pragma omp parallel for reduction(+:tmp)
        for (int j = 0; j < splb.local_size(); j++) {
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                tmp += kp->band_energy(splb.global_index(j), ispn) * kp->band_occupancy(splb.global_index(j), ispn);
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
#if defined(SIRIUS_USE_FP32)
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

    splindex_block<> splb(ctx_.num_bands(), n_blocks(ctx_.comm_band().size()), block_id(ctx_.comm_band().rank()));

    for (auto it : spl_num_kpoints_) {
        auto const& kp = this->get<T>(it.i);
        double tmp{0};
        #pragma omp parallel for reduction(+:tmp)
        for (int j = 0; j < splb.local_size(); j++) {
            for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
                tmp += ctx_.max_occupancy() * f(energy_fermi_ - kp->band_energy(splb.global_index(j), ispn));
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
#if defined(SIRIUS_USE_FP32)
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
        pout << hbar(80, '-') << std::endl;
        pout << std::endl;
        pout << "  ik                vk                    weight  num_gkvec";
        if (ctx_.full_potential()) {
            pout << "   gklo_basis_size";
        }
        pout << std::endl << hbar(80, '-') << std::endl;
    }

    for (auto it : spl_num_kpoints()) {
        int ik = it.i;
        pout << std::setw(4) << ik << ffmt(9, 4) << kpoints_[ik]->vk()[0] << ffmt(9, 4) << kpoints_[ik]->vk()[1]
             << ffmt(9, 4) << kpoints_[ik]->vk()[2] << ffmt(17, 6) << kpoints_[ik]->weight() << std::setw(11)
             << kpoints_[ik]->num_gkvec();

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
        if (!file_exists(name__)) {
            HDF5_tree(name__, hdf5_access_t::truncate);
        }
        HDF5_tree fout(name__, hdf5_access_t::read_write);
        fout.create_node("K_point_set");
        fout["K_point_set"].write("num_kpoints", num_kpoints());
    }
    ctx_.comm().barrier();
    for (int ik = 0; ik < num_kpoints(); ik++) {
        /* check if this ranks stores the k-point */
        if (ctx_.comm_k().rank() == spl_num_kpoints_.location(typename kp_index_t::global(ik)).ib) {
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
    RTE_THROW("not implemented");

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
//==     if (num_spins != ctx_.num_spins()) error_local(__FILE__, __LINE__, "wrong number of spins");
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

} // namespace sirius
