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
	void K_point_set::sync_band()
	{
	    PROFILE("sirius::K_point_set::sync_band");

	    sddk::mdarray<double, 3> data(ctx_.num_bands(), ctx_.num_spinors(), num_kpoints(),
		    get_memory_pool(sddk::memory_t::host), "K_point_set::sync_band.data");

	    int nb = ctx_.num_bands() * ctx_.num_spinors();
	    #pragma omp parallel
	    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
		int ik = spl_num_kpoints_[ikloc];
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

	template
	void
	K_point_set::sync_band<double, sync_band_t::energy>();

	template
	void
	K_point_set::sync_band<double, sync_band_t::occupancy>();

#if defined(USE_FP32)
	template
	void
	K_point_set::sync_band<float, sync_band_t::energy>();

	template
	void
	K_point_set::sync_band<float, sync_band_t::occupancy>();
#endif

	void K_point_set::create_k_mesh(r3::vector<int> k_grid__, r3::vector<int> k_shift__, int use_symmetry__)
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

	void K_point_set::initialize(std::vector<int> const& counts)
	{
	    if (this->initialized_) {
		RTE_THROW("K-point set is already initialized");
	    }
	    PROFILE("sirius::K_point_set::initialize");
	    /* distribute k-points along the 1-st dimension of the MPI grid */
	    if (counts.empty()) {
		sddk::splindex<sddk::splindex_t::block> spl_tmp(num_kpoints(), comm().size(), comm().rank());
		spl_num_kpoints_ = sddk::splindex<sddk::splindex_t::chunk>(num_kpoints(), comm().size(), comm().rank(), spl_tmp.counts());
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


	template<class F>
	double bisection_search(F&& f, double a, double b, double tol, int maxstep=1000)
	{
	    double x = (a+b)/2;
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

		x = (a + b) / 2.0;
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
	newton_minimization_chemical_potential(Nt&& N, DNt&& dN, D2Nt&& ddN, double mu0, double ne, double tol, int maxstep = 1000)
	{
	    struct {
		double mu; // chemical potential
		int iter; // newton information
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
		//double F = (Nf - ne) * (Nf - ne);
		double dF = 2 * (Nf - ne) * dNf;
		double ddF = 2 * dNf * dNf + 2 * (Nf - ne) * ddNf;
		mu = mu - alpha * dF / std::abs(ddF);

		res.ys.push_back(mu);

		if (std::abs(dF) < tol) {
		    res.iter = iter;
		    res.mu = mu;
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
	void K_point_set::find_band_occupancies()
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

	    #pragma omp parallel for reduction(min:emin) reduction(max:emax)
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
		auto F = [&compute_ne, ne_target, &f](double x) { return compute_ne(x, f) - ne_target; };
		energy_fermi_ = bisection_search(F, emin, emax, 1e-11);

		/* for cold and Methfessel Paxton smearing start newton minimization  */
		if (ctx_.smearing() == smearing::smearing_t::cold || ctx_.smearing() == smearing::smearing_t::methfessel_paxton) {
		    f        = smearing::occupancy(ctx_.smearing(), ctx_.smearing_width());
		    auto df  = smearing::occupancy_deriv(ctx_.smearing(), ctx_.smearing_width());
		    auto ddf = smearing::occupancy_deriv2(ctx_.smearing(), ctx_.smearing_width());
		    auto N   = [&](double mu) { return compute_ne(mu, f); };
		    auto dN  = [&](double mu) { return compute_ne(mu, df); };
		    auto ddN = [&](double mu) { return compute_ne(mu, ddf); };
		    auto res_newton =  newton_minimization_chemical_potential(N, dN, ddN, energy_fermi_, ne_target, tol, 300);
		    energy_fermi_ = res_newton.mu;
		    if (ctx_.verbosity() >= 2) {
			RTE_OUT(ctx_.out()) << "newton iteration converged after " << res_newton.iter << " steps\n";
		    }
		}
	    } catch(std::exception const& e) {
		if (ctx_.verbosity() >= 2) {
		    RTE_OUT(ctx_.out()) << e.what() << std::endl
			<< "fallback to bisection search" << std::endl;
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

	template
	void K_point_set::find_band_occupancies<double>();
#if defined(USE_FP32)
	template
	void K_point_set::find_band_occupancies<float>();
#endif

	template <typename T>
	double K_point_set::valence_eval_sum() const
	{
	    double eval_sum{0};

	    sddk::splindex<sddk::splindex_t::block> splb(ctx_.num_bands(), ctx_.comm_band().size(), ctx_.comm_band().rank());

	    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
		auto ik = spl_num_kpoints_[ikloc];
		auto const& kp = this->get<T>(ik);
		double tmp{0};
		#pragma omp parallel for reduction(+:tmp)
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

	double K_point_set::valence_eval_sum() const
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
	double K_point_set::entropy_sum() const
	{
	    double s_sum{0};

	    double ne_target = ctx_.unit_cell().num_valence_electrons() - ctx_.cfg().parameters().extra_charge();

	    bool only_occ = (ctx_.num_mag_dims() != 1 &&
			     std::abs(ctx_.num_bands() * ctx_.max_occupancy() - ne_target) < 1e-10);

	    if (only_occ) {
		return 0;
	    }

	    auto f = smearing::entropy(ctx_.smearing(), ctx_.smearing_width());

	    sddk::splindex<sddk::splindex_t::block> splb(ctx_.num_bands(), ctx_.comm_band().size(), ctx_.comm_band().rank());

	    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
		auto ik = spl_num_kpoints_[ikloc];
		auto const& kp = this->get<T>(ik);
		double tmp{0};
		#pragma omp parallel for reduction(+:tmp)
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

	double K_point_set::entropy_sum() const
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

	void K_point_set::print_info()
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

	void K_point_set::save(std::string const& name__) const
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
	void K_point_set::load()
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

	extern "C"{
	void wannier_setup_(const char*, int32_t*, int32_t*,
			    const double*, const double*, double*, int32_t*,//care! arg (4,5) changed with const
			    int32_t*, char(*) [3], double*, bool*, bool*,
			    int32_t*, int32_t*, int32_t*, int32_t*, int32_t*,
			    double*, int32_t*, int32_t*, int32_t*, double*,
			    double*, double*, int32_t*, int32_t*, double*,
			    size_t, size_t);

	void wannier_run_(const char*, int32_t*, int32_t*,
			  double*, double*, double*, int32_t*,
			  int32_t*, int32_t*, int32_t*, char(*) [3],
			  double*, bool*, std::complex<double>*, std::complex<double>*, double*,
			  std::complex<double>*, std::complex<double>*, bool*, double*,
			  double*, double*,
			  size_t, size_t);

	}


void write_Amn(sddk::mdarray<std::complex<double>,3>& Amn)              
{
    std::ofstream writeAmn;
    writeAmn.open("sirius.amn");
    std::string line; 
    double Ar, Ai;
    writeAmn << std::endl;
    writeAmn << std::endl;
   
    for(int ik=0; ik<Amn.size(2); ik++){
        for(int n=0; n<Amn.size(1); n++){
	    	for(int m=0; m<Amn.size(0); m++){
				writeAmn << std::fixed << std::setw(5) << m+1;
	    	    writeAmn << std::fixed << std::setw(5) << n+1;
	    	    writeAmn << std::fixed << std::setw(5) << ik+1;
        	    writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << Amn(m,n,ik).real();
        	    writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << Amn(m,n,ik).imag();
        	    writeAmn << std::fixed << std::setprecision(12) << std::setw(18) << abs(Amn(m,n,ik));
        	    writeAmn << std::endl;
	    	}
		}
    }
}


void write_Mmn(sddk::mdarray<std::complex<double>,4>& Mmn,      
    sddk::mdarray<int,2> &nnlist,                   
    sddk::mdarray<int32_t ,3> &nncell)                                
{
    std::ofstream writeMmn;
    writeMmn.open("sirius.mmn");
    writeMmn << std::endl;
    writeMmn << std::endl;
    for (int ik=0; ik<Mmn.size(3); ik++){
        for (int ib=0; ib<Mmn.size(2); ib++){
	    	writeMmn << std::setw(5) << ik+1;
	    	writeMmn << std::setw(5) << nnlist(ik,ib);
	    	writeMmn << std::setw(5) << nncell(0,ik,ib);
	    	writeMmn << std::setw(5) << nncell(1,ik,ib);
	    	writeMmn << std::setw(5) << nncell(2,ik,ib);
	    	writeMmn << std::endl;
	    	for (int n=0; n<Mmn.size(1); n++){
	    	    for (int m=0; m<Mmn.size(0); m++){
			    	writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << Mmn(m,n,ib,ik).real();
			    	writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << Mmn(m,n,ib,ik).imag();
			    	writeMmn << std::fixed << std::setprecision(12) << std::setw(18) << abs(Mmn(m,n,ib,ik));
			    	writeMmn << std::endl;
				}
	    	}
		}
    }
    writeMmn.close();
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
void K_point_set::generate_w90_coeffs()//sirius::K_point_set& k_set__)
{

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

    std::cout << "\n\n\nwannierization!!!!\n\n\n";
    
    //parameters needed for wannier_setup_
    size_t length_seedname = 256;              //aux variable for the length of a string 
    int32_t num_kpts;                          //input        
    int32_t num_bands_tot;                     //input      
    int32_t num_atoms;                         //input         
    size_t length_atomic_symbol = 3;           //aux, as expected from wannier90 lib          
    bool gamma_only;                           //input                   
    bool spinors;                              //input                    
    int32_t num_bands;                         //output                  
    int32_t num_wann;                          //output                      
    int32_t nntot;                             //output
    int32_t num_nnmax = 12;                    //aux variable for max number of neighbors             
                                               //fixed, as in pw2wannier or in wannier90 docs

    //initializing input variables from local variables
    num_kpts = this->num_kpoints();                              
    num_bands_tot = this->get<double>(0)->spinor_wave_functions().num_wf();      
    num_atoms = this->ctx().unit_cell().num_atoms();                      
    gamma_only = this->ctx().gamma_point();
    spinors = false;//right now, generate_wave_functions only works with noncolin (from SIRIUS docs)!
    //WARNING we need to compare with .win file!!!


    //non-scalar variables - definition + space allocation
    char seedname[length_seedname];                                         //input  
    //sddk::mdarray<int32_t ,1> mp_grid(3);                                 //input  
    sddk::mdarray<double,2> real_lattice(3,3);                              //input BOHR!      
    sddk::mdarray<double,2> recip_lattice(3,3);                             //input BOHR^{-1}!        
    sddk::mdarray<double,2> kpt_lattice(3,num_kpts);                        //input             
    char atomic_symbol[num_atoms][3];                                       //input               
    sddk::mdarray<double,2> atoms_cart(3,num_atoms);                        //input
    sddk::mdarray<int,2> nnlist(num_kpts,num_nnmax);                        //output                   
    sddk::mdarray<int32_t ,3> nncell(3,num_kpts,num_nnmax);                 //output                 
    sddk::mdarray<double,2> proj_site(3,num_bands_tot);                     //output                    
    sddk::mdarray<int32_t ,1> proj_l(num_bands_tot);                        //output               
    sddk::mdarray<int32_t ,1> proj_m(num_bands_tot);                        //output                   
    sddk::mdarray<int32_t ,1> proj_radial(num_bands_tot);                   //output                   
    sddk::mdarray<double,2> proj_z(3,num_bands_tot);                        //output                   
    sddk::mdarray<double,2> proj_x(3,num_bands_tot);                        //output                    
    sddk::mdarray<double,1> proj_zona(num_bands_tot);                       //output                   
    sddk::mdarray<int32_t ,1> exclude_bands(num_bands_tot);                 //output                   
    sddk::mdarray<int32_t ,1> proj_s(num_bands_tot);                        //output - optional        
    sddk::mdarray<double,2> proj_s_qaxis(3,num_bands_tot);                  //output - optional        
    //end non-scalar variables


    //initializing non-scalar variables
    std::string aux = "silicon"; 
    strcpy(seedname, aux.c_str());
    length_seedname = aux.length();

	//direct and reciprocal lattice vectors
	for(int ivec=0; ivec<3;ivec++){
		for(int icoor=0; icoor<3; icoor++){
			real_lattice(ivec,icoor) = ctx().unit_cell().lattice_vectors()(icoor,ivec)*bohr_radius;
		    recip_lattice(ivec,icoor) = ctx().unit_cell().reciprocal_lattice_vectors()(icoor,ivec)/bohr_radius;
		}
	}
   
    //copying the k fractional coordinates to a contiguous array
    for(int ik=0; ik < num_kpts; ik++){
	    for(int icoor=0; icoor<3; icoor++){
	    kpt_lattice(icoor, ik) = this->get<double>(ik)->vk()[icoor]; 
	    }
    }
    //initializing atomic_symbol and atomic_cart
    for(int iat=0; iat<num_atoms; iat++){
        std::fill(atomic_symbol[iat],atomic_symbol[iat]+3,' ');//check!!!!!!
        std::strcpy(atomic_symbol[iat], this->ctx().unit_cell().atom(iat).type().label().c_str());
        
	//position is saved in fractional coordinates, we need cartesian for wannier_setup_
        auto frac_coord = this->unit_cell().atom(iat).position();
        auto cart_coord = this->ctx().unit_cell().get_cartesian_coordinates(frac_coord);
        for (int icoor=0; icoor<3; icoor++){
	    atoms_cart(icoor,iat) = cart_coord[icoor]*bohr_radius;
	    }
    }
    //end parameters needed for wannier_setup_
    std::cout << "starting wannier_setup_\n";
    wannier_setup_(seedname,
                   this->ctx().cfg().parameters().ngridk().data(),                  // input              
                   &num_kpts,                                                       // input                      
                   real_lattice.at(sddk::memory_t::host),                           // input        
                   recip_lattice.at(sddk::memory_t::host),                          // input    
                   kpt_lattice.at(sddk::memory_t::host),                            // input            
                   &num_bands_tot,                                                  // input                     
                   &num_atoms,                                                      // input            
                   atomic_symbol,                                                   // input                
                   atoms_cart.at(sddk::memory_t::host),                             // input           
                   &gamma_only,                                                     // input                 
                   &spinors,                                                        // input                 
                   &nntot,                                                          // output                  
                   nnlist.at(sddk::memory_t::host),                                 // output           
                   nncell.at(sddk::memory_t::host),                                 // output                 
                   &num_bands,                                                      // output                         
                   &num_wann,                                                       // output                    
                   proj_site.at(sddk::memory_t::host),                              // output               
                   proj_l.at(sddk::memory_t::host),                                 // output                        
                   proj_m.at(sddk::memory_t::host),                                 // output                           
                   proj_radial.at(sddk::memory_t::host),                            // output                             
                   proj_z.at(sddk::memory_t::host),                                 // output                 
                   proj_x.at(sddk::memory_t::host),                                 // output                    
                   proj_zona.at(sddk::memory_t::host),                              // output          
                   exclude_bands.at(sddk::memory_t::host),                          // output               
                   proj_s.at(sddk::memory_t::host),                                 // output                 
                   proj_s_qaxis.at(sddk::memory_t::host),                           // output               
                   length_seedname,                                                 // aux-length of a string                
                   length_atomic_symbol);                                           // aux-length of a string                     

    std::cout << "wannier_setup succeeded.\n";
    std::cout << "number of neighbors: " << nntot << std::endl;
    num_wann = ctx_.unit_cell().num_ps_atomic_wf().first;
    num_bands = kpoints_[0]->spinor_wave_functions().num_wf(); 
    //allocating memory for overlap matrices 


    sddk::mdarray<std::complex<double>,4> Mmn(num_bands, num_bands, nntot, num_kpts);      //sirius2wannier  
    Mmn.zero();
    sddk::mdarray<std::complex<double>,3> Amn(num_bands, num_wann, num_kpts);              //sirius2wannier
    Amn.zero();

// 2nd step: compute <beta_{\xi'}^{\alpha}|  psi_{j,k+q}>; check how this is done in the Beta_projector class;
// Q-operator can be applied here. Look how this is done in Non_local_operator::apply();
// (look for Beta_projectors_base::inner() function; understand the "chunks" of beta-projectors
	
// 3nd step: copy wave-function at k+q (k') into an auxiliary wave-function object of G+k order and see how
// the G+k+q index can be reshuffled. Check the implementation of G-vector class which handles all the G- and G+k-
// indices

    std::vector<int> atoms(ctx_.unit_cell().num_atoms());
    std::iota(atoms.begin(), atoms.end(), 0);//we need to understand which orbitals to pick up, I am using every here
    

    std::unique_ptr<wf::Wave_functions<double>> Swf_k; 
    sddk::mdarray<std::complex<double>,3> psidotpsi(num_bands, num_bands, num_kpts);      //sirius2wannier
	sddk::mdarray<std::complex<double>,3> atdotat(num_wann, num_wann, num_kpts);      //sirius2wannier
	  
    psidotpsi.zero();
    atdotat.zero();
	std::cout << "Calculating Amn and Mmn...\n";
    //pw2wannier-like steps -> calculation of overlap matrices Amn(k) and Mmn(k,b)
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
    	int ik = spl_num_kpoints_[ikloc];
    	int nwf  = kpoints_[ik]->spinor_wave_functions().num_wf();
    	auto mem = ctx_.processing_unit_memory_t();
        auto q_op = (kpoints_[ik]->unit_cell_.augment()) ? std::make_unique<Q_operator<double>>(kpoints_[ik]->ctx_) : nullptr;
		kpoints_[ik]->beta_projectors().prepare();

	Swf_k = std::make_unique<wf::Wave_functions<double>>(kpoints_[ik]->gkvec_, wf::num_mag_dims(0), 			                                                                    wf::num_bands(nwf), ctx_.host_memory_t());
		apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, nwf), kpoints_[ik]->beta_projectors(),
                    (kpoints_[ik]->spinor_wave_functions()), q_op.get(), *Swf_k);

		/*********************************************************************
		 *                  Amn(k) = <psi_{mk} | w_{nk}>                     *
		 *********************************************************************/
		std::cout << "Calculating Amn. ik = " << ik << std::endl;
		//allocate space for atomic wfs, same that for hubbard
		//we should move the following 2 lines in initialize() function of kpoint class

		kpoints_[ik]->atomic_wave_functions_ = std::make_unique<wf::Wave_functions<double>>(kpoints_[ik]->gkvec_, wf::num_mag_dims(0),
				                                                                    wf::num_bands(num_wann), ctx_.host_memory_t());
		kpoints_[ik]->atomic_wave_functions_S_ = std::make_unique<wf::Wave_functions<double>>(kpoints_[ik]->gkvec_, wf::num_mag_dims(0),
				                                                                    wf::num_bands(num_wann), ctx_.host_memory_t());
		////defining pw expansion of atomic wave functions, defined in pseudopotential file
		kpoints_[ik]->generate_atomic_wave_functions(atoms, [&](int iat){ return &ctx_.unit_cell().atom_type(iat).indexb_wfs(); },
    	              ctx_.ps_atomic_wf_ri(), *(kpoints_[ik]->atomic_wave_functions_));
	
		//ORTHOGONALIZING -CHECK HUBBARD FUNCTION!

        apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, num_wann), kpoints_[ik]->beta_projectors(),
                *(kpoints_[ik]->atomic_wave_functions_), q_op.get(), *(kpoints_[ik]->atomic_wave_functions_S_));
		
		int BS = ctx_.cyclic_block_size();
        la::dmatrix<std::complex<double>> ovlp(num_wann, num_wann, ctx_.blacs_grid(), BS, BS);

        wf::inner(ctx_.spla_context(), mem, wf::spin_range(0), *(kpoints_[ik]->atomic_wave_functions_),
                wf::band_range(0, num_wann), *(kpoints_[ik]->atomic_wave_functions_S_), wf::band_range(0, num_wann), ovlp, 0, 0);
        
		auto B = std::get<0>(inverse_sqrt(ovlp, num_wann));
        wf::transform(ctx_.spla_context(), mem, *B, 0, 0, 1.0, *(kpoints_[ik]->atomic_wave_functions_),
                    wf::spin_index(0), wf::band_range(0, num_wann), 0.0, *(kpoints_[ik]->atomic_wave_functions_S_), wf::spin_index(0),
                    wf::band_range(0, num_wann));

        wf::copy(mem, *(kpoints_[ik]->atomic_wave_functions_S_), wf::spin_index(0), wf::band_range(0, num_wann),
                      *(kpoints_[ik]->atomic_wave_functions_), wf::spin_index(0), wf::band_range(0, num_wann));

        apply_S_operator<double, std::complex<double>>(mem, wf::spin_range(0), wf::band_range(0, num_wann), kpoints_[ik]->beta_projectors(),
                    *(kpoints_[ik]->atomic_wave_functions_), q_op.get(), *(kpoints_[ik]->atomic_wave_functions_S_));
        
		kpoints_[ik]->beta_projectors().dismiss();



    	for(int m=0;m<nwf; m++){
    	    for(int n=0;n<num_wann; n++){
    	           for (int ig_ik=0; ig_ik<kpoints_[ik]->gkvec_.get()->num_gvec(); ig_ik++){
    	                Amn(m, n, ik)+= std::conj(
							                std::conj(kpoints_[ik]->atomic_wave_functions().pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(n)))*
										    Swf_k->pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(m))
											     );
						                //std::conj(kpoints_[ik]->spinor_wave_functions().pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(m)))*
    	                                //kpoints_[ik]->atomic_wave_functions().pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(n));
    	                                //kpoints_[ik]->atomic_wave_functions_S_.pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(n));
		        }
		    }
    	}

		/*********************************************************************
		 *                 Mmn(k,b) = <u_{mk} | u_{n,k+b}>                   *
		 *********************************************************************/
		std::cout << "Calculating Mmn. ik = " << ik << std::endl;

		for (int ib=0; ib < nntot; ib++){
		    int ikpb = nnlist(ik,ib)-1; 
    	    std::unique_ptr<wf::Wave_functions<double>> aux_psi_kpb;
    	    aux_psi_kpb = std::make_unique<wf::Wave_functions<double>>(kpoints_[ik]->gkvec_, wf::num_mag_dims(0),
    	                                                           wf::num_bands(nwf),  
		                             						       ctx_.host_memory_t());
    	    aux_psi_kpb->zero(sddk::memory_t::host);

    	    //find for each G(k) the index in G(k+b)
    	    //if found, store in aux_psi_kpb(ig_ik) the matrix with the wavefunction at (ig_ikpb) 
    	    //equation to use after restoring the order: C_{n, k+b}(G) = C_{n, kb} (G+\tilde{G}) where kb is the copy of k+b in the FBZ 
    	    r3::vector<int> G;
    	    r3::vector<int> G2;
    	    for (int ig_ik=0; ig_ik<kpoints_[ik]->gkvec_.get()->num_gvec(); ig_ik++) {
    	        //compute the total vector to use to get the index in kpb
				G  = kpoints_[ik]->gkvec_.get()->gvec<sddk::index_domain_t::local>(ig_ik);
    	        G += r3::vector<int>(nncell(0,ik,ib), nncell(1,ik,ib), nncell(2,ik,ib));//care with this line!!	             
    	        
				int ig_ikpb = kpoints_[ikpb]->gkvec_->index_by_gvec(G);
				if(ig_ikpb == -1){
					continue;
				}
		        for(int iband=0; iband < nwf; iband++){
		        	aux_psi_kpb->pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(iband)) =
    	        		kpoints_[ikpb]->spinor_wave_functions().pw_coeffs(ig_ikpb, wf::spin_index(0), wf::band_index(iband));
				}
			}
			for(int m=0;m<nwf; m++){
    	        for(int n=0;n<nwf; n++){
    	            for (int ig_ik=0; ig_ik<kpoints_[ik]->gkvec_.get()->num_gvec(); ig_ik++){
    	                Mmn(m, n, ib, ik)+= std::conj(
							 					std::conj(aux_psi_kpb->pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(n)))*
											 	Swf_k->pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(m))
											     	 );

											//std::conj(kpoints_[ik]->spinor_wave_functions().pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(m)))*
    	                                    //aux_psi_kpb->pw_coeffs(ig_ik, wf::spin_index(0), wf::band_index(n));
    	            }
    	        }
		    }    
    	}//end ib

    std::cout << "checking orthogonality..." << std::endl;
		for(int n=0; n<num_wann; n++){
			for(int m=0; m<num_wann; m++){
				for(int ig=0; ig<kpoints_[ik]->gkvec_.get()->num_gvec(); ig++){
					atdotat(m,n,ik) += std::conj(kpoints_[ik]->atomic_wave_functions().pw_coeffs(ig, wf::spin_index(0), wf::band_index(m)))*
											//kpoints_[ik]->atomic_wave_functions().pw_coeffs(ig, wf::spin_index(0), wf::band_index(n));
											kpoints_[ik]->atomic_wave_functions_S().pw_coeffs(ig, wf::spin_index(0), wf::band_index(n));
			}
		}
	}


		for(int n=0; n<kpoints_[ik]->spinor_wave_functions().num_wf(); n++){
			for(int m=0; m<kpoints_[ik]->spinor_wave_functions().num_wf(); m++){
				for(int ig=0; ig<kpoints_[ik]->gkvec_.get()->num_gvec(); ig++){
					psidotpsi(m,n,ik) += std::conj(kpoints_[ik]->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(0), wf::band_index(m)))*
											//kpoints_[ik]->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(0), wf::band_index(n));
											Swf_k->pw_coeffs(ig, wf::spin_index(0), wf::band_index(n));
			}
		}
	}
    }//end ik
    write_Amn(Amn);
    write_Mmn(Mmn,nnlist,nncell);


    //check orthogonality



	std::ofstream atmat;
	atmat.open("orth_at.txt");
	for(int ik=0; ik<num_kpts; ik++){
		atmat << "ik= " << ik << std::endl;
		for(int n=0; n<num_wann; n++){
			for(int m=0; m<num_wann; m++){
		    	    atmat << std::fixed << std::setw(5) << m+1;
		    	    atmat << std::fixed << std::setw(5) << n+1;
			    	atmat << std::fixed << std::setprecision(12) << std::setw(18) << atdotat(m,n,ik).real();
			    	atmat << std::fixed << std::setprecision(12) << std::setw(18) << atdotat(m,n,ik).imag();
			    	atmat << std::fixed << std::setprecision(12) << std::setw(18) << abs(atdotat(m,n,ik));
					atmat << std::endl;
			}
		}
		atmat << std::endl;
	}
	atmat.close();

	std::ofstream psimat;
	psimat.open("orth_psi.txt");
	for(int ik=0; ik<num_kpts; ik++){
		psimat << "ik= " << ik << std::endl;
		for(int n=0; n<num_bands_tot; n++){
			for(int m=0; m<num_bands_tot; m++){
		    	    psimat << std::fixed << std::setw(5) << m+1;
		    	    psimat << std::fixed << std::setw(5) << n+1;
			    	psimat << std::fixed << std::setprecision(12) << std::setw(18) << psidotpsi(m,n,ik).real();
			    	psimat << std::fixed << std::setprecision(12) << std::setw(18) << psidotpsi(m,n,ik).imag();
			    	psimat << std::fixed << std::setprecision(12) << std::setw(18) << abs(psidotpsi(m,n,ik));
					psimat << std::endl;
			}
		}
		psimat << std::endl;
	}
	psimat.close();


    //compute wannier orbitals 
	//define additional arguments
    sddk::mdarray<double,2>               eigval(num_bands, num_kpts);                     //input
    sddk::mdarray<std::complex<double>,3> U_matrix(num_wann, num_wann, num_kpts);          //output  
    sddk::mdarray<std::complex<double>,3> U_dis(num_bands, num_wann, num_kpts);            //output  
    sddk::mdarray<bool,2>                 lwindow(num_bands, num_kpts);                    //output  
    sddk::mdarray<double,2>               wannier_centres(3, num_wann);                    //output  
    sddk::mdarray<double,1>               wannier_spreads(num_wann);                       //output  
    sddk::mdarray<double,1>               spread_loc(3);                                   //output-opt

	//Initialize eigval with the value of the energy dispersion
    for (int ikloc = 0; ikloc < spl_num_kpoints_.local_size(); ikloc++) {
		for(int iband=0; iband<num_bands; iband++){
			eigval(iband, ikloc) = kpoints_[ikloc]->band_energy(iband, 0)*ha2ev;//sirius saves energy in Hartree, we need it in eV
		}
	}


	U_matrix.zero();
	U_dis.zero();
	lwindow.zero();
	wannier_centres.zero();
	wannier_spreads.zero();
	spread_loc.zero();
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
		 		Mmn.at(sddk::memory_t::host),
		 		Amn.at(sddk::memory_t::host),
		 		eigval.at(sddk::memory_t::host),
		 		U_matrix.at(sddk::memory_t::host),
		 		U_dis.at(sddk::memory_t::host),
		 		lwindow.at(sddk::memory_t::host),
		 		wannier_centres.at(sddk::memory_t::host),
		 		wannier_spreads.at(sddk::memory_t::host),
		 		spread_loc.at(sddk::memory_t::host),
		 		length_seedname, 
		 		length_atomic_symbol);
    std::cout << "Wannier_run succeeded." << std::endl;


// 4th step: allocate resulting matrix M_{nn'}, compute contribution from C*C (1st part) using wf::inner() function;
// compute contribution from ultrasoft part using a matrix-matrix multiplication
//
// 5th step: parallelize over k-points
//
// 6ts step: parallelize over G+k vectors and k-points
}

} // namespace sirius
