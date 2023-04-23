// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file simulation_context.hpp
 *
 *  \brief Contains definition and implementation of Simulation_context class.
 */

#ifndef __SIMULATION_CONTEXT_HPP__
#define __SIMULATION_CONTEXT_HPP__

#include <algorithm>
#include <memory>
#include <spla/spla.hpp>

#include "simulation_parameters.hpp"
#include "mpi/mpi_grid.hpp"
#include "radial/radial_integrals.hpp"
#include "utils/utils.hpp"
#include "utils/env.hpp"
#include "density/augmentation_operator.hpp"
#include "gpu/acc.hpp"
#include "symmetry/rotation.hpp"
#include "fft/fft.hpp"

#ifdef SIRIUS_GPU
extern "C" void generate_phase_factors_gpu(int num_gvec_loc__, int num_atoms__, int const* gvec__,
                                           double const* atom_pos__, std::complex<double>* phase_factors__);
#endif

namespace sirius {

template <typename OUT>
void
print_memory_usage(OUT&& out__, std::string file_and_line__ = "")
{
    if (!env::print_memory_usage()) {
        return;
    }

    size_t VmRSS, VmHWM;
    utils::get_proc_status(&VmHWM, &VmRSS);

    std::stringstream s;
    s << "rank" << std::setfill('0') << std::setw(4) << mpi::Communicator::world().rank();
    out__ << "[" << s.str() << " at " << file_and_line__ << "] "
          << "VmHWM: " << (VmHWM >> 20) << " Mb, "
          << "VmRSS: " << (VmRSS >> 20) << " Mb";

    if (acc::num_devices() > 0) {
        size_t gpu_mem = acc::get_free_mem();
        out__ << ", GPU: " << (gpu_mem >> 20) << " Mb";
    }
    out__ << std::endl;


    std::vector<std::string> labels = {"host"};
    std::vector<sddk::memory_pool*> mp = {&get_memory_pool(sddk::memory_t::host)};
    
    int np{1};
    if (acc::num_devices() > 0) {
        labels.push_back("host pinned");
        labels.push_back("device");
        mp.push_back(&get_memory_pool(sddk::memory_t::host_pinned));
        mp.push_back(&get_memory_pool(sddk::memory_t::device));
        np = 3;
    }

    for (int i = 0; i < np; i++) {
        out__ << "[mem.pool] " << labels[i] << ": total capacity: " << (mp[i]->total_size() >> 20) << " Mb, "
              << "free: " << (mp[i]->free_size() >> 20) << " Mb, "
              << "num.blocks: " <<  mp[i]->num_blocks() << std::endl;
    }
}

/// Utility function to generate LAPW unit step function.
double unit_step_function_form_factors(double R__, double g__);

/// Simulation context is a set of parameters and objects describing a single simulation.
/** The order of initialization of the simulation context is the following: first, the default parameter
    values are set in the constructor, then (optionally) import() method is called and the parameters are
    overwritten with the those from the input file, and finally, the user sets the values with setter metods.
    Then the unit cell can be populated and the context can be initialized.
 */
class Simulation_context : public Simulation_parameters
{
  private:
    /// Communicator for this simulation.
    mpi::Communicator const& comm_;

    mpi::Communicator comm_k_;
    mpi::Communicator comm_band_;

    /// Auxiliary communicator for the coarse-grid FFT transformation.
    mpi::Communicator comm_ortho_fft_coarse_;

    /// Unit cell of the simulation.
    std::unique_ptr<Unit_cell> unit_cell_;

    /// MPI grid for this simulation.
    std::unique_ptr<mpi::Grid> mpi_grid_;

    /// 2D BLACS grid for distributed linear algebra operations.
    std::unique_ptr<la::BLACS_grid> blacs_grid_;

    /// Grid descriptor for the fine-grained FFT transform.
    fft::Grid fft_grid_;

    /// Fine-grained FFT for density and potential.
    /** This is the FFT driver to transform periodic functions such as density and potential on the fine-grained
     *  FFT grid. The transformation is parallel. */
    std::unique_ptr<spfft::Transform> spfft_transform_;
    std::unique_ptr<spfft::Grid> spfft_grid_;
#if defined(USE_FP32)
    std::unique_ptr<spfft::TransformFloat> spfft_transform_float_;
    std::unique_ptr<spfft::GridFloat> spfft_grid_float_;
#endif

    /// Grid descriptor for the coarse-grained FFT transform.
    fft::Grid fft_coarse_grid_;

    /// Coarse-grained FFT for application of local potential and density summation.
    std::unique_ptr<spfft::Transform> spfft_transform_coarse_;
    std::unique_ptr<spfft::Grid> spfft_grid_coarse_;
#if defined(USE_FP32)
    std::unique_ptr<spfft::TransformFloat> spfft_transform_coarse_float_;
    std::unique_ptr<spfft::GridFloat> spfft_grid_coarse_float_;
#endif

    /// G-vectors within the Gmax cutoff.
    std::shared_ptr<fft::Gvec> gvec_;

    std::shared_ptr<fft::Gvec_fft> gvec_fft_;

    /// G-vectors within the 2 * |Gmax^{WF}| cutoff.
    std::shared_ptr<fft::Gvec> gvec_coarse_;

    std::shared_ptr<fft::Gvec_fft> gvec_coarse_fft_;

    std::shared_ptr<fft::Gvec_shells> remap_gvec_;

    /// Creation time of the parameters.
    timeval start_time_;

    /// A tag string based on the the starting time.
    std::string start_time_tag_;

    /// 1D phase factors for each atom coordinate and G-vector index.
    sddk::mdarray<std::complex<double>, 3> phase_factors_;

    /// 1D phase factors of the symmetry operations.
    sddk::mdarray<std::complex<double>, 3> sym_phase_factors_;

    /// Phase factors for atom types.
    sddk::mdarray<std::complex<double>, 2> phase_factors_t_;

    /// Lattice coordinats of G-vectors in a GPU-friendly ordering.
    sddk::mdarray<int, 2> gvec_coord_;

    /// Volume of the initial unit cell.
    /** This is needed to estimate the new cutoff for radial integrals. */
    double omega0_;

    /// Initial lattice vectors.
    r3::matrix<double> lattice_vectors0_;

    /// Radial integrals of beta-projectors.
    std::unique_ptr<Radial_integrals_beta<false>> beta_ri_;

    /// Callback function provided by the host code to compute radial integrals of beta projectors.
    std::function<void(int, double, double*, int)> beta_ri_callback_{nullptr};

    /// Radial integrals of beta-projectors with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_beta<true>> beta_ri_djl_;

    std::function<void(int, double, double*, int)> beta_ri_djl_callback_{nullptr};

    /// Radial integrals of augmentation operator.
    std::unique_ptr<Radial_integrals_aug<false>> aug_ri_;

    /// Callback function provided by the host code to compute radial integrals of augmentation operator.
    std::function<void(int, double, double*, int, int)> aug_ri_callback_{nullptr};

    /// Radial integrals of augmentation operator with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_aug<true>> aug_ri_djl_;

    std::function<void(int, double, double*, int, int)> aug_ri_djl_callback_{nullptr};

    /// Radial integrals of atomic wave-functions.
    std::unique_ptr<Radial_integrals_atomic_wf<false>> ps_atomic_wf_ri_;

    /// Radial integrals of atomic wave-functions with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_atomic_wf<true>> ps_atomic_wf_ri_djl_;

    /// Radial integrals of pseudo-core charge density.
    std::unique_ptr<Radial_integrals_rho_core_pseudo<false>> ps_core_ri_;

    /// Radial integrals of pseudo-core charge density with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_rho_core_pseudo<true>> ps_core_ri_djl_;

    std::function<void(int, int, double*, double*)> rhoc_ri_callback_{nullptr};
    std::function<void(int, int, double*, double*)> rhoc_ri_djl_callback_{nullptr};

    std::function<void(int, int, double*, double*)> ps_rho_ri_callback_{nullptr};
    std::function<void(int, double, double*, int)> ps_atomic_wf_ri_callback_{nullptr};
    std::function<void(int, double, double*, int)> ps_atomic_wf_ri_djl_callback_{nullptr};

    /// Radial integrals of total pseudo-charge density.
    std::unique_ptr<Radial_integrals_rho_pseudo> ps_rho_ri_;

    /// Radial integrals of the local part of pseudopotential.
    std::unique_ptr<Radial_integrals_vloc<false>> vloc_ri_;

    /// Callback function to compute radial integrals of local potential.
    std::function<void(int, int, double*, double*)> vloc_ri_callback_{nullptr};

    /// Radial integrals of the local part of pseudopotential with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_vloc<true>> vloc_ri_djl_;

    std::function<void(int, int, double*, double*)> vloc_ri_djl_callback_{nullptr};

    /// List of real-space point indices for each of the atoms.
    std::vector<std::vector<std::pair<int, double>>> atoms_to_grid_idx_;

    /// Plane wave expansion coefficients of the step function.
    sddk::mdarray<std::complex<double>, 1> theta_pw_;

    /// Step function on the real-space grid.
    sddk::mdarray<double, 1> theta_;

    /// Augmentation operator for each atom type.
    /** The augmentation operator is used by Density, Potential, Q_operator, and Non_local_functor classes. */
    std::vector<std::unique_ptr<Augmentation_operator>> augmentation_op_;

    /// Standard eigen-value problem solver.
    std::unique_ptr<la::Eigensolver> std_evp_solver_;

    /// Generalized eigen-value problem solver.
    std::unique_ptr<la::Eigensolver> gen_evp_solver_;

    /// Type of host memory (pagable or page-locked) for the arrays that participate in host-to-device memory copy.
    sddk::memory_t host_memory_t_{sddk::memory_t::none};

    /// Callback function to compute band occupancies.
    std::function<void(void)> band_occ_callback_{nullptr};

    /// Callback function to compute effective potential.
    std::function<void(void)> veff_callback_{nullptr};

    /// Spla context.
    std::shared_ptr<::spla::Context> spla_ctx_{new ::spla::Context{SPLA_PU_HOST}};

    std::ostream* output_stream_{nullptr};
    std::ofstream output_file_stream_;

    /// External pointers to periodic functions.
    std::map<std::string, periodic_function_ptr_t<double>> pf_ext_ptr;

    mutable double evp_work_count_{0};
    mutable int num_loc_op_applied_{0};
    /// Total number of iterative solver steps.
    mutable int num_itsol_steps_{0};

    /// True if the context is already initialized.
    bool initialized_{false};

    /// Initialize FFT coarse and fine grids.
    void init_fft_grid();

    /// Initialize communicators.
    void init_comm();

    /// Unit step function is defined to be 1 in the interstitial and 0 inside muffin-tins.
    /** Unit step function is constructed from it's plane-wave expansion coefficients which are computed
     *  analytically:
     *  \f[
     *      \Theta({\bf r}) = \sum_{\bf G} \Theta({\bf G}) e^{i{\bf Gr}},
     *  \f]
     *  where
     *  \f[
     *      \Theta({\bf G}) = \frac{1}{\Omega} \int \Theta({\bf r}) e^{-i{\bf Gr}} d{\bf r} =
     *          \frac{1}{\Omega} \int_{\Omega} e^{-i{\bf Gr}} d{\bf r} - \frac{1}{\Omega} \int_{MT} e^{-i{\bf Gr}}
     *           d{\bf r} = \delta_{\bf G, 0} - \sum_{\alpha} \frac{1}{\Omega} \int_{MT_{\alpha}} e^{-i{\bf Gr}}
     *           d{\bf r}
     *  \f]
     *  Integralof a plane-wave over the muffin-tin volume is taken using the spherical expansion of the
     *  plane-wave around central point \f$ \tau_{\alpha} \f$:
     *  \f[ \int_{MT_{\alpha}} e^{-i{\bf Gr}} d{\bf r} = e^{-i{\bf G\tau_{\alpha}}}
     *   \int_{MT_{\alpha}} 4\pi \sum_{\ell m} (-i)^{\ell} j_{\ell}(Gr) Y_{\ell m}(\hat {\bf G}) Y_{\ell m}^{*}(\hat
     *   {\bf r}) r^2 \sin \theta dr d\phi d\theta
     *  \f]
     *  In the above integral only \f$ \ell=m=0 \f$ term survives. So we have:
     *  \f[
     *      \int_{MT_{\alpha}} e^{-i{\bf Gr}} d{\bf r} = 4\pi e^{-i{\bf G\tau_{\alpha}}} \Theta(\alpha, G)
     *  \f]
     *  where
     *  \f[
     *      \Theta(\alpha, G) = \int_{0}^{R_{\alpha}} \frac{\sin(Gr)}{Gr} r^2 dr =
     *          \left\{ \begin{array}{ll} \displaystyle R_{\alpha}^3 / 3 & G=0 \\
     *          \Big( \sin(GR_{\alpha}) - GR_{\alpha}\cos(GR_{\alpha}) \Big) / G^3 & G \ne 0 \end{array} \right.
     *  \f]
     *  are the so-called step function form factors. With this we have a final expression for the plane-wave
     *  coefficients of the unit step function:
     *  \f[ \Theta({\bf G}) = \delta_{\bf G, 0} - \sum_{\alpha}
     *   \frac{4\pi}{\Omega} e^{-i{\bf G\tau_{\alpha}}} \Theta(\alpha, G)
     *  \f]
     */
    void init_step_function();

    /// Find a list of real-space grid points around each atom.
    void init_atoms_to_grid_idx(double R__);

    /// Get the stsrting time stamp.
    void start()
    {
        gettimeofday(&start_time_, NULL);
        start_time_tag_ = utils::timestamp("%Y%m%d_%H%M%S");
    }

    /* copy constructor is forbidden */
    Simulation_context(Simulation_context const&) = delete;

  public:
    /// Create an empty simulation context with an explicit communicator.
    Simulation_context(mpi::Communicator const& comm__ = mpi::Communicator::world())
        : comm_(comm__)
    {
        unit_cell_ = std::make_unique<Unit_cell>(*this, comm_);
        start();
    }

    Simulation_context(mpi::Communicator const& comm__, mpi::Communicator const& comm_k__, mpi::Communicator const& comm_band__)
        : comm_(comm__)
        , comm_k_(comm_k__)
        , comm_band_(comm_band__)
    {
        unit_cell_ = std::make_unique<Unit_cell>(*this, comm_);
        start();
    }

    /// Create a simulation context with world communicator and load parameters from JSON string or JSON file.
    Simulation_context(std::string const& str__)
        : comm_(mpi::Communicator::world())
    {
        unit_cell_ = std::make_unique<Unit_cell>(*this, comm_);
        start();
        import(str__);
        unit_cell_->import(cfg().unit_cell());
    }

    explicit Simulation_context(nlohmann::json const& dict__)
        : comm_(mpi::Communicator::world())
    {
        unit_cell_ = std::make_unique<Unit_cell>(*this, comm_);
        start();
        import(dict__);
        unit_cell_->import(cfg().unit_cell());
    }

    // /// Create a simulation context with world communicator and load parameters from JSON string or JSON file.
    Simulation_context(std::string const& str__, mpi::Communicator const& comm__)
        : comm_(comm__)
    {
        unit_cell_ = std::make_unique<Unit_cell>(*this, comm_);
        start();
        import(str__);
        unit_cell_->import(cfg().unit_cell());
    }

    /// Destructor.
    ~Simulation_context()
    {
        if (!comm().is_finalized() && initialized_) {
            print_memory_usage(this->out(), FILE_LINE);
        }
    }

    /// Initialize the similation (can only be called once).
    void initialize();

    void print_info(std::ostream& out__) const;

    /// Update context after setting new lattice vectors or atomic coordinates.
    void update();

    auto const& atoms_to_grid_idx_map(int ia__) const
    {
        return atoms_to_grid_idx_[ia__];
    };

    auto& unit_cell()
    {
        return *unit_cell_;
    }

    /// Return const reference to unit cell object.
    auto const& unit_cell() const
    {
        return *unit_cell_;
    }

    /// Return const reference to Gvec object.
    auto const& gvec() const
    {
        return *gvec_;
    }

    /// Return shared pointer to Gvec object.
    auto gvec_sptr() const
    {
        return gvec_;
    }

    /// Return const reference to Gvec_fft object.
    auto const& gvec_fft() const
    {
        return *gvec_fft_;
    }

    /// Return shared pointer to Gvec_fft object.
    auto gvec_fft_sptr() const
    {
        return gvec_fft_;
    }

    auto const& gvec_coarse() const
    {
        return *gvec_coarse_;
    }

    auto const& gvec_coarse_sptr() const
    {
        return gvec_coarse_;
    }

    auto const& gvec_coarse_fft_sptr() const
    {
        return gvec_coarse_fft_;
    }

    auto const& remap_gvec() const
    {
        return *remap_gvec_;
    }

    auto const& blacs_grid() const
    {
        return *blacs_grid_;
    }

    /// Total communicator of the simulation.
    mpi::Communicator const& comm() const
    {
        return comm_;
    }

    /// Communicator between k-points.
    /** This communicator is used to split k-points */
    auto const& comm_k() const
    {
        return comm_k_;
    }

    /// Band parallelization communicator.
    /** This communicator is used to parallelize the band problem. However it is not necessarily used
        to create the BLACS grid. Diagonalization might be sequential. */
    auto const& comm_band() const
    {
        return comm_band_;
    }

    /// Communicator of the dense FFT grid.
    /** This communicator is passed to the spfft::Transform constructor. */
    auto const& comm_fft() const
    {
        /* use entire communicator of the simulation */
        return comm();
    }

    auto const& comm_ortho_fft() const
    {
        return mpi::Communicator::self();
    }

    /// Communicator of the coarse FFT grid.
    /** This communicator is passed to the spfft::Transform constructor. */
    auto const& comm_fft_coarse() const
    {
        if (cfg().control().fft_mode() == "serial") {
            return mpi::Communicator::self();
        } else {
            return mpi_grid_->communicator(1 << 0);
        }
    }

    /// Communicator, which is orthogonal to comm_fft_coarse within a band communicator.
    /** This communicator is used in reshuffling the wave-functions for the FFT-friendly distribution. It will be
        used to parallelize application of local Hamiltonian over bands. */
    auto const& comm_band_ortho_fft_coarse() const
    {
        if (cfg().control().fft_mode() == "serial") {
            return comm_band();
        } else {
            return mpi_grid_->communicator(1 << 1);
        }
    }

    auto const& comm_ortho_fft_coarse() const
    {
        return comm_ortho_fft_coarse_;
    }

    void create_storage_file() const;

    inline std::string const& start_time_tag() const
    {
        return start_time_tag_;
    }

    inline auto& std_evp_solver()
    {
        return *std_evp_solver_;
    }

    inline auto const& std_evp_solver() const
    {
        return *std_evp_solver_;
    }

    inline auto& gen_evp_solver()
    {
        return *gen_evp_solver_;
    }

    inline auto const& gen_evp_solver() const
    {
        return *gen_evp_solver_;
    }

    /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
    inline auto gvec_phase_factor(r3::vector<int> G__, int ia__) const
    {
        return phase_factors_(0, G__[0], ia__) * phase_factors_(1, G__[1], ia__) * phase_factors_(2, G__[2], ia__);
    }

    /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
    inline auto gvec_phase_factor(int ig__, int ia__) const
    {
        return gvec_phase_factor(gvec().gvec<sddk::index_domain_t::global>(ig__), ia__);
    }

    inline auto const& gvec_coord() const
    {
        return gvec_coord_;
    }

    /// Generate phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$ for all atoms of a given type.
    void generate_phase_factors(int iat__, sddk::mdarray<std::complex<double>, 2>& phase_factors__) const;

    /// Make periodic function out of form factors.
    /** Return vector of plane-wave coefficients */ // TODO: return mdarray
    template <sddk::index_domain_t index_domain, typename F>
    inline auto make_periodic_function(F&& form_factors__) const
    {
        PROFILE("sirius::Simulation_context::make_periodic_function");

        double fourpi_omega = fourpi / unit_cell().omega();

        int ngv = (index_domain == sddk::index_domain_t::local) ? gvec().count() : gvec().num_gvec();
        std::vector<std::complex<double>> f_pw(ngv, std::complex<double>(0, 0));

        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            /* global index of G-vector */
            int ig   = gvec().offset() + igloc;
            double g = gvec().gvec_len<sddk::index_domain_t::local>(igloc);

            int j = (index_domain == sddk::index_domain_t::local) ? igloc : ig;
            for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
                f_pw[j] += fourpi_omega * std::conj(phase_factors_t_(igloc, iat)) * form_factors__(iat, g);
            }
        }

        if (index_domain == sddk::index_domain_t::global) {
            comm_.allgather(&f_pw[0], gvec().count(), gvec().offset());
        }

        return f_pw;
    }

    /// Make periodic out of form factors computed for G-shells.
    template <sddk::index_domain_t index_domain>
    inline auto make_periodic_function(sddk::mdarray<double, 2>& form_factors__) const
    {
        PROFILE("sirius::Simulation_context::make_periodic_function");

        double fourpi_omega = fourpi / unit_cell().omega();

        int ngv = (index_domain == sddk::index_domain_t::local) ? gvec().count() : gvec().num_gvec();
        std::vector<std::complex<double>> f_pw(ngv, std::complex<double>(0, 0));

        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            /* global index of G-vector */
            int ig   = gvec().offset() + igloc;
            int igsh = gvec().shell(ig);

            int j = (index_domain == sddk::index_domain_t::local) ? igloc : ig;
            for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
                f_pw[j] += fourpi_omega * std::conj(phase_factors_t_(igloc, iat)) * form_factors__(igsh, iat);
            }
        }

        if (index_domain == sddk::index_domain_t::global) {
            comm_.allgather(&f_pw[0], gvec().count(), gvec().offset());
        }

        return f_pw;
    }

    /// Compute values of spherical Bessel functions at MT boundary.
    sddk::mdarray<double, 3> generate_sbessel_mt(int lmax__) const;

    /// Generate complex spherical harmoics for the local set of G-vectors.
    sddk::matrix<std::complex<double>> generate_gvec_ylm(int lmax__);

    /// Sum over the plane-wave coefficients and spherical harmonics that apperas in Poisson solver and finding of the
    /// MT boundary values.
    /** The following operation is performed:
     *  \f[
     *    q_{\ell m}^{\alpha} = \sum_{\bf G} 4\pi \rho({\bf G})
     *     e^{i{\bf G}{\bf r}_{\alpha}}i^{\ell}f_{\ell}^{\alpha}(G) Y_{\ell m}^{*}(\hat{\bf G})
     *  \f]
     */
    sddk::mdarray<std::complex<double>, 2> sum_fg_fl_yg(int lmax__, std::complex<double> const* fpw__, sddk::mdarray<double, 3>& fl__,
                                            sddk::matrix<std::complex<double>>& gvec_ylm__);

    inline auto const& beta_ri() const
    {
        return *beta_ri_;
    }

    inline auto const& beta_ri_djl() const
    {
        return *beta_ri_djl_;
    }

    inline auto const& aug_ri() const
    {
        return *aug_ri_;
    }

    inline auto const& aug_ri_djl() const
    {
        return *aug_ri_djl_;
    }

    inline auto const& ps_atomic_wf_ri() const
    {
        return *ps_atomic_wf_ri_;
    }

    inline auto const& ps_atomic_wf_ri_djl() const
    {
        return *ps_atomic_wf_ri_djl_;
    }

    inline auto const& ps_core_ri() const
    {
        return *ps_core_ri_;
    }

    inline auto const& ps_core_ri_djl() const
    {
        return *ps_core_ri_djl_;
    }

    inline auto const& ps_rho_ri() const
    {
        return *ps_rho_ri_;
    }

    inline auto const& vloc_ri() const
    {
        return *vloc_ri_;
    }

    inline auto const& vloc_ri_djl() const
    {
        return *vloc_ri_djl_;
    }

    /// Find the lambda parameter used in the Ewald summation.
    /** Lambda parameter scales the erfc function argument:
     *  \f[
     *    {\rm erf}(\sqrt{\lambda}x)
     *  \f]
     */
    double ewald_lambda() const;

    auto const& sym_phase_factors() const
    {
        return sym_phase_factors_;
    }

    inline bool initialized() const
    {
        return initialized_;
    }

    /// Return plane-wave coefficient of the step function.
    inline auto const& theta_pw(int ig__) const
    {
        return theta_pw_[ig__];
    }

    /// Return the value of the step function for the grid point ir.
    inline double theta(int ir__) const
    {
        return theta_[ir__];
    }

    /// Returns a constant pointer to the augmentation operator of a given atom type.
    inline auto const& augmentation_op(int iat__) const
    {
        RTE_ASSERT(augmentation_op_[iat__] != nullptr);
        return *augmentation_op_[iat__];
    }

    inline auto& augmentation_op(int iat__)
    {
        RTE_ASSERT(augmentation_op_[iat__] != nullptr);
        return *augmentation_op_[iat__];
    }

    /// Type of the host memory for arrays used in linear algebra operations.
    /** For CPU execution this is normal host memory, for GPU execution this is pinned memory. */
    inline auto host_memory_t() const
    {
        return host_memory_t_;
    }

    /// Return the memory type for processing unit.
    inline auto processing_unit_memory_t() const
    {
        return (this->processing_unit() == sddk::device_t::CPU) ? sddk::memory_t::host : sddk::memory_t::device;
    }

    /// Set the size of the fine-grained FFT grid.
    void fft_grid_size(std::array<int, 3> fft_grid_size__)
    {
        cfg().settings().fft_grid_size(fft_grid_size__);
    }

    template <typename T>
    fft::spfft_grid_type<T>& spfft_grid_coarse();

    template <typename T>
    fft::spfft_transform_type<T>& spfft();

    template <typename T>
    fft::spfft_transform_type<T> const& spfft() const;

    template <typename T>
    fft::spfft_transform_type<T>& spfft_coarse();

    template <typename T>
    fft::spfft_transform_type<T> const& spfft_coarse() const;

    auto const& fft_grid() const
    {
        return fft_grid_;
    }

    auto const& fft_coarse_grid() const
    {
        return fft_coarse_grid_;
    }

    auto const& spla_context() const
    {
        return *spla_ctx_;
    }

    auto& spla_context()
    {
        return *spla_ctx_;
    }

    inline double evp_work_count(double w__ = 0) const
    {
        evp_work_count_ += w__;
        return evp_work_count_;
    }

    /// Keep track of the total number of wave-functions to which the local operator was applied.
    inline int num_loc_op_applied(int n = 0) const
    {
        num_loc_op_applied_ += n;
        return num_loc_op_applied_;
    }

    inline int num_itsol_steps(int n = 0) const
    {
        num_itsol_steps_ += n;
        return num_itsol_steps_;
    }

    /// Set the callback function.
    inline void beta_ri_callback(void (*fptr__)(int, double, double*, int))
    {
        beta_ri_callback_ = fptr__;
    }

    inline void beta_ri_djl_callback(void (*fptr__)(int, double, double*, int))
    {
        beta_ri_djl_callback_ = fptr__;
    }

    /// Set the callback function.
    inline void aug_ri_callback(void (*fptr__)(int, double, double*, int, int))
    {
        aug_ri_callback_ = fptr__;
    }

    /// Set the callback function.
    inline void aug_ri_djl_callback(void (*fptr__)(int, double, double*, int, int))
    {
        aug_ri_djl_callback_ = fptr__;
    }

    inline void vloc_ri_callback(void (*fptr__)(int, int, double*, double*))
    {
        vloc_ri_callback_ = fptr__;
    }

    inline void ps_rho_ri_callback(void (*fptr__)(int, int, double*, double*))
    {
        ps_rho_ri_callback_ = fptr__;
    }

    inline void vloc_ri_djl_callback(void (*fptr__)(int, int, double*, double*))
    {
        vloc_ri_djl_callback_ = fptr__;
    }

    inline void rhoc_ri_callback(void (*fptr__)(int, int, double*, double*))
    {
        rhoc_ri_callback_ = fptr__;
    }

    inline void rhoc_ri_djl_callback(void (*fptr__)(int, int, double*, double*))
    {
        rhoc_ri_djl_callback_ = fptr__;
    }

    inline void ps_atomic_wf_ri_callback(void (*fptr__)(int, double, double*, int))
    {
        ps_atomic_wf_ri_callback_ = fptr__;
    }

    inline void ps_atomic_wf_ri_djl_callback(void (*fptr__)(int, double, double*, int))
    {
        ps_atomic_wf_ri_djl_callback_ = fptr__;
    }

    /// Set callback function to compute band occupations
    inline void band_occ_callback(void (*fptr__)(void))
    {
        band_occ_callback_ = fptr__;
    }

    inline std::function<void(void)> band_occ_callback() const
    {
        return band_occ_callback_;
    }

    inline void veff_callback(void (*fptr__)(void))
    {
        veff_callback_ = fptr__;
    }

    inline std::function<void(void)> veff_callback() const
    {
        return veff_callback_;
    }

    /// Export parameters of simulation context as a JSON dictionary.
    nlohmann::json serialize()
    {
        nlohmann::json dict;
        dict["config"] = cfg().dict();
        bool const cart_pos{false};
        dict["config"]["unit_cell"] = unit_cell().serialize(cart_pos);
        auto fftgrid                = {spfft_transform_coarse_->dim_x(), spfft_transform_coarse_->dim_y(),
                        spfft_transform_coarse_->dim_z()};
        dict["fft_coarse_grid"]     = fftgrid;
        dict["mpi_grid"]            = mpi_grid_dims();
        dict["omega"]               = unit_cell().omega();
        dict["chemical_formula"]    = unit_cell().chemical_formula();
        dict["num_atoms"]           = unit_cell().num_atoms();
        return dict;
    }

    /// Return output stream.
    inline std::ostream& out() const
    {
        RTE_ASSERT(output_stream_ != nullptr);
        return *output_stream_;
    }

    /// Return output stream based on the verbosity level.
    inline std::ostream& out(int level__) const
    {
        if (this->verbosity() >= level__) {
            return this->out();
        } else {
            return utils::null_stream();
        }
    }

    inline rte::ostream out(int level__, const char* label__) const
    {
        if (this->verbosity() >= level__) {
            return rte::ostream(this->out(), label__);
        } else {
            return rte::ostream(utils::null_stream(), label__);
        }
    }

    /// Print message from the stringstream.
    inline void message(int level__, char const* label__, std::stringstream const& s) const
    {
        if (this->verbosity() >= level__) {
            auto strings = ::rte::split(s.str());
            for (auto& e : strings) {
                this->out() << "[" << label__ << "] " << e << std::endl;
            }
        }
    }

    inline void set_periodic_function_ptr(std::string label__, periodic_function_ptr_t<double> ptr__)
    {
        pf_ext_ptr[label__] = ptr__;
    }

    inline auto periodic_function_ptr(std::string label__) const
    {
        periodic_function_ptr_t<double> const* ptr{nullptr};
        if (pf_ext_ptr.count(label__)) {
            ptr = &pf_ext_ptr.at(label__);
        }
        return ptr;
    }
};

} // namespace sirius

#endif
