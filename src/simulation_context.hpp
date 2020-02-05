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

/** \file simulation_context.hpp
 *
 *  \brief Contains definition and implementation of Simulation_context class.
 */

#ifndef __SIMULATION_CONTEXT_HPP__
#define __SIMULATION_CONTEXT_HPP__

#include <algorithm>

#include "simulation_parameters.hpp"
#include "mpi_grid.hpp"
#include "radial_integrals.hpp"
#include "utils/utils.hpp"
#include "Density/augmentation_operator.hpp"
#include "Potential/xc_functional.hpp"
#include "SDDK/GPU/acc.hpp"
#include "Symmetry/check_gvec.hpp"
#include "Symmetry/rotation.hpp"
#include "spfft/spfft.hpp"

#ifdef __GPU
extern "C" void generate_phase_factors_gpu(int num_gvec_loc__, int num_atoms__, int const* gvec__,
                                           double const* atom_pos__, double_complex* phase_factors__);
#endif

//#ifdef __GNUC__
//    #define __function_name__ __PRETTY_FUNCTION__
//#else
//    #define __function_name__ __func__
//#endif

#define __function_name__ __func__

namespace sirius {

/// Utility function to print a CPU and GPU memory utilization.
void print_memory_usage(const char* file__, int line__);

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
    /// Storage for various memory pools.
    mutable std::map<memory_t, memory_pool> memory_pool_;

    /// Communicator for this simulation.
    Communicator const& comm_;

    /// Auxiliary communicator for the fine-grid FFT transformation.
    /** This communicator is orthogonal to the FFT communicator for density and potential within the full
     *  communicator of the simulation context. In other words, comm_ortho_fft_ \otimes comm_fft() = ctx_.comm() */
    //Communicator comm_ortho_fft_{MPI_COMM_SELF};

    /// Auxiliary communicator for the coarse-grid FFT transformation.
    Communicator comm_ortho_fft_coarse_;

    /// Communicator, which is orthogonal to comm_fft_coarse within a band communicator.
    /** This communicator is used in reshuffling the wave-functions for the FFT-friendly distribution. It will be
        used to parallelize application of local Hamiltonian over bands. */
    Communicator comm_band_ortho_fft_coarse_;

    /// Unit cell of the simulation.
    Unit_cell unit_cell_;

    /// MPI grid for this simulation.
    std::unique_ptr<MPI_grid> mpi_grid_;

    /// 2D BLACS grid for distributed linear algebra operations.
    std::unique_ptr<BLACS_grid> blacs_grid_;

    /// Grid descriptor for the fine-grained FFT transform.
    sddk::FFT3D_grid fft_grid_;

    /// Fine-grained FFT for density and potential.
    /** This is the FFT driver to transform periodic functions such as density and potential on the fine-grained
     *  FFT grid. The transformation is parallel. */
    std::unique_ptr<spfft::Transform> spfft_transform_;
    std::unique_ptr<spfft::Grid> spfft_grid_;

    /// Grid descriptor for the coarse-grained FFT transform.
    sddk::FFT3D_grid fft_coarse_grid_;

    /// Coarse-grained FFT for application of local potential and density summation.
    std::unique_ptr<spfft::Transform> spfft_transform_coarse_;
    std::unique_ptr<spfft::Grid> spfft_grid_coarse_;

    /// G-vectors within the Gmax cutoff.
    std::unique_ptr<Gvec> gvec_;

    std::unique_ptr<Gvec_partition> gvec_partition_;

    /// G-vectors within the 2 * |Gmax^{WF}| cutoff.
    std::unique_ptr<Gvec> gvec_coarse_;

    std::unique_ptr<Gvec_partition> gvec_coarse_partition_;

    std::unique_ptr<Gvec_shells> remap_gvec_;

    /// Creation time of the parameters.
    timeval start_time_;

    /// A tag string based on the the starting time.
    std::string start_time_tag_;

    /// 1D phase factors for each atom coordinate and G-vector index.
    sddk::mdarray<double_complex, 3> phase_factors_;

    /// 1D phase factors of the symmetry operations.
    sddk::mdarray<double_complex, 3> sym_phase_factors_;

    /// Phase factors for atom types.
    sddk::mdarray<double_complex, 2> phase_factors_t_;

    /// Lattice coordinats of G-vectors in a GPU-friendly ordering.
    sddk::mdarray<int, 2> gvec_coord_;

    /// Theta and phi angles of G-vectors in GPU-friendly ordering.
    sddk::mdarray<double, 2> gvec_tp_;

    /// Volume of the initial unit cell.
    /** This is needed to estimate the new cutoff for radial integrals. */
    double omega0_;

    /// Radial integrals of beta-projectors.
    std::unique_ptr<Radial_integrals_beta<false>> beta_ri_;

    /// Radial integrals of beta-projectors with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_beta<true>> beta_ri_djl_;

    /// Radial integrals of augmentation operator.
    std::unique_ptr<Radial_integrals_aug<false>> aug_ri_;

    /// Radial integrals of augmentation operator with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_aug<true>> aug_ri_djl_;

    /// Radial integrals of atomic wave-functions.
    std::unique_ptr<Radial_integrals_atomic_wf<false>> atomic_wf_ri_;

    /// Radial integrals of atomic wave-functions with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_atomic_wf<true>> atomic_wf_ri_djl_;

    /// Radial integrals of hubbard wave-functions.
    std::unique_ptr<Radial_integrals_atomic_wf<false>> hubbard_wf_ri_;

    /// Radial integrals of hubbard wave-functions with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_atomic_wf<true>> hubbard_wf_ri_djl_;

    /// Radial integrals of pseudo-core charge density.
    std::unique_ptr<Radial_integrals_rho_core_pseudo<false>> ps_core_ri_;

    /// Radial integrals of pseudo-core charge density with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_rho_core_pseudo<true>> ps_core_ri_djl_;

    /// Radial integrals of total pseudo-charge density.
    std::unique_ptr<Radial_integrals_rho_pseudo> ps_rho_ri_;

    /// Radial integrals of the local part of pseudopotential.
    std::unique_ptr<Radial_integrals_vloc<false>> vloc_ri_;

    /// Radial integrals of the local part of pseudopotential with derivatives of spherical Bessel functions.
    std::unique_ptr<Radial_integrals_vloc<true>> vloc_ri_djl_;

    /// List of real-space point indices for each of the atoms.
    std::vector<std::vector<std::pair<int, double>>> atoms_to_grid_idx_;

    /// Plane wave expansion coefficients of the step function.
    sddk::mdarray<double_complex, 1> theta_pw_;

    /// Step function on the real-space grid.
    sddk::mdarray<double, 1> theta_;

    /// Augmentation operator for each atom type.
    /** The augmentation operator is used by Density, Potential, Q_operator, and Non_local_functor classes. */
    std::vector<std::unique_ptr<Augmentation_operator>> augmentation_op_;

    /// Standard eigen-value problem solver.
    std::unique_ptr<Eigensolver> std_evp_solver_;

    /// Generalized eigen-value problem solver.
    std::unique_ptr<Eigensolver> gen_evp_solver_;

    /// Type of host memory (pagable or page-locked) for the arrays that participate in host-to-device memory copy.
    memory_t host_memory_t_{memory_t::none};

    /// Type of preferred memory for wave-functions and related arrays.
    memory_t preferred_memory_t_{memory_t::none};

    /// Type of preferred memory for auxiliary wave-functions of the iterative solver.
    memory_t aux_preferred_memory_t_{memory_t::none};

    /// Type of BLAS linear algebra library.
    linalg_t blas_linalg_t_{linalg_t::none};

    mutable double evp_work_count_{0};
    mutable int num_loc_op_applied_{0};

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
    /// Create a simulation context with an explicit communicator and load parameters from JSON string or JSON file.
    Simulation_context(std::string const& str__, Communicator const& comm__)
        : comm_(comm__)
        , unit_cell_(*this, comm_)
    {
        start();
        import(str__);
        unit_cell_.import(unit_cell_input_);
    }

    /// Create an empty simulation context with an explicit communicator.
    Simulation_context(Communicator const& comm__ = Communicator::world())
        : comm_(comm__)
        , unit_cell_(*this, comm_)
    {
        start();
    }

    /// Create a simulation context with world communicator and load parameters from JSON string or JSON file.
    Simulation_context(std::string const& str__)
        : comm_(Communicator::world())
        , unit_cell_(*this, comm_)
    {
        start();
        import(str__);
        unit_cell_.import(unit_cell_input_);
    }

    /// Destructor.
    ~Simulation_context()
    {
        if (!comm().is_finalized()) {
            this->print_memory_usage(__FILE__, __LINE__);
        }
    }

    /// Initialize the similation (can only be called once).
    void initialize();

    void print_info() const;

    /// Print the memory usage.
    void print_memory_usage(const char* file__, int line__);

    /// Print message from the root rank.
    template <typename... Args>
    inline void message(int level__, char const* label__, Args... args) const
    {
        if (this->comm().rank() == 0 && this->control().verbosity_ >= level__) {
            if (label__) {
                std::printf("[%s] ", label__);
            }
            std::printf(args...);
        }
    }

    /// Update context after setting new lattice vectors or atomic coordinates.
    void update();

    std::vector<std::pair<int, double>> const& atoms_to_grid_idx_map(int ia__) const
    {
        return atoms_to_grid_idx_[ia__];
    };

    Unit_cell& unit_cell()
    {
        return unit_cell_;
    }

    Unit_cell const& unit_cell() const
    {
        return unit_cell_;
    }

    Gvec const& gvec() const
    {
        return *gvec_;
    }

    Gvec_partition const& gvec_partition() const
    {
        return *gvec_partition_;
    }

    Gvec const& gvec_coarse() const
    {
        return *gvec_coarse_;
    }

    Gvec_partition const& gvec_coarse_partition() const
    {
        return *gvec_coarse_partition_;
    }

    Gvec_shells const& remap_gvec() const
    {
        return *remap_gvec_;
    }

    BLACS_grid const& blacs_grid() const
    {
        return *blacs_grid_;
    }

    /// Total communicator of the simulation.
    Communicator const& comm() const
    {
        return comm_;
    }

    /// Communicator between k-points.
    /** This communicator is used to split k-points */
    Communicator const& comm_k() const
    {
        /* 1st dimension of the MPI grid is used for k-point distribution */
        return mpi_grid_->communicator(1 << 0);
    }

    /// Band parallelization communicator.
    /** This communicator is used to parallelize the band problem. However it is not necessarily used
        to create the BLACS grid. Diagonalization might be sequential. */
    Communicator const& comm_band() const
    {
        /* 2nd and 3rd dimensions of the MPI grid are used for parallelization inside k-point */
        return mpi_grid_->communicator(1 << 1 | 1 << 2);
    }

    /// Communicator of the dense FFT grid.
    /** This communicator is passed to the spfft::Transform constructor. */
    Communicator const& comm_fft() const
    {
        /* use entire communicator of the simulation */
        return comm();
    }

    Communicator const& comm_ortho_fft() const
    {
        return Communicator::self();
    }

    /// Communicator of the coarse FFT grid.
    /** This communicator is passed to the spfft::Transform constructor. */
    Communicator const& comm_fft_coarse() const
    {
        if (control().fft_mode_ == "serial") {
            return Communicator::self();
        } else {
            return comm_band();
        }
    }

    Communicator const& comm_ortho_fft_coarse() const
    {
        return comm_ortho_fft_coarse_;
    }

    Communicator const& comm_band_ortho_fft_coarse() const
    {
        return comm_band_ortho_fft_coarse_;
    }

    void create_storage_file() const;

    inline std::string const& start_time_tag() const
    {
        return start_time_tag_;
    }

    inline ev_solver_t std_evp_solver_type() const
    {
        return get_ev_solver_t(std_evp_solver_name());
    }

    inline ev_solver_t gen_evp_solver_type() const
    {
        return get_ev_solver_t(gen_evp_solver_name());
    }

    inline Eigensolver& std_evp_solver()
    {
        return* std_evp_solver_;
    }

    inline Eigensolver& gen_evp_solver()
    {
        return* gen_evp_solver_;
    }

    /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
    inline double_complex gvec_phase_factor(vector3d<int> G__, int ia__) const
    {
        return phase_factors_(0, G__[0], ia__) * phase_factors_(1, G__[1], ia__) * phase_factors_(2, G__[2], ia__);
    }

    /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
    inline double_complex gvec_phase_factor(int ig__, int ia__) const
    {
        return gvec_phase_factor(gvec().gvec(ig__), ia__);
    }

    inline sddk::mdarray<int, 2> const& gvec_coord() const
    {
        return gvec_coord_;
    }

    inline sddk::mdarray<double, 2> const& gvec_tp() const
    {
        return gvec_tp_;
    }

    /// Generate phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$ for all atoms of a given type.
    void generate_phase_factors(int iat__, mdarray<double_complex, 2>& phase_factors__) const;

    /// Make periodic function out of form factors.
    /** Return vector of plane-wave coefficients */ // TODO: return mdarray
    template <index_domain_t index_domain, typename F>
    inline std::vector<double_complex> make_periodic_function(F&& form_factors__) const
    {
        PROFILE("sirius::Simulation_context::make_periodic_function");

        double fourpi_omega = fourpi / unit_cell_.omega();

        int ngv = (index_domain == index_domain_t::local) ? gvec().count() : gvec().num_gvec();
        std::vector<double_complex> f_pw(ngv, double_complex(0, 0));

        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            /* global index of G-vector */
            int ig   = gvec().offset() + igloc;
            double g = gvec().gvec_len(ig);

            int j = (index_domain == index_domain_t::local) ? igloc : ig;
            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                f_pw[j] += fourpi_omega * std::conj(phase_factors_t_(igloc, iat)) * form_factors__(iat, g);
            }
        }

        if (index_domain == index_domain_t::global) {
            comm_.allgather(&f_pw[0], gvec().offset(), gvec().count());
        }

        return f_pw;
    }

    /// Compute values of spherical Bessel functions at MT boundary.
    sddk::mdarray<double, 3> generate_sbessel_mt(int lmax__) const;

    /// Generate complex spherical harmoics for the local set of G-vectors.
    matrix<double_complex> generate_gvec_ylm(int lmax__);

    /// Sum over the plane-wave coefficients and spherical harmonics that apperas in Poisson solver and finding of the
    /// MT boundary values.
    /** The following operation is performed:
     *  \f[
     *    q_{\ell m}^{\alpha} = \sum_{\bf G} 4\pi \rho({\bf G})
     *     e^{i{\bf G}{\bf r}_{\alpha}}i^{\ell}f_{\ell}^{\alpha}(G) Y_{\ell m}^{*}(\hat{\bf G})
     *  \f]
     */
    mdarray<double_complex, 2> sum_fg_fl_yg(int lmax__, double_complex const* fpw__, mdarray<double, 3>& fl__,
                                            matrix<double_complex>& gvec_ylm__);

    inline Radial_integrals_beta<false> const& beta_ri() const
    {
        return *beta_ri_;
    }

    inline Radial_integrals_beta<true> const& beta_ri_djl() const
    {
        return *beta_ri_djl_;
    }

    inline Radial_integrals_aug<false> const& aug_ri() const
    {
        return *aug_ri_;
    }

    inline Radial_integrals_aug<true> const& aug_ri_djl() const
    {
        return *aug_ri_djl_;
    }

    inline Radial_integrals_atomic_wf<false> const& atomic_wf_ri() const
    {
        return *atomic_wf_ri_;
    }

    inline Radial_integrals_atomic_wf<true> const& atomic_wf_djl() const
    {
        return *atomic_wf_ri_djl_;
    }

    inline Radial_integrals_atomic_wf<false> const& hubbard_wf_ri() const
    {
        return *hubbard_wf_ri_;
    }

    inline Radial_integrals_atomic_wf<true> const& hubbard_wf_djl() const
    {
        return *hubbard_wf_ri_djl_;
    }

    inline Radial_integrals_rho_core_pseudo<false> const& ps_core_ri() const
    {
        return *ps_core_ri_;
    }

    inline Radial_integrals_rho_core_pseudo<true> const& ps_core_ri_djl() const
    {
        return *ps_core_ri_djl_;
    }

    inline Radial_integrals_rho_pseudo const& ps_rho_ri() const
    {
        return *ps_rho_ri_;
    }

    inline Radial_integrals_vloc<false> const& vloc_ri() const
    {
        return *vloc_ri_;
    }

    inline Radial_integrals_vloc<true> const& vloc_ri_djl() const
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

    mdarray<double_complex, 3> const& sym_phase_factors() const
    {
        return sym_phase_factors_;
    }

    /// Return a reference to a memory pool.
    /** A memory pool is created when this function called for the first time. */
    memory_pool& mem_pool(memory_t M__) const
    {
        if (memory_pool_.count(M__) == 0) {
            memory_pool_.emplace(M__, std::move(memory_pool(M__)));
        }
        return memory_pool_.at(M__);
    }

    /// Get a default memory pool for a given device.
    memory_pool& mem_pool(device_t dev__)
    {
        switch (dev__) {
            case device_t::CPU: {
                return mem_pool(memory_t::host);
                break;
            }
            case device_t::GPU: {
                return mem_pool(memory_t::device);
                break;
            }
        }
        return mem_pool(memory_t::host); // make compiler happy
    }

    inline bool initialized() const
    {
        return initialized_;
    }

    /// Return plane-wave coefficient of the step function.
    inline double_complex const& theta_pw(int ig__) const
    {
        return theta_pw_[ig__];
    }

    /// Return the value of the step function for the grid point ir.
    inline double theta(int ir__) const
    {
        return theta_[ir__];
    }

    /// Returns a constant pointer to the augmentation operator of a given atom type.
    inline Augmentation_operator const* augmentation_op(int iat__) const
    {
        return augmentation_op_[iat__].get();
    }

    inline Augmentation_operator* augmentation_op(int iat__)
    {
        return augmentation_op_[iat__].get();
    }

    /// Type of the host memory for arrays used in linear algebra operations.
    inline memory_t host_memory_t() const
    {
        return host_memory_t_;
    }

    /// Type of preferred memory for the storage of hpsi, spsi, residuals and and related arrays.
    inline memory_t preferred_memory_t() const
    {
        return preferred_memory_t_;
    }

    /// Type of preferred memory for the storage of auxiliary wave-functions.
    inline memory_t aux_preferred_memory_t() const
    {
        return aux_preferred_memory_t_;
    }

    /// Linear algebra driver for the BLAS operations.
    linalg_t blas_linalg_t() const
    {
        return blas_linalg_t_;
    }

    /// Split local set of G-vectors into chunks.
    splindex<splindex_t::block> split_gvec_local() const;

    /// Set the size of the fine-grained FFT grid.
    void fft_grid_size(std::array<int, 3> fft_grid_size__)
    {
        settings_input_.fft_grid_size_ = fft_grid_size__;
    }

    spfft::Grid& spfft_grid_coarse()
    {
        return *spfft_grid_coarse_;
    }

    spfft::Transform& spfft()
    {
        return *spfft_transform_;
    }

    spfft::Transform const& spfft() const
    {
        return *spfft_transform_;
    }

    spfft::Transform& spfft_coarse()
    {
        return *spfft_transform_coarse_;
    }

    spfft::Transform const& spfft_coarse() const
    {
        return *spfft_transform_coarse_;
    }

    sddk::FFT3D_grid const& fft_grid() const
    {
        return fft_grid_;
    }

    sddk::FFT3D_grid const& fft_coarse_grid() const
    {
        return fft_coarse_grid_;
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

};


} // namespace sirius

#endif
