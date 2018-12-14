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

#include "version.hpp"
#include "simulation_parameters.hpp"
#include "mpi_grid.hpp"
#include "radial_integrals.hpp"
#include "utils/utils.hpp"
#include "Density/augmentation_operator.hpp"

#ifdef __GPU
#include "SDDK/GPU/cuda.hpp"

extern "C" void generate_phase_factors_gpu(int num_gvec_loc__, int num_atoms__, int const* gvec__,
                                           double const* atom_pos__, double_complex* phase_factors__);
#endif

namespace sirius {

/// Utility function to print a CPU and GPU memory utilization.
inline void print_memory_usage(const char* file__, int line__)
{
    size_t VmRSS, VmHWM;
    utils::get_proc_status(&VmHWM, &VmRSS);

    std::vector<char> str(2048);

    int n = snprintf(&str[0], 2048, "[rank%04i at line %i of file %s]", Communicator::world().rank(), line__, file__);

    n += snprintf(&str[n], 2048, " VmHWM: %i Mb, VmRSS: %i Mb", static_cast<int>(VmHWM >> 20),
                  static_cast<int>(VmRSS >> 20));

#ifdef __GPU
    size_t gpu_mem = acc::get_free_mem();
    n += snprintf(&str[n], 2048, ", GPU free memory: %i Mb", static_cast<int>(gpu_mem >> 20));
#endif

    printf("%s\n", &str[0]);
}

#define MEMORY_USAGE_INFO() print_memory_usage(__FILE__, __LINE__);

/// Simulation context is a set of parameters and objects describing a single simulation.
/** The order of initialization of the simulation context is the following: first, the default parameter
 *  values are set in the constructor, then (optionally) import() method is called and the parameters are
 *  overwritten with the those from the input file, and finally, the user sets the values with setter metods.
 *  Then the unit cell can be populated and the context can be initialized. */
class Simulation_context : public Simulation_parameters
{
  private:
    /// Communicator for this simulation.
    Communicator const& comm_;

    /// Auxiliary communicator for the fine-grid FFT transformation.
    /** This communicator is orthogonal to the FFT communicator for density and potential within the full
     *  communicator of the simulation context. In other words, comm_ortho_fft_ \otimes comm_fft() = ctx_.comm() */
    Communicator comm_ortho_fft_;

    /// Auxiliary communicator for the coarse-grid FFT transformation.
    Communicator comm_ortho_fft_coarse_;

    Communicator comm_band_ortho_fft_coarse_;

    /// Unit cell of the simulation.
    Unit_cell unit_cell_;

    /// MPI grid for this simulation.
    std::unique_ptr<MPI_grid> mpi_grid_;

    /// 2D BLACS grid for distributed linear algebra operations.
    std::unique_ptr<BLACS_grid> blacs_grid_;

    /// Fine-grained FFT for density and potential.
    /** This is the FFT driver to transform periodic functions such as density and potential on the fine-grained
     *  FFT grid. The transformation is parallel. */
    std::unique_ptr<FFT3D> fft_;

    /// Coarse-grained FFT for application of local potential and density summation.
    std::unique_ptr<FFT3D> fft_coarse_;

    /// G-vectors within the Gmax cutoff.
    std::unique_ptr<Gvec> gvec_;

    std::unique_ptr<Gvec_partition> gvec_partition_;

    /// G-vectors within the 2 * |Gmax^{WF}| cutoff.
    std::unique_ptr<Gvec> gvec_coarse_;

    std::unique_ptr<Gvec_partition> gvec_coarse_partition_;

    std::unique_ptr<remap_gvec_to_shells> remap_gvec_;

    /// Creation time of the parameters.
    timeval start_time_;

    /// A tag string based on the the starting time.
    std::string start_time_tag_;

    /// 1D phase factors for each atom coordinate and G-vector index.
    mdarray<double_complex, 3> phase_factors_;

    /// 1D phase factors of the symmetry operations.
    mdarray<double_complex, 3> sym_phase_factors_;

    /// Phase factors for atom types.
    mdarray<double_complex, 2> phase_factors_t_;

    /// Lattice coordinats of G-vectors in a GPU-friendly ordering.
    mdarray<int, 2> gvec_coord_;

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

    /// Storage for various memory pools.
    std::map<memory_t, memory_pool> memory_pool_;

    /// Plane wave expansion coefficients of the step function.
    mdarray<double_complex, 1> theta_pw_;

    /// Step function on the real-space grid.
    mdarray<double, 1> theta_;

    /// Augmentation operator for each atom type.
    std::vector<Augmentation_operator> augmentation_op_;

    memory_t host_memory_t_{memory_t::none};
    memory_t preferred_memory_t_{memory_t::none};
    linalg_t blas_linalg_t_{linalg_t::none};

    /// True if the context is already initialized.
    bool initialized_{false};

    /// Initialize FFT drivers.
    inline void init_fft()
    {
        PROFILE("sirius::Simulation_context::init_fft");

        auto rlv = unit_cell_.reciprocal_lattice_vectors();

        if (!(control().fft_mode_ == "serial" || control().fft_mode_ == "parallel")) {
            TERMINATE("wrong FFT mode");
        }

        /* create FFT driver for dense mesh (density and potential) */
        fft_ = std::unique_ptr<FFT3D>(new FFT3D(find_translations(pw_cutoff(), rlv), comm_fft(), processing_unit()));

        /* create FFT driver for coarse mesh */
        fft_coarse_ = std::unique_ptr<FFT3D>(
            new FFT3D(find_translations(2 * gk_cutoff(), rlv), comm_fft_coarse(), processing_unit()));

        /* create a list of G-vectors for corase FFT grid */
        gvec_coarse_ = std::unique_ptr<Gvec>(new Gvec(rlv, gk_cutoff() * 2, comm(), control().reduce_gvec_));

        gvec_coarse_partition_ = std::unique_ptr<Gvec_partition>(
            new Gvec_partition(*gvec_coarse_, comm_fft_coarse(), comm_ortho_fft_coarse()));

        /* create a list of G-vectors for dense FFT grid; G-vectors are divided between all available MPI ranks.*/
        gvec_ = std::unique_ptr<Gvec>(new Gvec(pw_cutoff(), *gvec_coarse_));

        gvec_partition_ = std::unique_ptr<Gvec_partition>(new Gvec_partition(*gvec_, comm_fft(), comm_ortho_fft()));

        remap_gvec_ = std::unique_ptr<remap_gvec_to_shells>(new remap_gvec_to_shells(comm(), gvec()));

        /* prepare fine-grained FFT driver for the entire simulation */
        fft_->prepare(*gvec_partition_);
    }

    /// Initialize communicators.
    inline void init_comm()
    {
        PROFILE("sirius::Simulation_context::init_comm");

        /* check MPI grid dimensions and set a default grid if needed */
        if (!control().mpi_grid_dims_.size()) {
            set_mpi_grid_dims({1, 1});
        }
        if (control().mpi_grid_dims_.size() != 2) {
            TERMINATE("wrong MPI grid");
        }

        int npr = control_input_.mpi_grid_dims_[0];
        int npc = control_input_.mpi_grid_dims_[1];
        int npb = npr * npc;
        int npk = comm_.size() / npb;
        if (npk * npb != comm_.size()) {
            std::stringstream s;
            s << "Can't divide " << comm_.size() << " ranks into groups of size " << npb;
            TERMINATE(s);
        }

        /* setup MPI grid */
        mpi_grid_ = std::unique_ptr<MPI_grid>(new MPI_grid({npk, npc, npr}, comm_));

        comm_ortho_fft_ = comm().split(comm_fft().rank());

        comm_ortho_fft_coarse_ = comm().split(comm_fft_coarse().rank());

        comm_band_ortho_fft_coarse_ = comm_band().split(comm_fft_coarse().rank());
    }

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
    void init_step_function()
    {
        auto v = make_periodic_function<index_domain_t::global>([&](int iat, double g) {
            auto R = unit_cell().atom_type(iat).mt_radius();
            if (g < 1e-12) {
                return std::pow(R, 3) / 3.0;
            } else {
                return (std::sin(g * R) - g * R * std::cos(g * R)) / std::pow(g, 3);
            }
        });

        theta_    = mdarray<double, 1>(fft().local_size());
        theta_pw_ = mdarray<double_complex, 1>(gvec().num_gvec());

        for (int ig = 0; ig < gvec().num_gvec(); ig++) {
            theta_pw_[ig] = -v[ig];
        }
        theta_pw_[0] += 1.0;

        std::vector<double_complex> ftmp(gvec_partition().gvec_count_fft());
        for (int i = 0; i < gvec_partition().gvec_count_fft(); i++) {
            ftmp[i] = theta_pw_[gvec_partition().idx_gvec(i)];
        }
        fft().transform<1>(ftmp.data());
        fft().output(&theta_[0]);

        double vit{0};
        for (int i = 0; i < fft().local_size(); i++) {
            vit += theta_[i];
        }
        vit *= (unit_cell().omega() / fft().size());
        fft().comm().allreduce(&vit, 1);

        if (std::abs(vit - unit_cell().volume_it()) > 1e-10) {
            std::stringstream s;
            s << "step function gives a wrong volume for IT region" << std::endl
              << "  difference with exact value : " << std::abs(vit - unit_cell().volume_it());
            if (comm().rank() == 0) {
                WARNING(s);
            }
        }
        if (control().print_checksum_) {
            double_complex z1 = theta_pw_.checksum();
            double d1         = theta_.checksum();
            fft().comm().allreduce(&d1, 1);
            if (comm().rank() == 0) {
                utils::print_checksum("theta", d1);
                utils::print_checksum("theta_pw", z1);
            }
        }
    }

    /// Get the stsrting time stamp.
    void start()
    {
        gettimeofday(&start_time_, NULL);
        start_time_tag_ = utils::timestamp("%Y%m%d_%H%M%S");
    }

    /// Find a list of real-space grid points around each atom.
    void init_atoms_to_grid_idx(double R__)
    {
        PROFILE("sirius::Simulation_context::init_atoms_to_grid_idx");

        atoms_to_grid_idx_.resize(unit_cell_.num_atoms());

        vector3d<double> delta(1.0 / fft_->size(0), 1.0 / fft_->size(1), 1.0 / fft_->size(2));

        int z_off = fft_->offset_z();
        vector3d<int> grid_beg(0, 0, z_off);
        vector3d<int> grid_end(fft_->size(0), fft_->size(1), z_off + fft_->local_size_z());
        std::vector<vector3d<double>> verts_cart{{-R__, -R__, -R__}, {R__, -R__, -R__}, {-R__, R__, -R__},
                                                 {R__, R__, -R__},   {-R__, -R__, R__}, {R__, -R__, R__},
                                                 {-R__, R__, R__},   {R__, R__, R__}};

        auto bounds_box = [&](vector3d<double> pos) {
            std::vector<vector3d<double>> verts;

            /* pos is a position of atom */
            for (auto v : verts_cart) {
                verts.push_back(pos + unit_cell_.get_fractional_coordinates(v));
            }

            std::pair<vector3d<int>, vector3d<int>> bounds_ind;

            for (int x : {0, 1, 2}) {
                std::sort(verts.begin(), verts.end(),
                          [x](vector3d<double>& a, vector3d<double>& b) { return a[x] < b[x]; });
                bounds_ind.first[x]  = std::max(static_cast<int>(verts[0][x] / delta[x]) - 1, grid_beg[x]);
                bounds_ind.second[x] = std::min(static_cast<int>(verts[5][x] / delta[x]) + 1, grid_end[x]);
            }

            return bounds_ind;
        };

        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {

            std::vector<std::pair<int, double>> atom_to_ind_map;

            for (int t0 = -1; t0 <= 1; t0++) {
                for (int t1 = -1; t1 <= 1; t1++) {
                    for (int t2 = -1; t2 <= 1; t2++) {
                        auto pos = unit_cell_.atom(ia).position() + vector3d<double>(t0, t1, t2);

                        /* find the small box around this atom */
                        auto box = bounds_box(pos);

                        for (int j0 = box.first[0]; j0 < box.second[0]; j0++) {
                            for (int j1 = box.first[1]; j1 < box.second[1]; j1++) {
                                for (int j2 = box.first[2]; j2 < box.second[2]; j2++) {
                                    auto v = pos - vector3d<double>(delta[0] * j0, delta[1] * j1, delta[2] * j2);
                                    auto r  = unit_cell_.get_cartesian_coordinates(v).length();
                                    if (r < R__) {
                                        auto ir = fft_->index_by_coord(j0, j1, j2 - z_off);
                                        atom_to_ind_map.push_back({ir, r});
                                    }
                                }
                            }
                        }
                    }
                }
            }

            atoms_to_grid_idx_[ia] = std::move(atom_to_ind_map);
        }
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
    Simulation_context(Communicator const& comm__)
        : comm_(comm__)
        , unit_cell_(*this, comm_)
    {
        start();
    }

    /// Create an empty simulation context with world communicator.
    Simulation_context()
        : comm_(Communicator::world())
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

    /// Initialize the similation (can only be called once).
    void initialize();

    void print_info() const;

    /// Print the memory usage.
    inline void print_memory_usage(const char* file__, int line__)
    {
        if (comm().rank() == 0 && control().print_memory_usage_) {
            sirius::print_memory_usage(file__, line__);

            printf("memory_t::host pool:        %li %li %li %li\n", mem_pool(memory_t::host).total_size() >> 20,
                                                                    mem_pool(memory_t::host).free_size() >> 20,
                                                                    mem_pool(memory_t::host).num_blocks(),
                                                                    mem_pool(memory_t::host).num_stored_ptr());

            printf("memory_t::host_pinned pool: %li %li %li %li\n", mem_pool(memory_t::host_pinned).total_size() >> 20,
                                                                    mem_pool(memory_t::host_pinned).free_size() >> 20,
                                                                    mem_pool(memory_t::host_pinned).num_blocks(),
                                                                    mem_pool(memory_t::host_pinned).num_stored_ptr());

            printf("memory_t::device pool:      %li %li %li %li\n", mem_pool(memory_t::device).total_size() >> 20,
                                                                    mem_pool(memory_t::device).free_size() >> 20,
                                                                    mem_pool(memory_t::device).num_blocks(),
                                                                    mem_pool(memory_t::device).num_stored_ptr());
        }
    }

    /// Update context after setting new lattice vectors or atomic coordinates.
    void update()
    {
        PROFILE("sirius::Simulation_context::update");

        gvec_->lattice_vectors(unit_cell().reciprocal_lattice_vectors());
        gvec_coarse_->lattice_vectors(unit_cell().reciprocal_lattice_vectors());

        unit_cell().update();

        if (unit_cell_.num_atoms() != 0 && use_symmetry() && control().verification_ >= 1) {
            unit_cell_.symmetry().check_gvec_symmetry(gvec(), comm());
            if (!full_potential()) {
                unit_cell_.symmetry().check_gvec_symmetry(gvec_coarse(), comm());
            }
        }

        init_atoms_to_grid_idx(control().rmt_max_);

        std::pair<int, int> limits(0, 0);
        for (int x : {0, 1, 2}) {
            limits.first  = std::min(limits.first, fft().limits(x).first);
            limits.second = std::max(limits.second, fft().limits(x).second);
        }

        phase_factors_ =
            mdarray<double_complex, 3>(3, limits, unit_cell().num_atoms(), memory_t::host, "phase_factors_");
        #pragma omp parallel for
        for (int i = limits.first; i <= limits.second; i++) {
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                auto pos = unit_cell_.atom(ia).position();
                for (int x : {0, 1, 2}) {
                    phase_factors_(x, i, ia) = std::exp(double_complex(0.0, twopi * (i * pos[x])));
                }
            }
        }

        phase_factors_t_ = mdarray<double_complex, 2>(gvec().count(), unit_cell().num_atom_types());
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            /* global index of G-vector */
            int ig = gvec().offset() + igloc;
            for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
                double_complex z(0, 0);
                for (int ia = 0; ia < unit_cell().atom_type(iat).num_atoms(); ia++) {
                    z += gvec_phase_factor(ig, unit_cell().atom_type(iat).atom_id(ia));
                }
                phase_factors_t_(igloc, iat) = z;
            }
        }

        if (use_symmetry()) {
            sym_phase_factors_ = mdarray<double_complex, 3>(3, limits, unit_cell().symmetry().num_mag_sym());

            #pragma omp parallel for
            for (int i = limits.first; i <= limits.second; i++) {
                for (int isym = 0; isym < unit_cell().symmetry().num_mag_sym(); isym++) {
                    auto t = unit_cell().symmetry().magnetic_group_symmetry(isym).spg_op.t;
                    for (int x : {0, 1, 2}) {
                        sym_phase_factors_(x, i, isym) = std::exp(double_complex(0.0, twopi * (i * t[x])));
                    }
                }
            }
        }
#if defined(__GPU)
        if (processing_unit() == device_t::GPU) {
            acc::set_device();
            gvec_coord_ = mdarray<int, 2>(gvec().count(), 3, memory_t::host, "gvec_coord_");
            gvec_coord_.allocate(memory_t::device);
            for (int igloc = 0; igloc < gvec().count(); igloc++) {
                int ig = gvec().offset() + igloc;
                auto G = gvec().gvec(ig);
                for (int x : {0, 1, 2}) {
                    gvec_coord_(igloc, x) = G[x];
                }
            }
            gvec_coord_.copy_to(memory_t::device);
        }
#endif
        if (full_potential()) {
            init_step_function();
        }

        if (!full_potential()) {
            augmentation_op_.clear();
            memory_pool* mp{nullptr};
            switch (processing_unit()) {
                case CPU: {
                    mp = &mem_pool(memory_t::host);
                    break;
                }
                case GPU: {
                    mp = &mem_pool(memory_t::host_pinned);
                    break;
                }
            }
            /* create augmentation operator Q_{xi,xi'}(G) here */
            for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
                augmentation_op_.push_back(
                    std::move(Augmentation_operator(unit_cell().atom_type(iat), gvec(), comm())));
                augmentation_op_.back().generate_pw_coeffs(aug_ri(), *mp);
            }
        }
    }

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

    inline FFT3D& fft() const
    {
        return *fft_;
    }

    inline FFT3D& fft_coarse() const
    {
        return *fft_coarse_;
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

    remap_gvec_to_shells const& remap_gvec() const
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
    Communicator const& comm_fft() const
    {
        /* 3rd dimension of MPI grid is used */
        return mpi_grid_->communicator(1 << 2);
    }

    Communicator const& comm_ortho_fft() const
    {
        return comm_ortho_fft_;
    }

    /// Communicator of the coarse FFT grid.
    Communicator const& comm_fft_coarse() const
    {
        if (control().fft_mode_ == "serial") {
            return Communicator::self();
        } else {
            return comm_fft();
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

    void create_storage_file() const
    {
        if (comm_.rank() == 0) {
            /* create new hdf5 file */
            HDF5_tree fout(storage_file_name, hdf5_access_t::truncate);
            fout.create_node("parameters");
            fout.create_node("effective_potential");
            fout.create_node("effective_magnetic_field");
            fout.create_node("density");
            fout.create_node("magnetization");

            for (int j = 0; j < num_mag_dims(); j++) {
                fout["magnetization"].create_node(j);
                fout["effective_magnetic_field"].create_node(j);
            }

            fout["parameters"].write("num_spins", num_spins());
            fout["parameters"].write("num_mag_dims", num_mag_dims());
            fout["parameters"].write("num_bands", num_bands());

            mdarray<int, 2> gv(3, gvec().num_gvec());
            for (int ig = 0; ig < gvec().num_gvec(); ig++) {
                auto G = gvec().gvec(ig);
                for (int x : {0, 1, 2}) {
                    gv(x, ig) = G[x];
                }
            }
            fout["parameters"].write("num_gvec", gvec().num_gvec());
            fout["parameters"].write("gvec", gv);

            fout.create_node("unit_cell");
            fout["unit_cell"].create_node("atoms");
            for (int j = 0; j < unit_cell().num_atoms(); j++) {
                fout["unit_cell"]["atoms"].create_node(j);
                fout["unit_cell"]["atoms"][j].write("mt_basis_size", unit_cell().atom(j).mt_basis_size());
            }
        }
        comm_.barrier();
    }

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

    template <typename T>
    inline std::unique_ptr<Eigensolver<T>> std_evp_solver()
    {
        return std::move(Eigensolver_factory<T>(std_evp_solver_type()));
    }

    template <typename T>
    inline std::unique_ptr<Eigensolver<T>> gen_evp_solver()
    {
        return std::move(Eigensolver_factory<T>(gen_evp_solver_type()));
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

    inline mdarray<int, 2> const& gvec_coord() const
    {
        return gvec_coord_;
    }

    /// Generate phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$ for all atoms of a given type.
    inline void generate_phase_factors(int iat__, mdarray<double_complex, 2>& phase_factors__) const
    {
        PROFILE("sirius::Simulation_context::generate_phase_factors");
        int na = unit_cell_.atom_type(iat__).num_atoms();
        switch (processing_unit_) {
            case device_t::CPU: {
                #pragma omp parallel for
                for (int igloc = 0; igloc < gvec().count(); igloc++) {
                    int ig = gvec().offset() + igloc;
                    for (int i = 0; i < na; i++) {
                        int ia                    = unit_cell().atom_type(iat__).atom_id(i);
                        phase_factors__(igloc, i) = gvec_phase_factor(ig, ia);
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(__GPU)
                acc::set_device();
                generate_phase_factors_gpu(gvec().count(), na, gvec_coord().at(memory_t::device),
                                           unit_cell().atom_coord(iat__).at(memory_t::device), phase_factors__.at(memory_t::device));
#else
                TERMINATE_NO_GPU
#endif
                break;
            }
        }
    }

    /// Make periodic function out of form factors.
    /** Return vector of plane-wave coefficients */ // TODO: return mdarray
    template <index_domain_t index_domain>
    inline std::vector<double_complex> make_periodic_function(std::function<double(int, double)> form_factors__) const
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

        return std::move(f_pw);
    }

    /// Compute values of spherical Bessel functions at MT boundary.
    inline mdarray<double, 3> generate_sbessel_mt(int lmax__) const
    {
        PROFILE("sirius::Simulation_context::generate_sbessel_mt");

        mdarray<double, 3> sbessel_mt(lmax__ + 1, gvec().count(), unit_cell().num_atom_types());
        for (int iat = 0; iat < unit_cell().num_atom_types(); iat++) {
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec().count(); igloc++) {
                auto gv = gvec().gvec_cart<index_domain_t::local>(igloc);
                gsl_sf_bessel_jl_array(lmax__, gv.length() * unit_cell().atom_type(iat).mt_radius(),
                                       &sbessel_mt(0, igloc, iat));
            }
        }
        return std::move(sbessel_mt);
    }

    /// Generate complex spherical harmoics for the local set of G-vectors.
    inline matrix<double_complex> generate_gvec_ylm(int lmax__)
    {
        PROFILE("sirius::Simulation_context::generate_gvec_ylm");

        matrix<double_complex> gvec_ylm(utils::lmmax(lmax__), gvec().count(), memory_t::host, "gvec_ylm");
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec().count(); igloc++) {
            auto rtp = SHT::spherical_coordinates(gvec().gvec_cart<index_domain_t::local>(igloc));
            SHT::spherical_harmonics(lmax__, rtp[1], rtp[2], &gvec_ylm(0, igloc));
        }
        return std::move(gvec_ylm);
    }

    /// Sum over the plane-wave coefficients and spherical harmonics that apperas in Poisson solver and finding of the
    /// MT boundary values.
    /** The following operation is performed:
     *  \f[
     *    q_{\ell m}^{\alpha} = \sum_{\bf G} 4\pi \rho({\bf G})
     *     e^{i{\bf G}{\bf r}_{\alpha}}i^{\ell}f_{\ell}^{\alpha}(G) Y_{\ell m}^{*}(\hat{\bf G})
     *  \f]
     */
    inline mdarray<double_complex, 2> sum_fg_fl_yg(int lmax__, double_complex const* fpw__, mdarray<double, 3>& fl__,
                                                   matrix<double_complex>& gvec_ylm__)
    {
        PROFILE("sirius::Simulation_context::sum_fg_fl_yg");

        int ngv_loc = gvec().count();

        int na_max{0};
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            na_max = std::max(na_max, unit_cell_.atom_type(iat).num_atoms());
        }

        int lmmax = utils::lmmax(lmax__);
        /* resuling matrix */
        mdarray<double_complex, 2> flm(lmmax, unit_cell().num_atoms());

        matrix<double_complex> phase_factors;
        matrix<double_complex> zm;
        matrix<double_complex> tmp;

        switch (processing_unit()) {
            case device_t::CPU: {
                auto& mp = mem_pool(memory_t::host);
                phase_factors = matrix<double_complex>(mp, ngv_loc, na_max);
                zm = matrix<double_complex>(mp, lmmax, ngv_loc);
                tmp = matrix<double_complex>(mp, lmmax, na_max);
                break;
            }
            case device_t::GPU: {
                auto& mp = mem_pool(memory_t::host);
                auto& mpd = mem_pool(memory_t::device);
                phase_factors = matrix<double_complex>(nullptr, ngv_loc, na_max);
                phase_factors.allocate(mpd);
                zm = matrix<double_complex>(mp, lmmax, ngv_loc);
                zm.allocate(mpd);
                tmp = matrix<double_complex>(mp, lmmax, na_max);
                tmp.allocate(mpd);
                break;
            }
        }

        std::vector<double_complex> zil(lmax__ + 1);
        for (int l = 0; l <= lmax__; l++) {
            zil[l] = std::pow(double_complex(0, 1), l);
        }

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            int na = unit_cell_.atom_type(iat).num_atoms();
            generate_phase_factors(iat, phase_factors);
            utils::timer t1("sirius::Simulation_context::sum_fg_fl_yg|zm");
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < ngv_loc; igloc++) {
                for (int l = 0, lm = 0; l <= lmax__; l++) {
                    double_complex z = fourpi * fl__(l, igloc, iat) * zil[l] * fpw__[igloc];
                    for (int m = -l; m <= l; m++, lm++) {
                        zm(lm, igloc) = z * std::conj(gvec_ylm__(lm, igloc));
                    }
                }
            }
            t1.stop();
            utils::timer t2("sirius::Simulation_context::sum_fg_fl_yg|mul");
            switch (processing_unit()) {
                case device_t::CPU: {
                    linalg<CPU>::gemm(0, 0, lmmax, na, ngv_loc, zm.at(memory_t::host), zm.ld(), phase_factors.at(memory_t::host),
                                      phase_factors.ld(), tmp.at(memory_t::host), tmp.ld());
                    break;
                }
                case device_t::GPU: {
#if defined(__GPU)
                    zm.copy_to(memory_t::device);
                    linalg<GPU>::gemm(0, 0, lmmax, na, ngv_loc, zm.at(memory_t::device), zm.ld(), phase_factors.at(memory_t::device),
                                      phase_factors.ld(), tmp.at(memory_t::device), tmp.ld());
                    tmp.copy_to(memory_t::host);
#endif
                    break;
                }
            }
            t2.stop();

            for (int i = 0; i < na; i++) {
                int ia = unit_cell_.atom_type(iat).atom_id(i);
                std::copy(&tmp(0, i), &tmp(0, i) + lmmax, &flm(0, ia));
            }
        }

        comm().allreduce(&flm(0, 0), (int)flm.size());

        return std::move(flm);
    }

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
    double ewald_lambda() const
    {
        /* alpha = 1 / (2*sigma^2), selecting alpha here for better convergence */
        double lambda{1};
        double gmax = pw_cutoff();
        double upper_bound{0};
        double charge = unit_cell_.num_electrons();

        /* iterate to find lambda */
        do {
            lambda += 0.1;
            upper_bound =
                charge * charge * std::sqrt(2.0 * lambda / twopi) * std::erfc(gmax * std::sqrt(1.0 / (4.0 * lambda)));
        } while (upper_bound < 1e-8);

        if (lambda < 1.5) {
            std::stringstream s;
            s << "ewald_lambda(): pw_cutoff is too small";
            WARNING(s);
        }
        return lambda;
    }

    mdarray<double_complex, 3> const& sym_phase_factors() const
    {
        return sym_phase_factors_;
    }

    /// Return a reference to a memory pool.
    /** A memory pool is created when this function called for the first time. */
    memory_pool& mem_pool(memory_t M__)
    {
        if (memory_pool_.count(M__) == 0) {
            memory_pool_.emplace(M__, std::move(memory_pool(M__)));
        }
        return memory_pool_.at(M__);
    }

    /// Get a defalt memory pool for a given device.
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

    /// Returns a constant reference to the augmentation operator of a given atom type.
    inline Augmentation_operator const& augmentation_op(int iat__) const
    {
        return augmentation_op_[iat__];
    }

    /// Returns a reference to the augmentation operator of a given atom type.
    inline Augmentation_operator& augmentation_op(int iat__)
    {
        return augmentation_op_[iat__];
    }

    /// Type of the host memory for arrays used in linear algebra operations.
    inline memory_t host_memory_t() const
    {
        return host_memory_t_;
    }

    /// Type of preferred memory for the storage of wave-functions and related arrays.
    inline memory_t preferred_memory_t() const
    {
        return preferred_memory_t_;
    }

    /// Linear algebra driver for the BLAS operations.
    inline linalg_t blas_linalg_t() const
    {
        return blas_linalg_t_;
    }
};

inline void Simulation_context::initialize()
{
    PROFILE("sirius::Simulation_context::initialize");

    /* can't initialize twice */
    if (initialized_) {
        TERMINATE("Simulation parameters are already initialized.");
    }
    /* Gamma-point calculation and non-collinear magnetism are not compatible */
    if (num_mag_dims() == 3) {
        set_gamma_point(false);
    }

    electronic_structure_method(parameters_input().electronic_structure_method_);
    set_core_relativity(parameters_input().core_relativity_);
    set_valence_relativity(parameters_input().valence_relativity_);

    /* get processing unit */
    std::string pu = control().processing_unit_;
    if (pu == "") {
#if defined(__GPU)
        pu = "gpu";
#else
        pu = "cpu";
#endif
    }
    set_processing_unit(pu);

    /* check if we can use a GPU device */
    if (processing_unit() == device_t::GPU) {
#if !defined(__GPU)
        TERMINATE_NO_GPU;
#endif
    }

    /* initialize MPI communicators */
    init_comm();

    switch (processing_unit()) {
        case device_t::CPU: {
            host_memory_t_ = memory_t::host;
            break;
        }
        case device_t::GPU: {
            //return memory_t::host;
            host_memory_t_ = memory_t::host_pinned;
            break;
        }
    }

    switch (processing_unit()) {
        case device_t::CPU: {
            preferred_memory_t_ = memory_t::host;
            break;
        }
        case device_t::GPU: {
            if (control_input_.memory_usage_ == "high") {
                preferred_memory_t_ = memory_t::device;
            }
            if (control_input_.memory_usage_ == "low" || control_input_.memory_usage_ == "medium") {
                preferred_memory_t_ = memory_t::host_pinned;
            }
            break;
        }
    }

    switch (processing_unit()) {
        case device_t::CPU: {
            blas_linalg_t_ = linalg_t::blas;
            break;
        }
        case device_t::GPU: {
            if (control_input_.memory_usage_ == "high") {
                blas_linalg_t_ = linalg_t::cublas;
            }
            if (control_input_.memory_usage_ == "low" || control_input_.memory_usage_ == "medium") {
                blas_linalg_t_ = linalg_t::cublasxt;
            }
            break;
        }
    }

    /* can't use reduced G-vectors in LAPW code */
    if (full_potential()) {
        control_input_.reduce_gvec_ = false;
    }

    if (!iterative_solver_input_.type_.size()) {
        if (full_potential()) {
            iterative_solver_input_.type_ = "exact";
        } else {
            iterative_solver_input_.type_ = "davidson";
        }
    }

    /* initialize variables related to the unit cell */
    unit_cell_.initialize();

    /* find the cutoff for G+k vectors (derived from rgkmax (aw_cutoff here) and minimum MT radius) */
    if (full_potential()) {
        set_gk_cutoff(aw_cutoff() / unit_cell_.min_mt_radius());
    }

    /* check the G+k cutoff */
    if (gk_cutoff() * 2 > pw_cutoff()) {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff() * 2;
        TERMINATE(s);
    }

    if (!full_potential()) {
        set_lmax_rho(unit_cell_.lmax() * 2);
        set_lmax_pot(unit_cell_.lmax() * 2);
        set_lmax_apw(-1);
    }

    /* initialize FFT interface */
    init_fft();

    int nbnd = static_cast<int>(unit_cell_.num_valence_electrons() / 2.0) +
               std::max(10, static_cast<int>(0.1 * unit_cell_.num_valence_electrons()));
    if (full_potential()) {
        /* take 10% of empty non-magnetic states */
        if (num_fv_states() < 0) {
            num_fv_states(nbnd);
        }
        if (num_fv_states() < static_cast<int>(unit_cell_.num_valence_electrons() / 2.0)) {
            std::stringstream s;
            s << "not enough first-variational states : " << num_fv_states();
            TERMINATE(s);
        }
    } else {
        if (num_mag_dims() == 3) {
            nbnd *= 2;
        }
        if (num_bands() < 0) {
            num_bands(nbnd);
        }
    }

    std::string evsn[] = {std_evp_solver_name(), gen_evp_solver_name()};
#if defined(__MAGMA)
    bool is_magma{true};
#else
    bool is_magma{false};
#endif
#if defined(__SCALAPACK)
    bool is_scalapack{true};
#else
    bool is_scalapack{false};
#endif
#if defined(__ELPA)
    bool is_elpa{true};
#else
    bool is_elpa{false};
#endif

    int npr = control_input_.mpi_grid_dims_[0];
    int npc = control_input_.mpi_grid_dims_[1];

    /* deduce the default eigen-value solver */
    for (int i : {0, 1}) {
        if (evsn[i] == "") {
            /* conditions for sequential diagonalization */
            if (comm_band().size() == 1 || npc == 1 || npr == 1 || !is_scalapack) {
                if (is_magma && num_bands() > 200) {
                    evsn[i] = "magma";
                } else {
                    evsn[i] = "lapack";
                }
            } else {
                if (is_scalapack) {
                    evsn[i] = "scalapack";
                }
                if (is_elpa) {
                    evsn[i] = "elpa1";
                }
            }
        }
    }

    std_evp_solver_name(evsn[0]);
    gen_evp_solver_name(evsn[1]);

    auto std_solver = std_evp_solver<double>();
    auto gen_solver = gen_evp_solver<double>();

    if (std_solver->is_parallel() != gen_solver->is_parallel()) {
        TERMINATE("both solvers must be sequential or parallel");
    }

    /* setup BLACS grid */
    if (std_solver->is_parallel()) {
        blacs_grid_ = std::unique_ptr<BLACS_grid>(new BLACS_grid(comm_band(), npr, npc));
    } else {
        blacs_grid_ = std::unique_ptr<BLACS_grid>(new BLACS_grid(Communicator::self(), 1, 1));
    }

    /* setup the cyclic block size */
    if (cyclic_block_size() < 0) {
        double a = std::min(std::log2(double(num_bands()) / blacs_grid_->num_ranks_col()),
                            std::log2(double(num_bands()) / blacs_grid_->num_ranks_row()));
        if (a < 1) {
            control_input_.cyclic_block_size_ = 2;
        } else {
            control_input_.cyclic_block_size_ =
                static_cast<int>(std::min(128.0, std::pow(2.0, static_cast<int>(a))) + 1e-12);
        }
    }

    if (!full_potential()) {
        /* add extra length to the cutoffs in order to interpolate radial integrals for q > cutoff */
        beta_ri_ = std::unique_ptr<Radial_integrals_beta<false>>(
            new Radial_integrals_beta<false>(unit_cell(), 2 * gk_cutoff(), settings().nprii_beta_));
        beta_ri_djl_ = std::unique_ptr<Radial_integrals_beta<true>>(
            new Radial_integrals_beta<true>(unit_cell(), 2 * gk_cutoff(), settings().nprii_beta_));
        aug_ri_ = std::unique_ptr<Radial_integrals_aug<false>>(
            new Radial_integrals_aug<false>(unit_cell(), 2 * pw_cutoff(), settings().nprii_aug_));
        aug_ri_djl_ = std::unique_ptr<Radial_integrals_aug<true>>(
            new Radial_integrals_aug<true>(unit_cell(), 2 * pw_cutoff(), settings().nprii_aug_));
        atomic_wf_ri_ = std::unique_ptr<Radial_integrals_atomic_wf<false>>(
            new Radial_integrals_atomic_wf<false>(unit_cell(), 2 * gk_cutoff(), 20));
        atomic_wf_ri_djl_ = std::unique_ptr<Radial_integrals_atomic_wf<true>>(
            new Radial_integrals_atomic_wf<true>(unit_cell(), 2 * gk_cutoff(), 20));
        ps_core_ri_ = std::unique_ptr<Radial_integrals_rho_core_pseudo<false>>(
            new Radial_integrals_rho_core_pseudo<false>(unit_cell(), 2 * pw_cutoff(), settings().nprii_rho_core_));
        ps_core_ri_djl_ = std::unique_ptr<Radial_integrals_rho_core_pseudo<true>>(
            new Radial_integrals_rho_core_pseudo<true>(unit_cell(), 2 * pw_cutoff(), settings().nprii_rho_core_));
        ps_rho_ri_ = std::unique_ptr<Radial_integrals_rho_pseudo>(
            new Radial_integrals_rho_pseudo(unit_cell(), 2 * pw_cutoff(), 20));
        vloc_ri_ = std::unique_ptr<Radial_integrals_vloc<false>>(
            new Radial_integrals_vloc<false>(unit_cell(), 2 * pw_cutoff(), settings().nprii_vloc_));
        vloc_ri_djl_ = std::unique_ptr<Radial_integrals_vloc<true>>(
            new Radial_integrals_vloc<true>(unit_cell(), 2 * pw_cutoff(), settings().nprii_vloc_));
    }

    if (control().verbosity_ >= 1 && comm().rank() == 0) {
        print_info();
    }

    if (control().verbosity_ >= 3) {
        pstdout pout(comm());
        if (comm().rank() == 0) {
            pout.printf("--- MPI rank placement ---\n");
        }
        pout.printf("rank: %3i, comm_band_rank: %3i, comm_k_rank: %3i, hostname: %s\n", comm().rank(),
                    comm_band().rank(), comm_k().rank(), utils::hostname().c_str());
    }

    update();

    if (comm_.rank() == 0 && control().print_memory_usage_) {
        MEMORY_USAGE_INFO();
    }

    initialized_ = true;
}

inline void Simulation_context::print_info() const
{
    tm const* ptm = localtime(&start_time_.tv_sec);
    char buf[100];
    strftime(buf, sizeof(buf), "%a, %e %b %Y %H:%M:%S", ptm);

    printf("\n");
    printf("SIRIUS version : %i.%i.%i\n", major_version, minor_version, revision);
    printf("git hash       : %s\n", git_hash);
    printf("git branch     : %s\n", git_branchname);
    printf("start time     : %s\n", buf);
    printf("\n");
    printf("number of MPI ranks           : %i\n", comm_.size());
    if (mpi_grid_) {
        printf("MPI grid                      :");
        for (int i = 0; i < mpi_grid_->num_dimensions(); i++) {
            printf(" %i", mpi_grid_->communicator(1 << i).size());
        }
        printf("\n");
    }
    printf("maximum number of OMP threads : %i\n", omp_get_max_threads());
    printf("number of MPI ranks per node  : %i\n", num_ranks_per_node());
    printf("page size (Kb)                : %li\n", utils::get_page_size() >> 10);
    printf("number of pages               : %li\n", utils::get_num_pages());
    printf("available memory (GB)         : %li\n", utils::get_total_memory() >> 30);

    std::string headers[]         = {"FFT context for density and potential", "FFT context for coarse grid"};
    double cutoffs[]              = {pw_cutoff(), 2 * gk_cutoff()};
    Communicator const* comms[]   = {&comm_fft(), &comm_fft_coarse()};
    FFT3D_grid const* fft_grids[] = {&fft(), &fft_coarse()};
    Gvec const* gvecs[]           = {&gvec(), &gvec_coarse()};

    printf("\n");
    for (int i = 0; i < 2; i++) {
        printf("%s\n", headers[i].c_str());
        printf("=====================================\n");
        printf("  comm size                             : %i\n", comms[i]->size());
        printf("  plane wave cutoff                     : %f\n", cutoffs[i]);
        printf("  grid size                             : %i %i %i   total : %i\n", fft_grids[i]->size(0),
               fft_grids[i]->size(1), fft_grids[i]->size(2), fft_grids[i]->size());
        printf("  grid limits                           : %i %i   %i %i   %i %i\n", fft_grids[i]->limits(0).first,
               fft_grids[i]->limits(0).second, fft_grids[i]->limits(1).first, fft_grids[i]->limits(1).second,
               fft_grids[i]->limits(2).first, fft_grids[i]->limits(2).second);
        printf("  number of G-vectors within the cutoff : %i\n", gvecs[i]->num_gvec());
        printf("  local number of G-vectors             : %i\n", gvecs[i]->count());
        printf("  number of G-shells                    : %i\n", gvecs[i]->num_shells());
        printf("\n");
    }

    unit_cell_.print_info(control().verbosity_);
    for (int i = 0; i < unit_cell_.num_atom_types(); i++) {
        unit_cell_.atom_type(i).print_info();
    }

    printf("\n");
    printf("total nuclear charge               : %i\n", unit_cell().total_nuclear_charge());
    printf("number of core electrons           : %f\n", unit_cell().num_core_electrons());
    printf("number of valence electrons        : %f\n", unit_cell().num_valence_electrons());
    printf("total number of electrons          : %f\n", unit_cell().num_electrons());
    printf("total number of aw basis functions : %i\n", unit_cell().mt_aw_basis_size());
    printf("total number of lo basis functions : %i\n", unit_cell().mt_lo_basis_size());
    printf("number of first-variational states : %i\n", num_fv_states());
    printf("number of bands                    : %i\n", num_bands());
    printf("number of spins                    : %i\n", num_spins());
    printf("number of magnetic dimensions      : %i\n", num_mag_dims());
    printf("lmax_apw                           : %i\n", lmax_apw());
    printf("lmax_rho                           : %i\n", lmax_rho());
    printf("lmax_pot                           : %i\n", lmax_pot());
    printf("lmax_rf                            : %i\n", unit_cell_.lmax());
    printf("smearing width                     : %f\n", smearing_width());
    printf("cyclic block size                  : %i\n", cyclic_block_size());
    printf("|G+k| cutoff                       : %f\n", gk_cutoff());

    std::string reln[] = {"valence relativity                 : ", "core relativity                    : "};

    relativity_t relt[] = {valence_relativity_, core_relativity_};
    for (int i = 0; i < 2; i++) {
        printf("%s", reln[i].c_str());
        switch (relt[i]) {
            case relativity_t::none: {
                printf("none\n");
                break;
            }
            case relativity_t::koelling_harmon: {
                printf("Koelling-Harmon\n");
                break;
            }
            case relativity_t::zora: {
                printf("zora\n");
                break;
            }
            case relativity_t::iora: {
                printf("iora\n");
                break;
            }
            case relativity_t::dirac: {
                printf("Dirac\n");
                break;
            }
        }
    }

    std::string evsn[] = {"standard eigen-value solver        : ", "generalized eigen-value solver     : "};

    ev_solver_t evst[] = {std_evp_solver_type(), gen_evp_solver_type()};
    for (int i = 0; i < 2; i++) {
        printf("%s", evsn[i].c_str());
        switch (evst[i]) {
            case ev_solver_t::lapack: {
                printf("LAPACK\n");
                break;
            }
#ifdef __SCALAPACK
            case ev_solver_t::scalapack: {
                printf("ScaLAPACK\n");
                break;
            }
            case ev_solver_t::elpa1: {
                printf("ELPA1\n");
                break;
            }
            case ev_solver_t::elpa2: {
                printf("ELPA2\n");
                break;
            }
#endif
            case ev_solver_t::magma: {
                printf("MAGMA\n");
                break;
            }
            case ev_solver_t::plasma: {
                printf("PLASMA\n");
                break;
            }
            default: {
                TERMINATE("wrong eigen-value solver");
            }
        }
    }

    printf("processing unit                    : ");
    switch (processing_unit()) {
        case device_t::CPU: {
            printf("CPU\n");
            break;
        }
        case device_t::GPU: {
            printf("GPU\n");
#ifdef __GPU
            acc::print_device_info(0);
#endif
            break;
        }
    }

    int i{1};
    printf("\n");
    printf("XC functionals\n");
    printf("==============\n");
    for (auto& xc_label : xc_functionals()) {
        XC_functional xc(xc_label, num_spins());
        printf("%i) %s: %s\n", i, xc_label.c_str(), xc.name().c_str());
        printf("%s\n", xc.refs().c_str());
        i++;
    }
}

} // namespace sirius

#endif
