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

/** \file k_point.hpp
 *
 *  \brief Contains definition of sirius::K_point class.
 */

#ifndef __K_POINT_HPP__
#define __K_POINT_HPP__

#include "matching_coefficients.hpp"
#include "Beta_projectors/beta_projectors.hpp"
#include "wave_functions.hpp"

namespace sirius {

/// K-point related variables and methods.
/** \image html wf_storage.png "Wave-function storage"
 *  \image html fv_eigen_vectors.png "First-variational eigen vectors" */
class K_point
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;

    /// Unit cell object.
    Unit_cell const& unit_cell_;

    /// Weight of k-point.
    double weight_;

    /// Fractional k-point coordinates.
    vector3d<double> vk_;

    /// List of G-vectors with |G+k| < cutoff.
    std::unique_ptr<Gvec> gkvec_;

    /// G-vector distribution for the FFT transformation.
    std::unique_ptr<Gvec_partition> gkvec_partition_;

    std::unique_ptr<spfft::Transform> spfft_transform_;

    /// First-variational eigen values
    std::vector<double> fv_eigen_values_;

    /// First-variational eigen vectors, distributed over 2D BLACS grid.
    dmatrix<double_complex> fv_eigen_vectors_;

    /// First-variational eigen vectors, distributed in slabs.
    std::unique_ptr<Wave_functions> fv_eigen_vectors_slab_;

    /// Lowest eigen-vectors of the LAPW overlap matrix with small aigen-values.
    std::unique_ptr<Wave_functions> singular_components_;

    /// Second-variational eigen vectors.
    /** Second-variational eigen-vectors are stored as one or two \f$ N_{fv} \times N_{fv} \f$ matrices in
     *  case of non-magnetic or collinear magnetic case or as a single \f$ 2 N_{fv} \times 2 N_{fv} \f$
     *  matrix in case of general non-collinear magnetism. */
    dmatrix<double_complex> sv_eigen_vectors_[2];

    /// Full-diagonalization eigen vectors.
    mdarray<double_complex, 2> fd_eigen_vectors_;

    /// First-variational states.
    std::unique_ptr<Wave_functions> fv_states_{nullptr};

    /// Two-component (spinor) wave functions describing the bands.
    std::unique_ptr<Wave_functions> spinor_wave_functions_{nullptr};

    /// Two-component (spinor) hubbard wave functions where the S matrix is applied (if ppus).
    std::unique_ptr<Wave_functions> hubbard_wave_functions_{nullptr}; // TODO: remove in future

    /// Scalar hubbard wave-functions.
    std::unique_ptr<Wave_functions> hubbard_wf_{nullptr};

    /// Band occupation numbers.
    mdarray<double, 2> band_occupancies_;

    /// Band energies.
    mdarray<double, 2> band_energies_;

    /// LAPW matching coefficients for the row G+k vectors.
    /** Used to setup the distributed LAPW Hamiltonian and overlap matrices. */
    std::unique_ptr<Matching_coefficients> alm_coeffs_row_{nullptr};

    /// LAPW matching coefficients for the column G+k vectors.
    /** Used to setup the distributed LAPW Hamiltonian and overlap matrices. */
    std::unique_ptr<Matching_coefficients> alm_coeffs_col_{nullptr};

    /// LAPW matching coefficients for the local set G+k vectors.
    std::unique_ptr<Matching_coefficients> alm_coeffs_loc_{nullptr};

    /// Mapping between local row and global G+k vecotor index.
    /** Used by matching_coefficients class. */
    std::vector<int> igk_row_;

    /// Mapping between local column and global G+k vecotor index.
    /** Used by matching_coefficients class. */
    std::vector<int> igk_col_;

    /// Mapping between local and global G+k vecotor index.
    /** Used by matching_coefficients class. */
    std::vector<int> igk_loc_;

    /// Number of G+k vectors distributed along rows of MPI grid
    int num_gkvec_row_{0};

    /// Number of G+k vectors distributed along columns of MPI grid
    int num_gkvec_col_{0};

    /// Offset of the local fraction of G+k vectors in the global index.
    int gkvec_offset_{0};

    /// Basis descriptors distributed between rows of the 2D MPI grid.
    /** This is a local array. Only MPI ranks belonging to the same column have identical copies of this array. */
    std::vector<lo_basis_descriptor> lo_basis_descriptors_row_;

    /// Basis descriptors distributed between columns of the 2D MPI grid.
    /** This is a local array. Only MPI ranks belonging to the same row have identical copies of this array. */
    std::vector<lo_basis_descriptor> lo_basis_descriptors_col_;

    /// List of columns of the Hamiltonian and overlap matrix lo block (local index) for a given atom.
    std::vector<std::vector<int>> atom_lo_cols_;

    /// list of rows of the Hamiltonian and overlap matrix lo block (local index) for a given atom
    std::vector<std::vector<int>> atom_lo_rows_;

    /// Imaginary unit to the power of l.
    std::vector<double_complex> zil_;

    /// Mapping between lm and l.
    std::vector<int> l_by_lm_;

    /// Column rank of the processors of ScaLAPACK/ELPA diagonalization grid.
    int rank_col_;

    /// Number of processors along the columns of the diagonalization grid.
    int num_ranks_col_;

    /// Row rank of the processors of ScaLAPACK/ELPA diagonalization grid.
    int rank_row_;

    /// Number of processors along the rows of the diagonalization grid.
    int num_ranks_row_;

    /// Beta projectors for a local set of G+k vectors.
    std::unique_ptr<Beta_projectors> beta_projectors_{nullptr};

    /// Beta projectors for row G+k vectors.
    /** Used to setup the full Hamiltonian in PP-PW case (for verification purpose only) */
    std::unique_ptr<Beta_projectors> beta_projectors_row_{nullptr};

    /// Beta projectors for column G+k vectors.
    /** Used to setup the full Hamiltonian in PP-PW case (for verification purpose only) */
    std::unique_ptr<Beta_projectors> beta_projectors_col_{nullptr};

    /// Preconditioner matrix for Chebyshev solver.
    mdarray<double_complex, 3> p_mtrx_;

    /// Communicator for parallelization inside k-point.
    /** This communicator is used to split G+k vectors and wave-functions. */
    Communicator const& comm_;

    /// Communicator between(!!) rows.
    Communicator const& comm_row_;

    /// Communicator between(!!) columns.
    Communicator const& comm_col_;

    /// Generate G+k and local orbital basis sets.
    void generate_gklo_basis();

    /// Test orthonormalization of first-variational states.
    void test_fv_states();

    /// Find G+k vectors within the cutoff.
    void generate_gkvec(double gk_cutoff__);

  public:
    /// Constructor
    K_point(Simulation_context& ctx__, double const* vk__, double weight__)
        : ctx_(ctx__)
        , unit_cell_(ctx_.unit_cell())
        , weight_(weight__)
        , comm_(ctx_.comm_band())
        , comm_row_(ctx_.blacs_grid().comm_row())
        , comm_col_(ctx_.blacs_grid().comm_col())
    {
        PROFILE("sirius::K_point::K_point");

        for (int x = 0; x < 3; x++) {
            vk_[x] = vk__[x];
        }

        band_occupancies_ = mdarray<double, 2>(ctx_.num_bands(), ctx_.num_spin_dims());
        band_occupancies_.zero();
        band_energies_ = mdarray<double, 2>(ctx_.num_bands(), ctx_.num_spin_dims());
        band_energies_.zero();

        num_ranks_row_ = comm_row_.size();
        num_ranks_col_ = comm_col_.size();

        rank_row_ = comm_row_.rank();
        rank_col_ = comm_col_.rank();
    }

    /// Initialize the k-point related arrays and data.
    void initialize(); // TODO: initialize from HDF5

    /// Update the reciprocal lattice vectors of the G+k array.
    void update();

    /// Generate first-variational states from eigen-vectors.
    /** APW+lo basis \f$ \varphi_{\mu {\bf k}}({\bf r}) = \{ \varphi_{\bf G+k}({\bf r}),
        \varphi_{j{\bf k}}({\bf r}) \} \f$ is used to expand first-variational wave-functions:

        \f[
        \psi_{i{\bf k}}({\bf r}) = \sum_{\mu} c_{\mu i}^{\bf k} \varphi_{\mu \bf k}({\bf r}) =
        \sum_{{\bf G}}c_{{\bf G} i}^{\bf k} \varphi_{\bf G+k}({\bf r}) +
        \sum_{j}c_{j i}^{\bf k}\varphi_{j{\bf k}}({\bf r})
        \f]

        Inside muffin-tins the expansion is converted into the following form:
        \f[
        \psi_{i {\bf k}}({\bf r})= \begin{array}{ll}
        \displaystyle \sum_{L} \sum_{\lambda=1}^{N_{\ell}^{\alpha}}
        F_{L \lambda}^{i {\bf k},\alpha}f_{\ell \lambda}^{\alpha}(r)
        Y_{\ell m}(\hat {\bf r}) & {\bf r} \in MT_{\alpha} \end{array}
        \f]

        Thus, the total number of coefficients representing a wave-funstion is equal
        to the number of muffin-tin basis functions of the form \f$ f_{\ell \lambda}^{\alpha}(r)
        Y_{\ell m}(\hat {\bf r}) \f$ plust the number of G+k plane waves.
        First-variational states are obtained from the first-variational eigen-vectors and
        LAPW matching coefficients.

        APW part:
        \f[
        \psi_{\xi j}^{\bf k} = \sum_{{\bf G}} Z_{{\bf G} j}^{\bf k} * A_{\xi}({\bf G+k})
        \f]
     */
    void generate_fv_states();

    /// Generate two-component spinor wave functions.
    /** In case of second-variational diagonalization spinor wave-functions are generated from the first-variational
        states and second-variational eigen-vectors. */
    void generate_spinor_wave_functions();

    void generate_atomic_wave_functions(const basis_functions_index& index, const int atom, const int offset,
                                        const bool hubbard, Wave_functions& phi);

    /// Generate plane-wave coefficients of the atomic wave-functions.
    /** Plane-wave coefficients of the atom-centered wave-functions
        \f$ \varphi^{\alpha}_{\ell m}({\bf r}) = \varphi^{\alpha}_{\ell}(r)R_{\ell m}(\theta, \phi) \f$
        are computed in the following way:
        \f[
        \varphi^{\alpha}_{\ell m}({\bf q}) = \frac{1}{\sqrt{\Omega}}
          \int e^{-i{\bf q}{\bf r}} \varphi^{\alpha}_{\ell m}({\bf r} - {\bf r}_{\alpha}) d{\bf r} =
          \frac{e^{-i{\bf q}{\bf r}_{\alpha}}}{\sqrt{\Omega}} \int e^{-i{\bf q}{\bf r}}
          \varphi^{\alpha}_{\ell}(r)R_{\ell m}(\theta, \phi) r^2 \sin\theta dr d\theta d\phi
        \f]
        where \f$ {\bf q} = {\bf G+k} \f$. Using the expansion of the plane wave in terms of spherical Bessel
        functions and real spherical harmonics:
        \f[
        e^{-i{\bf q}{\bf r}}=4\pi \sum_{\ell m} (-i)^\ell j_{\ell}(q r)R_{\ell m}({\bf \hat q})R_{\ell m}({\bf \hat r})
        \f]
        we arrive to the following expression:
        \f[
        \varphi^{\alpha}_{\ell m}({\bf q}) = e^{-i{\bf q}{\bf r}_{\alpha}} \frac{4\pi}{\sqrt{\Omega}} (-i)^\ell
          R_{\ell m}({\bf q}) \int \varphi^{\alpha}_{\ell}(r)  j_{\ell}(q r) r^2 dr
        \f]

        \note In the current implementation wave-functions are generated as scalars (without spin index). Spinor atomic
        wave-functions might be necessary in future for the more advanced LDA+U implementation.

        \param [in] atoms   List of atoms, for which the wave-functions are generated.
        \param [in] indexb  Lambda function that returns index of the basis functions for each atom type.
        \param [in] ri      Radial integrals of the product of sperical Bessel functions and atomic functions.
        \param [out] wf     Resulting wave-functions for the list of atoms. Output wave-functions must have
                            sufficient storage space.
     */
    void generate_atomic_wave_functions(std::vector<int> atoms__,
                                        std::function<sirius::basis_functions_index const*(int)> indexb__,
                                        Radial_integrals_atomic_wf<false> const& ri__, sddk::Wave_functions& wf__);

    void compute_gradient_wave_functions(Wave_functions& phi, const int starting_position_i, const int num_wf,
                                         Wave_functions& dphi, const int starting_position_j, const int direction);

    void generate_hubbard_orbitals();

    void copy_hubbard_orbitals_on_device()
        {
            if (ctx_.hubbard_correction() && is_device_memory(ctx_.preferred_memory_t()))
            {
                auto& mpd = ctx_.mem_pool(memory_t::device);
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    this->hubbard_wave_functions().pw_coeffs(ispn).allocate(mpd);
                    this->hubbard_wave_functions().pw_coeffs(ispn).copy_to(memory_t::device, 0, unit_cell_.num_wf_with_U().first);
                }
            }
        }

    void release_hubbard_orbitals_on_device()
    {
        if (ctx_.hubbard_correction() && is_device_memory(ctx_.preferred_memory_t()))
        {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                this->hubbard_wave_functions().pw_coeffs(ispn).deallocate(memory_t::device);
            }
        }
    }

    void orthogonalize_hubbard_orbitals(Wave_functions& phi__);

    /// Save data to HDF5 file.
    void save(std::string const& name__, int id__) const;

    void load(HDF5_tree h5in, int id);

    //== void save_wave_functions(int id);

    //== void load_wave_functions(int id);

    /// Collect distributed first-variational vectors into a global array.
    void get_fv_eigen_vectors(mdarray<double_complex, 2>& fv_evec__) const;

    /// Collect distributed second-variational vectors into a global array.
    void get_sv_eigen_vectors(mdarray<double_complex, 2>& sv_evec__) const
    {
        assert((int)sv_evec__.size(0) == ctx_.num_spins() * ctx_.num_fv_states());
        assert((int)sv_evec__.size(1) == ctx_.num_spins() * ctx_.num_fv_states());

        sv_evec__.zero();

        if (!ctx_.need_sv()) {
            for (int i = 0; i < ctx_.num_fv_states(); i++) {
                sv_evec__(i, i) = 1;
            }
            return;
        }

        int nsp = (ctx_.num_mag_dims() == 3) ? 1 : ctx_.num_spins();

        for (int ispn = 0; ispn < nsp; ispn++) {
            int offs = ctx_.num_fv_states() * ispn;
            for (int jloc = 0; jloc < sv_eigen_vectors_[ispn].num_cols_local(); jloc++) {
                int j = sv_eigen_vectors_[ispn].icol(jloc);
                for (int iloc = 0; iloc < sv_eigen_vectors_[ispn].num_rows_local(); iloc++) {
                    int i = sv_eigen_vectors_[ispn].irow(iloc);
                    sv_evec__(i + offs, j + offs) = sv_eigen_vectors_[ispn](iloc, jloc);
                }
            }
        }

        comm().allreduce(sv_evec__.at(memory_t::host), (int)sv_evec__.size());
    }

    /// Test orthonormalization of spinor wave-functions
    void test_spinor_wave_functions(int use_fft);

    /// Get the number of occupied bands for each spin channel.
    inline int num_occupied_bands(int ispn__ = -1) const
    {
        for (int j = ctx_.num_bands() - 1; j >= 0; j--) {
            if (std::abs(band_occupancy(j, ispn__)) > ctx_.min_occupancy() * ctx_.max_occupancy()) {
                return j + 1;
            }
        }
        return 0;
    }

    /// Total number of G+k vectors within the cutoff distance
    inline int num_gkvec() const
    {
        return gkvec_->num_gvec();
    }

    /// Get band energy.
    inline double band_energy(int j__, int ispn__) const
    {
        if (ctx_.num_mag_dims() == 3) {
            ispn__ = 0;
        }
        return band_energies_(j__, ispn__);
    }

    /// Set band energy.
    inline void band_energy(int j__, int ispn__, double e__)
    {
        if (ctx_.num_mag_dims() == 3) {
            ispn__ = 0;
        }
        band_energies_(j__, ispn__) = e__;
    }

    /// Get band occupancy.
    inline double band_occupancy(int j__, int ispn__) const
    {
        if (ctx_.num_mag_dims() == 3) {
            ispn__ = 0;
        }
        return band_occupancies_(j__, ispn__);
    }

    /// Set band occupancy.
    inline void band_occupancy(int j__, int ispn__, double occ__)
    {
        if (ctx_.num_mag_dims() == 3) {
            ispn__ = 0;
        }
        band_occupancies_(j__, ispn__) = occ__;
    }

    inline double fv_eigen_value(int i) const
    {
        return fv_eigen_values_[i];
    }

    void set_fv_eigen_values(double* eval)
    {
        std::memcpy(&fv_eigen_values_[0], eval, ctx_.num_fv_states() * sizeof(double));
    }

    /// Return weight of k-point.
    inline double weight() const
    {
        return weight_;
    }

    inline Wave_functions& fv_states()
    {
        assert(fv_states_ != nullptr);
        return *fv_states_;
    }

    inline Wave_functions& spinor_wave_functions()
    {
        assert(spinor_wave_functions_ != nullptr);
        return *spinor_wave_functions_;
    }

    inline Wave_functions& hubbard_wave_functions()
    {
        assert(hubbard_wave_functions_ != nullptr);
        return *hubbard_wave_functions_;
    }

    inline Wave_functions const& hubbard_wave_functions() const
    {
        assert(hubbard_wave_functions_ != nullptr);
        return *hubbard_wave_functions_;
    }

    inline Wave_functions& singular_components()
    {
        return *singular_components_;
    }

    inline vector3d<double> vk() const
    {
        return vk_;
    }

    /// Basis size of LAPW+lo method.
    /** The total LAPW+lo basis size is equal to the sum of the number of LAPW functions and the total number
     *  of the local orbitals. */
    inline int gklo_basis_size() const
    {
        return num_gkvec() + unit_cell_.mt_lo_basis_size();
    }

    /// Local number of G+k vectors in case of flat distribution.
    inline int num_gkvec_loc() const
    {
        return gkvec().count();
    }

    /// Return global index of G+k vector.
    inline int idxgk(int igkloc__) const
    {
        return gkvec_offset_ + igkloc__;
    }

    /// Local number of G+k vectors for each MPI rank in the row of the 2D MPI grid.
    inline int num_gkvec_row() const
    {
        return num_gkvec_row_;
    }

    /// Local number of local orbitals for each MPI rank in the row of the 2D MPI grid.
    inline int num_lo_row() const
    {
        return static_cast<int>(lo_basis_descriptors_row_.size());
    }

    /// Local number of basis functions for each MPI rank in the row of the 2D MPI grid.
    inline int gklo_basis_size_row() const
    {
        return num_gkvec_row() + num_lo_row();
    }

    /// Local number of G+k vectors for each MPI rank in the column of the 2D MPI grid.
    inline int num_gkvec_col() const
    {
        return num_gkvec_col_;
    }

    /// Local number of local orbitals for each MPI rank in the column of the 2D MPI grid.
    inline int num_lo_col() const
    {
        return static_cast<int>(lo_basis_descriptors_col_.size());
    }

    /// Local number of basis functions for each MPI rank in the column of the 2D MPI grid.
    inline int gklo_basis_size_col() const
    {
        return num_gkvec_col() + num_lo_col();
    }

    inline lo_basis_descriptor const& lo_basis_descriptor_col(int idx) const
    {
        assert(idx >= 0 && idx < (int)lo_basis_descriptors_col_.size());
        return lo_basis_descriptors_col_[idx];
    }

    inline lo_basis_descriptor const& lo_basis_descriptor_row(int idx) const
    {
        assert(idx >= 0 && idx < (int)lo_basis_descriptors_row_.size());
        return lo_basis_descriptors_row_[idx];
    }

    inline int igk_loc(int idx__) const
    {
        return igk_loc_[idx__];
    }

    inline std::vector<int> const& igk_loc() const
    {
        return igk_loc_;
    }

    inline int igk_row(int idx__) const
    {
        return igk_row_[idx__];
    }

    inline std::vector<int> const& igk_row() const
    {
        return igk_row_;
    }

    inline int igk_col(int idx__) const
    {
        return igk_col_[idx__];
    }

    inline std::vector<int> const& igk_col() const
    {
        return igk_col_;
    }

    inline int num_ranks_row() const
    {
        return num_ranks_row_;
    }

    inline int rank_row() const
    {
        return rank_row_;
    }

    inline int num_ranks_col() const
    {
        return num_ranks_col_;
    }

    inline int rank_col() const
    {
        return rank_col_;
    }

    /// Number of MPI ranks for a given k-point
    inline int num_ranks() const
    {
        return comm_.size();
    }

    inline int rank() const
    {
        return comm_.rank();
    }

    /// Return number of lo columns for a given atom
    inline int num_atom_lo_cols(int ia) const
    {
        return (int)atom_lo_cols_[ia].size();
    }

    /// Return local index (for the current MPI rank) of a column for a given atom and column index within an atom
    inline int lo_col(int ia, int i) const
    {
        return atom_lo_cols_[ia][i];
    }

    /// Return number of lo rows for a given atom
    inline int num_atom_lo_rows(int ia) const
    {
        return (int)atom_lo_rows_[ia].size();
    }

    /// Return local index (for the current MPI rank) of a row for a given atom and row index within an atom
    inline int lo_row(int ia, int i) const
    {
        return atom_lo_rows_[ia][i];
    }

    inline dmatrix<double_complex>& fv_eigen_vectors()
    {
        return fv_eigen_vectors_;
    }

    inline Wave_functions& fv_eigen_vectors_slab()
    {
        return *fv_eigen_vectors_slab_;
    }

    inline dmatrix<double_complex>& sv_eigen_vectors(int ispn)
    {
        return sv_eigen_vectors_[ispn];
    }

    inline mdarray<double_complex, 2>& fd_eigen_vectors()
    {
        return fd_eigen_vectors_;
    }

    void bypass_sv()
    {
        std::memcpy(&band_energies_[0], &fv_eigen_values_[0], ctx_.num_fv_states() * sizeof(double));
    }

    inline Gvec const& gkvec() const
    {
        return *gkvec_;
    }

    inline Gvec_partition const& gkvec_partition() const
    {
        return *gkvec_partition_;
    }

    inline Matching_coefficients const& alm_coeffs_row() const
    {
        return *alm_coeffs_row_;
    }

    inline Matching_coefficients const& alm_coeffs_col() const
    {
        return *alm_coeffs_col_;
    }

    inline Matching_coefficients const& alm_coeffs_loc() const
    {
        return *alm_coeffs_loc_;
    }

    inline Communicator const& comm() const
    {
        return comm_;
    }

    inline Communicator const& comm_row() const
    {
        return comm_row_;
    }

    inline Communicator const& comm_col() const
    {
        return comm_col_;
    }

    inline double_complex p_mtrx(int xi1, int xi2, int iat) const
    {
        return p_mtrx_(xi1, xi2, iat);
    }

    inline mdarray<double_complex, 3>& p_mtrx()
    {
        return p_mtrx_;
    }

    Beta_projectors& beta_projectors()
    {
        assert(beta_projectors_ != nullptr);
        return *beta_projectors_;
    }

    Beta_projectors const& beta_projectors() const
    {
        assert(beta_projectors_ != nullptr);
        return *beta_projectors_;
    }

    Beta_projectors& beta_projectors_row()
    {
        assert(beta_projectors_ != nullptr);
        return *beta_projectors_row_;
    }

    Beta_projectors& beta_projectors_col()
    {
        assert(beta_projectors_ != nullptr);
        return *beta_projectors_col_;
    }

    Simulation_context const& ctx() const
    {
        return ctx_;
    }

    /// Print message from the root rank.
    template <typename... Args>
    inline void message(int level__, char const* label__, Args... args) const
    {
        if (this->comm().rank() == 0 && this->ctx().control().verbosity_ >= level__) {
            if (label__) {
                std::printf("[%s] ", label__);
            }
            std::printf(args...);
        }
    }

    spfft::Transform& spfft_transform()
    {
        return *spfft_transform_;
    }

    spfft::Transform const& spfft_transform() const
    {
        return *spfft_transform_;
    }
};

} // namespace sirius

/** \page basis Basis functions for Kohn-Sham wave-functions expansion
 *
 *  \section basis1 LAPW+lo basis
 *
 *  LAPW+lo basis consists of two different sets of functions: LAPW functions \f$ \varphi_{{\bf G+k}} \f$ defined over
 *  entire unit cell:
 *  \f[
 *      \varphi_{{\bf G+k}}({\bf r}) = \left\{ \begin{array}{ll}
 *      \displaystyle \sum_{L} \sum_{\nu=1}^{O_{\ell}^{\alpha}} a_{L\nu}^{\alpha}({\bf G+k})u_{\ell \nu}^{\alpha}(r)
 *      Y_{\ell m}(\hat {\bf r}) & {\bf r} \in {\rm MT} \alpha \\
 *      \displaystyle \frac{1}{\sqrt  \Omega} e^{i({\bf G+k}){\bf r}} & {\bf r} \in {\rm I} \end{array} \right.
 *  \f]
 *  and Bloch sums of local orbitals defined inside muffin-tin spheres only:
 *  \f[
 *      \begin{array}{ll} \displaystyle \varphi_{j{\bf k}}({\bf r})=\sum_{{\bf T}} e^{i{\bf kT}}
 *      \varphi_{j}({\bf r - T}) & {\rm {\bf r} \in MT} \end{array}
 *  \f]
 *  Each local orbital is composed of radial and angular parts:
 *  \f[
 *      \varphi_{j}({\bf r}) = \phi_{\ell_j}^{\zeta_j,\alpha_j}(r) Y_{\ell_j m_j}(\hat {\bf r})
 *  \f]
 *  Radial part of local orbital is defined as a linear combination of radial functions (minimum two radial functions
 *  are required) such that local orbital vanishes at the sphere boundary:
 *  \f[
 *      \phi_{\ell}^{\zeta, \alpha}(r) = \sum_{p}\gamma_{p}^{\zeta,\alpha} u_{\ell \nu_p}^{\alpha}(r)
 *  \f]
 *
 *  Arbitrary number of local orbitals can be introduced for each angular quantum number (this is highlighted by
 *  the index \f$ \zeta \f$).
 *
 *  Radial functions are m-th order (with zero-order being a function itself) energy derivatives of the radial
 *  Schrödinger equation:
 *  \f[
 *      u_{\ell \nu}^{\alpha}(r) = \frac{\partial^{m_{\nu}}}{\partial^{m_{\nu}}E}u_{\ell}^{\alpha}(r,E)\Big|_{E=E_{\nu}}
 *  \f]
 */

/** \page data_dist K-point data distribution
 *
 *  \section data_dist1 "Panel" and "full" data storage
 *
 *  We have to deal with big arrays (matching coefficients, eigen vectors, wave functions, etc.) which may not fit
 *  into the memory of a single node. For some operations we need a "panel" distribution of data, where each
 *  MPI rank gets a local panel of block-cyclic distributed matrix. This way of storing data is necessary for the
 *  distributed PBLAS and ScaLAPACK operations.
 *
 */

#endif // __K_POINT_H__
