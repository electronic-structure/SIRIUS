/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file k_point.hpp
 *
 *  \brief Contains definition of sirius::K_point class.
 */

#ifndef __K_POINT_HPP__
#define __K_POINT_HPP__

#include "lapw/matching_coefficients.hpp"
#include "beta_projectors/beta_projectors.hpp"
#include "unit_cell/radial_functions_index.hpp"
#include "core/fft/fft.hpp"
#include "core/wf/wave_functions.hpp"

namespace sirius {

/// K-point related variables and methods.
/** \image html wf_storage.png "Wave-function storage"
 *  \image html fv_eigen_vectors.png "First-variational eigen vectors"
 *
 *  \tparam T  Precision of the wave-functions (float or double).
 */
template <typename T>
class K_point
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;

    /// Unit cell object.
    Unit_cell const& unit_cell_;

    /// Fractional k-point coordinates.
    r3::vector<double> vk_;

    /// Weight of k-point.
    double weight_{1.0};

    /// Communicator for parallelization inside k-point.
    /** This communicator is used to split G+k vectors and wave-functions. */
    mpi::Communicator const& comm_;

    /// List of G-vectors with |G+k| < cutoff.
    std::shared_ptr<fft::Gvec> gkvec_;

    std::shared_ptr<fft::Gvec> gkvec_row_;

    std::shared_ptr<fft::Gvec> gkvec_col_;

    /// G-vector distribution for the FFT transformation.
    std::shared_ptr<fft::Gvec_fft> gkvec_partition_;

    mutable std::unique_ptr<fft::spfft_transform_type<T>> spfft_transform_;

    /// First-variational eigen values
    mdarray<double, 1> fv_eigen_values_;

    /// First-variational eigen vectors, distributed over 2D BLACS grid.
    la::dmatrix<std::complex<T>> fv_eigen_vectors_;

    /// First-variational eigen vectors, distributed in slabs.
    std::unique_ptr<wf::Wave_functions<T>> fv_eigen_vectors_slab_;

    /// Lowest eigen-vectors of the LAPW overlap matrix with small aigen-values.
    std::unique_ptr<wf::Wave_functions<T>> singular_components_;

    /// Second-variational eigen vectors.
    /** Second-variational eigen-vectors are stored as one or two \f$ N_{fv} \times N_{fv} \f$ matrices in
     *  case of non-magnetic or collinear magnetic case or as a single \f$ 2 N_{fv} \times 2 N_{fv} \f$
     *  matrix in case of general non-collinear magnetism. */
    std::array<la::dmatrix<std::complex<T>>, 2> sv_eigen_vectors_;

    /// Full-diagonalization eigen vectors.
    mdarray<std::complex<T>, 2> fd_eigen_vectors_;

    /// First-variational states.
    std::unique_ptr<wf::Wave_functions<T>> fv_states_{nullptr};

    /// Two-component (spinor) wave functions describing the bands.
    std::unique_ptr<wf::Wave_functions<T>> spinor_wave_functions_{nullptr};

    /// Pseudopotential atmoic wave-functions (not orthogonalized).
    std::unique_ptr<wf::Wave_functions<T>> atomic_wave_functions_{nullptr};

    /// Pseudopotential atmoic wave-functions (not orthogonalized) with S-operator applied.
    std::unique_ptr<wf::Wave_functions<T>> atomic_wave_functions_S_{nullptr};

    /// Hubbard wave functions.
    std::unique_ptr<wf::Wave_functions<T>> hubbard_wave_functions_{nullptr};

    /// Hubbard wave functions with S-operator applied.
    std::unique_ptr<wf::Wave_functions<T>> hubbard_wave_functions_S_{nullptr};

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

    /// Number of G+k vectors distributed along rows of MPI grid
    int num_gkvec_row_{0};

    /// Number of G+k vectors distributed along columns of MPI grid
    int num_gkvec_col_{0};

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
    std::vector<std::complex<double>> zil_;

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
    std::unique_ptr<Beta_projectors<T>> beta_projectors_{nullptr};

    /// Beta projectors for row G+k vectors.
    /** Used to setup the full Hamiltonian in PP-PW case (for verification purpose only) */
    std::unique_ptr<Beta_projectors<T>> beta_projectors_row_{nullptr};

    /// Beta projectors for column G+k vectors.
    /** Used to setup the full Hamiltonian in PP-PW case (for verification purpose only) */
    std::unique_ptr<Beta_projectors<T>> beta_projectors_col_{nullptr};

    /// Communicator between(!!) rows.
    mpi::Communicator const& comm_row_;

    /// Communicator between(!!) columns.
    mpi::Communicator const& comm_col_;

    std::array<int, 2> ispn_map_{0, -1};

    /// Generate G+k and local orbital basis sets.
    void
    generate_gklo_basis();

    /// Find G+k vectors within the cutoff.
    void
    generate_gkvec(double gk_cutoff__);

    inline int
    get_ispn(int ispn__) const
    {
        RTE_ASSERT(ispn__ == 0 || ispn__ == 1);
        return ispn_map_[ispn__];
    }

    friend class K_point_set;

    void
    init0()
    {
        band_occupancies_ =
                mdarray<double, 2>({ctx_.num_bands(), ctx_.num_spinors()}, mdarray_label("band_occupancies"));
        band_occupancies_.zero();
        band_energies_ = mdarray<double, 2>({ctx_.num_bands(), ctx_.num_spinors()}, mdarray_label("band_energies"));
        band_energies_.zero();

        if (ctx_.num_mag_dims() == 1) {
            ispn_map_[1] = 1;
        } else if (ctx_.num_mag_dims() == 3) {
            ispn_map_[1] = 0;
        }
    }

  public:
    /// Constructor
    K_point(Simulation_context& ctx__, r3::vector<double> vk__, double weight__)
        : ctx_(ctx__)
        , unit_cell_(ctx_.unit_cell())
        , vk_(vk__)
        , weight_(weight__)
        , comm_(ctx_.comm_band())
        , rank_col_(ctx_.blacs_grid().comm_col().rank())
        , num_ranks_col_(ctx_.blacs_grid().comm_col().size())
        , rank_row_(ctx_.blacs_grid().comm_row().rank())
        , num_ranks_row_(ctx_.blacs_grid().comm_row().size())
        , comm_row_(ctx_.blacs_grid().comm_row())
        , comm_col_(ctx_.blacs_grid().comm_col())
    {
        this->init0();
        gkvec_ = std::make_shared<fft::Gvec>(vk_, unit_cell_.reciprocal_lattice_vectors(), ctx_.gk_cutoff(), comm_,
                                             ctx_.gamma_point());
    }

    /// Constructor
    K_point(Simulation_context& ctx__, std::shared_ptr<fft::Gvec> gkvec__, double weight__)
        : ctx_(ctx__)
        , unit_cell_(ctx_.unit_cell())
        , vk_(gkvec__->vk())
        , weight_(weight__)
        , comm_(ctx_.comm_band())
        , gkvec_(gkvec__)
        , rank_col_(ctx_.blacs_grid().comm_col().rank())
        , num_ranks_col_(ctx_.blacs_grid().comm_col().size())
        , rank_row_(ctx_.blacs_grid().comm_row().rank())
        , num_ranks_row_(ctx_.blacs_grid().comm_row().size())
        , comm_row_(ctx_.blacs_grid().comm_row())
        , comm_col_(ctx_.blacs_grid().comm_col())
    {
        this->init0();
    }

    /// Initialize the k-point related arrays and data.
    void
    initialize(); // TODO: initialize from HDF5

    /// Update the reciprocal lattice vectors of the G+k array.
    void
    update();

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
    void
    generate_fv_states();

    /// Generate two-component spinor wave functions.
    /** In case of second-variational diagonalization spinor wave-functions are generated from the first-variational
        states and second-variational eigen-vectors. */
    void
    generate_spinor_wave_functions();

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
    void
    generate_atomic_wave_functions(std::vector<int> atoms__, std::function<basis_functions_index const*(int)> indexb__,
                                   Radial_integrals_atomic_wf<false> const& ri__, wf::Wave_functions<T>& wf__);
    void
    generate_hubbard_orbitals();

    /// Save data to HDF5 file.
    void
    save(std::string const& name__, int id__) const;

    void
    load(HDF5_tree h5in, int id);

    //== void save_wave_functions(int id);

    //== void load_wave_functions(int id);

    /// Collect distributed first-variational vectors into a global array.
    void
    get_fv_eigen_vectors(mdarray<std::complex<T>, 2>& fv_evec__) const;

    /// Collect distributed second-variational vectors into a global array.
    void
    get_sv_eigen_vectors(mdarray<std::complex<T>, 2>& sv_evec__) const
    {
        RTE_ASSERT((int)sv_evec__.size(0) == ctx_.num_spins() * ctx_.num_fv_states());
        RTE_ASSERT((int)sv_evec__.size(1) == ctx_.num_spins() * ctx_.num_fv_states());

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
                    int i                         = sv_eigen_vectors_[ispn].irow(iloc);
                    sv_evec__(i + offs, j + offs) = sv_eigen_vectors_[ispn](iloc, jloc);
                }
            }
        }

        comm().allreduce(sv_evec__.at(memory_t::host), (int)sv_evec__.size());
    }

    inline auto const&
    gkvec() const
    {
        return *gkvec_;
    }

    /// Return shared pointer to gkvec object.
    inline auto
    gkvec_sptr() const
    {
        return gkvec_;
    }

    /// Total number of G+k vectors within the cutoff distance
    inline int
    num_gkvec() const
    {
        return gkvec_->num_gvec();
    }

    /// Local number of G+k vectors in case of flat distribution.
    inline int
    num_gkvec_loc() const
    {
        return gkvec().count();
    }

    /// Get the number of occupied bands for each spin channel.
    inline int
    num_occupied_bands(int ispn__ = -1) const
    {
        if (ctx_.num_mag_dims() == 3) {
            ispn__ = 0;
        }
        for (int j = ctx_.num_bands() - 1; j >= 0; j--) {
            if (std::abs(band_occupancy(j, ispn__)) > ctx_.min_occupancy() * ctx_.max_occupancy()) {
                return j + 1;
            }
        }
        return 0;
    }

    /// Get band energy.
    inline double
    band_energy(int j__, int ispn__) const
    {
        return band_energies_(j__, get_ispn(ispn__));
    }

    /// Set band energy.
    inline void
    band_energy(int j__, int ispn__, double e__)
    {
        band_energies_(j__, get_ispn(ispn__)) = e__;
    }

    inline auto
    band_energies(int ispn__) const
    {
        std::vector<double> result(ctx_.num_bands());
        for (int j = 0; j < ctx_.num_bands(); j++) {
            result[j] = this->band_energy(j, ispn__);
        }
        return result;
    }

    /// Get band occupancy.
    inline double
    band_occupancy(int j__, int ispn__) const
    {
        return band_occupancies_(j__, get_ispn(ispn__));
    }

    /// Set band occupancy.
    inline void
    band_occupancy(int j__, int ispn__, double occ__)
    {
        band_occupancies_(j__, get_ispn(ispn__)) = occ__;
    }

    inline double
    fv_eigen_value(int i) const
    {
        return fv_eigen_values_[i];
    }

    void
    set_fv_eigen_values(double* eval__)
    {
        std::copy(eval__, eval__ + ctx_.num_fv_states(), &fv_eigen_values_[0]);
    }

    /// Return weight of k-point.
    inline double
    weight() const
    {
        return weight_;
    }

    inline auto&
    fv_states()
    {
        RTE_ASSERT(fv_states_ != nullptr);
        return *fv_states_;
    }

    inline wf::Wave_functions<T> const&
    spinor_wave_functions() const
    {
        RTE_ASSERT(spinor_wave_functions_ != nullptr);
        return *spinor_wave_functions_;
    }

    inline wf::Wave_functions<T>&
    spinor_wave_functions()
    {
        RTE_ASSERT(spinor_wave_functions_ != nullptr);
        return *spinor_wave_functions_;
        // return const_cast<wf::Wave_functions<T>&>(static_cast<K_point const&>(*this).spinor_wave_functions());;
    }

    inline auto&
    spinor_wave_functions2()
    {
        RTE_ASSERT(spinor_wave_functions_ != nullptr);
        return *spinor_wave_functions_;
        // return const_cast<wf::Wave_functions<T>&>(static_cast<K_point const&>(*this).spinor_wave_functions());;
    }

    /// Return the initial atomic orbitals used to compute the hubbard wave functions. The S operator is applied on
    /// these functions.
    inline auto const&
    atomic_wave_functions_S() const
    {
        /* the S operator is applied on these functions */
        RTE_ASSERT(atomic_wave_functions_S_ != nullptr);
        return *atomic_wave_functions_S_;
    }

    inline auto&
    atomic_wave_functions_S()
    {
        return const_cast<wf::Wave_functions<T>&>(static_cast<K_point const&>(*this).atomic_wave_functions_S());
    }

    /// Return the initial atomic orbitals used to compute the hubbard wave functions.
    inline auto const&
    atomic_wave_functions() const
    {
        RTE_ASSERT(atomic_wave_functions_ != nullptr);
        return *atomic_wave_functions_;
    }

    /// Return the initial atomic orbitals used to compute the hubbard wave functions.
    inline auto&
    atomic_wave_functions()
    {
        return const_cast<wf::Wave_functions<T>&>(static_cast<K_point const&>(*this).atomic_wave_functions());
    }

    /// Return the actual hubbard wave functions used in the calculations.
    /** The S operator is applied on these functions. */
    inline auto const&
    hubbard_wave_functions_S() const
    {
        RTE_ASSERT(hubbard_wave_functions_S_ != nullptr);
        return *hubbard_wave_functions_S_;
    }

    inline auto&
    singular_components()
    {
        return *singular_components_;
    }

    inline auto
    vk() const
    {
        return vk_;
    }

    /// Basis size of LAPW+lo method.
    /** The total LAPW+lo basis size is equal to the sum of the number of LAPW functions and the total number
     *  of the local orbitals. */
    inline int
    gklo_basis_size() const
    {
        return num_gkvec() + unit_cell_.mt_lo_basis_size();
    }

    /// Local number of G+k vectors for each MPI rank in the row of the 2D MPI grid.
    inline int
    num_gkvec_row() const
    {
        return num_gkvec_row_;
    }

    /// Local number of local orbitals for each MPI rank in the row of the 2D MPI grid.
    inline int
    num_lo_row() const
    {
        return static_cast<int>(lo_basis_descriptors_row_.size());
    }

    /// Local number of basis functions for each MPI rank in the row of the 2D MPI grid.
    inline int
    gklo_basis_size_row() const
    {
        return num_gkvec_row() + num_lo_row();
    }

    /// Local number of G+k vectors for each MPI rank in the column of the 2D MPI grid.
    inline int
    num_gkvec_col() const
    {
        return num_gkvec_col_;
    }

    /// Local number of local orbitals for each MPI rank in the column of the 2D MPI grid.
    inline int
    num_lo_col() const
    {
        return static_cast<int>(lo_basis_descriptors_col_.size());
    }

    /// Local number of basis functions for each MPI rank in the column of the 2D MPI grid.
    inline int
    gklo_basis_size_col() const
    {
        return num_gkvec_col() + num_lo_col();
    }

    inline lo_basis_descriptor const&
    lo_basis_descriptor_col(int idx) const
    {
        RTE_ASSERT(idx >= 0 && idx < (int)lo_basis_descriptors_col_.size());
        return lo_basis_descriptors_col_[idx];
    }

    inline lo_basis_descriptor const&
    lo_basis_descriptor_row(int idx) const
    {
        RTE_ASSERT(idx >= 0 && idx < (int)lo_basis_descriptors_row_.size());
        return lo_basis_descriptors_row_[idx];
    }

    inline int
    num_ranks_row() const
    {
        return num_ranks_row_;
    }

    inline int
    rank_row() const
    {
        return rank_row_;
    }

    inline int
    num_ranks_col() const
    {
        return num_ranks_col_;
    }

    inline int
    rank_col() const
    {
        return rank_col_;
    }

    /// Number of MPI ranks for a given k-point
    inline int
    num_ranks() const
    {
        return comm_.size();
    }

    inline int
    rank() const
    {
        return comm_.rank();
    }

    /// Return number of lo columns for a given atom
    inline int
    num_atom_lo_cols(int ia) const
    {
        return (int)atom_lo_cols_[ia].size();
    }

    /// Return local index (for the current MPI rank) of a column for a given atom and column index within an atom
    inline int
    lo_col(int ia, int i) const
    {
        return atom_lo_cols_[ia][i];
    }

    /// Return number of lo rows for a given atom
    inline int
    num_atom_lo_rows(int ia) const
    {
        return (int)atom_lo_rows_[ia].size();
    }

    /// Return local index (for the current MPI rank) of a row for a given atom and row index within an atom
    inline int
    lo_row(int ia, int i) const
    {
        return atom_lo_rows_[ia][i];
    }

    inline auto&
    fv_eigen_vectors()
    {
        return fv_eigen_vectors_;
    }

    inline auto&
    fv_eigen_vectors_slab()
    {
        return *fv_eigen_vectors_slab_;
    }

    inline auto&
    sv_eigen_vectors(int ispn)
    {
        return sv_eigen_vectors_[ispn];
    }

    inline auto&
    fd_eigen_vectors()
    {
        return fd_eigen_vectors_;
    }

    void
    bypass_sv()
    {
        std::copy(&fv_eigen_values_[0], &fv_eigen_values_[0] + ctx_.num_fv_states(), &band_energies_[0]);
    }

    inline auto const&
    alm_coeffs_row() const
    {
        return *alm_coeffs_row_;
    }

    inline auto const&
    alm_coeffs_col() const
    {
        return *alm_coeffs_col_;
    }

    inline auto const&
    alm_coeffs_loc() const
    {
        return *alm_coeffs_loc_;
    }

    inline auto const&
    comm() const
    {
        return comm_;
    }

    inline auto const&
    comm_row() const
    {
        return comm_row_;
    }

    inline auto const&
    comm_col() const
    {
        return comm_col_;
    }

    auto
    beta_projectors() -> Beta_projectors<T>&
    {
        RTE_ASSERT(beta_projectors_ != nullptr);
        return *beta_projectors_;
    }

    auto
    beta_projectors() const -> const Beta_projectors<T>&
    {
        RTE_ASSERT(beta_projectors_ != nullptr);
        return *beta_projectors_;
    }

    auto&
    beta_projectors_row()
    {
        RTE_ASSERT(beta_projectors_ != nullptr);
        return *beta_projectors_row_;
    }

    auto&
    beta_projectors_col()
    {
        RTE_ASSERT(beta_projectors_ != nullptr);
        return *beta_projectors_col_;
    }

    auto const&
    ctx() const
    {
        return ctx_;
    }

    /// Return stdout stream for high verbosity output.
    /** This output will not be passed to a ctx_.out() stream. */
    std::ostream&
    out(int level__) const
    {
        if (ctx_.verbosity() >= level__ && this->comm().rank() == 0) {
            return std::cout;
        } else {
            return null_stream();
        }
    }

    auto&
    spfft_transform() const
    {
        return *spfft_transform_;
    }

    inline auto const&
    gkvec_fft() const
    {
        return *gkvec_partition_;
    }

    inline auto
    gkvec_fft_sptr() const
    {
        return gkvec_partition_;
    }

    inline auto const&
    gkvec_col() const
    {
        return *gkvec_col_;
    }

    inline auto const&
    gkvec_row() const
    {
        return *gkvec_row_;
    }
};

template <typename T>
inline auto
wave_function_factory(Simulation_context const& ctx__, K_point<T> const& kp__, wf::num_bands num_wf__,
                      wf::num_mag_dims num_md__, bool mt_part__)
{
    using wf_t = wf::Wave_functions<T>;
    std::unique_ptr<wf_t> wf{nullptr};
    if (mt_part__) {
        std::vector<int> num_mt_coeffs(ctx__.unit_cell().num_atoms());
        for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
            num_mt_coeffs[ia] = ctx__.unit_cell().atom(ia).mt_lo_basis_size();
        }
        wf = std::make_unique<wf_t>(kp__.gkvec_sptr(), num_mt_coeffs, num_md__, num_wf__, ctx__.host_memory_t());
    } else {
        wf = std::make_unique<wf_t>(kp__.gkvec_sptr(), num_md__, num_wf__, ctx__.host_memory_t());
    }

    return wf;
}

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
