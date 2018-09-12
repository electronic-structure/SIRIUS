// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Contains definition and partial implementation of sirius::K_point class.
 */

#ifndef __K_POINT_HPP__
#define __K_POINT_HPP__

#include "matching_coefficients.h"
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
        std::unique_ptr<Wave_functions> hubbard_wave_functions_{nullptr};

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
        inline void generate_gklo_basis();

        /// Test orthonormalization of first-variational states.
        inline void test_fv_states();

        inline double& band_energy_aux(int j__, int ispn__)
        {
            if (ctx_.num_mag_dims() == 3) {
                return band_energies_(j__, 0);
            } else {
                if (!(ispn__ == 0 || ispn__ == 1)) {
                    TERMINATE("wrong spin index");
                }
                return band_energies_(j__, ispn__);
            }
        }

        inline double& band_occupancy_aux(int j__, int ispn__)
        {
            if (ctx_.num_mag_dims() == 3) {
                return band_occupancies_(j__, 0);
            } else {
                if (!(ispn__ == 0 || ispn__ == 1)) {
                    TERMINATE("wrong spin index");
                }
                return band_occupancies_(j__, ispn__);
            }
        }

        /// Find G+k vectors within the cutoff.
        inline void generate_gkvec(double gk_cutoff__)
        {
            PROFILE("sirius::K_point::generate_gkvec");

            if (ctx_.full_potential() && (gk_cutoff__ * unit_cell_.max_mt_radius() > ctx_.lmax_apw()) &&
                comm_.rank() == 0 && ctx_.control().verbosity_ >= 0) {
                std::stringstream s;
                s << "G+k cutoff (" << gk_cutoff__ << ") is too large for a given lmax ("
                  << ctx_.lmax_apw() << ") and a maximum MT radius (" << unit_cell_.max_mt_radius() << ")" << std::endl
                  << "suggested minimum value for lmax : " << int(gk_cutoff__ * unit_cell_.max_mt_radius()) + 1;
                WARNING(s);
            }

            if (gk_cutoff__ * 2 > ctx_.pw_cutoff()) {
                std::stringstream s;
                s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
                  << "  pw cutoff : " << ctx_.pw_cutoff() << std::endl
                  << "  doubled G+k cutoff : " << gk_cutoff__ * 2;
                TERMINATE(s);
            }

            /* create G+k vectors; communicator of the coarse FFT grid is used because wave-functions will be transformed
             * only on the coarse grid; G+k-vectors will be distributed between MPI ranks assigned to the k-point */
            gkvec_ = std::unique_ptr<Gvec>(new Gvec(vk_, ctx_.unit_cell().reciprocal_lattice_vectors(), gk_cutoff__, comm(),
                                                    ctx_.gamma_point()));

            gkvec_partition_ = std::unique_ptr<Gvec_partition>(new Gvec_partition(*gkvec_, ctx_.comm_fft_coarse(),
                                                                                  ctx_.comm_band_ortho_fft_coarse()));

            gkvec_offset_ = gkvec().gvec_offset(comm().rank());
        }

    public:

        /// Constructor
        K_point(Simulation_context& ctx__,
                double const* vk__,
                double weight__)
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
        inline void initialize();

        inline void update()
        {
            PROFILE("sirius::K_point::update");

            gkvec_->lattice_vectors(ctx_.unit_cell().reciprocal_lattice_vectors());

            if (ctx_.full_potential()) {
                if (ctx_.iterative_solver_input().type_ == "exact") {
                    alm_coeffs_row_ = std::unique_ptr<Matching_coefficients>(
                        new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_row(), igk_row_, gkvec()));
                    alm_coeffs_col_ = std::unique_ptr<Matching_coefficients>(
                        new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_col(), igk_col_, gkvec()));
                }
                alm_coeffs_loc_ = std::unique_ptr<Matching_coefficients>(
                    new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_loc(), igk_loc_, gkvec()));
            }

            if (!ctx_.full_potential()) {
                /* compute |beta> projectors for atom types */
                beta_projectors_ = std::unique_ptr<Beta_projectors>(new Beta_projectors(ctx_, gkvec(), igk_loc_));

                if (ctx_.iterative_solver_input().type_ == "exact") {
                    beta_projectors_row_ = std::unique_ptr<Beta_projectors>(new Beta_projectors(ctx_, gkvec(), igk_row_));
                    beta_projectors_col_ = std::unique_ptr<Beta_projectors>(new Beta_projectors(ctx_, gkvec(), igk_col_));

                }

                //if (false) {
                //    p_mtrx_ = mdarray<double_complex, 3>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), unit_cell_.num_atom_types());
                //    p_mtrx_.zero();

                //    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                //        auto& atom_type = unit_cell_.atom_type(iat);

                //        if (!atom_type.pp_desc().augment) {
                //            continue;
                //        }
                //        int nbf = atom_type.mt_basis_size();
                //        int ofs = atom_type.offset_lo();

                //        matrix<double_complex> qinv(nbf, nbf);
                //        for (int xi1 = 0; xi1 < nbf; xi1++) {
                //            for (int xi2 = 0; xi2 < nbf; xi2++) {
                //                qinv(xi2, xi1) = ctx_.augmentation_op(iat).q_mtrx(xi2, xi1);
                //            }
                //        }
                //        linalg<CPU>::geinv(nbf, qinv);
                //
                //        /* compute P^{+}*P */
                //        linalg<CPU>::gemm(2, 0, nbf, nbf, num_gkvec_loc(),
                //                          beta_projectors_->beta_gk_t().at<CPU>(0, ofs), beta_projectors_->beta_gk_t().ld(),
                //                          beta_projectors_->beta_gk_t().at<CPU>(0, ofs), beta_projectors_->beta_gk_t().ld(),
                //                          &p_mtrx_(0, 0, iat), p_mtrx_.ld());
                //        comm().allreduce(&p_mtrx_(0, 0, iat), unit_cell_.max_mt_basis_size() * unit_cell_.max_mt_basis_size());

                //        for (int xi1 = 0; xi1 < nbf; xi1++) {
                //            for (int xi2 = 0; xi2 < nbf; xi2++) {
                //                qinv(xi2, xi1) += p_mtrx_(xi2, xi1, iat);
                //            }
                //        }
                //        /* compute (Q^{-1} + P^{+}*P)^{-1} */
                //        linalg<CPU>::geinv(nbf, qinv);
                //        for (int xi1 = 0; xi1 < nbf; xi1++) {
                //            for (int xi2 = 0; xi2 < nbf; xi2++) {
                //                p_mtrx_(xi2, xi1, iat) = qinv(xi2, xi1);
                //            }
                //        }
                //    }
                //}
            }

        }

        /// Generate first-variational states from eigen-vectors.
        /** First-variational states are obtained from the first-variational eigen-vectors and
         *  LAPW matching coefficients.
         *
         *  APW part:
         *  \f[
         *      \psi_{\xi j}^{\bf k} = \sum_{{\bf G}} Z_{{\bf G} j}^{\bf k} * A_{\xi}({\bf G+k})
         *  \f]
         */
        void generate_fv_states();

        #ifdef __GPU
        void generate_fv_states_aw_mt_gpu();
        #endif

        /// Generate two-component spinor wave functions
        inline void generate_spinor_wave_functions();

        inline void generate_atomic_wave_functions(const int num_ao__, Wave_functions &phi);

        inline void generate_atomic_wave_functions_aux(const int num_ao__, Wave_functions &phi, std::vector<int> &offset, bool hubbard);

        void compute_gradient_wave_functions(Wave_functions &phi,
                                             const int starting_position_i,
                                             const int num_wf,
                                             Wave_functions &dphi,
                                             const int starting_position_j,
                                             const int direction);

        void save(int id);

        void load(HDF5_tree h5in, int id);

        //== void save_wave_functions(int id);

        //== void load_wave_functions(int id);

        void get_fv_eigen_vectors(mdarray<double_complex, 2>& fv_evec);

        void get_sv_eigen_vectors(mdarray<double_complex, 2>& sv_evec);

        /// Test orthonormalization of spinor wave-functions
        void test_spinor_wave_functions(int use_fft);

        /// Get the number of occupied bands for each spin channel.
        int num_occupied_bands(int ispn__ = -1)
        {
            for (int j = ctx_.num_bands() - 1; j >= 0; j--) {
                if (std::abs(band_occupancy(j, ispn__) * weight()) > 1e-14) {
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

        /// Total number of muffin-tin and plane-wave expansion coefficients for the wave-functions.
        /** APW+lo basis \f$ \varphi_{\mu {\bf k}}({\bf r}) = \{ \varphi_{\bf G+k}({\bf r}),
         *  \varphi_{j{\bf k}}({\bf r}) \} \f$ is used to expand first-variational wave-functions:
         *
         *  \f[
         *      \psi_{i{\bf k}}({\bf r}) = \sum_{\mu} c_{\mu i}^{\bf k} \varphi_{\mu \bf k}({\bf r}) =
         *      \sum_{{\bf G}}c_{{\bf G} i}^{\bf k} \varphi_{\bf G+k}({\bf r}) +
         *      \sum_{j}c_{j i}^{\bf k}\varphi_{j{\bf k}}({\bf r})
         *  \f]
         *
         *  Inside muffin-tins the expansion is converted into the following form:
         *  \f[
         *      \psi_{i {\bf k}}({\bf r})= \begin{array}{ll}
         *      \displaystyle \sum_{L} \sum_{\lambda=1}^{N_{\ell}^{\alpha}}
         *      F_{L \lambda}^{i {\bf k},\alpha}f_{\ell \lambda}^{\alpha}(r)
         *      Y_{\ell m}(\hat {\bf r}) & {\bf r} \in MT_{\alpha} \end{array}
         *  \f]
         *
         *  Thus, the total number of coefficients representing a wave-funstion is equal
         *  to the number of muffin-tin basis functions of the form \f$ f_{\ell \lambda}^{\alpha}(r)
         *  Y_{\ell m}(\hat {\bf r}) \f$ plust the number of G+k plane waves. */
        //inline int wf_size() const // TODO: better name for this
        //{
        //    if (ctx_.full_potential()) {
        //        return unit_cell_.mt_basis_size() + num_gkvec();
        //    } else {
        //        return num_gkvec();
        //    }
        //}

        inline double& band_energy(int j__, int ispn__)
        {
            return band_energy_aux(j__, ispn__);
        }

        inline double band_energy(int j__, int ispn__) const
        {
            auto const& e = const_cast<K_point*>(this)->band_energy_aux(j__, ispn__);
            return e;
        }

        inline double& band_occupancy(int j__, int ispn__)
        {
            return band_occupancy_aux(j__, ispn__);
        }

        inline double band_occupancy(int j__, int ispn__) const
        {
            auto const& e = const_cast<K_point*>(this)->band_occupancy_aux(j__, ispn__);
            return e;
        }

        inline double fv_eigen_value(int i) const
        {
            return fv_eigen_values_[i];
        }

        void set_fv_eigen_values(double* eval)
        {
            std::memcpy(&fv_eigen_values_[0], eval, ctx_.num_fv_states() * sizeof(double));
        }

        inline double weight() const
        {
            return weight_;
        }

        inline Wave_functions& fv_states()
        {
            return *fv_states_;
        }

        inline Wave_functions& spinor_wave_functions()
        {
            return *spinor_wave_functions_;
        }

        inline Wave_functions& hubbard_wave_functions()
        {
            return *hubbard_wave_functions_;
        }

        inline Wave_functions const& hubbard_wave_functions() const
        {
            return *hubbard_wave_functions_;
        }

        inline void allocate_hubbard_wave_functions(int size)
        {
            if (hubbard_wave_functions_ != nullptr) {
                return;
            }
            const int num_sc = ctx_.num_mag_dims() == 3 ?  2 : 1;
            hubbard_wave_functions_ = std::unique_ptr<Wave_functions>(new Wave_functions(gkvec_partition(),
                                                                                         size,
                                                                                         num_sc));
        }

        inline bool hubbard_wave_functions_calculated()
        {
            return (hubbard_wave_functions_ != nullptr);
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
        inline int gklo_basis_size() const
        {
            return static_cast<int>(num_gkvec() + unit_cell_.mt_lo_basis_size());
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
            assert(idx >=0 && idx < (int)lo_basis_descriptors_col_.size());
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

        inline Matching_coefficients const& alm_coeffs_row()
        {
            return *alm_coeffs_row_;
        }

        inline Matching_coefficients const& alm_coeffs_col()
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
};

//== void K_point::check_alm(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm)
//== {
//==     static SHT* sht = NULL;
//==     if (!sht) sht = new SHT(ctx_.lmax_apw());
//== 
//==     Atom* atom = unit_cell_.atom(ia);
//==     Atom_type* type = atom->type();
//== 
//==     mdarray<double_complex, 2> z1(sht->num_points(), type->mt_aw_basis_size());
//==     for (int i = 0; i < type->mt_aw_basis_size(); i++)
//==     {
//==         int lm = type->indexb(i).lm;
//==         int idxrf = type->indexb(i).idxrf;
//==         double rf = atom->symmetry_class()->radial_function(atom->num_mt_points() - 1, idxrf);
//==         for (int itp = 0; itp < sht->num_points(); itp++)
//==         {
//==             z1(itp, i) = sht->ylm_backward(lm, itp) * rf;
//==         }
//==     }
//== 
//==     mdarray<double_complex, 2> z2(sht->num_points(), num_gkvec_loc);
//==     blas<CPU>::gemm(0, 2, sht->num_points(), num_gkvec_loc, type->mt_aw_basis_size(), z1.ptr(), z1.ld(),
//==                     alm.ptr(), alm.ld(), z2.ptr(), z2.ld());
//== 
//==     vector3d<double> vc = unit_cell_.get_cartesian_coordinates(unit_cell_.atom(ia)->position());
//==     
//==     double tdiff = 0;
//==     for (int igloc = 0; igloc < num_gkvec_loc; igloc++)
//==     {
//==         vector3d<double> gkc = gkvec_cart(igkglob(igloc));
//==         for (int itp = 0; itp < sht->num_points(); itp++)
//==         {
//==             double_complex aw_value = z2(itp, igloc);
//==             vector3d<double> r;
//==             for (int x = 0; x < 3; x++) r[x] = vc[x] + sht->coord(x, itp) * type->mt_radius();
//==             double_complex pw_value = exp(double_complex(0, Utils::scalar_product(r, gkc))) / sqrt(unit_cell_.omega());
//==             tdiff += abs(pw_value - aw_value);
//==         }
//==     }
//== 
//==     printf("atom : %i  absolute alm error : %e  average alm error : %e\n", 
//==            ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
//== }


//Periodic_function<double_complex>* K_point::spinor_wave_function_component(Band* band, int lmax, int ispn, int jloc)
//{
//    Timer t("sirius::K_point::spinor_wave_function_component");
//
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//
//    Periodic_function<double_complex, index_order>* func = 
//        new Periodic_function<double_complex, index_order>(ctx_, lmax);
//    func->allocate(ylm_component | it_component);
//    func->zero();
//    
//    if (basis_type == pwlo)
//    {
//        if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//
//        double fourpi_omega = fourpi / sqrt(ctx_.omega());
//        
//        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//        {
//            int igk = igkglob(igkloc);
//            double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//
//            // TODO: possilbe optimization with zgemm
//            for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//            {
//                int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//                
//                #pragma omp parallel for default(shared)
//                for (int lm = 0; lm < lmmax; lm++)
//                {
//                    int l = l_by_lm_(lm);
//                    double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                        func->f_ylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//                }
//            }
//        }
//
//        for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&func->f_ylm(0, 0, ia), lmmax * ctx_.max_num_mt_points(),
//                                ctx_.mpi_grid().communicator(1 << band->dim_row()));
//        }
//    }
//
//    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//    {
//        for (int i = 0; i < ctx_.atom(ia)->type()->mt_basis_size(); i++)
//        {
//            int lm = ctx_.atom(ia)->type()->indexb(i).lm;
//            int idxrf = ctx_.atom(ia)->type()->indexb(i).idxrf;
//            switch (index_order)
//            {
//                case angular_radial:
//                {
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(lm, ir, ia) += 
//                            spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//                case radial_angular:
//                {
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(ir, lm, ia) += 
//                            spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//            }
//        }
//    }
//
//    // in principle, wave function must have an overall e^{ikr} phase factor
//    ctx_.fft().input(num_gkvec(), &fft_index_[0], 
//                            &spinor_wave_functions_(ctx_.mt_basis_size(), ispn, jloc));
//    ctx_.fft().transform(1);
//    ctx_.fft().output(func->f_it());
//
//    for (int i = 0; i < ctx_.fft().size(); i++) func->f_it(i) /= sqrt(ctx_.omega());
//    
//    return func;
//}

//== void K_point::spinor_wave_function_component_mt(int lmax, int ispn, int jloc, mt_functions<double_complex>& psilm)
//== {
//==     Timer t("sirius::K_point::spinor_wave_function_component_mt");
//== 
//==     //int lmmax = Utils::lmmax_by_lmax(lmax);
//== 
//==     psilm.zero();
//==     
//==     //if (basis_type == pwlo)
//==     //{
//==     //    if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//== 
//==     //    double fourpi_omega = fourpi / sqrt(ctx_.omega());
//== 
//==     //    mdarray<double_complex, 2> zm(ctx_.max_num_mt_points(),  num_gkvec_row());
//== 
//==     //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    {
//==     //        int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//==     //        for (int l = 0; l <= lmax; l++)
//==     //        {
//==     //            #pragma omp parallel for default(shared)
//==     //            for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //            {
//==     //                int igk = igkglob(igkloc);
//==     //                double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//==     //                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia) * zil_[l];
//==     //                for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==     //                    zm(ir, igkloc) = z2 * (*sbessel_[igkloc])(ir, l, iat);
//==     //            }
//==     //            blas<CPU>::gemm(0, 2, ctx_.atom(ia)->num_mt_points(), (2 * l + 1), num_gkvec_row(),
//==     //                            &zm(0, 0), zm.ld(), &gkvec_ylm_(Utils::lm_by_l_m(l, -l), 0), gkvec_ylm_.ld(), 
//==     //                            &fylm(0, Utils::lm_by_l_m(l, -l), ia), fylm.ld());
//==     //        }
//==     //    }
//==     //    //for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //    //{
//==     //    //    int igk = igkglob(igkloc);
//==     //    //    double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//== 
//==     //    //    // TODO: possilbe optimization with zgemm
//==     //    //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    //    {
//==     //    //        int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//==     //    //        double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//==     //    //        
//==     //    //        #pragma omp parallel for default(shared)
//==     //    //        for (int lm = 0; lm < lmmax; lm++)
//==     //    //        {
//==     //    //            int l = l_by_lm_(lm);
//==     //    //            double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//==     //    //            for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==     //    //                fylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//==     //    //        }
//==     //    //    }
//==     //    //}
//== 
//==     //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    {
//==     //        Platform::allreduce(&fylm(0, 0, ia), lmmax * ctx_.max_num_mt_points(),
//==     //                            ctx_.mpi_grid().communicator(1 << band->dim_row()));
//==     //    }
//==     //}
//== 
//==     for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     {
//==         for (int i = 0; i < ctx_.atom(ia)->type()->mt_basis_size(); i++)
//==         {
//==             int lm = ctx_.atom(ia)->type()->indexb(i).lm;
//==             int idxrf = ctx_.atom(ia)->type()->indexb(i).idxrf;
//==             for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==             {
//==                 psilm(lm, ir, ia) += 
//==                     spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//==                     ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//==             }
//==         }
//==     }
//== }

inline void K_point::test_spinor_wave_functions(int use_fft)
{
    STOP();

//==     if (num_ranks() > 1) error_local(__FILE__, __LINE__, "test of spinor wave functions on multiple ranks is not implemented");
//== 
//==     std::vector<double_complex> v1[2];
//==     std::vector<double_complex> v2;
//== 
//==     if (use_fft == 0 || use_fft == 1) v2.resize(fft_->size());
//==     
//==     if (use_fft == 0) 
//==     {
//==         for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) v1[ispn].resize(num_gkvec());
//==     }
//==     
//==     if (use_fft == 1) 
//==     {
//==         for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) v1[ispn].resize(fft_->size());
//==     }
//==     
//==     double maxerr = 0;
//== 
//==     for (int j1 = 0; j1 < ctx_.num_bands(); j1++)
//==     {
//==         if (use_fft == 0)
//==         {
//==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==             {
//==                 fft_->input(num_gkvec(), gkvec_.index_map(),
//==                                        &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j1));
//==                 fft_->transform(1);
//==                 fft_->output(&v2[0]);
//== 
//==                 for (int ir = 0; ir < fft_->size(); ir++) v2[ir] *= ctx_.step_function()->theta_r(ir);
//==                 
//==                 fft_->input(&v2[0]);
//==                 fft_->transform(-1);
//==                 fft_->output(num_gkvec(), gkvec_.index_map(), &v1[ispn][0]); 
//==             }
//==         }
//==         
//==         if (use_fft == 1)
//==         {
//==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==             {
//==                 fft_->input(num_gkvec(), gkvec_.index_map(),
//==                                        &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j1));
//==                 fft_->transform(1);
//==                 fft_->output(&v1[ispn][0]);
//==             }
//==         }
//==        
//==         for (int j2 = 0; j2 < ctx_.num_bands(); j2++)
//==         {
//==             double_complex zsum(0, 0);
//==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==             {
//==                 for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==                 {
//==                     int offset_wf = unit_cell_.atom(ia)->offset_wf();
//==                     Atom_type* type = unit_cell_.atom(ia)->type();
//==                     Atom_symmetry_class* symmetry_class = unit_cell_.atom(ia)->symmetry_class();
//== 
//==                     for (int l = 0; l <= ctx_.lmax_apw(); l++)
//==                     {
//==                         int ordmax = type->indexr().num_rf(l);
//==                         for (int io1 = 0; io1 < ordmax; io1++)
//==                         {
//==                             for (int io2 = 0; io2 < ordmax; io2++)
//==                             {
//==                                 for (int m = -l; m <= l; m++)
//==                                 {
//==                                     zsum += conj(spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io1), ispn, j1)) *
//==                                             spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io2), ispn, j2) * 
//==                                             symmetry_class->o_radial_integral(l, io1, io2);
//==                                 }
//==                             }
//==                         }
//==                     }
//==                 }
//==             }
//==             
//==             if (use_fft == 0)
//==             {
//==                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==                {
//==                    for (int ig = 0; ig < num_gkvec(); ig++)
//==                        zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(unit_cell_.mt_basis_size() + ig, ispn, j2);
//==                }
//==             }
//==            
//==             if (use_fft == 1)
//==             {
//==                 for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==                 {
//==                     fft_->input(num_gkvec(), gkvec_.index_map(), &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j2));
//==                     fft_->transform(1);
//==                     fft_->output(&v2[0]);
//== 
//==                     for (int ir = 0; ir < fft_->size(); ir++)
//==                         zsum += std::conj(v1[ispn][ir]) * v2[ir] * ctx_.step_function()->theta_r(ir) / double(fft_->size());
//==                 }
//==             }
//==             
//==             if (use_fft == 2) 
//==             {
//==                 STOP();
//==                 //for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
//==                 //{
//==                 //    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
//==                 //    {
//==                 //        int ig3 = ctx_.gvec().index_g12(ig1, ig2);
//==                 //        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
//==                 //        {
//==                 //            zsum += std::conj(spinor_wave_functions_(unit_cell_.mt_basis_size() + ig1, ispn, j1)) * 
//==                 //                    spinor_wave_functions_(unit_cell_.mt_basis_size() + ig2, ispn, j2) * 
//==                 //                    ctx_.step_function()->theta_pw(ig3);
//==                 //        }
//==                 //    }
//==                 //}
//==             }
//== 
//==             zsum = (j1 == j2) ? zsum - double_complex(1.0, 0.0) : zsum;
//==             maxerr = std::max(maxerr, std::abs(zsum));
//==         }
//==     }
//==     std :: cout << "maximum error = " << maxerr << std::endl;
}

inline void K_point::save(int id)
{
    if (num_ranks() > 1) TERMINATE("writing of distributed eigen-vectors is not implemented");

    STOP();

    //if (ctx_.mpi_grid().root(1 << _dim_col_))
    //{
    //    HDF5_tree fout(storage_file_name, false);

    //    fout["K_set"].create_node(id);
    //    fout["K_set"][id].create_node("spinor_wave_functions");
    //    fout["K_set"][id].write("coordinates", &vk_[0], 3);
    //    fout["K_set"][id].write("band_energies", band_energies_);
    //    fout["K_set"][id].write("band_occupancies", band_occupancies_);
    //    if (num_ranks() == 1)
    //    {
    //        fout["K_set"][id].write("fv_eigen_vectors", fv_eigen_vectors_panel_.data());
    //        fout["K_set"][id].write("sv_eigen_vectors", sv_eigen_vectors_[0].data());
    //    }
    //}
    //
    //comm_col_.barrier();
    //
    //mdarray<double_complex, 2> wfj(NULL, wf_size(), ctx_.num_spins()); 
    //for (int j = 0; j < ctx_.num_bands(); j++)
    //{
    //    int rank = ctx_.spl_spinor_wf().local_rank(j);
    //    int offs = (int)ctx_.spl_spinor_wf().local_index(j);
    //    if (ctx_.mpi_grid().coordinate(_dim_col_) == rank)
    //    {
    //        HDF5_tree fout(storage_file_name, false);
    //        wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
    //        fout["K_set"][id]["spinor_wave_functions"].write(j, wfj);
    //    }
    //    comm_col_.barrier();
    //}
}

inline void K_point::load(HDF5_tree h5in, int id)
{
    STOP();
    //== band_energies_.resize(ctx_.num_bands());
    //== h5in[id].read("band_energies", band_energies_);

    //== band_occupancies_.resize(ctx_.num_bands());
    //== h5in[id].read("band_occupancies", band_occupancies_);
    //== 
    //== h5in[id].read_mdarray("fv_eigen_vectors", fv_eigen_vectors_panel_);
    //== h5in[id].read_mdarray("sv_eigen_vectors", sv_eigen_vectors_);
}

//== void K_point::save_wave_functions(int id)
//== {
//==     if (ctx_.mpi_grid().root(1 << _dim_col_))
//==     {
//==         HDF5_tree fout(storage_file_name, false);
//== 
//==         fout["K_points"].create_node(id);
//==         fout["K_points"][id].write("coordinates", &vk_[0], 3);
//==         fout["K_points"][id].write("mtgk_size", mtgk_size());
//==         fout["K_points"][id].create_node("spinor_wave_functions");
//==         fout["K_points"][id].write("band_energies", &band_energies_[0], ctx_.num_bands());
//==         fout["K_points"][id].write("band_occupancies", &band_occupancies_[0], ctx_.num_bands());
//==     }
//==     
//==     Platform::barrier(ctx_.mpi_grid().communicator(1 << _dim_col_));
//==     
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), ctx_.num_spins()); 
//==     for (int j = 0; j < ctx_.num_bands(); j++)
//==     {
//==         int rank = ctx_.spl_spinor_wf_col().location(_splindex_rank_, j);
//==         int offs = ctx_.spl_spinor_wf_col().location(_splindex_offs_, j);
//==         if (ctx_.mpi_grid().coordinate(_dim_col_) == rank)
//==         {
//==             HDF5_tree fout(storage_file_name, false);
//==             wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
//==             fout["K_points"][id]["spinor_wave_functions"].write_mdarray(j, wfj);
//==         }
//==         Platform::barrier(ctx_.mpi_grid().communicator(_dim_col_));
//==     }
//== }
//== 
//== void K_point::load_wave_functions(int id)
//== {
//==     HDF5_tree fin(storage_file_name, false);
//==     
//==     int mtgk_size_in;
//==     fin["K_points"][id].read("mtgk_size", &mtgk_size_in);
//==     if (mtgk_size_in != mtgk_size()) error_local(__FILE__, __LINE__, "wrong wave-function size");
//== 
//==     band_energies_.resize(ctx_.num_bands());
//==     fin["K_points"][id].read("band_energies", &band_energies_[0], ctx_.num_bands());
//== 
//==     band_occupancies_.resize(ctx_.num_bands());
//==     fin["K_points"][id].read("band_occupancies", &band_occupancies_[0], ctx_.num_bands());
//== 
//==     spinor_wave_functions_.set_dimensions(mtgk_size(), ctx_.num_spins(), 
//==                                           ctx_.spl_spinor_wf_col().local_size());
//==     spinor_wave_functions_.allocate();
//== 
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), ctx_.num_spins()); 
//==     for (int jloc = 0; jloc < ctx_.spl_spinor_wf_col().local_size(); jloc++)
//==     {
//==         int j = ctx_.spl_spinor_wf_col(jloc);
//==         wfj.set_ptr(&spinor_wave_functions_(0, 0, jloc));
//==         fin["K_points"][id]["spinor_wave_functions"].read_mdarray(j, wfj);
//==     }
//== }

inline void K_point::get_fv_eigen_vectors(mdarray<double_complex, 2>& fv_evec)
{
    assert((int)fv_evec.size(0) >= gklo_basis_size());
    assert((int)fv_evec.size(1) == ctx_.num_fv_states());
    
    fv_evec.zero();
    STOP();

    //for (int iloc = 0; iloc < (int)spl_fv_states_.local_size(); iloc++)
    //{
    //    int i = (int)spl_fv_states_[iloc];
    //    for (int jloc = 0; jloc < gklo_basis_size_row(); jloc++)
    //    {
    //        int j = gklo_basis_descriptor_row(jloc).id;
    //        fv_evec(j, i) = fv_eigen_vectors_(jloc, iloc);
    //    }
    //}
    //comm_.allreduce(fv_evec.at<CPU>(), (int)fv_evec.size());
}

inline void K_point::get_sv_eigen_vectors(mdarray<double_complex, 2>& sv_evec)
{
    assert((int)sv_evec.size(0) == ctx_.num_bands());
    assert((int)sv_evec.size(1) == ctx_.num_bands());

    sv_evec.zero();

    if (!ctx_.need_sv()) {
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            sv_evec(i, i) = 1;
        }
        return;
    }

    int nsp = (ctx_.num_mag_dims() == 3) ? 1 : ctx_.num_spins();

    for (int ispn = 0; ispn < nsp; ispn++)
    {
        int offs = ctx_.num_fv_states() * ispn;
        for (int jloc = 0; jloc < sv_eigen_vectors_[ispn].num_cols_local(); jloc++)
        {
            int j = sv_eigen_vectors_[ispn].icol(jloc);
            for (int iloc = 0; iloc < sv_eigen_vectors_[ispn].num_rows_local(); iloc++)
            {
                int i = sv_eigen_vectors_[ispn].irow(iloc);
                sv_evec(i + offs, j + offs) = sv_eigen_vectors_[ispn](iloc, jloc);
            }
        }
    }

    comm_.allreduce(sv_evec.at<CPU>(), (int)sv_evec.size());
}

#include "generate_fv_states.hpp"
#include "generate_spinor_wave_functions.hpp"
#include "generate_gklo_basis.hpp"
#include "initialize.hpp"
#include "test_fv_states.hpp"
#include "generate_atomic_wave_functions.hpp"

}

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
