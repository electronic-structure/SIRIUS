// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file k_point.h
 *   
 *  \brief Contains definition and partial implementation of sirius::K_point class.
 */

#ifndef __K_POINT_H__
#define __K_POINT_H__

#include "periodic_function.h"
#include "matching_coefficients.h"
#include "blacs_grid.h"

namespace sirius
{

/// K-point related variables and methods.
/** \image html wf_storage.png "Wave-function storage" */ // TODO: replace with proper image
class K_point
{
    private:

        /// Simulation context.
        Simulation_context& ctx_;

        /// Parameters of simulation.
        Simulation_parameters const& parameters_;
        
        /// Unit cell object.
        Unit_cell const& unit_cell_;

        /// 2D BLACS grid for diagonalization and 2D data distribution.
        BLACS_grid const& blacs_grid_;
        
        /// 1D BLACS grid for a "slab" data distribution.
        /** This grid is used to distribute G+k vector index and keep a whole band index */
        BLACS_grid blacs_grid_slab_;
        
        /// 1D BLACS grid for a "slice" data distribution.
        /** This grid is used to distribute band index and keep a whole G+k vector index */
        BLACS_grid blacs_grid_slice_;

        /// Alias for FFT driver.
        FFT3D<CPU>* fft_;

        /// Weight of k-point.
        double weight_;

        /// Fractional k-point coordinates.
        vector3d<double> vk_;
        
        Gvec gkvec_;

        Gvec pgkvec_;

        /// first-variational eigen values
        std::vector<double> fv_eigen_values_;

        /// First-variational eigen vectors, distributed over 2D BLACS grid.
        dmatrix<double_complex> fv_eigen_vectors_;
        
        /// Second-variational eigen vectors.
        /** Second-variational eigen-vectors are stored as one or two \f$ N_{fv} \times N_{fv} \f$ matrices in
         *  case of non-magnetic or collinear magnetic case or as a single \f$ 2 N_{fv} \times 2 N_{fv} \f$ 
         *  matrix in case of general non-collinear magnetism. */
        dmatrix<double_complex> sv_eigen_vectors_[2];

        /// Full-diagonalization eigen vectors.
        mdarray<double_complex, 2> fd_eigen_vectors_;

        /// Position of the G vector (from the G+k set) inside the coarse FFT buffer.
        std::vector<int> fft_index_coarse_;
       
        /// First-variational states in "slice" storage.
        dmatrix<double_complex> fv_states_slice_;

        /// First-variational states, distributed over rows and columns of the MPI grid
        /** Band index is distributed over columns and basis functions index is distributed 
         *  over rows of the MPI grid. */
        dmatrix<double_complex> fv_states_;

        /// Two-component (spinor) wave functions describing the bands.
        dmatrix<double_complex> spinor_wave_functions_[2];

        /// band occupation numbers
        std::vector<double> band_occupancies_;

        /// band energies
        std::vector<double> band_energies_; 

        /// phase factors \f$ e^{i ({\bf G+k}) {\bf r}_{\alpha}} \f$
        mdarray<double_complex, 2> gkvec_phase_factors_;

        /// spherical harmonics of G+k vectors
        mdarray<double_complex, 2> gkvec_ylm_;

        Matching_coefficients* alm_coeffs_row_;

        Matching_coefficients* alm_coeffs_col_;

        Matching_coefficients* alm_coeffs_;

        /// number of G+k vectors distributed along rows of MPI grid
        int num_gkvec_row_;
        
        /// number of G+k vectors distributed along columns of MPI grid
        int num_gkvec_col_;

        /// Short information about each G+k or lo basis function.
        /** This is a global array. Each MPI rank of the 2D grid has exactly the same copy. */
        std::vector<gklo_basis_descriptor> gklo_basis_descriptors_;

        /// Basis descriptors distributed along rows of the 2D MPI grid.
        /** This is a local array. Only MPI ranks belonging to the same row have identical copies of this array. */
        std::vector<gklo_basis_descriptor> gklo_basis_descriptors_row_;
        
        /// basis descriptors distributed along columns of the 2D MPI grid
        /** This is a local array. Only MPI ranks belonging to the same column have identical copies of this array. */
        std::vector<gklo_basis_descriptor> gklo_basis_descriptors_col_;

        splindex<block> spl_gkvec_;
            
        /// list of columns of the Hamiltonian and overlap matrix lo block (local index) for a given atom
        std::vector< std::vector<int> > atom_lo_cols_;

        /// list of rows of the Hamiltonian and overlap matrix lo block (local index) for a given atom
        std::vector< std::vector<int> > atom_lo_rows_;

        /// imaginary unit to the power of l
        std::vector<double_complex> zil_;

        /// mapping between lm and l
        std::vector<int> l_by_lm_;

        /// column rank of the processors of ScaLAPACK/ELPA diagonalization grid
        int rank_col_;

        /// number of processors along columns of the diagonalization grid
        int num_ranks_col_;

        int rank_row_;

        int num_ranks_row_;

        int num_ranks_;

        /// Phase-factor independent plane-wave coefficients of |beta> functions for atom types.
        matrix<double_complex> beta_gk_t_;

        /// Plane-wave coefficients of |beta> functions for atoms.
        matrix<double_complex> beta_gk_;

        mdarray<double_complex, 3> p_mtrx_;

        mdarray<double, 2> gkvec_row_;

        Communicator comm_;

        /// Communicator between(!!) rows.
        Communicator comm_row_;

        /// Communicator between(!!) columns.
        Communicator comm_col_;

        /// Build G+k and lo basis descriptors.
        void build_gklo_basis_descriptors();

        /// Distribute basis function index between rows and columns of MPI grid.
        void distribute_basis_index();
        
        /// Test orthonormalization of first-variational states.
        void test_fv_states();

        void init_gkvec_phase_factors(int num_gkvec__, std::vector<gklo_basis_descriptor>& desc__);
        
        /// Generate plane-wave coefficients for beta-projectors of atom types.
        void generate_beta_gk_t();

    public:

        Iterative_solver_input_section iterative_solver_input_section_;

        /// Constructor
        K_point(Simulation_context& ctx__,
                double* vk__,
                double weight__,
                BLACS_grid const& blacs_grid__);

        ~K_point()
        {
            if (alm_coeffs_row_ != nullptr) delete alm_coeffs_row_;
            if (alm_coeffs_col_ != nullptr) delete alm_coeffs_col_;
            if (alm_coeffs_ != nullptr) delete alm_coeffs_;
        }

        /// Initialize the k-point related arrays and data
        void initialize();

        /// Find G+k vectors within the cutoff
        void generate_gkvec(double gk_cutoff);

        /// Generate first-variational states from eigen-vectors
        void generate_fv_states();

        #ifdef __GPU
        void generate_fv_states_aw_mt_gpu();
        #endif

        /// Generate two-component spinor wave functions 
        void generate_spinor_wave_functions();

        Periodic_function<double_complex>* spinor_wave_function_component(int lmax, int ispn, int j);

        void save(int id);

        void load(HDF5_tree h5in, int id);

        //== void save_wave_functions(int id);

        //== void load_wave_functions(int id);

        void get_fv_eigen_vectors(mdarray<double_complex, 2>& fv_evec);
        
        void get_sv_eigen_vectors(mdarray<double_complex, 2>& sv_evec);
        
        /// Test orthonormalization of spinor wave-functions
        void test_spinor_wave_functions(int use_fft);

        /// Get the local list of occupied bands.
        occupied_bands_descriptor get_occupied_bands_list();

        /// Return G+k vector in fractional or Cartesian coordinates
        template <coordinates_t coord__>
        inline vector3d<double> gkvec(int igk__) const
        {
            auto G = gkvec_[igk__];
            auto Gk = vector3d<double>(G[0], G[1], G[2]) + vk_;
            switch (coord__)
            {
                case cartesian:
                {
                    return unit_cell_.reciprocal_lattice_vectors() * Gk;
                    break;
                }
                case fractional:
                {
                    return Gk;
                    break;
                }
                default:
                {
                    TERMINATE("wrong type of coordinates");
                    return vector3d<double>(); // make compiler happy
                }
            }
        }
        
        inline double_complex gkvec_phase_factor(int igk__, int ia__) const
        {
            return gkvec_phase_factors_(igk__, ia__);
        }

        /// Total number of G+k vectors within the cutoff distance
        inline int num_gkvec() const
        {
            return gkvec_.num_gvec();
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
         *  Y_{\ell m}(\hat {\bf r}) \f$ plust the number of G+k plane waves. 
         */ 
        inline int wf_size() const // TODO: better name for this
        {
            switch (ctx_.parameters().esm_type())
            {
                case full_potential_lapwlo:
                case full_potential_pwlo:
                {
                    return unit_cell_.mt_basis_size() + num_gkvec();
                    break;
                }
                case ultrasoft_pseudopotential:
                case norm_conserving_pseudopotential:
                {
                    return num_gkvec();
                    break;
                }
            }
            return -1; // make compiler happy
        }

        inline int wf_pw_offset() const
        {
            switch (ctx_.parameters().esm_type())
            {
                case full_potential_lapwlo:
                case full_potential_pwlo:
                {
                    return unit_cell_.mt_basis_size();
                    break;
                }
                case ultrasoft_pseudopotential:
                case norm_conserving_pseudopotential:
                {
                    return 0;
                    break;
                }
                default:
                {
                    terminate(__FILE__, __LINE__, "wrong esm_type");
                    return -1; //make compiler happy
                }
            }
        }

        inline void get_band_occupancies(double* band_occupancies) const
        {
            assert((int)band_occupancies_.size() == parameters_.num_bands());
            
            memcpy(band_occupancies, &band_occupancies_[0], parameters_.num_bands() * sizeof(double));
        }

        inline void set_band_occupancies(double* band_occupancies)
        {
            band_occupancies_.resize(parameters_.num_bands());
            memcpy(&band_occupancies_[0], band_occupancies, parameters_.num_bands() * sizeof(double));
        }

        inline void get_band_energies(double* band_energies) const
        {
            assert((int)band_energies_.size() == parameters_.num_bands());
            
            memcpy(band_energies, &band_energies_[0], parameters_.num_bands() * sizeof(double));
        }

        inline void set_band_energies(double* band_energies)
        {
            band_energies_.resize(parameters_.num_bands()); 
            memcpy(&band_energies_[0], band_energies, parameters_.num_bands() * sizeof(double));
        }

        inline double band_occupancy(int j) const
        {
            return band_occupancies_[j];
        }
        
        inline double band_energy(int j) const
        {
            return band_energies_[j];
        }

        inline double fv_eigen_value(int i) const
        {
            return fv_eigen_values_[i];
        }

        void set_fv_eigen_values(double* eval)
        {
            memcpy(&fv_eigen_values_[0], eval, parameters_.num_fv_states() * sizeof(double));
        }
        
        inline double weight() const
        {
            return weight_;
        }

        inline dmatrix<double_complex>& spinor_wave_functions(int ispn__)
        {
            return spinor_wave_functions_[ispn__];
        }

        inline int const* fft_index_coarse() const
        {
            return &fft_index_coarse_[0];
        }

        inline vector3d<double> vk() const
        {
            return vk_;
        }

        /// Basis size of our electronic structure method.
        /** In case of full-potential LAPW+lo or PW+lo method the total number of 
         *  basis functions is equal to the number of (augmented) plane-waves plus the number 
         *  of local orbitals. In case of plane-wave pseudopotential method this is just the 
         *  number of G+k vectors. 
         */
        inline int gklo_basis_size() const
        {
            return (int)gklo_basis_descriptors_.size();
        }
        
        /// Local number of basis functions for each MPI rank in the row of the 2D MPI grid.
        inline int gklo_basis_size_row() const
        {
            return (int)gklo_basis_descriptors_row_.size();
        }
        
        /// Local number of G+k vectors for each MPI rank in the row of the 2D MPI grid.
        inline int num_gkvec_row() const
        {
            return num_gkvec_row_;
        }

        /// Local number of local orbitals for each MPI rank in the row of the 2D MPI grid.
        inline int num_lo_row() const
        {
            return (int)gklo_basis_descriptors_row_.size() - num_gkvec_row_;
        }

        /// Local number of basis functions for each MPI rank in the column of the 2D MPI grid.
        inline int gklo_basis_size_col() const
        {
            return (int)gklo_basis_descriptors_col_.size();
        }
        
        /// Local number of G+k vectors for each MPI rank in the column of the 2D MPI grid.
        inline int num_gkvec_col() const
        {
            return num_gkvec_col_;
        }
        
        /// Local number of local orbitals for each MPI rank in the column of the 2D MPI grid.
        inline int num_lo_col() const
        {
            return (int)gklo_basis_descriptors_col_.size() - num_gkvec_col_;
        }

        inline gklo_basis_descriptor const& gklo_basis_descriptor_col(int idx) const
        {
            assert(idx >=0 && idx < (int)gklo_basis_descriptors_col_.size());
            return gklo_basis_descriptors_col_[idx];
        }
        
        inline gklo_basis_descriptor const& gklo_basis_descriptor_row(int idx) const
        {
            assert(idx >= 0 && idx < (int)gklo_basis_descriptors_row_.size());
            return gklo_basis_descriptors_row_[idx];
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
            return num_ranks_;
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
        
        inline dmatrix<double_complex>& fv_states()
        {
            return fv_states_;
        }

        inline dmatrix<double_complex>& fv_states_slice()
        {
            return fv_states_slice_;
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
            memcpy(&band_energies_[0], &fv_eigen_values_[0], parameters_.num_fv_states() * sizeof(double));
        }

        std::vector<double> get_pw_ekin() const
        {
            std::vector<double> pw_ekin(num_gkvec());
            for (int igk = 0; igk < num_gkvec(); igk++) pw_ekin[igk] = 0.5 * std::pow(gkvec<cartesian>(igk).length(), 2);
            return pw_ekin; 
        }

        inline Gvec const& gkvec() const
        {
            return gkvec_;
        }

        inline Gvec const& pgkvec() const
        {
            return pgkvec_;
        }

        inline matrix<double_complex> const& beta_gk_t() const
        {
            return beta_gk_t_;
        }

        inline matrix<double_complex>& beta_gk()
        {
            return beta_gk_;
        }

        inline Matching_coefficients* alm_coeffs_row()
        {
            return alm_coeffs_row_;
        }

        inline Matching_coefficients* alm_coeffs_col()
        {
            return alm_coeffs_col_;
        }

        inline Communicator& comm()
        {
            return comm_;
        }

        inline Communicator& comm_row()
        {
            return comm_row_;
        }

        inline Communicator& comm_col()
        {
            return comm_col_;
        }

        inline BLACS_grid const& blacs_grid() const
        {
            return blacs_grid_;
        }

        inline BLACS_grid const& blacs_grid_slab() const
        {
            return blacs_grid_slab_;
        }

        inline BLACS_grid const& blacs_grid_slice() const
        {
            return blacs_grid_slice_;
        }

        inline double_complex p_mtrx(int xi1, int xi2, int iat) const
        {
            return p_mtrx_(xi1, xi2, iat);
        }

        inline splindex<block>& spl_gkvec()
        {
            return spl_gkvec_;
        }

        inline int num_gkvec_loc() const
        {
            return (int)spl_gkvec_.local_size();
        }

        //void collect_all_gkvec(splindex<block>& spl_phi__,
        //                       double_complex const* phi_slab__,
        //                       double_complex* phi_slice__);

        //void collect_all_bands(splindex<block>& spl_phi__,
        //                       double_complex const* phi_slice__,
        //                       double_complex* phi_slab__);

        /// Generate beta-proectors for a block of atoms.
        void generate_beta_gk(int num_atoms__,
                              mdarray<double, 2>& atom_pos__,
                              mdarray<int, 2> const& beta_desc__,
                              matrix<double_complex>& beta_gk__);

        void generate_beta_phi(int nbeta__,
                               matrix<double_complex>& phi__,
                               int nphi__,
                               int offs__,
                               matrix<double_complex>& beta_gk__,
                               matrix<double_complex>& beta_phi__);

        void add_non_local_contribution(int num_atoms__,
                                        int num_beta__,
                                        mdarray<int, 2> const& beta_desc__,
                                        matrix<double_complex>& beta_gk__,
                                        mdarray<double_complex, 1>& op_mtrx_packed__,
                                        mdarray<int, 1> const& packed_mtrx_offset__,
                                        matrix<double_complex>& beta_phi__,
                                        matrix<double_complex>& op_phi__,
                                        int nphi__,
                                        int offs__,
                                        double_complex alpha,
                                        matrix<double_complex>& work__);
};

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

