// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

#include "global.h"
#include "periodic_function.h"
#include "matching_coefficients.h"
#include "blacs_grid.h"

#ifdef _GPU_
extern "C" void create_beta_gk_gpu(int num_atoms,
                                   int num_gkvec,
                                   int const* beta_desc,
                                   cuDoubleComplex const* beta_gk_t,
                                   double const* gkvec,
                                   double const* atom_pos,
                                   cuDoubleComplex* beta_gk);
#endif


namespace sirius
{

/// K-point related variables and methods.
/** \image html wf_storage.png "Wave-function storage" */ // TODO: replace with proper image
class K_point
{
    private:

        /// Global set of parameters.
        Global& parameters_;

        BLACS_grid const& blacs_grid_;

        /// Alias for FFT driver.
        FFT3D<CPU>* fft_;

        /// Weight of k-point.
        double weight_;

        /// Fractional k-point coordinates.
        vector3d<double> vk_;
        
        /// G+k vectors
        mdarray<double, 2> gkvec_;

        mdarray<double, 2> gkvec_gpu_;

        /// global index (in the range [0, N_G - 1]) of G-vector by the index of G+k vector in the range [0, N_Gk - 1]
        std::vector<int> gvec_index_;

        /// first-variational eigen values
        std::vector<double> fv_eigen_values_;

        /// First-variational eigen vectors, distributed over rows and columns of the MPI grid.
        dmatrix<double_complex> fv_eigen_vectors_panel_;
        
        /// Second-variational eigen vectors.
        /** Second-variational eigen-vectors are stored as one or two \f$ N_{fv} \times N_{fv} \f$ matrices in
         *  case of non-magnetic or collinear magnetic case or as a single \f$ 2 N_{fv} \times 2 N_{fv} \f$ 
         *  matrix in case of general non-collinear magnetism.
         */
        dmatrix<double_complex> sv_eigen_vectors_[2];

        /// Full-diagonalization eigen vectors.
        mdarray<double_complex, 2> fd_eigen_vectors_;

        /// Position of the G vector (from the G+k set) inside the FFT buffer.
        std::vector<int> fft_index_;

        std::vector<int> fft_index_coarse_;
       
        /// first-variational states, distributed over all ranks of the 2D MPI grid
        mdarray<double_complex, 2> fv_states_;
        
        /// first-variational states, distributed over rows and columns of the MPI grid
        /** Band index is distributed over columns and basis functions index is distributed 
         *  over rows of the MPI grid. 
         */
        dmatrix<double_complex> fv_states_panel_;

        /// two-component (spinor) wave functions describing the bands
        mdarray<double_complex, 3> spinor_wave_functions_;

        /// band occupation numbers
        std::vector<double> band_occupancies_;

        /// band energies
        std::vector<double> band_energies_; 

        /// phase factors \f$ e^{i ({\bf G+k}) {\bf r}_{\alpha}} \f$
        mdarray<double_complex, 2> gkvec_phase_factors_;

        /// spherical harmonics of G+k vectors
        mdarray<double_complex, 2> gkvec_ylm_;

        /// length of G+k vectors
        //std::vector<double> gkvec_len_;

        Matching_coefficients* alm_coeffs_row_;

        Matching_coefficients* alm_coeffs_col_;

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
            
        /// list of columns of the Hamiltonian and overlap matrix lo block (local index) for a given atom
        std::vector< std::vector<int> > atom_lo_cols_;

        /// list of rows of the Hamiltonian and overlap matrix lo block (local index) for a given atom
        std::vector< std::vector<int> > atom_lo_rows_;

        /// imaginary unit to the power of l
        std::vector<double_complex> zil_;

        /// mapping between lm and l
        std::vector<int> l_by_lm_;

        /// spherical bessel functions for G+k vectors  
        //std::vector< sbessel_pw<double>* > sbessel_;
        
        /// column rank of the processors of ScaLAPACK/ELPA diagonalization grid
        int rank_col_;

        /// number of processors along columns of the diagonalization grid
        int num_ranks_col_;

        int rank_row_;

        int num_ranks_row_;

        int num_ranks_;

        /// Phase-factor independent plane-wave coefficients of |beta> functions for atom types.
        mdarray<double_complex, 2> beta_gk_t_;

        /// Plane-wave coefficients of |beta> functions for atoms.
        dmatrix<double_complex> beta_pw_panel_;

        mdarray<double_complex, 3> p_mtrx_;

        mdarray<double, 2> gkvec_row_;

        Communicator comm_;

        /// Communicator between(!!) rows.
        Communicator comm_row_;

        /// Communicator between(!!) columns.
        Communicator comm_col_;

        /// block-cyclic distribution of the first-variational states along columns of the MPI grid
        splindex<block_cyclic> spl_fv_states_;
        
        /// additional splitting of the first-variational states along rows of the MPI grid
        splindex<block> sub_spl_fv_states_;

        /// block-cyclic distribution of the spinor wave-functions along columns of the MPI grid
        splindex<block_cyclic> spl_spinor_wf_;
       
        /// additional splitting of spinor wave-functions along rows of the MPI grid
        splindex<block> sub_spl_spinor_wf_;

        /// Initialize G+k related data
        void init_gkvec();
        
        /// Build APW+lo basis descriptors 
        void build_apwlo_basis_descriptors();

        /// Block-cyclic distribution of relevant arrays 
        void distribute_block_cyclic();
        
        /// Test orthonormalization of first-variational states
        void test_fv_states(int use_fft);

        void init_gkvec_ylm_and_len(int lmax__);
        
        void init_gkvec_phase_factors();

    public:

        /// Constructor
        K_point(Global& parameters__, double* vk__, double weight__, BLACS_grid const& blacs_grid__) 
            : parameters_(parameters__), 
              blacs_grid_(blacs_grid__),
              weight_(weight__),
              alm_coeffs_row_(nullptr),
              alm_coeffs_col_(nullptr)
        {
            for (int x = 0; x < 3; x++) vk_[x] = vk__[x];

            band_occupancies_ = std::vector<double>(parameters_.num_bands(), 1);

            comm_ = blacs_grid_.comm();
            comm_row_ = blacs_grid_.comm_row();
            comm_col_ = blacs_grid_.comm_col();
            
            num_ranks_ = comm_.size();
            num_ranks_row_ = comm_row_.size();
            num_ranks_col_ = comm_col_.size();

            rank_row_ = comm_row_.rank();
            rank_col_ = comm_col_.rank();

            fft_ = parameters_.reciprocal_lattice()->fft();

            /* distribue first-variational states along columns */
            spl_fv_states_ = splindex<block_cyclic>(parameters_.num_fv_states(), num_ranks_col_, rank_col_, blacs_grid_.cyclic_block_size());

            /* distribue spinor wave-functions along columns */
            spl_spinor_wf_ = splindex<block_cyclic>(parameters_.num_bands(), num_ranks_col_, rank_col_, blacs_grid_.cyclic_block_size());
            
            /* additionally split along rows */
            sub_spl_fv_states_ = splindex<block>(spl_fv_states_.local_size(), num_ranks_row_, rank_row_);
            sub_spl_spinor_wf_ = splindex<block>(spl_spinor_wf_.local_size(), num_ranks_row_, rank_row_);
        }

        ~K_point()
        {
            //== if (parameters_.esm_type() == full_potential_pwlo)
            //== {
            //==     for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) delete sbessel_[igkloc];
            //== }
            if (alm_coeffs_row_) delete alm_coeffs_row_;
            if (alm_coeffs_col_) delete alm_coeffs_col_;
        }

        /// Initialize the k-point related arrays and data
        void initialize();

        /// Update the relevant arrays in case of atom positions have been changed.
        void update();
        
        /// Find G+k vectors within the cutoff
        void generate_gkvec(double gk_cutoff);

        /// Generate first-variational states from eigen-vectors
        void generate_fv_states();

        #ifdef _GPU_
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
        
        /// Global index of G-vector by the index of G+k vector
        inline int gvec_index(int igk) 
        {
            assert(igk >= 0 && igk < (int)gvec_index_.size());
            
            return gvec_index_[igk];
        }
        
        /// Return G+k vector in fractional coordinates
        inline vector3d<double> gkvec(int igk)
        {
            assert(igk >= 0 && igk < (int)gkvec_.size(1));

            return vector3d<double>(gkvec_(0, igk), gkvec_(1, igk), gkvec_(2, igk));
        }

        inline vector3d<double> gkvec_cart(int igk)
        {
            return parameters_.reciprocal_lattice()->get_cartesian_coordinates(gkvec(igk));
        }

        inline double_complex gkvec_phase_factor(int igk, int ia)
        {
            return gkvec_phase_factors_(igk, ia);
        }

        //== /// Return length of a G+k vector
        //== inline double gkvec_len(int igk)
        //== {
        //==     assert(igk >= 0 && igk < (int)gkvec_len_.size());
        //==     return gkvec_len_[igk];
        //== }
                
        /// Total number of G+k vectors within the cutoff distance
        inline int num_gkvec() const
        {
            assert(gkvec_.size(1) == gvec_index_.size());
            return (int)gkvec_.size(1);
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
        inline int wf_size() // TODO: better name for this
        {
            switch (parameters_.esm_type())
            {
                case full_potential_lapwlo:
                case full_potential_pwlo:
                {
                    return parameters_.unit_cell()->mt_basis_size() + num_gkvec();
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

        inline int wf_pw_offset()
        {
            switch (parameters_.esm_type())
            {
                case full_potential_lapwlo:
                case full_potential_pwlo:
                {
                    return parameters_.unit_cell()->mt_basis_size();
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

        inline void get_band_occupancies(double* band_occupancies)
        {
            assert((int)band_occupancies_.size() == parameters_.num_bands());
            
            memcpy(band_occupancies, &band_occupancies_[0], parameters_.num_bands() * sizeof(double));
        }

        inline void set_band_occupancies(double* band_occupancies)
        {
            band_occupancies_.resize(parameters_.num_bands());
            memcpy(&band_occupancies_[0], band_occupancies, parameters_.num_bands() * sizeof(double));
        }

        inline void get_band_energies(double* band_energies)
        {
            assert((int)band_energies_.size() == parameters_.num_bands());
            
            memcpy(band_energies, &band_energies_[0], parameters_.num_bands() * sizeof(double));
        }

        inline void set_band_energies(double* band_energies)
        {
            band_energies_.resize(parameters_.num_bands()); 
            memcpy(&band_energies_[0], band_energies, parameters_.num_bands() * sizeof(double));
        }

        inline double band_occupancy(int j)
        {
            return band_occupancies_[j];
        }
        
        inline double band_energy(int j)
        {
            return band_energies_[j];
        }

        inline double fv_eigen_value(int i)
        {
            return fv_eigen_values_[i];
        }

        void set_fv_eigen_values(double* eval)
        {
            memcpy(&fv_eigen_values_[0], eval, parameters_.num_fv_states() * sizeof(double));
        }
        
        inline double weight()
        {
            return weight_;
        }

        inline double_complex& spinor_wave_function(int idxwf, int ispn, int j)
        {
            return spinor_wave_functions_(idxwf, ispn, j);
        }

        inline mdarray<double_complex, 3>& spinor_wave_functions()
        {
            return spinor_wave_functions_;
        }

        inline int* fft_index()
        {
            return &fft_index_[0];
        }

        inline int* fft_index_coarse()
        {
            return &fft_index_coarse_[0];
        }

        inline vector3d<double> vk()
        {
            return vk_;
        }

        inline double vk(int x)
        {
            return vk_[x];
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
        inline int lo_row(int ia, int i)
        {
            return atom_lo_rows_[ia][i];
        }

        inline dmatrix<double_complex>& fv_eigen_vectors_panel()
        {
            return fv_eigen_vectors_panel_;
        }
        
        inline mdarray<double_complex, 2>& fv_states()
        {
            return fv_states_;
        }

        inline dmatrix<double_complex>& fv_states_panel()
        {
            return fv_states_panel_;
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
            //== sv_eigen_vectors_[0].zero();
            //== for (int i = 0; i < parameters_.num_fv_states(); i++) sv_eigen_vectors_[0].set(i, i, complex_one);
        }

        std::vector<double> get_pw_ekin()
        {
            std::vector<double> pw_ekin(num_gkvec());
            for (int igk = 0; igk < num_gkvec(); igk++) pw_ekin[igk] = 0.5 * std::pow(gkvec_cart(igk).length(), 2);
            return pw_ekin; 
        }

        inline mdarray<double, 2>& gkvec_gpu()
        {
            return gkvec_gpu_;
        }

        inline mdarray<double, 2>& gkvec()
        {
            return gkvec_;
        }

        inline mdarray<double_complex, 2> const& beta_gk_t() const
        {
            return beta_gk_t_;
        }

        inline dmatrix<double_complex>& beta_pw_panel()
        {
            return beta_pw_panel_;
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

        inline splindex<block_cyclic>& spl_fv_states()
        {
            return spl_fv_states_;
        }

        inline int spl_fv_states(int icol_loc)
        {
            return static_cast<int>(spl_fv_states_[icol_loc]);
        }

        inline splindex<block>& sub_spl_fv_states()
        {
            return sub_spl_fv_states_;
        }

        inline int num_sub_bands()
        {
            return static_cast<int>(sub_spl_spinor_wf_.local_size());
        }

        inline int idxbandglob(int sub_index)
        {
            return static_cast<int>(spl_spinor_wf_[sub_spl_spinor_wf_[sub_index]]);
        }

        inline double_complex p_mtrx(int xi1, int xi2, int iat) const
        {
            return p_mtrx_(xi1, xi2, iat);
        }

        /// Generate beta-proectors for a block of atoms.
        void generate_beta_gk(int num_atoms__,
                              mdarray<double, 2>& atom_pos__,
                              mdarray<int, 2> const& beta_desc__,
                              matrix<double_complex>& beta_gk__)
        {
            Timer t("sirius::K_point::generate_beta_gk");

            if (parameters_.processing_unit() == CPU)
            {
                /* create beta projectors */
                #pragma omp parallel
                for (int i = 0; i < num_atoms__; i++)
                {
                    int ia = beta_desc__(3, i);
                    #pragma omp for
                    for (int xi = 0; xi < beta_desc__(0, i); xi++)
                    {
                        for (int igk_row = 0; igk_row < num_gkvec_row(); igk_row++)
                        {
                            beta_gk__(igk_row, beta_desc__(1, i) + xi) = 
                                beta_gk_t_(igk_row, beta_desc__(2, i) + xi) * conj(gkvec_phase_factor(igk_row, ia));
                        }
                    }
                }
            }
            if (parameters_.processing_unit() == GPU)
            {
                #ifdef _GPU_
                /* create beta projectors directly on GPU */
                create_beta_gk_gpu(num_atoms__,
                                   num_gkvec_row(),
                                   beta_desc__.at<GPU>(),
                                   beta_gk_t_.at<GPU>(),
                                   gkvec_row_.at<GPU>(),
                                   atom_pos__.at<GPU>(),
                                   beta_gk__.at<GPU>());
                #else
                TERMINATE_NO_GPU
                #endif
            }
        }

        void generate_beta_phi(int nbeta__,
                               matrix<double_complex>& phi__,
                               int nphi__,
                               int offs__,
                               matrix<double_complex>& beta_gk__,
                               matrix<double_complex>& beta_phi__)
        {
            Timer t("sirius::K_point::generate_beta_phi");
            #ifdef _GPU_
            #ifdef _GPU_DIRECT_
            // allrecue with gpu-direct is broken at the moment
            bool gpu_direct = false;
            #else
            bool gpu_direct = false;
            #endif
            #endif

            if (parameters_.processing_unit() == CPU)
            {
                /* compute <beta|phi> */
                linalg<CPU>::gemm(2, 0, nbeta__, nphi__, num_gkvec_row(), 
                                  beta_gk__.at<CPU>(), beta_gk__.ld(), 
                                  phi__.at<CPU>(0, offs__), phi__.ld(), 
                                  beta_phi__.at<CPU>(), beta_phi__.ld());
                comm_row().allreduce(beta_phi__.at<CPU>(), (int)beta_phi__.size());
            }

            if (parameters_.processing_unit() == GPU)
            {
                #ifdef _GPU_
                /* compute <beta|phi> */
                linalg<GPU>::gemm(2, 0, nbeta__, nphi__, num_gkvec_row(), 
                                  beta_gk__.at<GPU>(), beta_gk__.ld(), 
                                  phi__.at<GPU>(0, offs__), phi__.ld(), 
                                  beta_phi__.at<GPU>(), beta_phi__.ld());
                
                if (gpu_direct)
                {
                    comm_row().allreduce(beta_phi__.at<GPU>(), (int)beta_phi__.size());
                }
                else
                {
                    beta_phi__.copy_to_host();
                    comm_row().allreduce(beta_phi__.at<CPU>(), (int)beta_phi__.size());
                    beta_phi__.copy_to_device();
                }
                cuda_device_synchronize();
                #else
                TERMINATE_NO_GPU
                #endif
            }
        }

        void add_non_local_contribution(int num_atoms__,
                                        int num_beta__,
                                        mdarray<int, 2> const& beta_desc__,
                                        matrix<double_complex>& beta_gk__,
                                        mdarray<double_complex, 1>& op_mtrx_packed__,
                                        mdarray<int, 1> const& packed_mtrx_offset__,
                                        matrix<double_complex>& beta_phi__,
                                        matrix<double_complex>& phi__,
                                        int nphi__,
                                        int offs__,
                                        double_complex alpha,
                                        matrix<double_complex>& work__)
        {
            Timer t("sirius::K_point::add_non_local_contribution");

            if (parameters_.processing_unit() == CPU)
            {
                #pragma omp parallel for
                for (int i = 0; i < num_atoms__; i++)
                {
                    /* number of beta functions for a given atom */
                    int nbf = beta_desc__(0, i);
                    int ofs = beta_desc__(1, i);
                    int ia = beta_desc__(3, i);

                    /* compute O * <beta|phi> */
                    linalg<CPU>::gemm(0, 0, nbf, nphi__, nbf,
                                      op_mtrx_packed__.at<CPU>(packed_mtrx_offset__(ia)), nbf,
                                      beta_phi__.at<CPU>(ofs, 0), beta_phi__.ld(),
                                      work__.at<CPU>(ofs, 0), work__.ld());
                }
                
                /* compute <G+k|beta> * O * <beta|phi> and add to phi */
                linalg<CPU>::gemm(0, 0, num_gkvec_row(), nphi__, num_beta__, alpha,
                                  beta_gk__.at<CPU>(), beta_gk__.ld(), work__.at<CPU>(), work__.ld(), complex_one,
                                  phi__.at<CPU>(0, offs__), phi__.ld());
            }

            if (parameters_.processing_unit() == GPU)
            {
                #ifdef _GPU_
                #pragma omp parallel for
                for (int i = 0; i < num_atoms__; i++)
                {
                    /* number of beta functions for a given atom */
                    int nbf = beta_desc__(0, i);
                    int ofs = beta_desc__(1, i);
                    int ia = beta_desc__(3, i);

                    /* compute D*<beta|phi> */
                    linalg<GPU>::gemm(0, 0, nbf, nphi__, nbf, op_mtrx_packed__.at<GPU>(packed_mtrx_offset__(ia)), nbf, 
                                      beta_phi__.at<GPU>(ofs, 0), beta_phi__.ld(), work__.at<GPU>(ofs, 0), work__.ld(), 
                                      Platform::thread_id());

                }
                cuda_device_synchronize();
                
                double_complex beta = complex_one;
                /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
                linalg<GPU>::gemm(0, 0, num_gkvec_row(), nphi__, num_beta__, &alpha,
                                  beta_gk__.at<GPU>(), beta_gk__.ld(), work__.at<GPU>(), work__.ld(), &beta, 
                                  phi__.at<GPU>(0, offs__), phi__.ld());
                
                cuda_device_synchronize();
                #else
                TERMINATE_NO_GPU
                #endif
            }
        }
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

