#ifndef __K_POINT_H__
#define __K_POINT_H__

#include "global.h"
#include "periodic_function.h"

namespace sirius
{
// TODO: cleanup the mess with matching coefficients

/// K-point related variables and methods
/** \image html wf_storage.png "Wave-function storage" */ // TODO: replace with proper image
class K_point
{
    private:

        /// global set of parameters
        Global& parameters_;

        /// alias for FFT driver
        FFT3D<cpu>* fft_;

        /// weight of k-point
        double weight_;

        /// fractional k-point coordinates
        vector3d<double> vk_;
        
        /// G+k vectors
        mdarray<double, 2> gkvec_;

        mdarray<double, 2> gkvec_gpu_;

        /// global index (in the range [0, N_G - 1]) of G-vector by the index of G+k vector in the range [0, N_Gk - 1]
        std::vector<int> gvec_index_;

        /// first-variational eigen values
        std::vector<double> fv_eigen_values_;

        /// first-variational eigen vectors, distributed over rows and columns of the MPI grid
        dmatrix<double_complex> fv_eigen_vectors_panel_;
        
        /// second-variational eigen vectors
        /** Second-variational eigen-vectors are stored as one or two \f$ N_{fv} \times N_{fv} \f$ matrices in
         *  case of non-magnetic or collinear magnetic case or as a single \f$ 2 N_{fv} \times 2 N_{fv} \f$ 
         *  matrix in case of general non-collinear magnetism.
         */
        dmatrix<double_complex> sv_eigen_vectors_[2];

        /// full-diagonalization eigen vectors
        mdarray<double_complex, 2> fd_eigen_vectors_;

        /// position of the G vector (from the G+k set) inside the FFT buffer 
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

        /// precomputed values for the linear equations for matching coefficients
        mdarray<double_complex, 4> alm_b_;

        /// length of G+k vectors
        std::vector<double> gkvec_len_;

        /// number of G+k vectors distributed along rows of MPI grid
        int num_gkvec_row_;
        
        /// number of G+k vectors distributed along columns of MPI grid
        int num_gkvec_col_;

        /// short information about each G+k or lo basis function
        /** This is a global array. Each MPI rank of the 2D grid has exactly the same copy. */
        std::vector<gklo_basis_descriptor> gklo_basis_descriptors_;

        /// basis descriptors distributed along rows of the 2D MPI grid
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
        std::vector< sbessel_pw<double>* > sbessel_;
        
        /// column rank of the processors of ScaLAPACK/ELPA diagonalization grid
        int rank_col_;

        /// number of processors along columns of the diagonalization grid
        int num_ranks_col_;

        int rank_row_;

        int num_ranks_row_;

        int num_ranks_;

        /// phase-factor independent plane-wave coefficients of |beta> functions for atom types
        mdarray<double_complex, 2> beta_pw_t_;

        /// plane-wave coefficients of |beta> functions for atoms
        mdarray<double_complex, 2> beta_pw_a_;

        /// Generate matching coefficients for specific l-value
        template <int order, bool conjugate>
        void generate_matching_coefficients_l(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                              mdarray<double, 2>& A, mdarray<double_complex, 2>& alm);
        
        void check_alm(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm);

        /// Copy lo block from eigen-vector to wave-function
        inline void copy_lo_blocks(const double_complex* z, double_complex* vec);
        
        /// Copy plane wave block from eigen-vector to wave-function
        inline void copy_pw_block(const double_complex* z, double_complex* vec);

        /// Initialize G+k related data
        void init_gkvec();
        
        /// Build APW+lo basis descriptors 
        void build_apwlo_basis_descriptors();

        /// Block-cyclic distribution of relevant arrays 
        void distribute_block_cyclic();
        
        /// Test orthonormalization of first-variational states
        void test_fv_states(int use_fft);

        template <index_domain_t index_domain> 
        void init_gkvec_ylm_and_len(int lmax, int ngk);
        
        template <index_domain_t index_domain> 
        void init_gkvec_phase_factors(int ngk);

    public:

        /// Constructor
        K_point(Global& parameters__, double* vk__, double weight__) : parameters_(parameters__), weight_(weight__)
        {
            for (int x = 0; x < 3; x++) vk_[x] = vk__[x];

            band_occupancies_.resize(parameters_.num_bands());
            memset(&band_occupancies_[0], 0, parameters_.num_bands() * sizeof(double));
            
            num_ranks_row_ = parameters_.mpi_grid().dimension_size(_dim_row_);
            num_ranks_col_ = parameters_.mpi_grid().dimension_size(_dim_col_);

            num_ranks_ = num_ranks_row_ * num_ranks_col_;

            rank_row_ = parameters_.mpi_grid().coordinate(_dim_row_);
            rank_col_ = parameters_.mpi_grid().coordinate(_dim_col_);

            fft_ = parameters_.reciprocal_lattice()->fft();
        }

        ~K_point()
        {
            if (parameters_.esm_type() == full_potential_pwlo)
            {
                for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) delete sbessel_[igkloc];
            }
        }

        /// Initialize the k-point related arrays and data
        void initialize();

        /// Update the relevant arrays in case of atom positions have been changed.
        void update();
        
        /// Find G+k vectors within the cutoff
        void generate_gkvec(double gk_cutoff);

        /// Generate plane-wave matching coefficents for the radial solutions 
        /** At some point we need to compute the radial derivatives of the spherical Bessel functions at the 
         *  muffin-tin boundary. The following formula is used:
         *  \f[
         *      j_{{n}}^{{\prime}}(z)=-j_{{n+1}}(z)+(n/z)j_{{n}}(z)
         *  \f]
         */
        template <bool conjugate>
        void generate_matching_coefficients(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm)
        {
            Timer t("sirius::K_point::generate_matching_coefficients");

            Atom* atom = parameters_.unit_cell()->atom(ia);
            Atom_type* type = atom->type();

            assert(type->max_aw_order() <= 3);

            int iat = type->id();

            #pragma omp parallel default(shared)
            {
                mdarray<double, 2> A(3, 3);

                #pragma omp for
                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    int num_aw = (int)type->aw_descriptor(l).size();

                    for (int order = 0; order < num_aw; order++)
                    {
                        for (int dm = 0; dm < num_aw; dm++) A(dm, order) = atom->symmetry_class()->aw_surface_dm(l, order, dm);
                    }

                    switch (num_aw)
                    {
                        case 1:
                        {
                            generate_matching_coefficients_l<1, conjugate>(ia, iat, type, l, num_gkvec_loc, A, alm);
                            break;
                        }
                        case 2:
                        {
                            generate_matching_coefficients_l<2, conjugate>(ia, iat, type, l, num_gkvec_loc, A, alm);
                            break;
                        }
                        case 3:
                        {
                            generate_matching_coefficients_l<3, conjugate>(ia, iat, type, l, num_gkvec_loc, A, alm);
                            break;
                        }
                        default:
                        {
                            error_local(__FILE__, __LINE__, "wrong order of augmented wave");
                        }
                    }
                } //l
            }
            
            // check alm coefficients
            if (debug_level > 1) check_alm(num_gkvec_loc, ia, alm);
        }
        
        /// Generate plane-wave matching coefficents for the radial solutions 
        /** Normal layout of matching coefficients: G+k vectors are along rows, AW basis functions are 
         *  along columns. This layout is used to generate first-variational states. 
         *  Transposed layout: AW basis functions are along rows, G+k vectors are along columns. This layout
         *  is used to setup Hamiltonian and overlap matrices.
         */
        template <bool transpose>
        void generate_matching_coefficients(dmatrix<double_complex>& alm)
        {
            Timer t("sirius::K_point::generate_matching_coefficients_panel");

            int num_mt_aw_loc = transpose ? alm.num_rows_local() : alm.num_cols_local();
            int offset_col = apw_offset_col();

            #pragma omp parallel
            {
                mdarray<double, 2> A(3, 3);
                #pragma omp for
                for (int i = 0; i < num_mt_aw_loc; i++)
                {
                    int j = transpose ? alm.irow(i) : alm.icol(i);
                    int ia = parameters_.unit_cell()->mt_aw_basis_descriptor(j).ia;
                    int xi = parameters_.unit_cell()->mt_aw_basis_descriptor(j).xi;
                    Atom* atom = parameters_.unit_cell()->atom(ia);
                    Atom_type* type = atom->type();
                    int iat = type->id();
                    int l = type->indexb(xi).l;
                    int lm = type->indexb(xi).lm;
                    int order = type->indexb(xi).order; 

                    int num_aw = (int)type->aw_descriptor(l).size();

                    for (int order = 0; order < num_aw; order++)
                    {
                        for (int dm = 0; dm < num_aw; dm++) A(dm, order) = atom->symmetry_class()->aw_surface_dm(l, order, dm);
                    }
                    
                    switch (num_aw)
                    {
                        case 1:
                        {
                            A(0, 0) = 1.0 / A(0, 0);

                            double_complex zt;

                            if (transpose)
                            {
                                for (int igkloc = 0; igkloc < alm.num_cols_local(); igkloc++)
                                {
                                    zt = gkvec_phase_factors_(offset_col + igkloc, ia) * 
                                         alm_b_(0, offset_col + igkloc, l, iat) * A(0, 0);

                                    alm(i, igkloc) = conj(gkvec_ylm_(lm, offset_col + igkloc)) * zt;
                                }
                            }
                            else
                            {
                                for (int igkloc = 0; igkloc < alm.num_rows_local(); igkloc++)
                                {
                                    zt = gkvec_phase_factors_(igkloc, ia) * alm_b_(0, igkloc, l, iat) * A(0, 0);

                                    alm(igkloc, i) = conj(gkvec_ylm_(lm, igkloc)) * zt;
                                }
                            }
                            break;
                        }
                        case 2:
                        {
                            double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
                            std::swap(A(0, 0), A(1, 1));
                            A(0, 0) /= det;
                            A(1, 1) /= det;
                            A(0, 1) = -A(0, 1) / det;
                            A(1, 0) = -A(1, 0) / det;
                            
                            double_complex zt[2];
                            double_complex zb;

                            if (transpose)
                            {
                                for (int igkloc = 0; igkloc < alm.num_cols_local(); igkloc++)
                                {
                                    zt[0] = gkvec_phase_factors_(offset_col + igkloc, ia) * 
                                            alm_b_(0, offset_col + igkloc, l, iat);
                                    zt[1] = gkvec_phase_factors_(offset_col + igkloc, ia) * 
                                            alm_b_(1, offset_col + igkloc, l, iat);

                                    zb = A(order, 0) * zt[0] + A(order, 1) * zt[1];

                                    alm(i, igkloc) = conj(gkvec_ylm_(lm, offset_col + igkloc)) * zb;
                                }
                            }
                            else
                            {
                                for (int igkloc = 0; igkloc < alm.num_rows_local(); igkloc++)
                                {
                                    zt[0] = gkvec_phase_factors_(igkloc, ia) * alm_b_(0, igkloc, l, iat);
                                    zt[1] = gkvec_phase_factors_(igkloc, ia) * alm_b_(1, igkloc, l, iat);

                                    zb = A(order, 0) * zt[0] + A(order, 1) * zt[1];

                                    alm(igkloc, i) = conj(gkvec_ylm_(lm, igkloc)) * zb;
                                }
                            }
                            break;
                        }
                        default:
                        {
                            stop_here
                        }
                    }
                }
            }
        }

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

        /// Return length of a G+k vector
        inline double gkvec_len(int igk)
        {
            assert(igk >= 0 && igk < (int)gkvec_len_.size());
            return gkvec_len_[igk];
        }
                
        /// Total number of G+k vectors within the cutoff distance
        inline int num_gkvec()
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
                {
                    return num_gkvec();
                    break;
                }
                default:
                {
                    terminate(__FILE__, __LINE__, "wrong esm_type");
                    return -1; //make compiler happy
                }
            }
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
        inline int gklo_basis_size()
        {
            return (int)gklo_basis_descriptors_.size();
        }
        
        /// Local number of basis functions for each MPI rank in the row of the 2D MPI grid.
        inline int gklo_basis_size_row()
        {
            return (int)gklo_basis_descriptors_row_.size();
        }
        
        /// Local number of G+k vectors for each MPI rank in the row of the 2D MPI grid.
        inline int num_gkvec_row()
        {
            return num_gkvec_row_;
        }

        /// Local number of local orbitals for each MPI rank in the row of the 2D MPI grid.
        inline int num_lo_row()
        {
            return (int)gklo_basis_descriptors_row_.size() - num_gkvec_row_;
        }

        /// Local number of basis functions for each MPI rank in the column of the 2D MPI grid.
        inline int gklo_basis_size_col()
        {
            return (int)gklo_basis_descriptors_col_.size();
        }
        
        /// Local number of G+k vectors for each MPI rank in the column of the 2D MPI grid.
        inline int num_gkvec_col()
        {
            return num_gkvec_col_;
        }
        
        /// Local number of local orbitals for each MPI rank in the column of the 2D MPI grid.
        inline int num_lo_col()
        {
            return (int)gklo_basis_descriptors_col_.size() - num_gkvec_col_;
        }

        /// Local fraction of G+k vectors for a given MPI rank
        /** In case of distributed matrix setup row and column G+k vectors are combined. Row G+k vectors are first.*/
        inline int num_gkvec_loc() // TODO: this is probably obosolete
        {
            if (num_gkvec_row() == num_gkvec() && num_gkvec_col() == num_gkvec())
            {
                return num_gkvec();
            }
            else
            {
                return (num_gkvec_row() + num_gkvec_col());
            }
        } 
        
        /// Return the global index of the G+k vector by the local index
        inline int igkglob(int igkloc) // TODO: change name or change the local G+k row+col storage 
        {
            assert(igkloc >= 0 && igkloc < num_gkvec_loc());

            if (num_gkvec_row() == num_gkvec() && num_gkvec_col() == num_gkvec())
            {
                return igkloc;
            }
            else
            {
                // remember: row G+k vectors are first, column G+k vectors are second
                int igk = (igkloc < num_gkvec_row()) ? gklo_basis_descriptors_row_[igkloc].igk : 
                                                       gklo_basis_descriptors_col_[igkloc - num_gkvec_row()].igk;
                assert(igk >= 0);
                return igk;
            }
        }

        inline gklo_basis_descriptor& gklo_basis_descriptor_col(int idx)
        {
            assert(idx >=0 && idx < (int)gklo_basis_descriptors_col_.size());
            return gklo_basis_descriptors_col_[idx];
        }
        
        inline gklo_basis_descriptor& gklo_basis_descriptor_row(int idx)
        {
            assert(idx >= 0 && idx < (int)gklo_basis_descriptors_row_.size());
            return gklo_basis_descriptors_row_[idx];
        }
        
        inline int num_ranks_row()
        {
            return num_ranks_row_;
        }
        
        inline int rank_row()
        {
            return rank_row_;
        }
        
        inline int num_ranks_col()
        {
            return num_ranks_col_;
        }
        
        inline int rank_col()
        {
            return rank_col_;
        }
       
        /// Number of MPI ranks for a given k-point
        inline int num_ranks()
        {
            return num_ranks_;
        }

        /// Offset of column matching coefficients in the array. 
        /** In case of distributed matrix setup row and column apw coefficients 
          * are combined. Row coefficients are first.
          */
        inline int apw_offset_col()
        {
            return (num_ranks() > 1) ? num_gkvec_row() : 0;
        }

        /// Return number of lo columns for a given atom
        inline int num_atom_lo_cols(int ia)
        {
            return (int)atom_lo_cols_[ia].size();
        }

        /// Return local index (for the current MPI rank) of a column for a given atom and column index within an atom
        inline int lo_col(int ia, int i)
        {
            return atom_lo_cols_[ia][i];
        }
        
        /// Return number of lo rows for a given atom
        inline int num_atom_lo_rows(int ia)
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
            for (int igk = 0; igk < num_gkvec(); igk++) pw_ekin[igk] = 0.5 * pow(gkvec_len(igk), 2);
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

        inline mdarray<double_complex, 2>& beta_pw_t()
        {
            return beta_pw_t_;
        }

        inline double_complex& beta_pw_t(int igk, int idx)
        {
            return beta_pw_t_(igk, idx);
        }

        inline double_complex& beta_pw_a(int igk, int idx)
        {
            return beta_pw_a_(igk, idx);
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

#endif // __K_POINT_H__

