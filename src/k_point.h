namespace sirius
{

/// Descriptor of the APW+lo basis function
/** APW+lo basis consists of two different sets of functions: APW functions \f$ \varphi_{{\bf G+k}} \f$ defined over 
    entire unit cell:
    \f[
        \varphi_{{\bf G+k}}({\bf r}) = \left\{ \begin{array}{ll}
        \displaystyle \sum_{L} \sum_{\nu=1}^{O_{\ell}^{\alpha}} a_{L\nu}^{\alpha}({\bf G+k})u_{\ell \nu}^{\alpha}(r) 
        Y_{\ell m}(\hat {\bf r}) & {\bf r} \in {\rm MT} \alpha \\
        \displaystyle \frac{1}{\sqrt  \Omega} e^{i({\bf G+k}){\bf r}} & {\bf r} \in {\rm I} \end{array} \right.
    \f]  
    and Bloch sums of local orbitals defined inside muffin-tin spheres only:
    \f[
        \begin{array}{ll} \displaystyle \varphi_{j{\bf k}}({\bf r})=\sum_{{\bf T}} e^{i{\bf kT}} 
        \varphi_{j}({\bf r - T}) & {\rm {\bf r} \in MT} \end{array}
    \f]
    Each local orbital is composed of radial and angular parts:
    \f[
        \varphi_{j}({\bf r}) = \phi_{\ell_j}^{\zeta_j,\alpha_j}(r) Y_{\ell_j m_j}(\hat {\bf r})
    \f]
    Radial part of local orbital is defined as a linear combination of radial functions (minimum two radial functions 
    are required) such that local orbital vanishes at the sphere boundary:
    \f[
        \phi_{\ell}^{\zeta, \alpha}(r) = \sum_{p}\gamma_{p}^{\zeta,\alpha} u_{\ell \nu_p}^{\alpha}(r)  
    \f]
    
    Arbitrary number of local orbitals may be introduced for each angular quantum number.

    Radial functions are m-th order (with zero-order being a function itself) energy derivatives of the radial 
    Schrödinger equation:
    \f[
        u_{\ell \nu}^{\alpha}(r) = \frac{\partial^{m_{\nu}}}{\partial^{m_{\nu}}E}u_{\ell}^{\alpha}(r,E)\Big|_{E=E_{\nu}}
    \f]
*/
struct apwlo_basis_descriptor
{
    /// global index of the descriptor
    int idxglob;

    /// index of G+k vector
    int igk;

    /// G+k vector in lattice coordinates
    vector3d<double> gkvec;

    /// G+k vector in Cartesian coordinates
    vector3d<double> gkvec_cart;

    /// G vector index for the G+k vector
    int ig;

    /// index of atom if this is a local orbital descriptor
    int ia;

    /// index of l
    int l;

    /// combined lm index
    int lm;

    /// order of a lo radial function for a given orbital quantum number l
    int order;

    /// index of a lo radial function
    int idxrf;
};

/// K-point related variables and methods
/** \image html wf_storage.png "Wave-function storage" */
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

        /// global index (in the range [0, N_G - 1]) of G-vector by the index of G+k vector in the range [0, N_Gk - 1]
        std::vector<int> gvec_index_;

        /// first-variational eigen values
        std::vector<double> fv_eigen_values_;

        /// first-variational eigen vectors
        mdarray<complex16, 2> fv_eigen_vectors_;
        
        /// second-variational eigen vectors
        mdarray<complex16, 2> sv_eigen_vectors_;

        /// full-diagonalization eigen vectors
        mdarray<complex16, 2> fd_eigen_vectors_;

        /// position of the G vector (from the G+k set) inside the FFT buffer 
        std::vector<int> fft_index_;
       
        /// first-variational states, distributed along the columns of the MPI grid
        mdarray<complex16, 2> fv_states_col_;
       
        /// first-variational states, distributed along the rows of the MPI grid
        mdarray<complex16, 2> fv_states_row_;

        /// two-component (spinor) wave functions describing the bands
        mdarray<complex16, 3> spinor_wave_functions_;

        /// band occupation numbers
        std::vector<double> band_occupancies_;

        /// band energies
        std::vector<double> band_energies_; 

        /// phase factors \f$ e^{i ({\bf G+k}) {\bf r}_{\alpha}} \f$
        mdarray<complex16, 2> gkvec_phase_factors_;

        /// spherical harmonics of G+k vectors
        mdarray<complex16, 2> gkvec_ylm_;

        /// precomputed values for the linear equations for matching coefficients
        mdarray<complex16, 4> alm_b_;

        /// length of G+k vectors
        std::vector<double> gkvec_len_;

        /// number of G+k vectors distributed along rows of MPI grid
        int num_gkvec_row_;
        
        /// number of G+k vectors distributed along columns of MPI grid
        int num_gkvec_col_;

        /// short information about each APW+lo basis function
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_;

        /// row APW+lo basis descriptors
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_row_;
        
        /// column APW+lo basis descriptors
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_col_;
            
        /// list of columns of the Hamiltonian and overlap matrix lo block (local index) for a given atom
        std::vector< std::vector<int> > atom_lo_cols_;

        /// list of rows of the Hamiltonian and overlap matrix lo block (local index) for a given atom
        std::vector< std::vector<int> > atom_lo_rows_;

        /// imaginary unit to the power of l
        std::vector<complex16> zil_;

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

        /// in case of pseudopotential: radial integrals of |beta> functions
        //mdarray<double, 3> beta_radial_integrals_;
        
        /// phase-factor independent plane-wave coefficients of |beta> functions 
        mdarray<complex16, 3> beta_pw_;

        /// Generate matching coefficients for specific l-value
        template <int order, bool conjugate>
        void generate_matching_coefficients_l(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                              mdarray<double, 2>& A, mdarray<complex16, 2>& alm);
        
        void check_alm(int num_gkvec_loc, int ia, mdarray<complex16, 2>& alm);

        /// Copy lo block from eigen-vector to wave-function
        inline void copy_lo_blocks(const complex16* z, complex16* vec);
        
        /// Copy plane wave block from eigen-vector to wave-function
        inline void copy_pw_block(const complex16* z, complex16* vec);

        /// Initialize G+k related data
        void init_gkvec();
        
        /// Build APW+lo basis descriptors 
        void build_apwlo_basis_descriptors();

        /// Block-cyclic distribution of relevant arrays 
        void distribute_block_cyclic();
        
        /// Test orthonormalization of first-variational states
        /** Important: distribute_fv_states_row() must be called prior to calling this function.*/
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
            if (parameters_.basis_type() == pwlo)
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
            muffin-tin boundary. The following formula is used:
            \f[
                j_{{n}}^{{\prime}}(z)=-j_{{n+1}}(z)+(n/z)j_{{n}}(z)
            \f]
        */
        template <bool conjugate>
        void generate_matching_coefficients(int num_gkvec_loc, int ia, mdarray<complex16, 2>& alm);
        
        /// Generate first-variational states from eigen-vectors
        void generate_fv_states();

        /// Distribute fv states over rows of the MPI grid 
        void distribute_fv_states_row();
        
        /// Generate two-component spinor wave functions 
        void generate_spinor_wave_functions();

        Periodic_function<complex16>* spinor_wave_function_component(int lmax, int ispn, int j);

        //== void spinor_wave_function_component_mt(int lmax, int ispn, int jloc, mt_functions<complex16>& psilm);
        
        void generate_beta_pw(complex16* beta_pw__, int ia);
        
        void save(int id);

        void load(HDF5_tree h5in, int id);

        //== void save_wave_functions(int id);

        //== void load_wave_functions(int id);

        void get_fv_eigen_vectors(mdarray<complex16, 2>& fv_evec);
        
        void get_sv_eigen_vectors(mdarray<complex16, 2>& sv_evec);
        
        /// Test orthonormalization of spinor wave-functions
        void test_spinor_wave_functions(int use_fft);
        
        /// APW+lo basis size
        /** Total number of APW+lo basis functions is equal to the number of augmented plane-waves plus
            the number of local orbitals. */
        inline int apwlo_basis_size()
        {
            return (int)apwlo_basis_descriptors_.size();
        }
        
        /// Global index of G-vector by the index of G+k vector
        inline int gvec_index(int igk) 
        {
            assert(igk >= 0 && igk < (int)gvec_index_.size());
            
            return gvec_index_[igk];
        }
        
        /// Return G+k vector in fractional coordinates
        inline vector3d<double> gkvec(int igk)
        {
            assert(igk >= 0 && igk < gkvec_.size(1));

            return vector3d<double>(gkvec_(0, igk), gkvec_(1, igk), gkvec_(2, igk));
        }

        inline vector3d<double> gkvec_cart(int igk)
        {
            return parameters_.reciprocal_lattice()->get_cartesian_coordinates(gkvec(igk));
        }

        inline complex16 gkvec_phase_factor(int igk, int ia)
        {
            return gkvec_phase_factors_(igk, ia);
        }

        /// Return length of a G+k vector
        inline double gkvec_len(int igk)
        {
            return gkvec_len_[igk];
        }
                
        /// Total number of G+k vectors within the cutoff distance
        inline int num_gkvec()
        {
            assert(gkvec_.size(1) == (int)gvec_index_.size());

            return gkvec_.size(1);
        }

        /// Total number of muffin-tin and plane-wave expansion coefficients for the wave-functions.
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
        */ 
        inline int wf_size()
        {
            return (parameters_.unit_cell()->mt_basis_size() + num_gkvec());
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

        inline complex16& spinor_wave_function(int idxwf, int ispn, int j)
        {
            return spinor_wave_functions_(idxwf, ispn, j);
        }

        inline int* fft_index()
        {
            return &fft_index_[0];
        }

        inline vector3d<double> vk()
        {
            return vk_;
        }

        inline double vk(int x)
        {
            return vk_[x];
        }
        
        /// Number of APW+lo basis functions distributed along rows of MPI grid
        inline int apwlo_basis_size_row()
        {
            return (int)apwlo_basis_descriptors_row_.size();
        }
        
        /// Number of G+k vectors along the rows of the matrix
        inline int num_gkvec_row()
        {
            return num_gkvec_row_;
        }

        /// Number of local orbitals along the rows of the matrix
        inline int num_lo_row()
        {
            return (int)apwlo_basis_descriptors_row_.size() - num_gkvec_row_;
        }

        /// Number of APW+lo basis functions distributed along columns of MPI grid
        inline int apwlo_basis_size_col()
        {
            return (int)apwlo_basis_descriptors_col_.size();
        }
        
        /// Number of G+k vectors along the columns of the matrix
        inline int num_gkvec_col()
        {
            return num_gkvec_col_;
        }
        
        /// Number of local orbitals along the columns of the matrix
        inline int num_lo_col()
        {
            return (int)apwlo_basis_descriptors_col_.size() - num_gkvec_col_;
        }

        /// Local fraction of G+k vectors for a given MPI rank
        /** In case of distributed matrix setup row and column G+k vectors are combined. Row G+k vectors are first.*/
        inline int num_gkvec_loc()
        {
            if ((num_gkvec_row() == num_gkvec()) && (num_gkvec_col() == num_gkvec()))
            {
                return num_gkvec();
            }
            else
            {
                return (num_gkvec_row() + num_gkvec_col());
            }
        } 
        
        /// Return the global index of the G+k vector by the local index
        inline int igkglob(int igkloc)
        {
            if ((num_gkvec_row() == num_gkvec()) && (num_gkvec_col() == num_gkvec()))
            {
                return igkloc;
            }
            else
            {
                int igk = (igkloc < num_gkvec_row()) ? apwlo_basis_descriptors_row_[igkloc].igk : 
                                                       apwlo_basis_descriptors_col_[igkloc - num_gkvec_row()].igk;
                assert(igk >= 0);
                return igk;
            }
        }

        inline apwlo_basis_descriptor& apwlo_basis_descriptors_col(int idx)
        {
            return apwlo_basis_descriptors_col_[idx];
        }
        
        inline apwlo_basis_descriptor& apwlo_basis_descriptors_row(int idx)
        {
            return apwlo_basis_descriptors_row_[idx];
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
        /** In case of distributed matrix setup row and column apw coefficients are combined. Row coefficients are first.*/
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

        inline mdarray<complex16, 2>& fv_eigen_vectors()
        {
            return fv_eigen_vectors_;
        }
        
        inline mdarray<complex16, 2>& fv_states_col()
        {
            return fv_states_col_;
        }
        
        inline mdarray<complex16, 2>& fv_states_row()
        {
            return fv_states_row_;
        }

        inline mdarray<complex16, 2>& sv_eigen_vectors()
        {
            return sv_eigen_vectors_;
        }
        
        inline mdarray<complex16, 2>& fd_eigen_vectors()
        {
            return fd_eigen_vectors_;
        }

        void bypass_sv()
        {
            memcpy(&band_energies_[0], &fv_eigen_values_[0], parameters_.num_fv_states() * sizeof(double));
            sv_eigen_vectors_.zero();
            for (int icol = 0; icol < parameters_.spl_fv_states_col().local_size(); icol++)
            {
                int i = parameters_.spl_fv_states_col(icol);
                for (int irow = 0; irow < parameters_.spl_fv_states_row().local_size(); irow++)
                {
                    if (parameters_.spl_fv_states_row(irow) == i) sv_eigen_vectors_(irow, icol) = complex16(1, 0);
                }
            }
        }

        std::vector<double> get_pw_ekin()
        {
            std::vector<double> pw_ekin(num_gkvec());
            for (int igk = 0; igk < num_gkvec(); igk++) pw_ekin[igk] = 0.5 * pow(gkvec_len(igk), 2);
            return pw_ekin; 
        }
};

#include "k_point.hpp"

};

