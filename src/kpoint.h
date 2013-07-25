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
    // TODO: add G+k vector in lattice and Cartesian coordinates
    int idxglob;
    int igk;
    int ig;
    int ia;
    int l;
    int lm;
    int order;
    int idxrf;

    double gkvec[3];
    double gkvec_cart[3];

};

class kpoint
{
    private:

        /// global set of parameters
        Global& parameters_;

        /// weight of k-point
        double weight_;

        /// fractional k-point coordinates
        double vk_[3];
        
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
        
        /// Number of G+k vectors distributed along columns of MPI grid
        int num_gkvec_col_;

        /// Short information about each APW+lo basis function
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_;

        /// Row APW+lo basis descriptors
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_row_;
        
        /// Column APW+lo basis descriptors
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_col_;
            
        /// List of columns (lo block) for a given atom
        std::vector< std::vector<int> > icol_by_atom_;

        /// List of rows (lo block) for a given atom
        std::vector< std::vector<int> > irow_by_atom_;

        /// Imaginary unit to the power of l
        std::vector<complex16> zil_;

        /// Mapping between lm and l
        mdarray<int, 1> l_by_lm_;

        /// Spherical bessel functions for G+k vectors  
        std::vector< sbessel_pw<double>* > sbessel_;
        
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
        /** In case of distributed matrix setup row and column G+k vectors are combined. */
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

        /// Generate matching coefficients for specific l-value
        template <int order>
        void generate_matching_coefficients_l(int ia, int iat, AtomType* type, int l, int num_gkvec_loc, 
                                              mdarray<double, 2>& A, mdarray<complex16, 2>& alm);
        
        /// Apply the muffin-tin part of the first-variational Hamiltonian to the apw basis function
        /** The following vector is computed:
            \f[
              b_{L_2 \nu_2}^{\alpha}({\bf G'}) = \sum_{L_1 \nu_1} \sum_{L_3} 
                a_{L_1\nu_1}^{\alpha*}({\bf G'}) 
                \langle u_{\ell_1\nu_1}^{\alpha} | h_{L3}^{\alpha} |  u_{\ell_2\nu_2}^{\alpha}  
                \rangle  \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle +  
                \frac{1}{2} \sum_{\nu_1} a_{L_2\nu_1}^{\alpha *}({\bf G'})
                u_{\ell_2\nu_1}^{\alpha}(R_{\alpha})
                u_{\ell_2\nu_2}^{'\alpha}(R_{\alpha})R_{\alpha}^{2}
            \f] 
        */
        void apply_hmt_to_apw(int num_gkvec_row, int ia, mdarray<complex16, 2>& alm, mdarray<complex16, 2>& halm);

        void check_alm(int num_gkvec_loc, int ia, mdarray<complex16, 2>& alm);
        
        void set_fv_h_o_apw_lo(AtomType* type, Atom* atom, int ia, int apw_offset_col, mdarray<complex16, 2>& alm, 
                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);

        void set_fv_h_o_it(PeriodicFunction<double>* effective_potential, 
                           mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);

        void set_fv_h_o_lo_lo(mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);

        template <processing_unit_t pu>
        void set_fv_h_o_pw_lo(PeriodicFunction<double>* effective_potential, int num_ranks,
                              mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);

        void solve_fv_evp_1stage(Band* band, mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);

        void solve_fv_evp_2stage(mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);

        /// Generate first-variational states
        /** 1. setup H and O \n 
            2. solve \$ H\psi = E\psi \$ \n
            3. senerate wave-functions from eigen-vectors

            \param [in] band Pointer to Band class
            \param [in] effective_potential Pointer to effective potential 

        */
        void generate_fv_states(Band* band, PeriodicFunction<double>* effective_potential);

        inline void copy_lo_blocks(const int apwlo_basis_size_row, const int num_gkvec_row, 
                                   const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row, 
                                   const complex16* z, complex16* vec);
        
        inline void copy_pw_block(const int num_gkvec, const int num_gkvec_row, 
                                  const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row, 
                                  const complex16* z, complex16* vec);

        void generate_spinor_wave_functions(Band* band);

        /// Find G+k vectors within the cutoff
        void generate_gkvec();

        /// Initialize G+k related data
        void init_gkvec();

        /// Build APW+lo basis descriptors 
        void build_apwlo_basis_descriptors();

        /// Block-cyclic distribution of relevant arrays 
        void distribute_block_cyclic(Band* band);
        
        void test_fv_states(Band* band, int use_fft);

        void test_spinor_wave_functions(int use_fft);

    public:

        /// Constructor
        kpoint(Global& parameters__, double* vk__, double weight__) : parameters_(parameters__), weight_(weight__)
        {
            for (int x = 0; x < 3; x++) vk_[x] = vk__[x];

            band_occupancies_.resize(parameters_.num_bands());
            memset(&band_occupancies_[0], 0, parameters_.num_bands() * sizeof(double));
        }

        ~kpoint()
        {
            if (basis_type == pwlo)
            {
                for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) delete sbessel_[igkloc];
            }
        }

        void initialize(Band* band);

        /// Generate plane-wave matching coefficents for the radial solutions 

        /** The matching coefficients are conjugated!. This is done in favor of the convenient overlap 
            matrix construnction.
        */
        void generate_matching_coefficients(int num_gkvec_loc, int ia, mdarray<complex16, 2>& alm);
        
        /// Setup the Hamiltonian and overlap matrices in APW+lo basis
        /** The Hamiltonian matrix has the following expression:
            \f[
                H_{\mu' \mu} = \langle \varphi_{\mu'} | \hat H | \varphi_{\mu} \rangle
            \f]

            \f[
                H_{\mu' \mu}=\langle \varphi_{\mu' } | \hat H | \varphi_{\mu } \rangle  = 
                \left( \begin{array}{cc} 
                   H_{\bf G'G} & H_{{\bf G'}j} \\
                   H_{j'{\bf G}} & H_{j'j}
                \end{array} \right)
            \f]
            
            The overlap matrix has the following expression:
            \f[
                O_{\mu' \mu} = \langle \varphi_{\mu'} | \varphi_{\mu} \rangle
            \f]
            APW-APW block:
            \f[
                O_{{\bf G'} {\bf G}}^{\bf k} = \sum_{\alpha} \sum_{L\nu} a_{L\nu}^{\alpha *}({\bf G'+k}) 
                a_{L\nu}^{\alpha}({\bf G+k})
            \f]
            
            APW-lo block:
            \f[
                O_{{\bf G'} j}^{\bf k} = \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'+k}) 
                \langle u_{\ell_j \nu'}^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j} \rangle
            \f]

            lo-APW block:
            \f[
                O_{j' {\bf G}}^{\bf k} = 
                \sum_{\nu'} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | u_{\ell_{j'} \nu'}^{\alpha_{j'}} \rangle
                a_{\ell_{j'} m_{j'} \nu'}^{\alpha_{j'}}({\bf G+k}) 
            \f]

            lo-lo block:
            \f[
                O_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | 
                \phi_{\ell_{j}}^{\zeta_{j} \alpha_{j}} \rangle \delta_{\alpha_{j'} \alpha_j} 
                \delta_{\ell_{j'} \ell_j} \delta_{m_{j'} m_j}
            \f]

        */
        template <processing_unit_t pu, basis_t basis>
        void set_fv_h_o(PeriodicFunction<double>* effective_potential, int num_ranks,
                        mdarray<complex16, 2>& h, mdarray<complex16, 2>& o);
        
        /// Solve \f$ \hat H \psi = E \psi \f$ and find the eigen-states of the Hamiltonian
        void find_eigen_states(Band* band, PeriodicFunction<double>* effective_potential, 
                               PeriodicFunction<double>* effective_magnetic_field[3]);

        template <processing_unit_t pu, basis_t basis>
        void ibs_force(Band* band, mdarray<double, 2>& ffac, mdarray<double, 2>& force);

        PeriodicFunction<complex16>* spinor_wave_function_component(Band* band, int lmax, int ispn, int j);

        void spinor_wave_function_component_mt(Band* band, int lmax, int ispn, int jloc, mdarray<complex16, 3>& fylm);
        
        void save_wave_functions(int id, Band* band__);

        void load_wave_functions(int id, Band* band__);

        void get_fv_eigen_vectors(mdarray<complex16, 2>& fv_evec);
        
        void get_sv_eigen_vectors(mdarray<complex16, 2>& sv_evec);
        
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
        
        /// Pointer to G+k vector
        inline double* gkvec(int igk)
        {
            assert(igk >= 0 && igk < gkvec_.size(1));

            return &gkvec_(0, igk);
        }

        inline complex16 gkvec_phase_factor(int igk, int ia)
        {
            return gkvec_phase_factors_(igk, ia);
        }
                
        /// Total number of G+k vectors within the cutoff distance
        inline int num_gkvec()
        {
            assert(gkvec_.size(1) == (int)gvec_index_.size());

            return gkvec_.size(1);
        }

        /// Total number of muffin-tin and plane-wave expansion coefficients for the first-variational state
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

            Thus, the total number of coefficients representing a first-variational state is equal
            to the number of muffi-tin basis functions of the form \f$ f_{\ell \lambda}^{\alpha}(r) 
            Y_{\ell m}(\hat {\bf r}) \f$ plust the number of G+k plane waves. 
        */ 
        inline int mtgk_size() // TODO: find a better name for this
        {
            return (parameters_.mt_basis_size() + num_gkvec());
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

        inline double* vk()
        {
            return vk_;
        }

        inline double vk(int x)
        {
            return vk_[x];
        }
};

#include "kpoint_cpu.hpp"
#ifdef _GPU_
#include "kpoint_gpu.hpp"
#endif

void kpoint::initialize(Band* band)
{
    Timer t("sirius::kpoint::initialize");
    
    zil_.resize(parameters_.lmax() + 1);
    for (int l = 0; l <= parameters_.lmax(); l++) zil_[l] = pow(complex16(0, 1), l);
    
    l_by_lm_.set_dimensions(Utils::lmmax_by_lmax(parameters_.lmax()));
    l_by_lm_.allocate();
    for (int l = 0, lm = 0; l <= parameters_.lmax(); l++)
    {
        for (int m = -l; m <= l; m++, lm++) l_by_lm_(lm) = l;
    }

    generate_gkvec();

    build_apwlo_basis_descriptors();

    distribute_block_cyclic(band);
    
    init_gkvec();
    
    icol_by_atom_.resize(parameters_.num_atoms());
    irow_by_atom_.resize(parameters_.num_atoms());

    for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[icol].ia;
        icol_by_atom_[ia].push_back(icol);
    }
    
    for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
    {
        int ia = apwlo_basis_descriptors_row_[irow].ia;
        irow_by_atom_[ia].push_back(irow);
    }
    
    if (basis_type == pwlo)
    {
        sbessel_.resize(num_gkvec_loc()); 
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        {
            sbessel_[igkloc] = new sbessel_pw<double>(parameters_, parameters_.lmax_pw());
            sbessel_[igkloc]->interpolate(gkvec_len_[igkloc]);
        }
    }
    
    fv_eigen_values_.resize(parameters_.num_fv_states());

    fv_eigen_vectors_.set_dimensions(apwlo_basis_size_row(), band->spl_fv_states_col().local_size());
    fv_eigen_vectors_.allocate();
    
    fv_states_col_.set_dimensions(mtgk_size(), band->spl_fv_states_col().local_size());
    fv_states_col_.allocate();
    
    if (band->num_ranks() == 1)
    {
        fv_states_row_.set_dimensions(mtgk_size(), parameters_.num_fv_states());
        fv_states_row_.set_ptr(fv_states_col_.get_ptr());
    }
    else
    {
        fv_states_row_.set_dimensions(mtgk_size(), band->spl_fv_states_row().local_size());
        fv_states_row_.allocate();
    }
    
    sv_eigen_vectors_.set_dimensions(band->spl_fv_states_row().local_size(), band->spl_spinor_wf_col().local_size());
    sv_eigen_vectors_.allocate();

    band_energies_.resize(parameters_.num_bands());

    spinor_wave_functions_.set_dimensions(mtgk_size(), parameters_.num_spins(), 
                                          band->spl_spinor_wf_col().local_size());

    if (band->sv())
    {
        spinor_wave_functions_.allocate();
    }
    else
    {
        spinor_wave_functions_.set_ptr(fv_states_col_.get_ptr());
    }
}

// TODO: add a switch to return conjuagted or normal coefficients
void kpoint::generate_matching_coefficients(int num_gkvec_loc, int ia, mdarray<complex16, 2>& alm)
{
    Timer t("sirius::kpoint::generate_matching_coefficients");

    Atom* atom = parameters_.atom(ia);
    AtomType* type = atom->type();

    assert(type->max_aw_order() <= 2);

    int iat = parameters_.atom_type_index_by_id(type->id());

    #pragma omp parallel default(shared)
    {
        mdarray<double, 2> A(2, 2);

        #pragma omp for
        for (int l = 0; l <= parameters_.lmax_apw(); l++)
        {
            int num_aw = (int)type->aw_descriptor(l).size();

            for (int order = 0; order < num_aw; order++)
            {
                for (int dm = 0; dm < num_aw; dm++)
                {
                    A(dm, order) = atom->symmetry_class()->aw_surface_dm(l, order, dm);
                }
            }

            switch (num_aw)
            {
                case 1:
                {
                    generate_matching_coefficients_l<1>(ia, iat, type, l, num_gkvec_loc, A, alm);
                    break;
                }
                case 2:
                {
                    generate_matching_coefficients_l<2>(ia, iat, type, l, num_gkvec_loc, A, alm);
                    break;
                }
                default:
                {
                    error(__FILE__, __LINE__, "wrong order of augmented wave", fatal_err);
                }
            }
        } //l
    }
    
    // check alm coefficients
    if (debug_level > 1) check_alm(num_gkvec_loc, ia, alm);
}

void kpoint::check_alm(int num_gkvec_loc, int ia, mdarray<complex16, 2>& alm)
{
    static SHT* sht = NULL;
    if (!sht)
    {
        sht = new SHT(parameters_.lmax_apw());
    }

    Atom* atom = parameters_.atom(ia);
    AtomType* type = parameters_.atom(ia)->type();

    mdarray<complex16, 2> z1(sht->num_points(), type->mt_aw_basis_size());
    for (int i = 0; i < type->mt_aw_basis_size(); i++)
    {
        int lm = type->indexb(i).lm;
        int idxrf = type->indexb(i).idxrf;
        double rf = atom->symmetry_class()->radial_function(atom->num_mt_points() - 1, idxrf);
        for (int itp = 0; itp < sht->num_points(); itp++)
        {
            z1(itp, i) = sht->ylm_backward(lm, itp) * rf;
        }
    }

    mdarray<complex16, 2> z2(sht->num_points(), num_gkvec_loc);
    blas<cpu>::gemm(0, 2, sht->num_points(), num_gkvec_loc, type->mt_aw_basis_size(), z1.get_ptr(), z1.ld(),
                    alm.get_ptr(), alm.ld(), z2.get_ptr(), z2.ld());

    double vc[3];
    parameters_.get_coordinates<cartesian, direct>(parameters_.atom(ia)->position(), vc);
    
    double tdiff = 0;
    for (int igloc = 0; igloc < num_gkvec_loc; igloc++)
    {
        double gkc[3];
        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(igkglob(igloc)), gkc);
        for (int itp = 0; itp < sht->num_points(); itp++)
        {
            complex16 aw_value = z2(itp, igloc);
            double r[3];
            for (int x = 0; x < 3; x++) r[x] = vc[x] + sht->coord(x, itp) * type->mt_radius();
            complex16 pw_value = exp(complex16(0, Utils::scalar_product(r, gkc))) / sqrt(parameters_.omega());
            tdiff += abs(pw_value - aw_value);
        }
    }

    printf("atom : %i  absolute alm error : %e  average alm error : %e\n", 
           ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
}

void kpoint::apply_hmt_to_apw(int num_gkvec_row, int ia, mdarray<complex16, 2>& alm, mdarray<complex16, 2>& halm)
{
    Timer t("sirius::kpoint::apply_hmt_to_apw");
    
    Atom* atom = parameters_.atom(ia);
    AtomType* type = atom->type();
    
    #pragma omp parallel default(shared)
    {
        std::vector<complex16> zv(num_gkvec_row);
        
        #pragma omp for
        for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
        {
            memset(&zv[0], 0, num_gkvec_row * sizeof(complex16));

            int lm2 = type->indexb(j2).lm;
            int idxrf2 = type->indexb(j2).idxrf;

            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++)
            {
                int lm1 = type->indexb(j1).lm;
                int idxrf1 = type->indexb(j1).idxrf;
                
                complex16 zsum = parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm2,  
                                                                          atom->h_radial_integrals(idxrf1, idxrf2));
                
                if (abs(zsum) > 1e-14) 
                {
                    for (int ig = 0; ig < num_gkvec_row; ig++) zv[ig] += zsum * alm(ig, j1); 
                }
            } // j1
             
            int l2 = type->indexb(j2).l;
            int order2 = type->indexb(j2).order;
            
            for (int order1 = 0; order1 < (int)type->aw_descriptor(l2).size(); order1++)
            {
                double t1 = 0.5 * pow(type->mt_radius(), 2) * 
                            atom->symmetry_class()->aw_surface_dm(l2, order1, 0) * 
                            atom->symmetry_class()->aw_surface_dm(l2, order2, 1);
                
                for (int ig = 0; ig < num_gkvec_row; ig++) 
                    zv[ig] += t1 * alm(ig, type->indexb_by_lm_order(lm2, order1));
            }
            
            memcpy(&halm(0, j2), &zv[0], num_gkvec_row * sizeof(complex16));
        } // j2
    }
}

void kpoint::set_fv_h_o_apw_lo(AtomType* type, Atom* atom, int ia, int apw_offset_col, mdarray<complex16, 2>& alm, 
                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::kpoint::set_fv_h_o_apw_lo");
    
    // apw-lo block
    for (int i = 0; i < (int)icol_by_atom_[ia].size(); i++)
    {
        int icol = icol_by_atom_[ia][i];

        int l = apwlo_basis_descriptors_col_[icol].l;
        int lm = apwlo_basis_descriptors_col_[icol].lm;
        int idxrf = apwlo_basis_descriptors_col_[icol].idxrf;
        int order = apwlo_basis_descriptors_col_[icol].order;
        
        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
        {
            int lm1 = type->indexb(j1).lm;
            int idxrf1 = type->indexb(j1).idxrf;
                    
            complex16 zsum = parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm, atom->h_radial_integrals(idxrf, idxrf1));

            if (abs(zsum) > 1e-14)
            {
                for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++) h(igkloc, icol) += zsum * alm(igkloc, j1);
            }
        }

        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
        {
            for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
            {
                o(igkloc, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order) * 
                                   alm(igkloc, type->indexb_by_lm_order(lm, order1));
            }
        }
    }

    std::vector<complex16> ztmp(num_gkvec_col());
    // lo-apw block
    for (int i = 0; i < (int)irow_by_atom_[ia].size(); i++)
    {
        int irow = irow_by_atom_[ia][i];

        int l = apwlo_basis_descriptors_row_[irow].l;
        int lm = apwlo_basis_descriptors_row_[irow].lm;
        int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;
        int order = apwlo_basis_descriptors_row_[irow].order;

        memset(&ztmp[0], 0, num_gkvec_col() * sizeof(complex16));
    
        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
        {
            int lm1 = type->indexb(j1).lm;
            int idxrf1 = type->indexb(j1).idxrf;
                    
            complex16 zsum = parameters_.gaunt().sum_L3_complex_gaunt(lm, lm1, atom->h_radial_integrals(idxrf, idxrf1));

            if (abs(zsum) > 1e-14)
            {
                for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
                    ztmp[igkloc] += zsum * conj(alm(apw_offset_col + igkloc, j1));
            }
        }

        for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++) h(irow, igkloc) += ztmp[igkloc]; 

        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
        {
            for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
            {
                o(irow, igkloc) += atom->symmetry_class()->o_radial_integral(l, order, order1) * 
                                   conj(alm(apw_offset_col + igkloc, type->indexb_by_lm_order(lm, order1)));
            }
        }
    }
}

void kpoint::set_fv_h_o_it(PeriodicFunction<double>* effective_potential, 
                           mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::kpoint::set_fv_h_o_it");

    #pragma omp parallel for default(shared)
    for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
    {
        double v2c[3];
        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_col_[igkloc2].igk), v2c);

        for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
        {
            int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
                                             apwlo_basis_descriptors_col_[igkloc2].ig);
            double v1c[3];
            parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_row_[igkloc1].igk), v1c);
            
            double t1 = 0.5 * Utils::scalar_product(v1c, v2c);
                               
            h(igkloc1, igkloc2) += (effective_potential->f_pw(ig12) + t1 * parameters_.step_function_pw(ig12));
            o(igkloc1, igkloc2) += parameters_.step_function_pw(ig12);
        }
    }
}

void kpoint::set_fv_h_o_lo_lo(mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::kpoint::set_fv_h_o_lo_lo");

    // lo-lo block
    #pragma omp parallel for default(shared)
    for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[icol].ia;
        int lm2 = apwlo_basis_descriptors_col_[icol].lm; 
        int idxrf2 = apwlo_basis_descriptors_col_[icol].idxrf; 

        for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
        {
            if (ia == apwlo_basis_descriptors_row_[irow].ia)
            {
                Atom* atom = parameters_.atom(ia);
                int lm1 = apwlo_basis_descriptors_row_[irow].lm; 
                int idxrf1 = apwlo_basis_descriptors_row_[irow].idxrf; 

                h(irow, icol) += parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm2, 
                                                                          atom->h_radial_integrals(idxrf1, idxrf2));

                if (lm1 == lm2)
                {
                    int l = apwlo_basis_descriptors_row_[irow].l;
                    int order1 = apwlo_basis_descriptors_row_[irow].order; 
                    int order2 = apwlo_basis_descriptors_col_[icol].order; 
                    o(irow, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order2);
                }
            }
        }
    }
}

inline void kpoint::copy_lo_blocks(const int apwlo_basis_size_row, const int num_gkvec_row, 
                                   const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row, 
                                   const complex16* z, complex16* vec)
{
    for (int j = num_gkvec_row; j < apwlo_basis_size_row; j++)
    {
        int ia = apwlo_basis_descriptors_row[j].ia;
        int lm = apwlo_basis_descriptors_row[j].lm;
        int order = apwlo_basis_descriptors_row[j].order;
        vec[parameters_.atom(ia)->offset_wf() + parameters_.atom(ia)->type()->indexb_by_lm_order(lm, order)] = z[j];
    }
}

inline void kpoint::copy_pw_block(const int num_gkvec, const int num_gkvec_row, 
                                  const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row, 
                                  const complex16* z, complex16* vec)
{
    memset(vec, 0, num_gkvec * sizeof(complex16));

    for (int j = 0; j < num_gkvec_row; j++) vec[apwlo_basis_descriptors_row[j].igk] = z[j];
}

void kpoint::solve_fv_evp_1stage(Band* band, mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer *t1 = new Timer("sirius::kpoint::generate_fv_states:genevp");
    generalized_evp* solver = NULL;

    switch (parameters_.eigen_value_solver())
    {
        case lapack:
        {
            solver = new generalized_evp_lapack(-1.0);
            break;
        }
        case scalapack:
        {
            solver = new generalized_evp_scalapack(parameters_.cyclic_block_size(), band->num_ranks_row(), 
                                                   band->num_ranks_col(), band->blacs_context(), -1.0);
            break;
        }
        case elpa:
        {
            solver = new generalized_evp_elpa(parameters_.cyclic_block_size(), apwlo_basis_size_row(), 
                                              band->num_ranks_row(), band->rank_row(),
                                              apwlo_basis_size_col(), band->num_ranks_col(), 
                                              band->rank_col(), band->blacs_context(), 
                                              parameters_.mpi_grid().communicator(1 << band->dim_row()),
                                              parameters_.mpi_grid().communicator(1 << band->dim_col()),
                                              parameters_.mpi_grid().communicator(1 << band->dim_col() | 
                                                                                  1 << band->dim_row()));
            break;
        }
        case magma:
        {
            solver = new generalized_evp_magma();
            break;
        }
        default:
        {
            error(__FILE__, __LINE__, "eigen value solver is not defined", fatal_err);
        }
    }

    solver->solve(apwlo_basis_size(), parameters_.num_fv_states(), h.get_ptr(), h.ld(), o.get_ptr(), o.ld(), 
                  &fv_eigen_values_[0], fv_eigen_vectors_.get_ptr(), fv_eigen_vectors_.ld());

    delete solver;
    delete t1;
}

void kpoint::solve_fv_evp_2stage(mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    if (parameters_.eigen_value_solver() != lapack) error(__FILE__, __LINE__, "implemented for LAPACK only");
    
    standard_evp_lapack s;

    std::vector<double> o_eval(apwlo_basis_size());
    
    mdarray<complex16, 2> o_tmp(apwlo_basis_size(), apwlo_basis_size());
    memcpy(o_tmp.get_ptr(), o.get_ptr(), o.size() * sizeof(complex16));
    mdarray<complex16, 2> o_evec(apwlo_basis_size(), apwlo_basis_size());
 
    s.solve(apwlo_basis_size(), o_tmp.get_ptr(), o_tmp.ld(), &o_eval[0], o_evec.get_ptr(), o_evec.ld());

    int num_dependent_apwlo = 0;
    for (int i = 0; i < apwlo_basis_size(); i++) 
    {
        if (fabs(o_eval[i]) < 1e-4) 
        {
            num_dependent_apwlo++;
        }
        else
        {
            o_eval[i] = 1.0 / sqrt(o_eval[i]);
        }
    }

    //std::cout << "num_dependent_apwlo = " << num_dependent_apwlo << std::endl;

    mdarray<complex16, 2> h_tmp(apwlo_basis_size(), apwlo_basis_size());
    // compute h_tmp = Z^{h.c.} * H
    blas<cpu>::gemm(2, 0, apwlo_basis_size(), apwlo_basis_size(), apwlo_basis_size(), o_evec.get_ptr(), 
                    o_evec.ld(), h.get_ptr(), h.ld(), h_tmp.get_ptr(), h_tmp.ld());
    // compute \tilda H = Z^{h.c.} * H * Z = h_tmp * Z
    blas<cpu>::gemm(0, 0, apwlo_basis_size(), apwlo_basis_size(), apwlo_basis_size(), h_tmp.get_ptr(), 
                    h_tmp.ld(), o_evec.get_ptr(), o_evec.ld(), h.get_ptr(), h.ld());

    int reduced_apwlo_basis_size = apwlo_basis_size() - num_dependent_apwlo;
    
    for (int i = 0; i < reduced_apwlo_basis_size; i++)
    {
        for (int j = 0; j < reduced_apwlo_basis_size; j++)
        {
            double d = o_eval[num_dependent_apwlo + j] * o_eval[num_dependent_apwlo + i];
            h(num_dependent_apwlo + j, num_dependent_apwlo + i) *= d;
        }
    }

    std::vector<double> h_eval(reduced_apwlo_basis_size);
    s.solve(reduced_apwlo_basis_size, &h(num_dependent_apwlo, num_dependent_apwlo), h.ld(), &h_eval[0], 
            h_tmp.get_ptr(), h_tmp.ld());

    for (int i = 0; i < reduced_apwlo_basis_size; i++)
    {
        for (int j = 0; j < reduced_apwlo_basis_size; j++) h_tmp(j, i) *= o_eval[num_dependent_apwlo + j];
    }

    for (int i = 0; i < parameters_.num_fv_states(); i++) fv_eigen_values_[i] = h_eval[i];

    blas<cpu>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), reduced_apwlo_basis_size, 
                    &o_evec(0, num_dependent_apwlo), o_evec.ld(), h_tmp.get_ptr(), h_tmp.ld(), 
                    fv_eigen_vectors_.get_ptr(), fv_eigen_vectors_.ld());
}

void kpoint::generate_fv_states(Band* band, PeriodicFunction<double>* effective_potential)
{
    Timer t("sirius::kpoint::generate_fv_states");

    mdarray<complex16, 2> h(apwlo_basis_size_row(), apwlo_basis_size_col());
    mdarray<complex16, 2> o(apwlo_basis_size_row(), apwlo_basis_size_col());
    
    // Magma requires special allocation
    #ifdef _MAGMA_
    if (parameters_.eigen_value_solver() == magma)
    {
        h.pin_memory();
        o.pin_memory();
    }
    #endif
   
    // setup Hamiltonian and overlap
    switch (parameters_.processing_unit())
    {
        case cpu:
        {
            set_fv_h_o<cpu, basis_type>(effective_potential, band->num_ranks(), h, o);
            break;
        }
        #ifdef _GPU_
        case gpu:
        {
            set_fv_h_o<gpu, basis_type>(effective_potential, band->num_ranks(), h, o);
            break;
        }
        #endif
        default:
        {
            error(__FILE__, __LINE__, "wrong processing unit");
        }
    }
    
    // TODO: move debug code to a separate function
    if ((debug_level > 0) && (parameters_.eigen_value_solver() == lapack))
    {
        Utils::check_hermitian("h", h);
        Utils::check_hermitian("o", o);
    }

    //sirius_io::hdf5_write_matrix("h.h5", h);
    //sirius_io::hdf5_write_matrix("o.h5", o);
    
    //Utils::write_matrix("h.txt", true, h);
    //Utils::write_matrix("o.txt", true, o);

    //** if (verbosity_level > 1)
    //** {
    //**     double h_max = 0;
    //**     double o_max = 0;
    //**     int h_irow = 0;
    //**     int h_icol = 0;
    //**     int o_irow = 0;
    //**     int o_icol = 0;
    //**     std::vector<double> h_diag(apwlo_basis_size(), 0);
    //**     std::vector<double> o_diag(apwlo_basis_size(), 0);
    //**     for (int icol = 0; icol < apwlo_basis_size_col(); icol++)
    //**     {
    //**         int idxglob = apwlo_basis_descriptors_col_[icol].idxglob;
    //**         for (int irow = 0; irow < apwlo_basis_size_row(); irow++)
    //**         {
    //**             if (apwlo_basis_descriptors_row_[irow].idxglob == idxglob)
    //**             {
    //**                 h_diag[idxglob] = abs(h(irow, icol));
    //**                 o_diag[idxglob] = abs(o(irow, icol));
    //**             }
    //**             if (abs(h(irow, icol)) > h_max)
    //**             {
    //**                 h_max = abs(h(irow, icol));
    //**                 h_irow = irow;
    //**                 h_icol = icol;
    //**             }
    //**             if (abs(o(irow, icol)) > o_max)
    //**             {
    //**                 o_max = abs(o(irow, icol));
    //**                 o_irow = irow;
    //**                 o_icol = icol;
    //**             }
    //**         }
    //**     }

    //**     Platform::allreduce(&h_diag[0], apwlo_basis_size(),
    //**                         parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));
    //**     
    //**     Platform::allreduce(&o_diag[0], apwlo_basis_size(),
    //**                         parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));
    //**     
    //**     if (parameters_.mpi_grid().root(1 << band->dim_row() | 1 << band->dim_col()))
    //**     {
    //**         std::stringstream s;
    //**         s << "h_diag : ";
    //**         for (int i = 0; i < apwlo_basis_size(); i++) s << h_diag[i] << " ";
    //**         s << std::endl;
    //**         s << "o_diag : ";
    //**         for (int i = 0; i < apwlo_basis_size(); i++) s << o_diag[i] << " ";
    //**         warning(__FILE__, __LINE__, s, 0);
    //**     }

    //**     std::stringstream s;
    //**     s << "h_max " << h_max << " irow, icol : " << h_irow << " " << h_icol << std::endl;
    //**     s << " (row) igk, ig, ia, l, lm, irdrf, order : " << apwlo_basis_descriptors_row_[h_irow].igk << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].ig << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].ia << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].l << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].lm << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].idxrf << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].order 
    //**                                                       << std::endl;
    //**     s << " (col) igk, ig, ia, l, lm, irdrf, order : " << apwlo_basis_descriptors_col_[h_icol].igk << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].ig << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].ia << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].l << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].lm << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].idxrf << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].order 
    //**                                                       << std::endl;

    //**     s << "o_max " << o_max << " irow, icol : " << o_irow << " " << o_icol << std::endl;
    //**     s << " (row) igk, ig, ia, l, lm, irdrf, order : " << apwlo_basis_descriptors_row_[o_irow].igk << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].ig << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].ia << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].l << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].lm << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].idxrf << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].order 
    //**                                                       << std::endl;
    //**     s << " (col) igk, ig, ia, l, lm, irdrf, order : " << apwlo_basis_descriptors_col_[o_icol].igk << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].ig << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].ia << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].l << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].lm << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].idxrf << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].order 
    //**                                                       << std::endl;
    //**     warning(__FILE__, __LINE__, s, 0);
    //** }
    
    assert(apwlo_basis_size() > parameters_.num_fv_states());
    
    //== fv_eigen_values_.resize(parameters_.num_fv_states());

    //== fv_eigen_vectors_.set_dimensions(apwlo_basis_size_row(), band->spl_fv_states_col().local_size());
    //== fv_eigen_vectors_.allocate();
   
    // debug scalapack
    //** std::vector<double> fv_eigen_values_glob(parameters_.num_fv_states());
    //** if ((debug_level > 2) && 
    //**     (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa))
    //** {
    //**     mdarray<complex16, 2> h_glob(apwlo_basis_size(), apwlo_basis_size());
    //**     mdarray<complex16, 2> o_glob(apwlo_basis_size(), apwlo_basis_size());
    //**     mdarray<complex16, 2> fv_eigen_vectors_glob(apwlo_basis_size(), parameters_.num_fv_states());

    //**     h_glob.zero();
    //**     o_glob.zero();

    //**     for (int icol = 0; icol < apwlo_basis_size_col(); icol++)
    //**     {
    //**         int j = apwlo_basis_descriptors_col_[icol].idxglob;
    //**         for (int irow = 0; irow < apwlo_basis_size_row(); irow++)
    //**         {
    //**             int i = apwlo_basis_descriptors_row_[irow].idxglob;
    //**             h_glob(i, j) = h(irow, icol);
    //**             o_glob(i, j) = o(irow, icol);
    //**         }
    //**     }
    //**     
    //**     Platform::allreduce(h_glob.get_ptr(), (int)h_glob.size(), 
    //**                         parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));
    //**     
    //**     Platform::allreduce(o_glob.get_ptr(), (int)o_glob.size(), 
    //**                         parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));

    //**     Utils::check_hermitian("h_glob", h_glob);
    //**     Utils::check_hermitian("o_glob", o_glob);
    //**     
    //**     generalized_evp_lapack lapack_solver(-1.0);

    //**     lapack_solver.solve(apwlo_basis_size(), parameters_.num_fv_states(), h_glob.get_ptr(), h_glob.ld(), 
    //**                         o_glob.get_ptr(), o_glob.ld(), &fv_eigen_values_glob[0], fv_eigen_vectors_glob.get_ptr(),
    //**                         fv_eigen_vectors_glob.ld());
    //** }
    
    if (fix_apwlo_linear_dependence)
    {
        solve_fv_evp_2stage(h, o);
    }
    else
    {
        solve_fv_evp_1stage(band, h, o);
    }
        
    #ifdef _MAGMA_
    if (parameters_.eigen_value_solver() == magma)
    {
        h.unpin_memory();
        o.unpin_memory();
    }
    #endif
   
    h.deallocate();
    o.deallocate();

    //** if ((debug_level > 2) && (parameters_.eigen_value_solver() == scalapack))
    //** {
    //**     double d = 0.0;
    //**     for (int i = 0; i < parameters_.num_fv_states(); i++) 
    //**         d += fabs(fv_eigen_values_[i] - fv_eigen_values_glob[i]);
    //**     std::stringstream s;
    //**     s << "Totoal eigen-value difference : " << d;
    //**     warning(__FILE__, __LINE__, s, 0);
    //** }
    
    // generate first-variational wave-functions
    //==fv_states_col_.set_dimensions(mtgk_size(), band->spl_fv_states_col().local_size());
    //==fv_states_col_.allocate();
    fv_states_col_.zero();

    mdarray<complex16, 2> alm(num_gkvec_row(), parameters_.max_mt_aw_basis_size());
    
    Timer *t2 = new Timer("sirius::kpoint::generate_fv_states:wf");
    if (basis_type == apwlo)
    {
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            Atom* atom = parameters_.atom(ia);
            AtomType* type = atom->type();
            
            generate_matching_coefficients(num_gkvec_row(), ia, alm);

            blas<cpu>::gemm(2, 0, type->mt_aw_basis_size(), band->spl_fv_states_col().local_size(),
                            num_gkvec_row(), &alm(0, 0), alm.ld(), &fv_eigen_vectors_(0, 0), 
                            fv_eigen_vectors_.ld(), &fv_states_col_(atom->offset_wf(), 0), 
                            fv_states_col_.ld());
        }
    }

    for (int j = 0; j < band->spl_fv_states_col().local_size(); j++)
    {
        copy_lo_blocks(apwlo_basis_size_row(), num_gkvec_row(), apwlo_basis_descriptors_row_, 
                       &fv_eigen_vectors_(0, j), &fv_states_col_(0, j));

        copy_pw_block(num_gkvec(), num_gkvec_row(), apwlo_basis_descriptors_row_, 
                      &fv_eigen_vectors_(0, j), &fv_states_col_(parameters_.mt_basis_size(), j));
    }

    for (int j = 0; j < band->spl_fv_states_col().local_size(); j++)
    {
        Platform::allreduce(&fv_states_col_(0, j), mtgk_size(), 
                            parameters_.mpi_grid().communicator(1 << band->dim_row()));
    }
    delete t2;
}

void kpoint::generate_spinor_wave_functions(Band* band)
{
    Timer t("sirius::kpoint::generate_spinor_wave_functions");

    spinor_wave_functions_.zero();
    
    if (parameters_.num_mag_dims() != 3)
    {
        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        {
            blas<cpu>::gemm(0, 0, mtgk_size(), band->spl_fv_states_col().local_size(), 
                            band->spl_fv_states_row().local_size(), 
                            &fv_states_row_(0, 0), fv_states_row_.ld(), 
                            &sv_eigen_vectors_(0, ispn * band->spl_fv_states_col().local_size()), 
                            sv_eigen_vectors_.ld(), 
                            &spinor_wave_functions_(0, ispn, ispn * band->spl_fv_states_col().local_size()), 
                            spinor_wave_functions_.ld() * parameters_.num_spins());
        }
    }
    else
    {
        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        {
            blas<cpu>::gemm(0, 0, mtgk_size(), band->spl_spinor_wf_col().local_size(), 
                            band->num_fv_states_row(ispn), 
                            &fv_states_row_(0, band->offs_fv_states_row(ispn)), fv_states_row_.ld(), 
                            &sv_eigen_vectors_(ispn * band->num_fv_states_row_up(), 0), 
                            sv_eigen_vectors_.ld(), 
                            &spinor_wave_functions_(0, ispn, 0), 
                            spinor_wave_functions_.ld() * parameters_.num_spins());
        }
    }
    
    for (int i = 0; i < band->spl_spinor_wf_col().local_size(); i++)
    {
        Platform::allreduce(&spinor_wave_functions_(0, 0, i), 
                            spinor_wave_functions_.size(0) * spinor_wave_functions_.size(1), 
                            parameters_.mpi_grid().communicator(1 << band->dim_row()));
    }
}

void kpoint::generate_gkvec()
{
    double gk_cutoff = parameters_.aw_cutoff() / parameters_.min_mt_radius();

    if ((gk_cutoff * parameters_.max_mt_radius() > double(parameters_.lmax_apw())) && basis_type == apwlo)
    {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff << ") is too large for a given lmax (" 
          << parameters_.lmax_apw() << ")" << std::endl
          << "minimum value for lmax : " << int(gk_cutoff * parameters_.max_mt_radius()) + 1;
        error(__FILE__, __LINE__, s);
    }

    if (gk_cutoff * 2 > parameters_.pw_cutoff())
        error(__FILE__, __LINE__, "aw cutoff is too large for a given plane-wave cutoff");

    std::vector< std::pair<double, int> > gkmap;

    // find G-vectors for which |G+k| < cutoff
    for (int ig = 0; ig < parameters_.num_gvec(); ig++)
    {
        double vgk[3];
        for (int x = 0; x < 3; x++) vgk[x] = parameters_.gvec(ig)[x] + vk_[x];

        double v[3];
        parameters_.get_coordinates<cartesian, reciprocal>(vgk, v);
        double gklen = Utils::vector_length(v);

        if (gklen <= gk_cutoff) gkmap.push_back(std::pair<double, int>(gklen, ig));
    }

    std::sort(gkmap.begin(), gkmap.end());

    gkvec_.set_dimensions(3, (int)gkmap.size());
    gkvec_.allocate();

    gvec_index_.resize(gkmap.size());

    for (int ig = 0; ig < (int)gkmap.size(); ig++)
    {
        gvec_index_[ig] = gkmap[ig].second;
        for (int x = 0; x < 3; x++) gkvec_(x, ig) = parameters_.gvec(gkmap[ig].second)[x] + vk_[x];
    }
    
    fft_index_.resize(num_gkvec());
    for (int ig = 0; ig < num_gkvec(); ig++) fft_index_[ig] = parameters_.fft_index(gvec_index_[ig]);
}

void kpoint::init_gkvec()
{
    gkvec_phase_factors_.set_dimensions(num_gkvec_loc(), parameters_.num_atoms());
    gkvec_phase_factors_.allocate();

    int lmax = std::max(parameters_.lmax_apw(), parameters_.lmax_pw());

    gkvec_ylm_.set_dimensions(Utils::lmmax_by_lmax(lmax), num_gkvec_loc());
    gkvec_ylm_.allocate();

    #pragma omp parallel for default(shared)
    for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
    {
        int igk = igkglob(igkloc);
        double v[3];
        double vs[3];

        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(igk), v);
        SHT::spherical_coordinates(v, vs); // vs = {r, theta, phi}

        SHT::spherical_harmonics(lmax, vs[1], vs[2], &gkvec_ylm_(0, igkloc));

        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            double phase = twopi * Utils::scalar_product(gkvec(igk), parameters_.atom(ia)->position());

            gkvec_phase_factors_(igkloc, ia) = exp(complex16(0.0, phase));
        }
    }
    
    gkvec_len_.resize(num_gkvec_loc());
    for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
    {
        int igk = igkglob(igkloc);
        double v[3];
        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(igk), v);
        gkvec_len_[igkloc] = Utils::vector_length(v);
    }
   
    if (basis_type == apwlo)
    {
        alm_b_.set_dimensions(parameters_.lmax_apw() + 1, parameters_.num_atom_types(), num_gkvec_loc(), 2);
        alm_b_.allocate();
        alm_b_.zero();

        // compute values of spherical Bessel functions and first derivative at MT boundary
        mdarray<double, 2> sbessel_mt(parameters_.lmax_apw() + 2, 2);
        sbessel_mt.zero();

        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        {
            for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
            {
                double R = parameters_.atom_type(iat)->mt_radius();

                double gkR = gkvec_len_[igkloc] * R;

                gsl_sf_bessel_jl_array(parameters_.lmax_apw() + 1, gkR, &sbessel_mt(0, 0));
                
                // Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                    sbessel_mt(l, 1) = -sbessel_mt(l + 1, 0) * gkvec_len_[igkloc] + (l / R) * sbessel_mt(l, 0);
                
                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    double f = fourpi / sqrt(parameters_.omega());
                    alm_b_(l, iat, igkloc, 0) = zil_[l] * f * sbessel_mt(l, 0); 
                    alm_b_(l, iat, igkloc, 1) = zil_[l] * f * sbessel_mt(l, 1);
                }
            }
        }
    }
}

void kpoint::build_apwlo_basis_descriptors()
{
    assert(apwlo_basis_descriptors_.size() == 0);

    apwlo_basis_descriptor apwlobd;

    // G+k basis functions
    for (int igk = 0; igk < num_gkvec(); igk++)
    {
        apwlobd.igk = igk;
        apwlobd.ig = gvec_index(igk);
        apwlobd.ia = -1;
        apwlobd.lm = -1;
        apwlobd.l = -1;
        apwlobd.order = -1;
        apwlobd.idxrf = -1;
        apwlobd.idxglob = (int)apwlo_basis_descriptors_.size();
        apwlo_basis_descriptors_.push_back(apwlobd);
    }

    // local orbital basis functions
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        Atom* atom = parameters_.atom(ia);
        AtomType* type = atom->type();
    
        int lo_index_offset = type->mt_aw_basis_size();
        
        for (int j = 0; j < type->mt_lo_basis_size(); j++) 
        {
            int l = type->indexb(lo_index_offset + j).l;
            int lm = type->indexb(lo_index_offset + j).lm;
            int order = type->indexb(lo_index_offset + j).order;
            int idxrf = type->indexb(lo_index_offset + j).idxrf;
            apwlobd.igk = -1;
            apwlobd.ig = -1;
            apwlobd.ia = ia;
            apwlobd.lm = lm;
            apwlobd.l = l;
            apwlobd.order = order;
            apwlobd.idxrf = idxrf;
            apwlobd.idxglob = (int)apwlo_basis_descriptors_.size();
            apwlo_basis_descriptors_.push_back(apwlobd);
        }
    }
    
    // ckeck if we count basis functions correctly
    if ((int)apwlo_basis_descriptors_.size() != (num_gkvec() + parameters_.mt_lo_basis_size()))
    {
        std::stringstream s;
        s << "(L)APW+lo basis descriptors array has a wrong size" << std::endl
          << "size of apwlo_basis_descriptors_ : " << apwlo_basis_descriptors_.size() << std::endl
          << "num_gkvec : " << num_gkvec() << std::endl 
          << "mt_lo_basis_size : " << parameters_.mt_lo_basis_size();
        error(__FILE__, __LINE__, s);
    }
}

/// Block-cyclic distribution of relevant arrays 
void kpoint::distribute_block_cyclic(Band* band)
{
    // distribute APW+lo basis between rows
    splindex<block_cyclic> spl_row(apwlo_basis_size(), band->num_ranks_row(), band->rank_row(), 
                                   parameters_.cyclic_block_size());
    apwlo_basis_descriptors_row_.resize(spl_row.local_size());
    for (int i = 0; i < spl_row.local_size(); i++)
        apwlo_basis_descriptors_row_[i] = apwlo_basis_descriptors_[spl_row[i]];

    // distribute APW+lo basis between columns
    splindex<block_cyclic> spl_col(apwlo_basis_size(), band->num_ranks_col(), band->rank_col(), 
                                   parameters_.cyclic_block_size());
    apwlo_basis_descriptors_col_.resize(spl_col.local_size());
    for (int i = 0; i < spl_col.local_size(); i++)
        apwlo_basis_descriptors_col_[i] = apwlo_basis_descriptors_[spl_col[i]];
    
    #if defined(_SCALAPACK) || defined(_ELPA_)
    if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa)
    {
        int nr = linalg<scalapack>::numroc(apwlo_basis_size(), parameters_.cyclic_block_size(), 
                                           band->rank_row(), 0, band->num_ranks_row());
        
        if (nr != apwlo_basis_size_row()) 
            error(__FILE__, __LINE__, "numroc returned a different local row size");

        int nc = linalg<scalapack>::numroc(apwlo_basis_size(), parameters_.cyclic_block_size(), 
                                           band->rank_col(), 0, band->num_ranks_col());
        
        if (nc != apwlo_basis_size_col()) 
            error(__FILE__, __LINE__, "numroc returned a different local column size");
    }
    #endif

    // get the number of row- and column- G+k-vectors
    num_gkvec_row_ = 0;
    for (int i = 0; i < apwlo_basis_size_row(); i++)
        if (apwlo_basis_descriptors_row_[i].igk != -1) num_gkvec_row_++;
    
    num_gkvec_col_ = 0;
    for (int i = 0; i < apwlo_basis_size_col(); i++)
        if (apwlo_basis_descriptors_col_[i].igk != -1) num_gkvec_col_++;
}

void kpoint::find_eigen_states(Band* band, PeriodicFunction<double>* effective_potential, 
                               PeriodicFunction<double>* effective_magnetic_field[3])
{
    assert(band != NULL);
    
    Timer t("sirius::kpoint::find_eigen_states");

    if (band->num_ranks() > 1 && 
        (parameters_.eigen_value_solver() == lapack || parameters_.eigen_value_solver() == magma))
    {
        error(__FILE__, __LINE__, "Can't use more than one MPI rank for LAPACK or MAGMA eigen-value solver");
    }

    generate_fv_states(band, effective_potential);
    
    // distribute fv states along rows of the MPI grid
    if (band->num_ranks() != 1)
    {
        // ===========================================================================================
        // Initially fv states are distributed along the colums of the MPI grid and aligned such that
        // each MPI column rank has exactly the same number of fv states. But this does not imply that
        // the distribution is the same for row MPI ranks, because MPI grid can be rectangular.
        // ===========================================================================================
        for (int i = 0; i < band->spl_fv_states_row().local_size(); i++)
        {
            // index of fv state in the range [0...num_fv_states)
            int ist = (band->spl_fv_states_row(i) % parameters_.num_fv_states());
            
            // find local column lindex of fv state
            int offset_col = band->spl_fv_states_col().location(_splindex_offs_, ist);
            
            // find column MPI rank which stores this fv state 
            int rank_col = band->spl_fv_states_col().location(_splindex_rank_, ist);

            // if this rank stores this fv state, then copy it
            if (rank_col == band->rank_col())
                memcpy(&fv_states_row_(0, i), &fv_states_col_(0, offset_col), mtgk_size() * sizeof(complex16));

            // send fv state to all column MPI ranks
            Platform::bcast(&fv_states_row_(0, i), mtgk_size(), 
                            parameters_.mpi_grid().communicator(1 << band->dim_col()), rank_col); 
        }
    }

    if (debug_level > 1) test_fv_states(band, 0);

    band->solve_sv(parameters_, mtgk_size(), num_gkvec(), fft_index(), &fv_eigen_values_[0], 
                   fv_states_row_, fv_states_col_, effective_magnetic_field, &band_energies_[0],
                   sv_eigen_vectors_);

    if (band->sv()) generate_spinor_wave_functions(band);

    /*for (int i = 0; i < 3; i++)
        test_spinor_wave_functions(i); */
}

//PeriodicFunction<complex16>* kpoint::spinor_wave_function_component(Band* band, int lmax, int ispn, int jloc)
//{
//    Timer t("sirius::kpoint::spinor_wave_function_component");
//
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//
//    PeriodicFunction<complex16, index_order>* func = 
//        new PeriodicFunction<complex16, index_order>(parameters_, lmax);
//    func->allocate(ylm_component | it_component);
//    func->zero();
//    
//    if (basis_type == pwlo)
//    {
//        if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//
//        double fourpi_omega = fourpi / sqrt(parameters_.omega());
//        
//        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//        {
//            int igk = igkglob(igkloc);
//            complex16 z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//
//            // TODO: possilbe optimization with zgemm
//            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//            {
//                int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//                complex16 z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//                
//                #pragma omp parallel for default(shared)
//                for (int lm = 0; lm < lmmax; lm++)
//                {
//                    int l = l_by_lm_(lm);
//                    complex16 z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                        func->f_ylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//                }
//            }
//        }
//
//        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&func->f_ylm(0, 0, ia), lmmax * parameters_.max_num_mt_points(),
//                                parameters_.mpi_grid().communicator(1 << band->dim_row()));
//        }
//    }
//
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
//        {
//            int lm = parameters_.atom(ia)->type()->indexb(i).lm;
//            int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
//            switch (index_order)
//            {
//                case angular_radial:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(lm, ir, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//                case radial_angular:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(ir, lm, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//            }
//        }
//    }
//
//    // in principle, wave function must have an overall e^{ikr} phase factor
//    parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//                            &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, jloc));
//    parameters_.fft().transform(1);
//    parameters_.fft().output(func->f_it());
//
//    for (int i = 0; i < parameters_.fft().size(); i++) func->f_it(i) /= sqrt(parameters_.omega());
//    
//    return func;
//}

//void kpoint::spinor_wave_function_component_mt(Band* band, int lmax, int ispn, int jloc, mdarray<complex16, 3>& fylm)
//{
//    Timer t("sirius::kpoint::spinor_wave_function_component_mt");
//
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//
//    fylm.zero();
//    
//    if (basis_type == pwlo)
//    {
//        if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//
//        double fourpi_omega = fourpi / sqrt(parameters_.omega());
//
//        mdarray<complex16, 2> zm(parameters_.max_num_mt_points(),  num_gkvec_row());
//
//        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//        {
//            int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//            for (int l = 0; l <= lmax; l++)
//            {
//                #pragma omp parallel for default(shared)
//                for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//                {
//                    int igk = igkglob(igkloc);
//                    complex16 z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//                    complex16 z2 = z1 * gkvec_phase_factors_(igkloc, ia) * zil_[l];
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                        zm(ir, igkloc) = z2 * (*sbessel_[igkloc])(ir, l, iat);
//                }
//                blas<cpu>::gemm(0, 2, parameters_.atom(ia)->num_mt_points(), (2 * l + 1), num_gkvec_row(),
//                                &zm(0, 0), zm.ld(), &gkvec_ylm_(Utils::lm_by_l_m(l, -l), 0), gkvec_ylm_.ld(), 
//                                &fylm(0, Utils::lm_by_l_m(l, -l), ia), fylm.ld());
//            }
//        }
//        //for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//        //{
//        //    int igk = igkglob(igkloc);
//        //    complex16 z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//
//        //    // TODO: possilbe optimization with zgemm
//        //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//        //    {
//        //        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//        //        complex16 z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//        //        
//        //        #pragma omp parallel for default(shared)
//        //        for (int lm = 0; lm < lmmax; lm++)
//        //        {
//        //            int l = l_by_lm_(lm);
//        //            complex16 z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//        //            for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//        //                fylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//        //        }
//        //    }
//        //}
//
//        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&fylm(0, 0, ia), lmmax * parameters_.max_num_mt_points(),
//                                parameters_.mpi_grid().communicator(1 << band->dim_row()));
//        }
//    }
//
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
//        {
//            int lm = parameters_.atom(ia)->type()->indexb(i).lm;
//            int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
//            switch (index_order)
//            {
//                case angular_radial:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        fylm(lm, ir, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//                case radial_angular:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        fylm(ir, lm, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//            }
//        }
//    }
//}

void kpoint::test_fv_states(Band* band, int use_fft)
{
    std::vector<complex16> v1;
    std::vector<complex16> v2;
    
    if (use_fft == 0) 
    {
        v1.resize(num_gkvec());
        v2.resize(parameters_.fft().size());
    }
    
    if (use_fft == 1) 
    {
        v1.resize(parameters_.fft().size());
        v2.resize(parameters_.fft().size());
    }
    
    double maxerr = 0;

    for (int j1 = 0; j1 < band->spl_fv_states_col().local_size(); j1++)
    {
        if (use_fft == 0)
        {
            parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                    &fv_states_col_(parameters_.mt_basis_size(), j1));
            parameters_.fft().transform(1);
            parameters_.fft().output(&v2[0]);

            for (int ir = 0; ir < parameters_.fft().size(); ir++)
                v2[ir] *= parameters_.step_function(ir);
            
            parameters_.fft().input(&v2[0]);
            parameters_.fft().transform(-1);
            parameters_.fft().output(num_gkvec(), &fft_index_[0], &v1[0]); 
        }
        
        if (use_fft == 1)
        {
            parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                    &fv_states_col_(parameters_.mt_basis_size(), j1));
            parameters_.fft().transform(1);
            parameters_.fft().output(&v1[0]);
        }
       
        for (int j2 = 0; j2 < band->spl_fv_states_row().local_size(); j2++)
        {
            complex16 zsum(0.0, 0.0);
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                int offset_wf = parameters_.atom(ia)->offset_wf();
                AtomType* type = parameters_.atom(ia)->type();
                AtomSymmetryClass* symmetry_class = parameters_.atom(ia)->symmetry_class();

                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    int ordmax = type->indexr().num_rf(l);
                    for (int io1 = 0; io1 < ordmax; io1++)
                        for (int io2 = 0; io2 < ordmax; io2++)
                            for (int m = -l; m <= l; m++)
                                zsum += conj(fv_states_col_(offset_wf + 
                                                            type->indexb_by_l_m_order(l, m, io1), j1)) *
                                             fv_states_row_(offset_wf + 
                                                            type->indexb_by_l_m_order(l, m, io2), j2) * 
                                             symmetry_class->o_radial_integral(l, io1, io2);
                }
            }
            
            if (use_fft == 0)
            {
               for (int ig = 0; ig < num_gkvec(); ig++)
                   zsum += conj(v1[ig]) * fv_states_row_(parameters_.mt_basis_size() + ig, j2);
            }
           
            if (use_fft == 1)
            {
                parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                   &fv_states_row_(parameters_.mt_basis_size(), j2));
                parameters_.fft().transform(1);
                parameters_.fft().output(&v2[0]);

                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                    zsum += conj(v1[ir]) * v2[ir] * parameters_.step_function(ir) / double(parameters_.fft().size());
            }
            
            if (use_fft == 2) 
            {
                for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
                {
                    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
                    {
                        int ig3 = parameters_.index_g12(gvec_index(ig1), gvec_index(ig2));
                        zsum += conj(fv_states_col_(parameters_.mt_basis_size() + ig1, j1)) * 
                                     fv_states_row_(parameters_.mt_basis_size() + ig2, j2) * 
                                parameters_.step_function_pw(ig3);
                    }
               }
            }

            if (band->spl_fv_states_col()[j1] == (band->spl_fv_states_row()[j2] % parameters_.num_fv_states()))
                zsum = zsum - complex16(1, 0);
           
            maxerr = std::max(maxerr, abs(zsum));
        }
    }

    Platform::allreduce(&maxerr, 1, 
                        parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));

    if (parameters_.mpi_grid().side(1 << 0)) 
        printf("k-point: %f %f %f, interstitial integration : %i, maximum error : %18.10e\n", 
               vk_[0], vk_[1], vk_[2], use_fft, maxerr);
}

//** void kpoint::test_spinor_wave_functions(int use_fft)
//** {
//**     std::vector<complex16> v1[2];
//**     std::vector<complex16> v2;
//** 
//**     if (use_fft == 0 || use_fft == 1)
//**         v2.resize(parameters_.fft().size());
//**     
//**     if (use_fft == 0) 
//**         for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             v1[ispn].resize(num_gkvec());
//**     
//**     if (use_fft == 1) 
//**         for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             v1[ispn].resize(parameters_.fft().size());
//**     
//**     double maxerr = 0;
//** 
//**     for (int j1 = 0; j1 < parameters_.num_bands(); j1++)
//**     {
//**         if (use_fft == 0)
//**         {
//**             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             {
//**                 parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//**                                    &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j1));
//**                 parameters_.fft().transform(1);
//**                 parameters_.fft().output(&v2[0]);
//** 
//**                 for (int ir = 0; ir < parameters_.fft().size(); ir++)
//**                     v2[ir] *= parameters_.step_function(ir);
//**                 
//**                 parameters_.fft().input(&v2[0]);
//**                 parameters_.fft().transform(-1);
//**                 parameters_.fft().output(num_gkvec(), &fft_index_[0], &v1[ispn][0]); 
//**             }
//**         }
//**         
//**         if (use_fft == 1)
//**         {
//**             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             {
//**                 parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//**                                    &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j1));
//**                 parameters_.fft().transform(1);
//**                 parameters_.fft().output(&v1[ispn][0]);
//**             }
//**         }
//**        
//**         for (int j2 = 0; j2 < parameters_.num_bands(); j2++)
//**         {
//**             complex16 zsum(0.0, 0.0);
//**             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             {
//**                 for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//**                 {
//**                     int offset_wf = parameters_.atom(ia)->offset_wf();
//**                     AtomType* type = parameters_.atom(ia)->type();
//**                     AtomSymmetryClass* symmetry_class = parameters_.atom(ia)->symmetry_class();
//** 
//**                     for (int l = 0; l <= parameters_.lmax_apw(); l++)
//**                     {
//**                         int ordmax = type->indexr().num_rf(l);
//**                         for (int io1 = 0; io1 < ordmax; io1++)
//**                             for (int io2 = 0; io2 < ordmax; io2++)
//**                                 for (int m = -l; m <= l; m++)
//**                                     zsum += conj(spinor_wave_functions_(offset_wf + 
//**                                                                         type->indexb_by_l_m_order(l, m, io1),
//**                                                                         ispn, j1)) *
//**                                                  spinor_wave_functions_(offset_wf + 
//**                                                                         type->indexb_by_l_m_order(l, m, io2), 
//**                                                                         ispn, j2) * 
//**                                                  symmetry_class->o_radial_integral(l, io1, io2);
//**                     }
//**                 }
//**             }
//**             
//**             if (use_fft == 0)
//**             {
//**                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**                {
//**                    for (int ig = 0; ig < num_gkvec(); ig++)
//**                        zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(parameters_.mt_basis_size() + ig, ispn, j2);
//**                }
//**             }
//**            
//**             if (use_fft == 1)
//**             {
//**                 for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**                 {
//**                     parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//**                                        &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j2));
//**                     parameters_.fft().transform(1);
//**                     parameters_.fft().output(&v2[0]);
//** 
//**                     for (int ir = 0; ir < parameters_.fft().size(); ir++)
//**                         zsum += conj(v1[ispn][ir]) * v2[ir] * parameters_.step_function(ir) / double(parameters_.fft().size());
//**                 }
//**             }
//**             
//**             if (use_fft == 2) 
//**             {
//**                 for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
//**                 {
//**                     for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
//**                     {
//**                         int ig3 = parameters_.index_g12(gvec_index(ig1), gvec_index(ig2));
//**                         for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**                             zsum += conj(spinor_wave_functions_(parameters_.mt_basis_size() + ig1, ispn, j1)) * 
//**                                          spinor_wave_functions_(parameters_.mt_basis_size() + ig2, ispn, j2) * 
//**                                     parameters_.step_function_pw(ig3);
//**                     }
//**                }
//**            }
//** 
//**            zsum = (j1 == j2) ? zsum - complex16(1.0, 0.0) : zsum;
//**            maxerr = std::max(maxerr, abs(zsum));
//**         }
//**     }
//**     std :: cout << "maximum error = " << maxerr << std::endl;
//** }

void kpoint::save_wave_functions(int id, Band* band__)
{
    if (parameters_.mpi_grid().root(1 << band__->dim_col()))
    {
        hdf5_tree fout("sirius.h5", false);

        fout["kpoints"].create_node(id);
        fout["kpoints"][id].write("coordinates", vk_, 3);
        fout["kpoints"][id].write("mtgk_size", mtgk_size());
        fout["kpoints"][id].create_node("spinor_wave_functions");
        fout["kpoints"][id].write("band_energies", &band_energies_[0], parameters_.num_bands());
        fout["kpoints"][id].write("band_occupancies", &band_occupancies_[0], parameters_.num_bands());
    }
    
    Platform::barrier(parameters_.mpi_grid().communicator(1 << band__->dim_col()));
    
    mdarray<complex16, 2> wfj(NULL, mtgk_size(), parameters_.num_spins()); 
    for (int j = 0; j < parameters_.num_bands(); j++)
    {
        int rank = band__->spl_spinor_wf_col().location(_splindex_rank_, j);
        int offs = band__->spl_spinor_wf_col().location(_splindex_offs_, j);
        if (parameters_.mpi_grid().coordinate(band__->dim_col()) == rank)
        {
            hdf5_tree fout("sirius.h5", false);
            wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
            fout["kpoints"][id]["spinor_wave_functions"].write(j, wfj);
        }
        Platform::barrier(parameters_.mpi_grid().communicator(1 << band__->dim_col()));
    }
}

void kpoint::load_wave_functions(int id, Band* band__)
{
    hdf5_tree fin("sirius.h5", false);
    
    int mtgk_size_in;
    fin["kpoints"][id].read("mtgk_size", &mtgk_size_in);
    if (mtgk_size_in != mtgk_size()) error(__FILE__, __LINE__, "wrong wave-function size");

    band_energies_.resize(parameters_.num_bands());
    fin["kpoints"][id].read("band_energies", &band_energies_[0], parameters_.num_bands());

    band_occupancies_.resize(parameters_.num_bands());
    fin["kpoints"][id].read("band_occupancies", &band_occupancies_[0], parameters_.num_bands());

    spinor_wave_functions_.set_dimensions(mtgk_size(), parameters_.num_spins(), 
                                          band__->spl_spinor_wf_col().local_size());
    spinor_wave_functions_.allocate();

    mdarray<complex16, 2> wfj(NULL, mtgk_size(), parameters_.num_spins()); 
    for (int jloc = 0; jloc < band__->spl_spinor_wf_col().local_size(); jloc++)
    {
        int j = band__->spl_spinor_wf_col(jloc);
        wfj.set_ptr(&spinor_wave_functions_(0, 0, jloc));
        fin["kpoints"][id]["spinor_wave_functions"].read(j, wfj);
    }
}

void kpoint::get_fv_eigen_vectors(mdarray<complex16, 2>& fv_evec)
{
    assert(fv_evec.size(0) >= fv_eigen_vectors_.size(0));
    assert(fv_evec.size(1) <= fv_eigen_vectors_.size(1));

    for (int i = 0; i < fv_evec.size(1); i++)
        memcpy(&fv_evec(0, i), &fv_eigen_vectors_(0, i), fv_eigen_vectors_.size(0) * sizeof(complex16));
}

void kpoint::get_sv_eigen_vectors(mdarray<complex16, 2>& sv_evec)
{
    assert(sv_evec.size(0) == parameters_.num_bands());
    assert(sv_evec.size(1) == parameters_.num_bands());
    assert(sv_eigen_vectors_.size(1) == parameters_.num_bands());

    sv_evec.zero();

    if (parameters_.num_mag_dims() == 0)
    {
        assert(sv_eigen_vectors_.size(0) == parameters_.num_fv_states());

        for (int i = 0; i < sv_evec.size(1); i++)
            memcpy(&sv_evec(0, i), &sv_eigen_vectors_(0, i), sv_eigen_vectors_.size(0) * sizeof(complex16));
    }
    if (parameters_.num_mag_dims() == 1)
    {
        assert(sv_eigen_vectors_.size(0) == parameters_.num_fv_states());

        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        {
            for (int i = 0; i < parameters_.num_fv_states(); i++)
            {
                memcpy(&sv_evec(ispn * parameters_.num_fv_states(), ispn * parameters_.num_fv_states() + i), 
                       &sv_eigen_vectors_(0, ispn * parameters_.num_fv_states() + i), 
                       sv_eigen_vectors_.size(0) * sizeof(complex16));
            }
        }
    }
    if (parameters_.num_mag_dims() == 3)
    {
        assert(sv_eigen_vectors_.size(0) == parameters_.num_bands());
        for (int i = 0; i < parameters_.num_bands(); i++)
            memcpy(&sv_evec(0, i), &sv_eigen_vectors_(0, i), sv_eigen_vectors_.size(0) * sizeof(complex16));
    }
}

};

