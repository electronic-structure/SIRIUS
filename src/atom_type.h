#ifndef __ATOM_TYPE_H__
#define __ATOM_TYPE_H__

namespace sirius {

/// describes single atomic level
struct atomic_level_descriptor
{
    /// principal quantum number
    int n;

    /// angular momentum quantum number
    int l;
    
    /// quantum number k
    int k;
    
    /// level occupancy
    int occupancy;
};

/// describes radial solution
struct radial_solution_descriptor
{
    /// principal quantum number
    int n;
    
    /// angular momentum quantum number
    int l;
    
    /// order of energy derivative
    int dme;
    
    /// energy of the solution
    double enu;
    
    /// automatically determine energy
    bool auto_enu;
};

/// set of radial solution descriptors, used to construct augmented waves or local orbitals
typedef std::vector<radial_solution_descriptor> radial_solution_descriptor_set;

class radial_functions_index
{
    public:

        struct radial_function_index_descriptor
        {
            int l;

            int order;

            int idxlo;

            radial_function_index_descriptor(int l, 
                                             int order, 
                                             int idxlo =  -1) : l(l), 
                                                                order(order), 
                                                                idxlo(idxlo)
            {
                assert(l >= 0);
                assert(order >= 0);
            }
        };

    private: 

        std::vector<radial_function_index_descriptor> radial_function_index_descriptors_;

        mdarray<int,2> index_by_l_order_;

        mdarray<int,1> index_by_idxlo_;

        /// number of radial functions for each angular momentum quantum number
        std::vector<int> num_rf_;

        // maximum number of radial functions across all angular momentums
        int max_num_rf_;
    
    public:

        void init(const int lmax,
                  const std::vector<radial_solution_descriptor_set>& aw_descriptors, 
                  const std::vector<radial_solution_descriptor_set>& lo_descriptors)
        {
            assert((int)aw_descriptors.size() == lmax + 1);

            num_rf_ = std::vector<int>(lmax + 1, 0);
            max_num_rf_ = 0;

            radial_function_index_descriptors_.clear();

            for (int l = 0; l <= lmax; l++)
            {
                assert(aw_descriptors[l].size() <= 2);

                num_rf_[l] = (int)aw_descriptors[l].size();
                for (int order = 0; order < (int)aw_descriptors[l].size(); order++)
                   radial_function_index_descriptors_.push_back(radial_function_index_descriptor(l, order));
            }

            for (int idxlo = 0; idxlo < (int)lo_descriptors.size(); idxlo++)
            {
                int l = lo_descriptors[idxlo][0].l;
                radial_function_index_descriptors_.push_back(radial_function_index_descriptor(l, num_rf_[l], idxlo));
                num_rf_[l]++;
            }

            for (int l = 0; l <= lmax; l++)
                max_num_rf_ = std::max(max_num_rf_, num_rf_[l]);

            index_by_l_order_.set_dimensions(lmax + 1, max_num_rf_);
            index_by_l_order_.allocate();

            if (lo_descriptors.size())
            {
                index_by_idxlo_.set_dimensions((int)lo_descriptors.size());
                index_by_idxlo_.allocate();
            }

            for (int i = 0; i < (int)radial_function_index_descriptors_.size(); i++)
            {
                int l = radial_function_index_descriptors_[i].l;
                int order = radial_function_index_descriptors_[i].order;
                int idxlo = radial_function_index_descriptors_[i].idxlo;
                index_by_l_order_(l, order) = i;
                if (idxlo >= 0) index_by_idxlo_(idxlo) = i; 
            }
        }

        inline int size()
        {
            return (int)radial_function_index_descriptors_.size();
        }

        inline radial_function_index_descriptor& operator[](int i)
        {
            assert(i >= 0 && i < (int)radial_function_index_descriptors_.size());

            return radial_function_index_descriptors_[i];
        }

        inline int index_by_l_order(int l, int order)
        {
            return index_by_l_order_(l, order);
        }

        inline int index_by_idxlo(int idxlo)
        {
            return index_by_idxlo_(idxlo);
        }

        /*!
            \brief Number of radial functions for a given orbital quantum number.
        */
        inline int num_rf(int l)
        {
            assert(l >= 0 && l < (int)num_rf_.size());
            
            return num_rf_[l];
        }

        /*!
            \brief Maximum number of radial functions for one orbital quantum number.
        */
        inline int max_num_rf()
        {
            return max_num_rf_;
        }
};

class basis_functions_index
{
    public: 

        struct basis_function_index_descriptor
        {
            int l;

            int m;

            int lm;

            int order;

            int idxlo;

            int idxrf;
            
            basis_function_index_descriptor(int l, 
                                            int m, 
                                            int order, 
                                            int idxlo, 
                                            int idxrf) : l(l), 
                                                         m(m), 
                                                         order(order), 
                                                         idxlo(idxlo), 
                                                         idxrf(idxrf) 
            {
                assert(l >= 0);
                assert(m >= -l && m <= l);
                assert(order >= 0);
                assert(idxrf >= 0);

                lm = Utils::lm_by_l_m(l, m);
            }
        };

    private:

        std::vector<basis_function_index_descriptor> basis_function_index_descriptors_; 
       
        mdarray<int,2> index_by_lm_order_;
       
        /// number of augmented wave basis functions
        int size_aw_;
       
        /// number of local orbital basis functions
        int size_lo_;

    public:

        basis_functions_index() : size_aw_(-1),
                                  size_lo_(-1)
        {
        }
        
        void init(const int lmax,
                 radial_functions_index& indexr)
        {
            basis_function_index_descriptors_.clear();

            for (int idxrf = 0; idxrf < indexr.size(); idxrf++)
            {
                int l = indexr[idxrf].l;
                int order = indexr[idxrf].order;
                int idxlo = indexr[idxrf].idxlo;
                for (int m = -l; m <= l; m++)
                    basis_function_index_descriptors_.push_back(basis_function_index_descriptor(l, m, order, idxlo, idxrf));
            }

            index_by_lm_order_.set_dimensions(Utils::lmmax_by_lmax(lmax), indexr.max_num_rf());
            index_by_lm_order_.allocate();

            for (int i = 0; i < (int)basis_function_index_descriptors_.size(); i++)
            {
                int lm = basis_function_index_descriptors_[i].lm;
                int order = basis_function_index_descriptors_[i].order;
                index_by_lm_order_(lm, order) = i;
               
                // get number of aw basis functions
                if (basis_function_index_descriptors_[i].idxlo < 0) size_aw_ = i + 1;
            }

            size_lo_ = (int)basis_function_index_descriptors_.size() - size_aw_;

            assert(size_aw_ > 0);
            assert(size_lo_ >= 0);
        } 

        /*! 
            \brief Return total number of MT basis functions.
        */
        inline int size()
        {
            return (int)basis_function_index_descriptors_.size();
        }

        inline int size_aw()
        {
            return size_aw_;
        }

        inline int size_lo()
        {
            return size_lo_;
        }
        
        inline int index_by_l_m_order(int l, int m, int order)
        {
            return index_by_lm_order_(Utils::lm_by_l_m(l, m), order);
        }
        
        inline int index_by_lm_order(int lm, int order)
        {
            return index_by_lm_order_(lm, order);
        }
        
        inline basis_function_index_descriptor& operator[](int i)
        {
            assert(i >= 0 && i < (int)basis_function_index_descriptors_.size());

            return basis_function_index_descriptors_[i];
        }
};

class AtomType
{
    private:
        
        /// unique id of atom type
        int id_;
    
        /// label of the input file
        std::string label_;

        /// chemical element symbol
        std::string symbol_;

        /// chemical element name
        std::string name_;
        
        /// nucleus charge
        int zn_;

        /// atom mass
        double mass_;

        /// muffin-tin radius
        double mt_radius_;

        /// number of muffin-tin points
        int num_mt_points_;
        
        /// beginning of the radial grid
        double radial_grid_origin_;
        
        /// effective infinity distance
        double radial_grid_infinity_;
        
        /// radial grid
        RadialGrid radial_grid_;

        /// list of atomic levels 
        std::vector<atomic_level_descriptor> atomic_levels_;

        /// number of core levels
        int num_core_levels_;

        /// number of core electrons
        int num_core_electrons_;

        /// number of valence electrons
        int num_valence_electrons_;
        
        /// default augmented wave configuration
        radial_solution_descriptor_set aw_default_l_;
        
        /// augmented wave configuration for specific l
        std::vector<radial_solution_descriptor_set> aw_specific_l_;

        /// list of radial descriptor sets used to construct augmented waves 
        std::vector<radial_solution_descriptor_set> aw_descriptors_;
        
        /// list of radial descriptor sets used to construct local orbitals
        std::vector<radial_solution_descriptor_set> lo_descriptors_;
        
        /// density of a free atom
        std::vector<double> free_atom_density_;
        
        /// potential of a free atom
        std::vector<double> free_atom_potential_;

        /// maximum number of aw radial functions across angular momentums
        int max_aw_order_;

        radial_functions_index indexr_;
        
        basis_functions_index indexb_;
       
        // forbid copy constructor
        AtomType(const AtomType& src);
        
        // forbid assignment operator
        AtomType& operator=(const AtomType& src);
        
        void read_input()
        {
            std::string fname = label_ + std::string(".json");
            JsonTree parser(fname);
            parser["name"] >> name_;
            parser["symbol"] >> symbol_;
            parser["mass"] >> mass_;
            parser["number"] >> zn_;
            parser["rmin"] >> radial_grid_origin_;
            parser["rmax"] >> radial_grid_infinity_;
            parser["rmt"] >> mt_radius_;
            parser["nrmt"] >> num_mt_points_;
            std::string core_str;
            parser["core"] >> core_str;
            if (int size = (int)core_str.size())
            {
                if (size % 2)
                {
                    std::string s = std::string("wrong core configuration string : ") + core_str;
                    error(__FILE__, __LINE__, s);
                }
                int j = 0;
                while (j < size)
                {
                    char c1 = core_str[j++];
                    char c2 = core_str[j++];
                    
                    int n = -1;
                    int l = -1;
                    
                    std::istringstream iss(std::string(1, c1));
                    iss >> n;
                    
                    if (n <= 0 || iss.fail())
                    {
                        std::string s = std::string("wrong principal quantum number : " ) + std::string(1, c1);
                        error(__FILE__, __LINE__, s);
                    }
                    
                    switch(c2)
                    {
                        case 's':
                            l = 0;
                            break;

                        case 'p':
                            l = 1;
                            break;

                        case 'd':
                            l = 2;
                            break;

                        case 'f':
                            l = 3;
                            break;

                        default:
                            std::string s = std::string("wrong angular momentum label : " ) + std::string(1, c2);
                            error(__FILE__, __LINE__, s);
                    }

                    atomic_level_descriptor level;
                    if (!full_relativistic_core)
                    {
                        level.n = n;
                        level.l = l;
                        level.k = -1;
                        level.occupancy = 2 * (2 * l + 1);
                        atomic_levels_.push_back(level);
                    }
                    //TODO: add nlk levels here
                }

            }
            num_core_levels_ = (int)atomic_levels_.size();
            
            radial_solution_descriptor rsd;

            // default augmented wave basis
            rsd.n = -1;
            rsd.l = -1;
            for (int order = 0; order < parser["valence"][0]["basis"].size(); order++)
            {
                parser["valence"][0]["basis"][order]["enu"] >> rsd.enu;
                parser["valence"][0]["basis"][order]["dme"] >> rsd.dme;
                parser["valence"][0]["basis"][order]["auto"] >> rsd.auto_enu;
                aw_default_l_.push_back(rsd);
            }
            
            radial_solution_descriptor_set rsd_set;
            for (int j = 1; j < parser["valence"].size(); j++)
            {
                rsd.l = parser["valence"][j]["l"].get<int>();
                rsd.n = parser["valence"][j]["n"].get<int>();
                rsd_set.clear();
                for (int order = 0; order < parser["valence"][j]["basis"].size(); order++)
                {
                    parser["valence"][j]["basis"][order]["enu"] >> rsd.enu;
                    parser["valence"][j]["basis"][order]["dme"] >> rsd.dme;
                    parser["valence"][j]["basis"][order]["auto"] >> rsd.auto_enu;
                    rsd_set.push_back(rsd);
                }
                aw_specific_l_.push_back(rsd_set);
            }

            for (int j = 0; j < parser["lo"].size(); j++)
            {
                rsd.l = parser["lo"][j]["l"].get<int>();
                rsd_set.clear();
                for (int order = 0; order < parser["lo"][j]["basis"].size(); order++)
                {
                    parser["lo"][j]["basis"][order]["n"] >> rsd.n;
                    parser["lo"][j]["basis"][order]["enu"] >> rsd.enu;
                    parser["lo"][j]["basis"][order]["dme"] >> rsd.dme;
                    parser["lo"][j]["basis"][order]["auto"] >> rsd.auto_enu;
                    rsd_set.push_back(rsd);
                }
                lo_descriptors_.push_back(rsd_set);
            }
        }
    
        void rebuild_aw_descriptors(int lmax)
        {
            aw_descriptors_.clear();
            for (int l = 0; l <= lmax; l++)
            {
                aw_descriptors_.push_back(aw_default_l_);
                for (int ord = 0; ord < (int)aw_descriptors_[l].size(); ord++)
                {
                    aw_descriptors_[l][ord].n = l + 1;
                    aw_descriptors_[l][ord].l = l;
                }
            }

            for (int i = 0; i < (int)aw_specific_l_.size(); i++)
            {
                int l = aw_specific_l_[i][0].l;
                aw_descriptors_[l] = aw_specific_l_[i];
            }
        }

    public:
        
        AtomType(const char* symbol__, 
                 const char* name__, 
                 int zn__, 
                 double mass__, 
                 std::vector<atomic_level_descriptor>& levels__) : symbol_(std::string(symbol__)),
                                                                   name_(std::string(name__)),
                                                                   zn_(zn__),
                                                                   mass_(mass__),
                                                                   mt_radius_(2.0),
                                                                   num_mt_points_(2000),
                                                                   atomic_levels_(levels__)
                                                         
        {
            radial_grid_.init(exponential_grid, num_mt_points_, 1e-6 / zn_, mt_radius_, 20.0 + 0.25 * zn_); 
        }

        AtomType(int id__, 
                 const std::string& label__) : id_(id__),
                                               label_(label__)
        {
            if (label_.size() == 0)
                error(__FILE__, __LINE__, "empty atom label");

            read_input();
           
            // add valence levels to the list of core levels
            atomic_level_descriptor level;
            
            int nl_occ[7][4];
            memset(&nl_occ[0][0], 0, 28 * sizeof(int));

            // scan the list of atomic levels and get the occupancy of (n,l) shell which is 2 * (2 * l + 1)
            for (int ist = 0; ist < 28; ist++)
            {
                int n = atomic_conf[zn_ - 1][ist][0];
                int l = atomic_conf[zn_ - 1][ist][1];

                if (n != -1)
                    nl_occ[n - 1][l] += atomic_conf[zn_ - 1][ist][3];
            }

            // scan all levels of the atom
            for (int n = 0; n < 7; n++)
                for (int l = 0; l < 4; l++)
                {
                    level.n = n + 1;
                    level.l = l;
                    level.k = -1;
                    level.occupancy = nl_occ[n][l];

                    if (level.occupancy)
                    {
                        // check if this is not a core level
                        bool found = false;
                        for (int ist = 0; ist < (int)atomic_levels_.size(); ist++)
                        {
                            if (atomic_levels_[ist].n == level.n && atomic_levels_[ist].l == level.l)
                            {
                                found = true;
                                break;
                            }
                        }
                        // add a valence level
                        if (!found)
                            atomic_levels_.push_back(level);
                    }
                }

            num_core_electrons_ = 0;
            for (int i = 0; i < num_core_levels_; i++)
                num_core_electrons_ += atomic_levels_[i].occupancy;

            num_valence_electrons_ = zn_ - num_core_electrons_;
        }
        
        void init(int lmax)
        {
            radial_grid_.init(exponential_grid, num_mt_points_, radial_grid_origin_, mt_radius_, radial_grid_infinity_); 
            
            rebuild_aw_descriptors(lmax);

            max_aw_order_ = 0;
            for (int l = 0; l <= lmax; l++)
                max_aw_order_ = std::max(max_aw_order_, (int)aw_descriptors_[l].size());
            
            if (max_aw_order_ > 2)
                error(__FILE__, __LINE__, "maximum aw order is > 2");

            indexr_.init(lmax, aw_descriptors_, lo_descriptors_);
            indexb_.init(lmax, indexr_);
            
            free_atom_density_ = std::vector<double>(radial_grid_.size());
            free_atom_potential_ = std::vector<double>(radial_grid_.size());
        }

        double solve_free_atom(double solver_tol, double energy_tol, double charge_tol, std::vector<double>& enu)
        {
            Timer t("sirius::AtomType::solve_free_atom");
            
            RadialSolver solver(false, -1.0 * zn_, radial_grid_);
            
            solver.set_tolerance(solver_tol);
            
            std::vector<double> veff(radial_grid_.size());
            std::vector<double> vnuc(radial_grid_.size());
            for (int i = 0; i < radial_grid_.size(); i++)
            {
                vnuc[i] = -1.0 * zn_ / radial_grid_[i];
                veff[i] = vnuc[i];
            }
    
            Spline<double> rho(radial_grid_.size(), radial_grid_);
    
            Spline<double> f(radial_grid_.size(), radial_grid_);
    
            std::vector<double> vh(radial_grid_.size());
            std::vector<double> vxc(radial_grid_.size());
            std::vector<double> exc(radial_grid_.size());
            std::vector<double> g1;
            std::vector<double> g2;
            std::vector<double> rho_old;
    
            enu.resize(atomic_levels_.size());
    
            double energy_tot = 0.0;
            double energy_tot_old;
            double charge_rms;
            double energy_diff;
    
            double beta = 0.9;
            
            bool converged = false;
            
            for (int ist = 0; ist < (int)atomic_levels_.size(); ist++)
                enu[ist] = -1.0 * zn_ / 2 / pow(atomic_levels_[ist].n, 2);
            
            for (int iter = 0; iter < 100; iter++)
            {
                rho_old = rho.data_points();
                
                memset(&rho[0], 0, rho.size() * sizeof(double));
                #pragma omp parallel default(shared)
                {
                    std::vector<double> rho_t(rho.size());
                    memset(&rho_t[0], 0, rho.size() * sizeof(double));
                    std::vector<double> p;
                
                    #pragma omp for
                    for (int ist = 0; ist < (int)atomic_levels_.size(); ist++)
                    {
                        solver.bound_state(atomic_levels_[ist].n, atomic_levels_[ist].l, veff, enu[ist], p);
                    
                        for (int i = 0; i < radial_grid_.size(); i++)
                            rho_t[i] += atomic_levels_[ist].occupancy * pow(y00 * p[i] / radial_grid_[i], 2);
                    }

                    #pragma omp critical
                    for (int i = 0; i < rho.size(); i++)
                        rho[i] += rho_t[i];
                } 
                
                charge_rms = 0.0;
                for (int i = 0; i < radial_grid_.size(); i++)
                    charge_rms += pow(rho[i] - rho_old[i], 2);
                charge_rms = sqrt(charge_rms / radial_grid_.size());
                
                rho.interpolate();
                
                // compute Hartree potential
                rho.integrate(g2, 2);
                double t1 = rho.integrate(g1, 1);

                for (int i = 0; i < radial_grid_.size(); i++)
                    vh[i] = fourpi * (g2[i] / radial_grid_[i] + t1 - g1[i]);
                
                // compute XC potential and energy
                libxc_interface::getxc(rho.size(), &rho[0], &vxc[0], &exc[0]);

                for (int i = 0; i < radial_grid_.size(); i++)
                    veff[i] = (1 - beta) * veff[i] + beta * (vnuc[i] + vh[i] + vxc[i]);
                
                // kinetic energy
                for (int i = 0; i < radial_grid_.size(); i++)
                    f[i] = veff[i] * rho[i];
                f.interpolate();
                
                double eval_sum = 0.0;
                for (int ist = 0; ist < (int)atomic_levels_.size(); ist++)
                    eval_sum += atomic_levels_[ist].occupancy * enu[ist];

                double energy_kin = eval_sum - fourpi * f.integrate(2);

                // xc energy
                for (int i = 0; i < radial_grid_.size(); i++)
                    f[i] = exc[i] * rho[i];
                f.interpolate();
                double energy_xc = fourpi * f.integrate(2); 
                
                // electron-nuclear energy
                for (int i = 0; i < radial_grid_.size(); i++)
                    f[i] = vnuc[i] * rho[i];
                f.interpolate();
                double energy_enuc = fourpi * f.integrate(2); 

                // Coulomb energy
                for (int i = 0; i < radial_grid_.size(); i++)
                    f[i] = vh[i] * rho[i];
                f.interpolate();
                double energy_coul = 0.5 * fourpi * f.integrate(2);
                
                energy_tot_old = energy_tot;

                energy_tot = energy_kin + energy_xc + energy_coul + energy_enuc; 
                
                energy_diff = fabs(energy_tot - energy_tot_old);
                
                if (energy_diff < energy_tol && charge_rms < charge_tol) 
                { 
                    converged = true;
                    break;
                }
                
                beta = std::max(beta * 0.95, 0.01);
            }
            
            if (!converged)
            {
                printf("energy_diff : %18.10f   charge_rms : %18.10f   beta : %18.10f\n", energy_diff, charge_rms, beta);
                std::stringstream s;
                s << "atom " << symbol_ << " is not converged" << std::endl
                  << "  energy difference : " << energy_diff << std::endl
                  << "  charge difference : " << charge_rms;
                error(__FILE__, __LINE__, s);
            }
            
            free_atom_density_ = rho.data_points();
            
            free_atom_potential_ = veff;
            
            return energy_tot;
        }

        void print_info()
        {
            printf("symbol         : %s\n", symbol_.c_str());
            printf("name           : %s\n", name_.c_str());
            printf("zn             : %i\n", zn_);
            printf("mass           : %f\n", mass_);
            printf("mt_radius      : %f\n", mt_radius_);
            printf("num_mt_points  : %i\n", num_mt_points_);
            printf("\n");
            printf("number of core electrons    : %i\n", num_core_electrons_);
            printf("number of valence electrons : %i\n", num_valence_electrons_);
            printf("\n");
            printf("atomic levels (n, l, k, occupancy)\n");
            printf("core\n");
            for (int i = 0; i < num_core_levels_; i++)
            {
                printf("%i  %i  %i  %i\n", atomic_levels_[i].n, atomic_levels_[i].l, atomic_levels_[i].k,
                                           atomic_levels_[i].occupancy);
            }
            printf("valence\n");
            for (int i = num_core_levels_; i < (int)atomic_levels_.size(); i++)
            {
                printf("%i  %i  %i  %i\n", atomic_levels_[i].n, atomic_levels_[i].l, atomic_levels_[i].k,
                                           atomic_levels_[i].occupancy);
            }
            
            printf("\n");
            printf("default augmented wave basis\n");
            printf("[");
            for (int order = 0; order < (int)aw_default_l_.size(); order++)
            {
                if (order) printf(", ");
                printf("{enu : %f, dme : %i, auto : %i}", aw_default_l_[order].enu, 
                                                          aw_default_l_[order].dme, 
                                                          aw_default_l_[order].auto_enu);
            }
            printf("]\n");

            printf("augmented wave basis for specific l\n");
            for (int j = 0; j < (int)aw_specific_l_.size(); j++)
            {
                printf("[");
                for (int order = 0; order < (int)aw_specific_l_[j].size(); order++)
                {
                    if (order) printf(", ");
                    printf("{l : %i, n : %i, enu : %f, dme : %i, auto : %i}", aw_specific_l_[j][order].l,
                                                                              aw_specific_l_[j][order].n,
                                                                              aw_specific_l_[j][order].enu,
                                                                              aw_specific_l_[j][order].dme,
                                                                              aw_specific_l_[j][order].auto_enu);
                }
                printf("]\n");
            }

            printf("augmented wave basis\n");
            for (int j = 0; j < (int)aw_descriptors_.size(); j++)
            {
                printf("[");
                for (int order = 0; order < (int)aw_descriptors_[j].size(); order++)
                { 
                    if (order) printf(", ");
                    printf("{l : %i, n : %i, enu : %f, dme : %i, auto : %i}", aw_descriptors_[j][order].l,
                                                                              aw_descriptors_[j][order].n,
                                                                              aw_descriptors_[j][order].enu,
                                                                              aw_descriptors_[j][order].dme,
                                                                              aw_descriptors_[j][order].auto_enu);
                }
                printf("]\n");
            }
            printf("maximum order of aw : %i\n", max_aw_order_);

            printf("local orbitals\n");
            for (int j = 0; j < (int)lo_descriptors_.size(); j++)
            {
                printf("[");
                for (int order = 0; order < (int)lo_descriptors_[j].size(); order++)
                {
                    if (order) printf(", ");
                    printf("{l : %i, n : %i, enu : %f, dme : %i, auto : %i}", lo_descriptors_[j][order].l,
                                                                              lo_descriptors_[j][order].n,
                                                                              lo_descriptors_[j][order].enu,
                                                                              lo_descriptors_[j][order].dme,
                                                                              lo_descriptors_[j][order].auto_enu);
                }
                printf("]\n");
            }

            printf("\n");
            printf("total number of radial functions : %i\n", indexr().size());
            printf("maximum number of radial functions per orbital quantum number: %i\n", indexr().max_num_rf());
            printf("total number of basis functions : %i\n", indexb().size());
            printf("number of aw basis functions : %i\n", indexb().size_aw());
            printf("number of lo basis functions : %i\n", indexb().size_lo());
            radial_grid().print_info();
            printf("\n");
        }
        
        const std::string& label()
        {
            return label_;
        }

        inline int id()
        {
            return id_;
        }
        
        inline int zn()
        {
            return zn_;
        }
        
        const std::string& symbol()
        { 
            return symbol_;
        }

        const std::string& name()
        { 
            return name_;
        }
        
        inline double mass()
        {
            return mass_;
        }
        
        inline double mt_radius()
        {
            return mt_radius_;
        }
        
        inline void set_mt_radius(double mt_radius__)
        {
            mt_radius_ = mt_radius__;
            radial_grid_.init(exponential_grid, num_mt_points_, radial_grid_origin_, mt_radius_, radial_grid_infinity_); 
        }
        
        inline int num_mt_points()
        {
            return num_mt_points_;
        }
        
        inline RadialGrid& radial_grid()
        {
            return radial_grid_;
        }
        
        inline double radial_grid(int ir)
        {
            return radial_grid_[ir];
        }
        
        inline int num_atomic_levels()
        {
            return (int)atomic_levels_.size();
        }    
        
        inline int num_core_levels()
        {
            return num_core_levels_;
        }    
        
        inline atomic_level_descriptor& atomic_level(int idx)
        {
            return atomic_levels_[idx];
        }
        
        inline int num_core_electrons()
        {
            return num_core_electrons_;
        }
        
        inline int num_valence_electrons()
        {
            return num_valence_electrons_;
        }
        
        inline double free_atom_density(const int idx)
        {
            return free_atom_density_[idx];
        }
        
        inline double* free_atom_density_ptr()
        {
            return &free_atom_density_[0];
        }
        
        inline double free_atom_potential(const int idx)
        {
            return free_atom_potential_[idx];
        }
        
        inline double* free_atom_potential_ptr()
        {
            return &free_atom_potential_[0];
        }
        
        inline int num_aw_descriptors()
        {
            return (int)aw_descriptors_.size();
        }

        inline radial_solution_descriptor_set& aw_descriptor(int idx)
        {
            return aw_descriptors_[idx];
        }
        
        inline int num_lo_descriptors()
        {
            return (int)lo_descriptors_.size();
        }

        inline radial_solution_descriptor_set& lo_descriptor(int idx)
        {
            return lo_descriptors_[idx];
        }

        inline int max_aw_order()
        {
            return max_aw_order_;
        }

        inline radial_functions_index& indexr()
        {
            return indexr_;
        }
        
        inline radial_functions_index::radial_function_index_descriptor& indexr(int i)
        {
            assert(i >= 0 && i < (int)indexr_.size());

            return indexr_[i];
        }

        inline basis_functions_index& indexb()
        {
            return indexb_;
        }

        inline basis_functions_index::basis_function_index_descriptor& indexb(int i)
        {
            assert(i >= 0 && i < (int)indexb_.size());
            
            return indexb_[i];
        }

        inline int indexb_by_l_m_order(int l, int m, int order)
        {
            return indexb_.index_by_l_m_order(l, m, order);
        }
        
        inline int indexb_by_lm_order(int lm, int order)
        {
            return indexb_.index_by_lm_order(lm, order);
        }

        inline int mt_aw_basis_size()
        {
            return indexb_.size_aw();
        }
        
        inline int mt_lo_basis_size()
        {
            return indexb_.size_lo();
        }

        inline int mt_basis_size()
        {
            return indexb_.size();
        }

        inline int mt_radial_basis_size()
        {
            return indexr_.size();
        }
};

};

#endif // __ATOM_TYPE_H__

