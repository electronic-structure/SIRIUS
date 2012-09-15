#ifndef __ATOM_TYPE_H__
#define __ATOM_TYPE_H__

namespace sirius {

/// describes single atomic level
struct atomic_level
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

            radial_function_index_descriptor(int l, int order, int idxlo =  -1) : l(l), order(order), idxlo(idxlo)
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

                num_rf_[l] = aw_descriptors[l].size();
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

            index_by_idxlo_.set_dimensions(lo_descriptors.size());
            index_by_idxlo_.allocate();

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
            return radial_function_index_descriptors_.size();
        }

        inline radial_function_index_descriptor& operator[](int i)
        {
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

        inline int num_rf(int l)
        {
            assert(l >= 0 && l < (int)num_rf_.size());
            return num_rf_[l];
        }

        inline int max_num_rf()
        {
            return max_num_rf_;
        }
};

class basis_functions_index
{
    private:

        class basis_function_index_descriptor
        {
            private:

                int l_;

                int m_;

                int lm_;

                int order_;

                int idxlo_;

                int idxrf_;
            
            public:

                basis_function_index_descriptor(int l__, int m__, int order__, int idxlo__, int idxrf__) : l_(l__), m_(m__), order_(order__), 
                                                                                                           idxlo_(idxlo__), idxrf_(idxrf__) 
                {
                    lm_ = lm_by_l_m(l_, m_);
                }
        };


    public:

       std::vector<basis_function_index_descriptor> basis_function_index_descriptors_; 

       void init(radial_functions_index& indexr)
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
       }

       inline int size()
       {
           return basis_function_index_descriptors_.size();
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
        std::vector<atomic_level> levels_nl_;

        /// number of core levels
        int num_core_levels_nl_;

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
            if (int size = core_str.size())
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
                    
                    int n;
                    int l;
                    
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

                    atomic_level level;
                    level.n = n;
                    level.l = l;
                    level.k = -1;
                    level.occupancy = 2 * (2 * l + 1);
                    levels_nl_.push_back(level);
                }

            }
            num_core_levels_nl_ = levels_nl_.size();
            
            radial_solution_descriptor rsd;
            radial_solution_descriptor_set rsd_set;

            for (int j = 0; j < parser["valence"].size(); j++)
            {
                int l = parser["valence"][j]["l"].get<int>();
                if (l == -1)
                {
                    rsd.n = -1;
                    rsd.l = -1;
                    for (int order = 0; order < parser["valence"][j]["basis"].size(); order++)
                    {
                        parser["valence"][j]["basis"][order]["enu"] >> rsd.enu;
                        parser["valence"][j]["basis"][order]["dme"] >> rsd.dme;
                        parser["valence"][j]["basis"][order]["auto"] >> rsd.auto_enu;
                        aw_default_l_.push_back(rsd);
                    }
                }
                else
                {
                    rsd.l = l;
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
            }

            for (int j = 0; j < parser["lo"].size(); j++)
            {
                rsd.l = parser["lo"][j]["l"].get<int>();
                rsd.n = parser["lo"][j]["n"].get<int>();
                rsd_set.clear();
                for (int order = 0; order < parser["lo"][j]["basis"].size(); order++)
                {
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
        
        AtomType(const char* _symbol, 
                 const char* _name, 
                 int _zn, 
                 double _mass, 
                 std::vector<atomic_level>& _levels) : symbol_(std::string(_symbol)),
                                                       name_(std::string(_name)),
                                                       zn_(_zn),
                                                       mass_(_mass),
                                                       mt_radius_(2.0),
                                                       num_mt_points_(2000),
                                                       levels_nl_(_levels)
                                                         
        {
            radial_grid_.init(exponential_grid, num_mt_points_, 1e-6 / zn_, mt_radius_, 20.0 + 0.25 * zn_); 
        }

        AtomType(int id_, 
                 const std::string& label) : id_(id_),
                                             label_(label)
        {
            if (label.size() != 0)
            {
                read_input();
                
                atomic_level level;
                
                int nl_occ[7][4];
                memset(&nl_occ[0][0], 0, 28 * sizeof(int));

                for (int ist = 0; ist < 28; ist++)
                {
                    int n = atomic_conf[zn_][ist][0];
                    int l = atomic_conf[zn_][ist][1];

                    if (n != -1)
                        nl_occ[n - 1][l] += atomic_conf[zn_ - 1][ist][3];
                }

                for (int n = 0; n < 7; n++)
                    for (int l = 0; l < 4; l++)
                    {
                        level.n = n + 1;
                        level.l = l;
                        if ((level.occupancy = nl_occ[n][l]))
                        {
                            bool found = false;
                            for (int ist = 0; ist < (int)levels_nl_.size(); ist++)
                            {
                                if (levels_nl_[ist].n == level.n && levels_nl_[ist].l == level.l)
                                {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                                levels_nl_.push_back(level);
                        }
                    }

                    num_core_electrons_ = 0;
                    for (int i = 0; i < num_core_levels_nl_; i++)
                        num_core_electrons_ += levels_nl_[i].occupancy;

                    num_valence_electrons_ = zn_ - num_core_electrons_;
            }
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
        
        inline void set_mt_radius(double _mt_radius)
        {
            mt_radius_ = _mt_radius;
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
        
        inline int num_levels_nl()
        {
            return levels_nl_.size();
        }    
        
        inline atomic_level& level_nl(int idx)
        {
            return levels_nl_[idx];
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
        
        inline double free_atom_potential(const int idx)
        {
            return free_atom_potential_[idx];
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
            indexb_.init(indexr_);
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
    
            enu.resize(levels_nl_.size());
    
            double energy_tot = 0.0;
            double energy_tot_old;
            double charge_rms;
            double energy_diff;
    
            double beta = 0.9;
            
            bool converged = false;
            
            for (int ist = 0; ist < (int)levels_nl_.size(); ist++)
                enu[ist] = -1.0 * zn_ / 2 / pow(levels_nl_[ist].n, 2);
            
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
                    for (int ist = 0; ist < (int)levels_nl_.size(); ist++)
                    {
                        solver.bound_state(levels_nl_[ist].n, levels_nl_[ist].l, veff, enu[ist], p);
                    
                        for (int i = 0; i < radial_grid_.size(); i++)
                            rho_t[i] += levels_nl_[ist].occupancy * pow(y00 * p[i] / radial_grid_[i], 2);
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
                xc_potential::get(rho.size(), &rho[0], &vxc[0], &exc[0]);

                for (int i = 0; i < radial_grid_.size(); i++)
                    veff[i] = (1 - beta) * veff[i] + beta * (vnuc[i] + vh[i] + vxc[i]);
                
                // kinetic energy
                for (int i = 0; i < radial_grid_.size(); i++)
                    f[i] = veff[i] * rho[i];
                f.interpolate();
                
                double eval_sum = 0.0;
                for (int ist = 0; ist < (int)levels_nl_.size(); ist++)
                    eval_sum += levels_nl_[ist].occupancy * enu[ist];

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
            printf("\n");
            std::cout << "name          : " << name_ << std::endl;
            std::cout << "symbol        : " << symbol_ << std::endl;
            std::cout << "zn            : " << zn_ << std::endl;
            std::cout << "mass          : " << mass_ << std::endl;
            std::cout << "mt_radius     : " << mt_radius_ << std::endl;
            std::cout << "num_mt_points : " << num_mt_points_ << std::endl;
            
            printf("number of core electrons : %i\n", num_core_electrons_);
            printf("number of valence electrons : %i\n", num_valence_electrons_);
            
            std::cout << "core levels (n,l,occupancy) " << std::endl;
            for (int i = 0; i < num_core_levels_nl_; i++)
                std::cout << "  " << levels_nl_[i].n << " " << levels_nl_[i].l << " " << levels_nl_[i].occupancy << std::endl;
            
            std::cout << "default augmented wave basis" << std::endl;
            for (int order = 0; order < (int)aw_default_l_.size(); order++)
                std::cout << "  enu : " << aw_default_l_[order].enu 
                          << "  dme : " << aw_default_l_[order].dme
                          << "  auto : " << aw_default_l_[order].auto_enu << std::endl;

            std::cout << "augmented wave basis for specific l" << std::endl;
            for (int j = 0; j < (int)aw_specific_l_.size(); j++)
            {
                for (int order = 0; order < (int)aw_specific_l_[j].size(); order++)
                    std::cout << "  n : " << aw_specific_l_[j][order].n
                              << "  l : " << aw_specific_l_[j][order].l
                              << "  enu : " << aw_specific_l_[j][order].enu 
                              << "  dme : " << aw_specific_l_[j][order].dme
                              << "  auto : " << aw_specific_l_[j][order].auto_enu << std::endl;
            }

            printf("augmented wave basis\n");
            for (int j = 0; j < (int)aw_descriptors_.size(); j++)
            {
                printf("n : %i   l : %i   ", aw_descriptors_[j][0].n, aw_descriptors_[j][0].l);
                for (int order = 0; order < (int)aw_descriptors_[j].size(); order++)
                    printf("{enu %f dme %i auto %i} ", aw_descriptors_[j][order].enu, 
                                                       aw_descriptors_[j][order].dme, 
                                                       aw_descriptors_[j][order].auto_enu);
                printf("\n");
            }
            printf("maximum order of aw : %i\n", max_aw_order_);

            std::cout << "local orbitals" << std::endl;
            for (int j = 0; j < (int)lo_descriptors_.size(); j++)
            {
                for (int order = 0; order < (int)lo_descriptors_[j].size(); order++)
                    std::cout << "  n : " << lo_descriptors_[j][order].n
                              << "  l : " << lo_descriptors_[j][order].l
                              << "  enu : " << lo_descriptors_[j][order].enu
                              << "  dme : " << lo_descriptors_[j][order].dme
                              << "  auto : " << lo_descriptors_[j][order].auto_enu << std::endl;
            }

            printf("total number of radial functions : %i\n", indexr_.size());
            printf("maximum number of radial functions : %i\n", indexr_.max_num_rf());
            printf("total number of basis functions : %i\n", indexb_.size());
            radial_grid_.print_info();
        }

        inline int num_aw_descriptors()
        {
            return aw_descriptors_.size();
        }

        inline radial_solution_descriptor_set& aw_descriptor(int idx)
        {
            return aw_descriptors_[idx];
        }
        
        inline int num_lo_descriptors()
        {
            return lo_descriptors_.size();
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
            return indexr_[i];
        }

        inline basis_functions_index& indexb()
        {
            return indexb_;
        }
};

};

#endif // __ATOM_TYPE_H__

