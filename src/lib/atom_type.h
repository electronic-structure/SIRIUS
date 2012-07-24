#ifndef __ATOM_TYPE_H__
#define __ATOM_TYPE_H__

namespace sirius {

/// describes single atomic level of the radial Schroedinger equation
struct atomic_level_nl
{
    /// principal quantum number
    int n;

    /// angular momentum quantum number
    int l;
    
    /// level occupancy
    int occupancy;
};

/// describes single atomic level of the radial Dirac equation
struct atomic_level_nlk 
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

/// collection of radial solutions for a given angular momentum quantum number l
class radial_l_solutions_descriptor
{
    public:

        radial_l_solutions_descriptor()
        {
        }
        
        radial_l_solutions_descriptor(int l) : l(l)
        {
        }
    
        /// angular momentum quantum number
        int l;

        /// list of radial solution descriptors
        std::vector<radial_solution_descriptor> radial_solution_descriptors;
};

class AtomType
{
    private:
    
        std::string label_;
        std::string symbol_;
        std::string name_;
        
        /// nucleus charge
        int zn_;

        /// atom mass
        double mass_;

        /// muffin-tin radius
        double mt_radius_;

        /// number of muffin-tin points
        int mt_num_points_;

        /// list of core levels
        std::vector<atomic_level_nlk> core_;
        
        /// default APW configuration
        radial_l_solutions_descriptor apw_default_l_;
        
        /// APW configuration for specific l
        std::vector<radial_l_solutions_descriptor> apw_specific_l_;

        /// list of radial descriptors used to construct apw functions 
        std::vector<radial_l_solutions_descriptor> apw_descriptors_;
        
        /// list of radial descriptors used to construct local orbitals
        std::vector<radial_l_solutions_descriptor> lo_descriptors_;
       
        /// radial grid
        RadialGrid radial_grid_;

        void read_input()
        {
            std::string fname = label_ + std::string(".json");
            JsonTree parser(fname);
            parser["name"] >> name_;
            parser["symbol"] >> symbol_;
            parser["mass"] >> mass_;
            parser["number"] >> zn_;
            double origin = parser["rmin"].get<double>();
            double infinity= parser["rmax"].get<double>();
            parser["rmt"] >> mt_radius_;
            parser["nrmt"] >> mt_num_points_;
            radial_grid_.init(exponential_grid, mt_num_points_, origin, mt_radius_, infinity); 
            std::string core_str;
            parser["core"] >> core_str;
            if (int size = core_str.size())
            {
                if (size % 2)
                    stop(std::cout << "wrong core configuration string : " << core_str);
                
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
                        stop(std::cout << "wrong principal quantum number : " << c1);
                    
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
                            stop(std::cout << "wrong angular momentum quantum number : " << c2);
                    }

                    atomic_level_nlk level;
                    for (int i = 0; i < 28; i++)
                    {
                        if (atomic_conf[zn_ - 1][i][0] == n && atomic_conf[zn_ - 1][i][1] == l)
                        {
                            level.n = n;
                            level.l = l;
                            level.k = atomic_conf[zn_ - 1][i][2];
                            level.occupancy = atomic_conf[zn_ - 1][i][3];
                            core_.push_back(level);
                        }
                    }
                }

            }
            
            radial_solution_descriptor rsd;
            radial_l_solutions_descriptor rlsd;

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
                        apw_default_l_.radial_solution_descriptors.push_back(rsd);
                    }
                }
                else
                {
                    rsd.l = l;
                    rsd.n = parser["valence"][j]["n"].get<int>();
                    rlsd.radial_solution_descriptors.clear();
                    rlsd.l = l;
                    for (int order = 0; order < parser["valence"][j]["basis"].size(); order++)
                    {
                        parser["valence"][j]["basis"][order]["enu"] >> rsd.enu;
                        parser["valence"][j]["basis"][order]["dme"] >> rsd.dme;
                        parser["valence"][j]["basis"][order]["auto"] >> rsd.auto_enu;
                        rlsd.radial_solution_descriptors.push_back(rsd);
                    }
                    apw_specific_l_.push_back(rlsd);
                }
            }

            for (int j = 0; j < parser["lo"].size(); j++)
            {
                rsd.l = parser["lo"][j]["l"].get<int>();
                rsd.n = parser["lo"][j]["n"].get<int>();
                rlsd.radial_solution_descriptors.clear();
                rlsd.l = rsd.l;
                for (int order = 0; order < parser["lo"][j]["basis"].size(); order++)
                {
                    parser["lo"][j]["basis"][order]["enu"] >> rsd.enu;
                    parser["lo"][j]["basis"][order]["dme"] >> rsd.dme;
                    parser["lo"][j]["basis"][order]["auto"] >> rsd.auto_enu;
                    rlsd.radial_solution_descriptors.push_back(rsd);
                }
                lo_descriptors_.push_back(rlsd);
            }
        }
    
    public:

        AtomType(const std::string& label) : label_(label)
        {
            read_input();
        }

        void rebuild_apw_descriptors(int lmax)
        {
            apw_descriptors_.clear();
            for (int l = 0; l <= lmax; l++)
            {
                apw_descriptors_.push_back(apw_default_l_);

            }
        }

        void print_info()
        {
            std::cout << "name          : " << name_ << std::endl;
            std::cout << "symbol        : " << symbol_ << std::endl;
            std::cout << "zn            : " << zn_ << std::endl;
            std::cout << "mass          : " << mass_ << std::endl;
            std::cout << "mt_radius     : " << mt_radius_ << std::endl;
            std::cout << "mt_num_points : " << mt_num_points_ << std::endl;
            
            std::cout << "core levels (n,l,k,occupancy) " << std::endl;
            for (int i = 0; i < (int)core_.size(); i++)
                std::cout << "  " << core_[i].n << " " << core_[i].l << " " << core_[i].k << " " << core_[i].occupancy << std::endl;
            
            std::cout << "default APW basis" << std::endl;
            for (int order = 0; order < (int)apw_default_l_.radial_solution_descriptors.size(); order++)
                std::cout << "  enu : " << apw_default_l_.radial_solution_descriptors[order].enu 
                          << "  dme : " << apw_default_l_.radial_solution_descriptors[order].dme
                          << "  auto : " << apw_default_l_.radial_solution_descriptors[order].auto_enu << std::endl;

            std::cout << "APW basis for specific l" << std::endl;
            for (int j = 0; j < apw_specific_l_.size(); j++)
            {
                std::cout << "l : " << apw_specific_l_[j].l << std::endl;
                for (int order = 0; order < (int)apw_specific_l_[j].radial_solution_descriptors.size(); order++)
                    std::cout << "  n : " << apw_specific_l_[j].radial_solution_descriptors[order].n
                              << "  enu : " << apw_specific_l_[j].radial_solution_descriptors[order].enu 
                              << "  dme : " << apw_specific_l_[j].radial_solution_descriptors[order].dme
                              << "  auto : " << apw_specific_l_[j].radial_solution_descriptors[order].auto_enu << std::endl;
            }

            std::cout << "local orbitals" << std::endl;
            for (int j = 0; j < lo_descriptors_.size(); j++)
            {
                std::cout << "l : " << lo_descriptors_[j].l << std::endl;
                for (int order = 0; order < (int)lo_descriptors_[j].radial_solution_descriptors.size(); order++)
                    std::cout << "  n : " << lo_descriptors_[j].radial_solution_descriptors[order].n
                              << "  enu : " << lo_descriptors_[j].radial_solution_descriptors[order].enu
                              << "  dme : " << lo_descriptors_[j].radial_solution_descriptors[order].dme
                              << "  auto : " << lo_descriptors_[j].radial_solution_descriptors[order].auto_enu << std::endl;
            }

            radial_grid_.print_info();
        }

};
    
};

#endif // __ATOM_TYPE_H__

