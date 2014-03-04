#include "atom_type.h"

namespace sirius {

Atom_type::Atom_type(int id__) 
    : id_(id__), 
      zn_(0), 
      num_mt_points_(0), 
      radial_grid_(NULL), 
      esm_type_(full_potential_lapwlo), 
      initialized_(false)
{
}

Atom_type::Atom_type(const char* symbol__, const char* name__, int zn__, double mass__, 
                     std::vector<atomic_level_descriptor>& levels__) 
    : symbol_(std::string(symbol__)), 
      name_(std::string(name__)), 
      zn_(zn__), 
      mass_(mass__), 
      mt_radius_(2.0), 
      num_mt_points_(2000 + zn__ * 50), 
      atomic_levels_(levels__), 
      esm_type_(full_potential_lapwlo), 
      initialized_(false)
{
    radial_grid_ = new Radial_grid(default_radial_grid_t, num_mt_points_, 1e-6 / zn_, mt_radius_, 20.0 + 0.25 * zn_); 
}

Atom_type::Atom_type(int id__, const std::string label__, const std::string file_name__, electronic_structure_method_t esm_type__) 
    : id_(id__), 
      zn_(0), 
      mass_(0), 
      num_mt_points_(0), 
      radial_grid_(NULL), 
      esm_type_(esm_type__), 
      label_(label__),
      file_name_(file_name__),
      initialized_(false)
{
}

Atom_type::~Atom_type()
{
    delete radial_grid_;
}

void Atom_type::init(int lmax)
{
    if (initialized_) error_local(__FILE__, __LINE__, "can't initialize twice");
    
    if (file_name_.length() > 0)
    {
        if (!Utils::file_exists(file_name_))
        {
            std::stringstream s;
            s << "file " + file_name_ + " doesn't exist";
            error_global(__FILE__, __LINE__, s);
        }
        else
        {
            read_input(file_name_);
        }
    }

    //==============================================
    // add valence levels to the list of core levels
    //==============================================
    atomic_level_descriptor level;

    for (int ist = 0; ist < 28; ist++)
    {
        bool found = false;
        level.n = atomic_conf[zn_ - 1][ist][0];
        level.l = atomic_conf[zn_ - 1][ist][1];
        level.k = atomic_conf[zn_ - 1][ist][2];
        level.occupancy = double(atomic_conf[zn_ - 1][ist][3]);
        level.core = false;

        if (level.n != -1)
        {
            for (int jst = 0; jst < (int)atomic_levels_.size(); jst++)
            {
                if ((atomic_levels_[jst].n == level.n) &&
                    (atomic_levels_[jst].l == level.l) &&
                    (atomic_levels_[jst].k == level.k)) found = true;
            }
            if (!found) atomic_levels_.push_back(level);
        }
    }

    if (zn_ == 0) error_local(__FILE__, __LINE__, "zero atom charge");

    // set default radial grid if it was not done by user
    if (radial_grid_ == NULL) set_radial_grid();
    
    if (esm_type_ == full_potential_lapwlo)
    {
        // initialize aw descriptors if they were not set manually
        if (aw_descriptors_.size() == 0) init_aw_descriptors(lmax);

        if ((int)aw_descriptors_.size() != (lmax + 1)) error_local(__FILE__, __LINE__, "wrong size of augmented wave descriptors");

        max_aw_order_ = 0;
        for (int l = 0; l <= lmax; l++) max_aw_order_ = std::max(max_aw_order_, (int)aw_descriptors_[l].size());

        if (max_aw_order_ > 3) error_local(__FILE__, __LINE__, "maximum aw order > 3");
    }

    indexr_.init(aw_descriptors_, lo_descriptors_);
    indexb_.init(indexr_);
    
    // allocate Q matrix
    if (esm_type_ == ultrasoft_pseudopotential)
    {
        uspp_.q_mtrx.set_dimensions(mt_basis_size(), mt_basis_size());
        uspp_.q_mtrx.allocate();
    }
   
    // get the number of core electrons
    num_core_electrons_ = 0;
    if (esm_type_ == full_potential_lapwlo || esm_type_ == full_potential_pwlo)
    {
        for (int i = 0; i < (int)atomic_levels_.size(); i++) 
        {
            if (atomic_levels_[i].core) num_core_electrons_ += atomic_levels_[i].occupancy;
        }
    }

    // get number of valence electrons
    num_valence_electrons_ = zn_ - num_core_electrons_;
    
    initialized_ = true;
}

void Atom_type::set_radial_grid(int num_points, double* points)
{
    if (num_mt_points_ == 0) error_local(__FILE__, __LINE__, "number of muffin-tin points is zero");
    if (radial_grid_) delete radial_grid_;
    if (num_points < 0 && points == NULL)
    {
        radial_grid_ = new Radial_grid(default_radial_grid_t, num_mt_points_, radial_grid_origin_, mt_radius_, radial_grid_infinity_); 
    }
    else
    {
        radial_grid_ = new Radial_grid(num_points, num_mt_points_, mt_radius_, points);
    }
    free_atom_density_.resize(radial_grid_->size());
    free_atom_potential_.resize(radial_grid_->size());
}

void Atom_type::init_aw_descriptors(int lmax)
{
    assert(lmax >= -1);

    if (lmax >= 0 && aw_default_l_.size() == 0) error_local(__FILE__, __LINE__, "default AW descriptor is empty"); 

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
        if (l < lmax) aw_descriptors_[l] = aw_specific_l_[i];
    }
}

void Atom_type::add_aw_descriptor(int n, int l, double enu, int dme, int auto_enu)
{
    if ((int)aw_descriptors_.size() < (l + 1)) aw_descriptors_.resize(l + 1, radial_solution_descriptor_set());
    
    radial_solution_descriptor rsd;
    
    rsd.n = n;
    if (n == -1)
    {
        // default pqn value for any l
        rsd.n = l + 1;
        for (int ist = 0; ist < num_atomic_levels(); ist++)
        {
            // take next level after the core
            if (atomic_level(ist).core && atomic_level(ist).l == l) rsd.n = atomic_level(ist).n + 1;
        }
    }
    
    rsd.l = l;
    rsd.dme = dme;
    rsd.enu = enu;
    rsd.auto_enu = auto_enu;
    aw_descriptors_[l].push_back(rsd);
}

void Atom_type::add_lo_descriptor(int ilo, int n, int l, double enu, int dme, int auto_enu)
{
    if ((int)lo_descriptors_.size() == ilo) 
    {
        lo_descriptors_.push_back(local_orbital_descriptor());
        lo_descriptors_[ilo].type = lo_rs;
        lo_descriptors_[ilo].l = l;
    }
    else
    {
        if (l != lo_descriptors_[ilo].l) error_local(__FILE__, __LINE__, "wrong angular quantum number");
    }
    
    radial_solution_descriptor rsd;
    
    rsd.n = n;
    if (n == -1)
    {
        // default value for any l
        rsd.n = l + 1;
        for (int ist = 0; ist < num_atomic_levels(); ist++)
        {
            if (atomic_level(ist).core && atomic_level(ist).l == l)
            {   
                // take next level after the core
                rsd.n = atomic_level(ist).n + 1;
            }
        }
    }
    
    rsd.l = l;
    rsd.dme = dme;
    rsd.enu = enu;
    rsd.auto_enu = auto_enu;
    lo_descriptors_[ilo].rsd_set.push_back(rsd);
}

double Atom_type::solve_free_atom(double solver_tol, double energy_tol, double charge_tol, std::vector<double>& enu)
{
    Timer t("sirius::Atom_type::solve_free_atom");
    
    free_atom_radial_functions_.set_dimensions(radial_grid_->size(), (int)atomic_levels_.size());
    free_atom_radial_functions_.allocate();

    Radial_solver solver(false, -1.0 * zn_, *radial_grid_);
    libxc_interface xci;

    solver.set_tolerance(solver_tol);
    
    std::vector<double> veff(radial_grid_->size());
    std::vector<double> vnuc(radial_grid_->size());
    for (int i = 0; i < radial_grid_->size(); i++)
    {
        vnuc[i] = -1.0 * zn_ / radial_grid(i);
        veff[i] = vnuc[i];
    }

    Spline<double> rho(radial_grid_->size(), *radial_grid_);

    Spline<double> f(radial_grid_->size(), *radial_grid_);

    std::vector<double> vh(radial_grid_->size());
    std::vector<double> vxc(radial_grid_->size());
    std::vector<double> exc(radial_grid_->size());
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
        enu[ist] = -1.0 * zn_ / 2 / pow(double(atomic_levels_[ist].n), 2);
    
    for (int iter = 0; iter < 200; iter++)
    {
        rho_old = rho.data_points();
        
        memset(&rho[0], 0, rho.num_points() * sizeof(double));
        #pragma omp parallel default(shared)
        {
            std::vector<double> p(rho.num_points());
            std::vector<double> rho_t(rho.num_points());
            memset(&rho_t[0], 0, rho.num_points() * sizeof(double));
        
            #pragma omp for
            for (int ist = 0; ist < (int)atomic_levels_.size(); ist++)
            {
                solver.bound_state(atomic_levels_[ist].n, atomic_levels_[ist].l, veff, enu[ist], p);
            
                for (int i = 0; i < radial_grid_->size(); i++)
                {
                    free_atom_radial_functions_(i, ist) = p[i] * radial_grid_->rinv(i);
                    rho_t[i] += atomic_levels_[ist].occupancy * 
                                pow(y00 * free_atom_radial_functions_(i, ist), 2);
                }
            }

            #pragma omp critical
            for (int i = 0; i < rho.num_points(); i++) rho[i] += rho_t[i];
        } 
        
        charge_rms = 0.0;
        for (int i = 0; i < radial_grid_->size(); i++) charge_rms += pow(rho[i] - rho_old[i], 2);
        charge_rms = sqrt(charge_rms / radial_grid_->size());
        
        rho.interpolate();

        // compute Hartree potential
        rho.integrate(g2, 2);
        double t1 = rho.integrate(g1, 1);

        for (int i = 0; i < radial_grid_->size(); i++) vh[i] = fourpi * (g2[i] / radial_grid(i) + t1 - g1[i]);
        
        // compute XC potential and energy
        xci.getxc(rho.num_points(), &rho[0], &vxc[0], &exc[0]);
        
        for (int i = 0; i < radial_grid_->size(); i++)
            veff[i] = (1 - beta) * veff[i] + beta * (vnuc[i] + vh[i] + vxc[i]);
        
        // kinetic energy
        for (int i = 0; i < radial_grid_->size(); i++) f[i] = veff[i] * rho[i];
        f.interpolate();
        
        double eval_sum = 0.0;
        for (int ist = 0; ist < (int)atomic_levels_.size(); ist++)
            eval_sum += atomic_levels_[ist].occupancy * enu[ist];

        double energy_kin = eval_sum - fourpi * f.integrate(2);

        // xc energy
        for (int i = 0; i < radial_grid_->size(); i++) f[i] = exc[i] * rho[i];
        f.interpolate();
        double energy_xc = fourpi * f.integrate(2); 
        
        // electron-nuclear energy
        for (int i = 0; i < radial_grid_->size(); i++) f[i] = vnuc[i] * rho[i];
        f.interpolate();
        double energy_enuc = fourpi * f.integrate(2); 

        // Coulomb energy
        for (int i = 0; i < radial_grid_->size(); i++) f[i] = vh[i] * rho[i];
        f.interpolate();
        double energy_coul = 0.5 * fourpi * f.integrate(2);
        
        energy_tot_old = energy_tot;

        energy_tot = energy_kin + energy_xc + energy_coul + energy_enuc; 
        
        energy_diff = fabs(energy_tot - energy_tot_old);
        
        if (energy_diff < energy_tol && charge_rms < charge_tol) 
        { 
            converged = true;
            printf("Free atom (id = %i) converged in %i iterations.\n", id(), iter);
            break;
        }
        
        beta = std::max(beta * 0.95, 0.005);
    }

    if (!converged)
    {
        printf("energy_diff : %18.10f   charge_rms : %18.10f   beta : %18.10f\n", energy_diff, charge_rms, beta);
        std::stringstream s;
        s << "atom " << symbol_ << " is not converged" << std::endl
          << "  energy difference : " << energy_diff << std::endl
          << "  charge difference : " << charge_rms;
        error_local(__FILE__, __LINE__, s);
    }
    
    free_atom_density_ = rho.data_points();
    
    free_atom_potential_ = veff;
    
    return energy_tot;
}

void Atom_type::print_info()
{
    printf("\n");
    printf("symbol         : %s\n", symbol_.c_str());
    printf("name           : %s\n", name_.c_str());
    printf("zn             : %i\n", zn_);
    printf("mass           : %f\n", mass_);
    printf("mt_radius      : %f\n", mt_radius_);
    printf("num_mt_points  : %i\n", num_mt_points_);
    printf("\n");
    printf("number of core electrons    : %f\n", num_core_electrons_);
    printf("number of valence electrons : %f\n", num_valence_electrons_);

    if (esm_type_ == full_potential_lapwlo || esm_type_ == full_potential_pwlo)
    {
        printf("\n");
        printf("atomic levels (n, l, k, occupancy, core)\n");
        for (int i = 0; i < (int)atomic_levels_.size(); i++)
        {
            printf("%i  %i  %i  %8.4f %i\n", atomic_levels_[i].n, atomic_levels_[i].l, atomic_levels_[i].k,
                                              atomic_levels_[i].occupancy, atomic_levels_[i].core);
        }
        printf("\n");
        printf("local orbitals\n");
        for (int j = 0; j < (int)lo_descriptors_.size(); j++)
        {
            switch (lo_descriptors_[j].type)
            {
                case lo_rs:
                {
                    printf("radial solutions   [");
                    for (int order = 0; order < (int)lo_descriptors_[j].rsd_set.size(); order++)
                    {
                        if (order) printf(", ");
                        printf("{l : %2i, n : %2i, enu : %f, dme : %i, auto : %i}", lo_descriptors_[j].rsd_set[order].l,
                                                                                    lo_descriptors_[j].rsd_set[order].n,
                                                                                    lo_descriptors_[j].rsd_set[order].enu,
                                                                                    lo_descriptors_[j].rsd_set[order].dme,
                                                                                    lo_descriptors_[j].rsd_set[order].auto_enu);
                    }
                    printf("]\n");
                    break;
                }
                case lo_cp:
                {
                    printf("confined polynomial {l : %2i, p1 : %i, p2 : %i}\n", lo_descriptors_[j].l, 
                                                                                lo_descriptors_[j].p1, 
                                                                                lo_descriptors_[j].p2);
                    break;
                }
            }
        }
    }

    if (esm_type_ == full_potential_lapwlo)
    {
        printf("\n");
        printf("augmented wave basis\n");
        for (int j = 0; j < (int)aw_descriptors_.size(); j++)
        {
            printf("[");
            for (int order = 0; order < (int)aw_descriptors_[j].size(); order++)
            { 
                if (order) printf(", ");
                printf("{l : %2i, n : %2i, enu : %f, dme : %i, auto : %i}", aw_descriptors_[j][order].l,
                                                                            aw_descriptors_[j][order].n,
                                                                            aw_descriptors_[j][order].enu,
                                                                            aw_descriptors_[j][order].dme,
                                                                            aw_descriptors_[j][order].auto_enu);
            }
            printf("]\n");
        }
        printf("maximum order of aw : %i\n", max_aw_order_);
    }

    printf("\n");
    printf("total number of radial functions : %i\n", indexr().size());
    printf("maximum number of radial functions per orbital quantum number: %i\n", indexr().max_num_rf());
    printf("total number of basis functions : %i\n", indexb().size());
    printf("number of aw basis functions : %i\n", indexb().size_aw());
    printf("number of lo basis functions : %i\n", indexb().size_lo());
    radial_grid().print_info();
}
        
void Atom_type::read_input_core(JSON_tree& parser)
{
    std::string core_str;
    parser["core"] >> core_str;
    if (int size = (int)core_str.size())
    {
        if (size % 2)
        {
            std::string s = std::string("wrong core configuration string : ") + core_str;
            error_local(__FILE__, __LINE__, s);
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
                error_local(__FILE__, __LINE__, s);
            }
            
            switch (c2)
            {
                case 's':
                {
                    l = 0;
                    break;
                }
                case 'p':
                {
                    l = 1;
                    break;
                }
                case 'd':
                {
                    l = 2;
                    break;
                }
                case 'f':
                {
                    l = 3;
                    break;
                }
                default:
                {
                    std::string s = std::string("wrong angular momentum label : " ) + std::string(1, c2);
                    error_local(__FILE__, __LINE__, s);
                }
            }

            atomic_level_descriptor level;
            level.n = n;
            level.l = l;
            level.core = true;
            for (int ist = 0; ist < 28; ist++)
            {
                if ((level.n == atomic_conf[zn_ - 1][ist][0]) && (level.l == atomic_conf[zn_ - 1][ist][1]))
                {
                    level.k = atomic_conf[zn_ - 1][ist][2]; 
                    level.occupancy = double(atomic_conf[zn_ - 1][ist][3]);
                    atomic_levels_.push_back(level);
                }
            }
        }
    }
}

void Atom_type::read_input_aw(JSON_tree& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;
    
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
    
    for (int j = 1; j < parser["valence"].size(); j++)
    {
        parser["valence"][j]["l"] >> rsd.l;
        parser["valence"][j]["n"] >> rsd.n;
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
    
void Atom_type::read_input_lo(JSON_tree& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;
    
    int l;
    for (int j = 0; j < parser["lo"].size(); j++)
    {
        parser["lo"][j]["l"] >> l;

        if (parser["lo"][j].exist("basis"))
        {
            local_orbital_descriptor lod;
            lod.type = lo_rs;
            lod.l = l;
            rsd.l = l;
            rsd_set.clear();
            for (int order = 0; order < parser["lo"][j]["basis"].size(); order++)
            {
                parser["lo"][j]["basis"][order]["n"] >> rsd.n;
                parser["lo"][j]["basis"][order]["enu"] >> rsd.enu;
                parser["lo"][j]["basis"][order]["dme"] >> rsd.dme;
                parser["lo"][j]["basis"][order]["auto"] >> rsd.auto_enu;
                rsd_set.push_back(rsd);
            }
            lod.rsd_set = rsd_set;
            lo_descriptors_.push_back(lod);
        }
        if (parser["lo"][j].exist("polynom"))
        {
            local_orbital_descriptor lod;
            lod.type = lo_cp;
            lod.l = l;

            std::vector<int> p1;
            std::vector<int> p2;
            
            parser["lo"][j]["polynom"]["p1"] >> p1;
            if (parser["lo"][j]["polynom"].exist("p2")) 
            {
                parser["lo"][j]["polynom"]["p2"] >> p2;
            }
            else
            {
                p2.push_back(2);
            }

            for (int i = 0; i < (int)p2.size(); i++)
            {
                for (int j = 0; j < (int)p1.size(); j++)
                {
                    lod.p1 = p1[j];
                    lod.p2 = p2[i];
                    lo_descriptors_.push_back(lod);
                }
            }
        }

    }
}
    
void Atom_type::read_input(const std::string& fname)
{
    JSON_tree parser(fname);

    if (esm_type_ == ultrasoft_pseudopotential)
    {
        parser["uspp"]["header"]["element"] >> symbol_;

        double zp;
        parser["uspp"]["header"]["zp"] >> zp;
        zn_ = int(zp + 1e-10);

        int nmesh;
        parser["uspp"]["header"]["nmesh"] >> nmesh;

        parser["uspp"]["radial_grid"] >> uspp_.r;

        parser["uspp"]["vloc"] >> uspp_.vloc;

        uspp_.core_charge_density = parser["uspp"]["core_charge_density"].get(std::vector<double>(nmesh, 0));

        parser["uspp"]["total_charge_density"] >> uspp_.total_charge_density;

        if ((int)uspp_.r.size() != nmesh)
        {
            error_local(__FILE__, __LINE__, "wrong mesh size");
        }
        if ((int)uspp_.vloc.size() != nmesh || 
            (int)uspp_.core_charge_density.size() != nmesh || 
            (int)uspp_.total_charge_density.size() != nmesh)
        {
            std::cout << uspp_.vloc.size()  << " " << uspp_.core_charge_density.size() << " " << uspp_.total_charge_density.size() << std::endl;
            error_local(__FILE__, __LINE__, "wrong array size");
        }

        num_mt_points_ = nmesh;
        mt_radius_ = uspp_.r[nmesh - 1];
        
        set_radial_grid(nmesh, &uspp_.r[0]);

        parser["uspp"]["header"]["lmax"] >> uspp_.lmax;
        parser["uspp"]["header"]["nbeta"] >> uspp_.num_beta_radial_functions;

        parser["uspp"]["non_local"]["Q"]["num_q_coefs"] >> uspp_.num_q_coefs;

        parser["uspp"]["non_local"]["Q"]["q_functions_inner_radii"] >> uspp_.q_functions_inner_radii;

        uspp_.q_coefs.set_dimensions(uspp_.num_q_coefs, 2 * uspp_.lmax + 1, 
                                     uspp_.num_beta_radial_functions,  uspp_.num_beta_radial_functions); 
        uspp_.q_coefs.allocate();

        uspp_.q_radial_functions.set_dimensions(num_mt_points_, uspp_.num_beta_radial_functions * (uspp_.num_beta_radial_functions + 1) / 2);
        uspp_.q_radial_functions.allocate();

        for (int j = 0; j < uspp_.num_beta_radial_functions; j++)
        {
            for (int i = 0; i <= j; i++)
            {
                int idx = j * (j + 1) / 2 + i;

                std::vector<int> ij;
                parser["uspp"]["non_local"]["Q"]["qij"][idx]["ij"] >> ij;
                if (ij[0] != i || ij[1] != j) 
                {
                    std::stringstream s;
                    s << "wrong ij indices" << std::endl
                      << "i = " << i << " j = " << j << " idx = " << idx << std::endl
                      << "ij = " << ij[0] << " " << ij[1];
                    error_local(__FILE__, __LINE__, s);
                }

                std::vector<double> qfcoef;
                parser["uspp"]["non_local"]["Q"]["qij"][idx]["q_coefs"] >> qfcoef;

                int k = 0;
                for (int l = 0; l <= 2 * uspp_.lmax; l++)
                {
                    for (int n = 0; n < uspp_.num_q_coefs; n++) 
                    {
                        if (k >= (int)qfcoef.size()) error_local(__FILE__, __LINE__, "wrong size of qfcoef");
                        uspp_.q_coefs(n, l, i, j) = uspp_.q_coefs(n, l, j, i) = qfcoef[k++];
                    }
                }

                std::vector<double> qfunc;
                parser["uspp"]["non_local"]["Q"]["qij"][idx]["q_radial_function"] >> qfunc;
                if ((int)qfunc.size() != num_mt_points_) error_local(__FILE__, __LINE__, "wrong size of qfunc");
                memcpy(&uspp_.q_radial_functions(0, idx), &qfunc[0], num_mt_points_ * sizeof(double)); 
            }
        }

        uspp_.beta_radial_functions.set_dimensions(num_mt_points_, uspp_.num_beta_radial_functions);
        uspp_.beta_radial_functions.allocate();
        uspp_.beta_radial_functions.zero();

        uspp_.num_beta_radial_points.resize(uspp_.num_beta_radial_functions);
        uspp_.beta_l.resize(uspp_.num_beta_radial_functions);

        local_orbital_descriptor lod;
        for (int i = 0; i < uspp_.num_beta_radial_functions; i++)
        {
            parser["uspp"]["non_local"]["beta"][i]["kbeta"] >> uspp_.num_beta_radial_points[i];
            std::vector<double> beta;
            parser["uspp"]["non_local"]["beta"][i]["beta"] >> beta;
            if ((int)beta.size() != uspp_.num_beta_radial_points[i]) error_local(__FILE__, __LINE__, "wrong size of beta function");
            memcpy(&uspp_.beta_radial_functions(0, i), &beta[0], beta.size() * sizeof(double)); 
 
            parser["uspp"]["non_local"]["beta"][i]["lll"] >> uspp_.beta_l[i];
            
            // think of |beta> functions as of local orbitals
            lod.l = uspp_.beta_l[i];
            lo_descriptors_.push_back(lod);
        }

        uspp_.d_mtrx_ion.set_dimensions(uspp_.num_beta_radial_functions, uspp_.num_beta_radial_functions);
        uspp_.d_mtrx_ion.allocate();
        uspp_.d_mtrx_ion.zero();

        for (int k = 0; k < parser["uspp"]["non_local"]["D"].size(); k++)
        {
            double d;
            std::vector<int> ij;
            parser["uspp"]["non_local"]["D"][k]["ij"] >> ij;
            parser["uspp"]["non_local"]["D"][k]["d_ion"] >> d;
            uspp_.d_mtrx_ion(ij[0], ij[1]) = d;
            uspp_.d_mtrx_ion(ij[1], ij[0]) = d;
        }
        
    }

    if (esm_type_ == full_potential_lapwlo || esm_type_ == full_potential_pwlo)
    {
        parser["name"] >> name_;
        parser["symbol"] >> symbol_;
        parser["mass"] >> mass_;
        parser["number"] >> zn_;
        parser["rmin"] >> radial_grid_origin_;
        parser["rmax"] >> radial_grid_infinity_;
        parser["rmt"] >> mt_radius_;
        parser["nrmt"] >> num_mt_points_;

        read_input_core(parser);

        read_input_aw(parser);

        read_input_lo(parser);
    }
}

void Atom_type::sync_free_atom(int rank)
{
    assert(free_atom_potential_.size() != 0);
    assert(free_atom_density_.size() != 0);

    Platform::bcast(&free_atom_density_[0], radial_grid().size(), rank);
    Platform::bcast(&free_atom_potential_[0], radial_grid().size(), rank);
}

void Atom_type::fix_q_radial_function(int l, int i, int j, double* qrf)
{
    for (int ir = 0; ir < num_mt_points(); ir++)
    {
        double x = radial_grid(ir);
        double x2 = x * x;
        if (x < uspp_.q_functions_inner_radii[l])
        {
            qrf[ir] = uspp_.q_coefs(0, l, i, j);
            for (int n = 1; n < uspp_.num_q_coefs; n++) qrf[ir] += uspp_.q_coefs(n, l, i, j) * pow(x2, n);
            qrf[ir] *= pow(x, l + 2);
        }
    }
}

}
