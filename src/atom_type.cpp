#include "atom_type.h"

namespace sirius {

void Atom_type::init(int offset_lo__)
{
    PROFILE();

    /* check if the class instance was already initialized */
    if (initialized_) {
        TERMINATE("can't initialize twice");
    }

    offset_lo_ = offset_lo__;

    /* read data from file if it exists */
    if (file_name_.length() > 0) {
        if (!Utils::file_exists(file_name_)) {
            std::stringstream s;
            s << "file " + file_name_ + " doesn't exist";
            TERMINATE(s);
        } else {
            read_input(file_name_);
        }
    }

    /* add valence levels to the list of core levels */
    if (parameters_.full_potential()) {
        atomic_level_descriptor level;
        for (int ist = 0; ist < 28; ist++) {
            bool found = false;
            level.n = atomic_conf[zn_ - 1][ist][0];
            level.l = atomic_conf[zn_ - 1][ist][1];
            level.k = atomic_conf[zn_ - 1][ist][2];
            level.occupancy = double(atomic_conf[zn_ - 1][ist][3]);
            level.core = false;

            if (level.n != -1) {
                for (size_t jst = 0; jst < atomic_levels_.size(); jst++) {
                    if (atomic_levels_[jst].n == level.n &&
                        atomic_levels_[jst].l == level.l &&
                        atomic_levels_[jst].k == level.k) {
                        found = true;
                    }
                }
                if (!found) {
                    atomic_levels_.push_back(level);
                }
            }
        }
    }
    
    /* check the nuclear charge */
    if (zn_ == 0) {
        TERMINATE("zero atom charge");
    }

    /* set default radial grid if it was not done by user */
    if (radial_grid_.num_points() == 0) {
        set_radial_grid();
    }
    
    if (parameters_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
        /* initialize free atom density and potential */
        //== init_free_atom(false);

        /* initialize aw descriptors if they were not set manually */
        if (aw_descriptors_.size() == 0) {
            init_aw_descriptors(parameters_.lmax_apw());
        }

        if (static_cast<int>(aw_descriptors_.size()) != (parameters_.lmax_apw() + 1)) {
            TERMINATE("wrong size of augmented wave descriptors");
        }

        max_aw_order_ = 0;
        for (int l = 0; l <= parameters_.lmax_apw(); l++) {
            max_aw_order_ = std::max(max_aw_order_, (int)aw_descriptors_[l].size());
        }

        if (max_aw_order_ > 3) {
            TERMINATE("maximum aw order > 3");
        }
    }

    if (!parameters_.full_potential()) {
        local_orbital_descriptor lod;
        for (int i = 0; i < uspp_.num_beta_radial_functions; i++) {
            /* think of |beta> functions as of local orbitals */
            lod.l = uspp_.beta_l[i];
            lo_descriptors_.push_back(lod);
        }
    }
    
    /* initialize index of radial functions */
    indexr_.init(aw_descriptors_, lo_descriptors_);

    /* initialize index of muffin-tin basis functions */
    indexb_.init(indexr_);
    
    /* get the number of core electrons */
    num_core_electrons_ = 0;
    if (parameters_.full_potential()) {
        for (size_t i = 0; i < atomic_levels_.size(); i++) {
            if (atomic_levels_[i].core) {
                num_core_electrons_ += atomic_levels_[i].occupancy;
            }
        }
    }

    /* get number of valence electrons */
    num_valence_electrons_ = zn_ - num_core_electrons_;

    int lmmax_pot = Utils::lmmax(parameters_.lmax_pot());
    auto l_by_lm = Utils::l_by_lm(parameters_.lmax_pot());

    /* index the non-zero radial integrals */
    std::vector< std::pair<int, int> > non_zero_elements;

    for (int lm = 0; lm < lmmax_pot; lm++)
    {
        int l = l_by_lm[lm];

        for (int i2 = 0; i2 < indexr().size(); i2++)
        {
            int l2 = indexr(i2).l;
            
            for (int i1 = 0; i1 <= i2; i1++)
            {
                int l1 = indexr(i1).l;
                if ((l + l1 + l2) % 2 == 0)
                {
                    if (lm) non_zero_elements.push_back(std::pair<int, int>(i2, lm + lmmax_pot * i1));
                    for (int j = 0; j < parameters_.num_mag_dims(); j++)
                    {
                        int offs = (j + 1) * lmmax_pot * indexr().size();
                        non_zero_elements.push_back(std::pair<int, int>(i2, lm + lmmax_pot * i1 + offs));
                    }
                }
            }
        }
    }
    idx_radial_integrals_ = mdarray<int, 2>(2, non_zero_elements.size());
    for (size_t j = 0; j < non_zero_elements.size(); j++)
    {
        idx_radial_integrals_(0, j) = non_zero_elements[j].first;
        idx_radial_integrals_(1, j) = non_zero_elements[j].second;
    }

    if (parameters_.processing_unit() == GPU && parameters_.full_potential())
    {
        #ifdef __GPU
        idx_radial_integrals_.allocate(memory_t::device);
        idx_radial_integrals_.copy_to_device();
        rf_coef_ = mdarray<double, 3>(num_mt_points_, 4, indexr().size(), memory_t::host_pinned | memory_t::device);
        vrf_coef_ = mdarray<double, 3>(num_mt_points_, 4, lmmax_pot * indexr().size() * (parameters_.num_mag_dims() + 1), memory_t::host_pinned | memory_t::device); 
        #else
        TERMINATE_NO_GPU
        #endif
    }
    
    initialized_ = true;
}

void Atom_type::init_free_atom(bool smooth)
{
    /* check if atomic file exists */
    if (!Utils::file_exists(file_name_)) {
        std::stringstream s;
        //s << "file " + file_name_ + " doesn't exist";
        s << "Free atom density and potential for atom " << label_ << " are not initialized";
        WARNING(s);
        return;
    }
    
    json parser;
    std::ifstream(file_name_) >> parser;

    /* create free atom radial grid */
    auto fa_r = parser["free_atom"]["radial_grid"].get<std::vector<double>>();
    free_atom_radial_grid_ = Radial_grid(fa_r);
    /* read density and potential */
    auto v = parser["free_atom"]["density"].get<std::vector<double>>();
    free_atom_density_ = Spline<double>(free_atom_radial_grid_, v);
    /* smooth free atom density inside the muffin-tin sphere */
    if (smooth) {
        /* find point on the grid close to the muffin-tin radius */
        int irmt = idx_rmt_free_atom();
    
        mdarray<double, 1> b(2);
        mdarray<double, 2> A(2, 2);
        double R = free_atom_radial_grid_[irmt];
        A(0, 0) = std::pow(R, 2);
        A(0, 1) = std::pow(R, 3);
        A(1, 0) = 2 * R;
        A(1, 1) = 3 * std::pow(R, 2);
        
        b(0) = free_atom_density_[irmt];
        b(1) = free_atom_density_.deriv(1, irmt);

        linalg<CPU>::gesv<double>(2, 1, A.at<CPU>(), 2, b.at<CPU>(), 2);
       
        //== /* write initial density */
        //== std::stringstream sstr;
        //== sstr << "free_density_" << id_ << ".dat";
        //== FILE* fout = fopen(sstr.str().c_str(), "w");

        //== for (int ir = 0; ir < free_atom_radial_grid().num_points(); ir++)
        //== {
        //==     fprintf(fout, "%f %f \n", free_atom_radial_grid(ir), free_atom_density_[ir]);
        //== }
        //== fclose(fout);
        
        /* make smooth free atom density inside muffin-tin */
        for (int i = 0; i <= irmt; i++) {
            free_atom_density_[i] = b(0) * std::pow(free_atom_radial_grid(i), 2) + 
                                    b(1) * std::pow(free_atom_radial_grid(i), 3);
        }

        /* interpolate new smooth density */
        free_atom_density_.interpolate();

        //== /* write smoothed density */
        //== sstr.str("");
        //== sstr << "free_density_modified_" << id_ << ".dat";
        //== fout = fopen(sstr.str().c_str(), "w");

        //== for (int ir = 0; ir < free_atom_radial_grid().num_points(); ir++)
        //== {
        //==     fprintf(fout, "%f %f \n", free_atom_radial_grid(ir), free_atom_density_[ir]);
        //== }
        //== fclose(fout);
   }
}

void Atom_type::print_info() const
{
    printf("\n");
    printf("symbol         : %s\n", symbol_.c_str());
    printf("name           : %s\n", name_.c_str());
    printf("zn             : %i\n", zn_);
    printf("mass           : %f\n", mass_);
    printf("mt_radius      : %f\n", mt_radius_);
    printf("num_mt_points  : %i\n", num_mt_points_);
    printf("grid_origin    : %f\n", radial_grid_[0]);
    printf("grid_name      : %s\n", radial_grid_.grid_type_name().c_str());
    printf("\n");
    printf("number of core electrons    : %f\n", num_core_electrons_);
    printf("number of valence electrons : %f\n", num_valence_electrons_);

    if (parameters_.full_potential())
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
            printf("[");
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
        }

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
}

void Atom_type::read_input_core(json const& parser)
{
    std::string core_str = parser["core"];
    if (int size = (int)core_str.size()) {
        if (size % 2) {
            std::string s = std::string("wrong core configuration string : ") + core_str;
            TERMINATE(s);
        }
        int j = 0;
        while (j < size) {
            char c1 = core_str[j++];
            char c2 = core_str[j++];

            int n = -1;
            int l = -1;

            std::istringstream iss(std::string(1, c1));
            iss >> n;

            if (n <= 0 || iss.fail()) {
                std::string s = std::string("wrong principal quantum number : ") + std::string(1, c1);
                TERMINATE(s);
            }

            switch (c2) {
                case 's': {
                    l = 0;
                    break;
                }
                case 'p': {
                    l = 1;
                    break;
                }
                case 'd': {
                    l = 2;
                    break;
                }
                case 'f': {
                    l = 3;
                    break;
                }
                default: {
                    std::string s = std::string("wrong angular momentum label : ") + std::string(1, c2);
                    TERMINATE(s);
                }
            }

            atomic_level_descriptor level;
            level.n    = n;
            level.l    = l;
            level.core = true;
            for (int ist = 0; ist < 28; ist++) {
                if ((level.n == atomic_conf[zn_ - 1][ist][0]) && (level.l == atomic_conf[zn_ - 1][ist][1])) {
                    level.k         = atomic_conf[zn_ - 1][ist][2];
                    level.occupancy = double(atomic_conf[zn_ - 1][ist][3]);
                    atomic_levels_.push_back(level);
                }
            }
        }
    }
}

void Atom_type::read_input_aw(json const& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;

    /* default augmented wave basis */
    rsd.n = -1;
    rsd.l = -1;
    for (size_t order = 0; order < parser["valence"][0]["basis"].size(); order++) {
        rsd.enu      = parser["valence"][0]["basis"][order]["enu"];
        rsd.dme      = parser["valence"][0]["basis"][order]["dme"];
        rsd.auto_enu = parser["valence"][0]["basis"][order]["auto"];
        aw_default_l_.push_back(rsd);
    }

    for (size_t j = 1; j < parser["valence"].size(); j++) {
        rsd.l = parser["valence"][j]["l"];
        rsd.n = parser["valence"][j]["n"];
        rsd_set.clear();
        for (size_t order = 0; order < parser["valence"][j]["basis"].size(); order++) {
            rsd.enu      = parser["valence"][j]["basis"][order]["enu"];
            rsd.dme      = parser["valence"][j]["basis"][order]["dme"];
            rsd.auto_enu = parser["valence"][j]["basis"][order]["auto"];
            rsd_set.push_back(rsd);
        }
        aw_specific_l_.push_back(rsd_set);
    }
}

void Atom_type::read_input_lo(json const& parser)
{
    radial_solution_descriptor rsd;
    radial_solution_descriptor_set rsd_set;

    int l;
    for (size_t j = 0; j < parser["lo"].size(); j++) {
        l = parser["lo"][j]["l"];

        local_orbital_descriptor lod;
        lod.l = l;
        rsd.l = l;
        rsd_set.clear();
        for (size_t order = 0; order < parser["lo"][j]["basis"].size(); order++) {
            rsd.n        = parser["lo"][j]["basis"][order]["n"];
            rsd.enu      = parser["lo"][j]["basis"][order]["enu"];
            rsd.dme      = parser["lo"][j]["basis"][order]["dme"];
            rsd.auto_enu = parser["lo"][j]["basis"][order]["auto"];
            rsd_set.push_back(rsd);
        }
        lod.rsd_set = rsd_set;
        lo_descriptors_.push_back(lod);
    }
}

void Atom_type::read_pseudo_uspp(json const& parser)
{
    symbol_ = parser["pseudo_potential"]["header"]["element"];

    double zp;
    zp  = parser["pseudo_potential"]["header"]["z_valence"];
    zn_ = int(zp + 1e-10);

    int nmesh;
    nmesh = parser["pseudo_potential"]["header"]["mesh_size"];

    uspp_.r = parser["pseudo_potential"]["radial_grid"].get<std::vector<double>>();

    uspp_.vloc = parser["pseudo_potential"]["local_potential"].get<std::vector<double>>();

    uspp_.core_charge_density = parser["pseudo_potential"].value("core_charge_density", std::vector<double>(nmesh, 0));

    uspp_.total_charge_density = parser["pseudo_potential"]["total_charge_density"].get<std::vector<double>>();

    if ((int)uspp_.r.size() != nmesh) {
        TERMINATE("wrong mesh size");
    }
    if ((int)uspp_.vloc.size() != nmesh ||
        (int)uspp_.core_charge_density.size() != nmesh ||
        (int)uspp_.total_charge_density.size() != nmesh) {
        std::cout << uspp_.vloc.size() << " " << uspp_.core_charge_density.size() << " " << uspp_.total_charge_density.size() << std::endl;
        TERMINATE("wrong array size");
    }

    num_mt_points_ = nmesh;
    mt_radius_     = uspp_.r[nmesh - 1];

    set_radial_grid(nmesh, &uspp_.r[0]);

    uspp_.num_beta_radial_functions = parser["pseudo_potential"]["header"]["number_of_proj"];

    uspp_.beta_radial_functions = mdarray<double, 2>(num_mt_points_, uspp_.num_beta_radial_functions);
    uspp_.beta_radial_functions.zero();

    uspp_.num_beta_radial_points.resize(uspp_.num_beta_radial_functions);
    uspp_.beta_l.resize(uspp_.num_beta_radial_functions);

    int lmax_beta = 0;
    local_orbital_descriptor lod;
    for (int i = 0; i < uspp_.num_beta_radial_functions; i++) {
        auto beta = parser["pseudo_potential"]["beta_projectors"][i]["radial_function"].get<std::vector<double>>();
        if ((int)beta.size() > num_mt_points_) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << beta.size() << std::endl
              << "radial grid size: " << num_mt_points_;
            TERMINATE(s);
        }
        uspp_.num_beta_radial_points[i] = static_cast<int>(beta.size());
        std::memcpy(&uspp_.beta_radial_functions(0, i), &beta[0], uspp_.num_beta_radial_points[i] * sizeof(double));

        uspp_.beta_l[i] = parser["pseudo_potential"]["beta_projectors"][i]["angular_momentum"];
        lmax_beta       = std::max(lmax_beta, uspp_.beta_l[i]);
    }

    uspp_.d_mtrx_ion = mdarray<double, 2>(uspp_.num_beta_radial_functions, uspp_.num_beta_radial_functions);
    uspp_.d_mtrx_ion.zero();
    auto dion = parser["pseudo_potential"]["D_ion"].get<std::vector<double>>();

    for (int i = 0; i < uspp_.num_beta_radial_functions; i++) {
        for (int j = 0; j < uspp_.num_beta_radial_functions; j++) {
            uspp_.d_mtrx_ion(i, j) = dion[j * uspp_.num_beta_radial_functions + i];
        }
    }

    if (parser["pseudo_potential"].count("augmentation")) {
        uspp_.augmentation_        = true;
        uspp_.q_radial_functions_l = mdarray<double, 3>(num_mt_points_, uspp_.num_beta_radial_functions * (uspp_.num_beta_radial_functions + 1) / 2, 2 * lmax_beta + 1);
        uspp_.q_radial_functions_l.zero();

        for (size_t k = 0; k < parser["pseudo_potential"]["augmentation"].size(); k++) {
            int i    = parser["pseudo_potential"]["augmentation"][k]["i"];
            int j    = parser["pseudo_potential"]["augmentation"][k]["j"];
            int idx  = j * (j + 1) / 2 + i;
            int l    = parser["pseudo_potential"]["augmentation"][k]["angular_momentum"];
            auto qij = parser["pseudo_potential"]["augmentation"][k]["radial_function"].get<std::vector<double>>();
            if ((int)qij.size() != num_mt_points_) {
                TERMINATE("wrong size of qij");
            }

            std::memcpy(&uspp_.q_radial_functions_l(0, idx, l), &qij[0], num_mt_points_ * sizeof(double));
        }
    }

    /* read starting wave functions ( UPF CHI ) */
    if (parser["pseudo_potential"].count("atomic_wave_functions")) {
        size_t nwf = parser["pseudo_potential"]["atomic_wave_functions"].size();
        for (size_t k = 0; k < nwf; k++) {
            std::pair<int, std::vector<double>> wf;
            wf.second = parser["pseudo_potential"]["atomic_wave_functions"][k]["radial_function"].get<std::vector<double>>();

            if ((int)wf.second.size() != num_mt_points_) {
                std::stringstream s;
                s << "wrong size of atomic functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
                  << "size of atomic radial functions in the file: " << wf.second.size() << std::endl
                  << "radial grid size: " << num_mt_points_;
                TERMINATE(s);
            }
            wf.first = parser["pseudo_potential"]["atomic_wave_functions"][k]["angular_momentum"];
            uspp_.atomic_pseudo_wfs_.push_back(wf);

            /* read occupation of the function */
            double occ = parser["pseudo_potential"]["atomic_wave_functions"][k]["occupation"];
            uspp_.atomic_pseudo_wfs_occ_.push_back(occ);
        }
    }

    uspp_.is_initialized = true;
}

void Atom_type::read_pseudo_paw(json const& parser)
{
    if (!uspp_.is_initialized) {
        TERMINATE("Ultrasoft or base part of PAW is not initialized");
    }

    /* read core energy */
    paw_.core_energy = parser["pseudo_potential"]["header"]["paw_core_energy"];

    /* cutoff index */
    paw_.cutoff_radius_index = parser["pseudo_potential"]["header"]["cutoff_radius_index"];

    /* read augmentation multipoles and integrals */
    paw_.aug_integrals = parser["pseudo_potential"]["paw_data"]["aug_integrals"].get<std::vector<double>>();

    paw_.aug_multopoles = parser["pseudo_potential"]["paw_data"]["aug_multipoles"].get<std::vector<double>>();

    /* read core density and potential */
    paw_.all_elec_core_charge = parser["pseudo_potential"]["paw_data"]["ae_core_charge_density"].get<std::vector<double>>();

    paw_.all_elec_loc_potential = parser["pseudo_potential"]["paw_data"]["ae_local_potential"].get<std::vector<double>>();

    /* read occupations */
    paw_.occupations = parser["pseudo_potential"]["paw_data"]["occupations"].get<std::vector<double>>();

    /* setups for reading AE and PS basis wave functions */
    int num_wfc = uspp_.num_beta_radial_functions;

    paw_.all_elec_wfc = mdarray<double, 2>(num_mt_points_, num_wfc);
    paw_.pseudo_wfc   = mdarray<double, 2>(num_mt_points_, num_wfc);

    paw_.all_elec_wfc.zero();
    paw_.pseudo_wfc.zero();

    /* read ae and ps wave functions */
    for (int i = 0; i < num_wfc; i++) {
        /* read ae wave func */
        auto wfc = parser["pseudo_potential"]["paw_data"]["ae_wfc"][i]["radial_function"].get<std::vector<double>>();

        if ((int)wfc.size() > num_mt_points_) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points_;
            TERMINATE(s);
        }

        // TODO: check if this is OK
        std::memcpy(&paw_.all_elec_wfc(0, i), wfc.data(), (paw_.cutoff_radius_index + 100) * sizeof(double));

        /* read ps wave func */
        wfc.clear();

        wfc = parser["pseudo_potential"]["paw_data"]["ps_wfc"][i]["radial_function"].get<std::vector<double>>();

        if ((int)wfc.size() > num_mt_points_) {
            std::stringstream s;
            s << "wrong size of beta functions for atom type " << symbol_ << " (label: " << label_ << ")" << std::endl
              << "size of beta radial functions in the file: " << wfc.size() << std::endl
              << "radial grid size: " << num_mt_points_;
            TERMINATE(s);
        }

        std::memcpy(&paw_.pseudo_wfc(0, i), wfc.data(), (paw_.cutoff_radius_index + 100) * sizeof(double));
    }
}

void Atom_type::read_input(const std::string& fname)
{
    json parser;
    std::ifstream(fname) >> parser;

    if (!parameters_.full_potential()) {
        read_pseudo_uspp(parser);

        if (parameters_.esm_type() == electronic_structure_method_t::paw_pseudopotential) {
            read_pseudo_paw(parser);
        }
    }

    if (parameters_.full_potential()) {
        name_               = parser["name"];
        symbol_             = parser["symbol"];
        mass_               = parser["mass"];
        zn_                 = parser["number"];
        radial_grid_origin_ = parser["rmin"];
        mt_radius_          = parser["rmt"];
        num_mt_points_      = parser["nrmt"];

        read_input_core(parser);

        read_input_aw(parser);

        read_input_lo(parser);
    }
}
        
}
