#include "atom_type.h"

namespace sirius {

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

    if (!parser["pseudo_potential"]["augmentation"].empty()) {
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
    if (!parser["pseudo_potential"]["atomic_wave_functions"].empty()) {
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
