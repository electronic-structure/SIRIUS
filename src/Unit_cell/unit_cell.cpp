#include "unit_cell.hpp"

namespace sirius {
std::vector<double> Unit_cell::find_mt_radii()
{
    if (nearest_neighbours_.size() == 0) {
        TERMINATE("array of nearest neighbours is empty");
    }

    std::vector<double> Rmt(num_atom_types(), 1e10);

    if (parameters_.auto_rmt() == 1) {
        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();
            if (nearest_neighbours_[ia].size() > 1) {
                int ja  = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja).type_id();
                /* don't allow spheres to touch: take a smaller value than half a distance */
                double R = std::min(parameters_.rmt_max(), 0.95 * nearest_neighbours_[ia][1].distance / 2);
                /* take minimal R for the given atom type */
                Rmt[id1] = std::min(R, Rmt[id1]);
                Rmt[id2] = std::min(R, Rmt[id2]);
            } else {
                Rmt[id1] = parameters_.rmt_max();
            }
        }
    }

    if (parameters_.auto_rmt() == 2) {
        std::vector<double> scale(num_atom_types(), 1e10);

        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();
            if (nearest_neighbours_[ia].size() > 1) {
                int ja  = nearest_neighbours_[ia][1].atom_id;
                int id2 = atom(ja).type_id();

                double d   = nearest_neighbours_[ia][1].distance;
                double s   = 0.95 * d / (atom_type(id1).mt_radius() + atom_type(id2).mt_radius());
                scale[id1] = std::min(s, scale[id1]);
                scale[id2] = std::min(s, scale[id2]);
            } else {
                scale[id1] = parameters_.rmt_max() / atom_type(id1).mt_radius();
            }
        }

        for (int iat = 0; iat < num_atom_types(); iat++) {
            Rmt[iat] = std::min(parameters_.rmt_max(), atom_type(iat).mt_radius() * scale[iat]);
        }
    }

    /* Suppose we have 3 different atoms. First we determint Rmt between 1st and 2nd atom,
     * then we determine Rmt between (let's say) 2nd and 3rd atom and at this point we reduce
     * the Rmt of the 2nd atom. This means that the 1st atom gets a possibility to expand if
     * it is far from the 3rd atom. */
    bool inflate = true;

    if (inflate) {
        std::vector<bool> scale_Rmt(num_atom_types(), true);
        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();

            if (nearest_neighbours_[ia].size() > 1) {
                int ja      = nearest_neighbours_[ia][1].atom_id;
                int id2     = atom(ja).type_id();
                double dist = nearest_neighbours_[ia][1].distance;

                if (Rmt[id1] + Rmt[id2] > dist * 0.94) {
                    scale_Rmt[id1] = false;
                    scale_Rmt[id2] = false;
                }
            }
        }

        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();
            if (nearest_neighbours_[ia].size() > 1) {
                int ja      = nearest_neighbours_[ia][1].atom_id;
                int id2     = atom(ja).type_id();
                double dist = nearest_neighbours_[ia][1].distance;

                if (scale_Rmt[id1] && !scale_Rmt[id2]) {
                    Rmt[id1]       = std::min(parameters_.rmt_max(), 0.95 * (dist - Rmt[id2]));
                    scale_Rmt[id1] = false;
                }
            }
        }
    }

    for (int i = 0; i < num_atom_types(); i++) {
        if (Rmt[i] < 0.3) {
            std::stringstream s;
            s << "muffin-tin radius for atom type " << i << " (" << atom_types_[i].label()
              << ") is too small: " << Rmt[i];
            TERMINATE(s);
        }
    }

    return Rmt;
}

bool Unit_cell::check_mt_overlap(int& ia__, int& ja__)
{
    if (num_atoms() != 0 && nearest_neighbours_.size() == 0) {
        TERMINATE("array of nearest neighbours is empty");
    }

    for (int ia = 0; ia < num_atoms(); ia++) {
        /* first atom is always the central one itself */
        if (nearest_neighbours_[ia].size() <= 1) {
            continue;
        }

        int ja      = nearest_neighbours_[ia][1].atom_id;
        double dist = nearest_neighbours_[ia][1].distance;

        if (atom(ia).mt_radius() + atom(ja).mt_radius() >= dist) {
            ia__ = ia;
            ja__ = ja;
            return true;
        }
    }

    return false;
}

void Unit_cell::print_info(int verbosity__) const
{
    std::printf("\n");
    std::printf("Unit cell\n");
    for (int i = 0; i < 80; i++) {
        std::printf("-");
    }
    std::printf("\n");

    std::printf("lattice vectors\n");
    for (int i = 0; i < 3; i++) {
        std::printf("  a%1i : %18.10f %18.10f %18.10f \n", i + 1, lattice_vectors_(0, i), lattice_vectors_(1, i),
               lattice_vectors_(2, i));
    }
    std::printf("reciprocal lattice vectors\n");
    for (int i = 0; i < 3; i++) {
        std::printf("  b%1i : %18.10f %18.10f %18.10f \n", i + 1, reciprocal_lattice_vectors_(0, i),
               reciprocal_lattice_vectors_(1, i), reciprocal_lattice_vectors_(2, i));
    }
    std::printf("\n");
    std::printf("unit cell volume : %18.8f [a.u.^3]\n", omega());
    std::printf("1/sqrt(omega)    : %18.8f\n", 1.0 / sqrt(omega()));
    std::printf("MT volume        : %f (%5.2f%%)\n", volume_mt(), volume_mt() * 100 / omega());
    std::printf("IT volume        : %f (%5.2f%%)\n", volume_it(), volume_it() * 100 / omega());

    std::printf("\n");
    std::printf("number of atom types : %i\n", num_atom_types());
    for (int i = 0; i < num_atom_types(); i++) {
        int id = atom_type(i).id();
        std::printf("type id : %i   symbol : %2s   mt_radius : %10.6f, num_atoms: %i\n", id, atom_type(i).symbol().c_str(),
               atom_type(i).mt_radius(), atom_type(i).num_atoms());
    }

    std::printf("total number of atoms : %i\n", num_atoms());
    std::printf("number of symmetry classes : %i\n", num_atom_symmetry_classes());
    if (!parameters_.full_potential()) {
        std::printf("number of PAW atoms : %i\n", num_paw_atoms());
    }
    if (verbosity__ >= 2) {
        std::printf("\n");
        std::printf("atom id              position                    vector_field        type id    class id\n");
        std::printf("----------------------------------------------------------------------------------------\n");
        for (int i = 0; i < num_atoms(); i++) {
            auto pos = atom(i).position();
            auto vf  = atom(i).vector_field();
            std::printf("%6i      %f %f %f   %f %f %f   %6i      %6i\n", i, pos[0], pos[1], pos[2], vf[0], vf[1], vf[2],
                   atom(i).type_id(), atom(i).symmetry_class_id());
        }

        std::printf("\n");
        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            std::printf("class id : %i   atom id : ", ic);
            for (int i = 0; i < atom_symmetry_class(ic).num_atoms(); i++) {
                std::printf("%i ", atom_symmetry_class(ic).atom_id(i));
            }
            std::printf("\n");
        }
        std::printf("\n");
        std::printf("atom id              position (Cartesian, a.u.)\n");
        std::printf("----------------------------------------------------------------------------------------\n");
        for (int i = 0; i < num_atoms(); i++) {
            auto pos = atom(i).position();
            auto vc  = get_cartesian_coordinates(pos);
            std::printf("%6i      %18.12f %18.12f %18.12f\n", i, vc[0], vc[1], vc[2]);
        }

        std::printf("\n");
        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            std::printf("class id : %i   atom id : ", ic);
            for (int i = 0; i < atom_symmetry_class(ic).num_atoms(); i++) {
                std::printf("%i ", atom_symmetry_class(ic).atom_id(i));
            }
            std::printf("\n");
        }
    }
    std::printf("\nminimum bond length: %20.12f\n", min_bond_length());
    if (!parameters_.full_potential()) {
        std::printf("\nnumber of pseudo wave-functions: %i\n", this->num_ps_atomic_wf());
    }
    print_symmetry_info(verbosity__);
}

unit_cell_parameters_descriptor Unit_cell::unit_cell_parameters()
{
    unit_cell_parameters_descriptor d;

    vector3d<double> v0(lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0));
    vector3d<double> v1(lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1));
    vector3d<double> v2(lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2));

    d.a = v0.length();
    d.b = v1.length();
    d.c = v2.length();

    d.alpha = std::acos(dot(v1, v2) / d.b / d.c) * 180 / pi;
    d.beta  = std::acos(dot(v0, v2) / d.a / d.c) * 180 / pi;
    d.gamma = std::acos(dot(v0, v1) / d.a / d.b) * 180 / pi;

    return d;
}

void Unit_cell::write_cif()
{
    if (comm_.rank() == 0) {
        FILE* fout = fopen("unit_cell.cif", "w");

        auto d = unit_cell_parameters();

        fprintf(fout, "_cell_length_a %f\n", d.a);
        fprintf(fout, "_cell_length_b %f\n", d.b);
        fprintf(fout, "_cell_length_c %f\n", d.c);
        fprintf(fout, "_cell_angle_alpha %f\n", d.alpha);
        fprintf(fout, "_cell_angle_beta %f\n", d.beta);
        fprintf(fout, "_cell_angle_gamma %f\n", d.gamma);

        // fprintf(fout, "loop_\n");
        // fprintf(fout, "_symmetry_equiv_pos_as_xyz\n");

        fprintf(fout, "loop_\n");
        fprintf(fout, "_atom_site_label\n");
        fprintf(fout, "_atom_type_symbol\n");
        fprintf(fout, "_atom_site_fract_x\n");
        fprintf(fout, "_atom_site_fract_y\n");
        fprintf(fout, "_atom_site_fract_z\n");
        for (int ia = 0; ia < num_atoms(); ia++) {
            auto pos = atom(ia).position();
            fprintf(fout, "%i %s %f %f %f\n", ia + 1, atom(ia).type().label().c_str(), pos[0], pos[1], pos[2]);
        }
        fclose(fout);
    }
}

json Unit_cell::serialize(bool cart_pos__) const
{
    json dict;

    dict["lattice_vectors"] = {{lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0)},
                               {lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1)},
                               {lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2)}};
    dict["atom_types"]      = json::array();
    for (int iat = 0; iat < num_atom_types(); iat++) {
        dict["atom_types"].push_back(atom_type(iat).label());
    }
    dict["atom_files"] = json::object();
    for (int iat = 0; iat < num_atom_types(); iat++) {
        dict["atom_files"][atom_type(iat).label()] = atom_type(iat).file_name();
    }
    dict["atoms"] = json::object();
    for (int iat = 0; iat < num_atom_types(); iat++) {
        dict["atoms"][atom_type(iat).label()] = json::array();
        for (int i = 0; i < atom_type(iat).num_atoms(); i++) {
            int ia = atom_type(iat).atom_id(i);
            auto v = atom(ia).position();
            /* convert to Cartesian coordinates */
            if (cart_pos__) {
                v = lattice_vectors_ * v;
            }
            dict["atoms"][atom_type(iat).label()].push_back({v[0], v[1], v[2]});
        }
    }
    return dict;
}

void Unit_cell::find_nearest_neighbours(double cluster_radius)
{
    PROFILE("sirius::Unit_cell::find_nearest_neighbours");

    auto max_frac_coord = find_translations(cluster_radius, lattice_vectors_);

    nearest_neighbours_.clear();
    nearest_neighbours_.resize(num_atoms());

    #pragma omp parallel for default(shared)
    for (int ia = 0; ia < num_atoms(); ia++) {
        auto iapos = get_cartesian_coordinates(atom(ia).position());

        std::vector<nearest_neighbour_descriptor> nn;

        std::vector<std::pair<double, int>> nn_sort;

        for (int i0 = -max_frac_coord[0]; i0 <= max_frac_coord[0]; i0++) {
            for (int i1 = -max_frac_coord[1]; i1 <= max_frac_coord[1]; i1++) {
                for (int i2 = -max_frac_coord[2]; i2 <= max_frac_coord[2]; i2++) {
                    nearest_neighbour_descriptor nnd;
                    nnd.translation[0] = i0;
                    nnd.translation[1] = i1;
                    nnd.translation[2] = i2;

                    auto vt = get_cartesian_coordinates<int>(nnd.translation);

                    for (int ja = 0; ja < num_atoms(); ja++) {
                        nnd.atom_id = ja;

                        auto japos = get_cartesian_coordinates(atom(ja).position());

                        vector3d<double> v = japos + vt - iapos;

                        nnd.distance = v.length();

                        if (nnd.distance <= cluster_radius) {
                            nn.push_back(nnd);

                            nn_sort.push_back(std::pair<double, int>(nnd.distance, (int)nn.size() - 1));
                        }
                    }
                }
            }
        }

        std::sort(nn_sort.begin(), nn_sort.end());
        nearest_neighbours_[ia].resize(nn.size());
        for (int i = 0; i < (int)nn.size(); i++) {
            nearest_neighbours_[ia][i] = nn[nn_sort[i].second];
        }
    }

    if (parameters_.control().print_neighbors_ && comm_.rank() == 0) {
        std::printf("Nearest neighbors\n");
        std::printf("=================\n");
        for (int ia = 0; ia < num_atoms(); ia++) {
            std::printf("Central atom: %s (%i)\n", atom(ia).type().symbol().c_str(), ia);
            for (int i = 0; i < 80; i++) {
                std::printf("-");
            }
            std::printf("\n");
            std::printf("atom (  id)       D [a.u.]    translation\n");
            for (int i = 0; i < 80; i++) {
                std::printf("-");
            }
            std::printf("\n");
            for (int i = 0; i < (int)nearest_neighbours_[ia].size(); i++) {
                int ja = nearest_neighbours_[ia][i].atom_id;
                std::printf("%4s (%4i)   %12.6f  %4i %4i %4i\n", atom(ja).type().symbol().c_str(), ja,
                       nearest_neighbours_[ia][i].distance, nearest_neighbours_[ia][i].translation[0],
                       nearest_neighbours_[ia][i].translation[1], nearest_neighbours_[ia][i].translation[2]);
            }
            std::printf("\n");
        }
    }
}

bool Unit_cell::is_point_in_mt(vector3d<double> vc, int& ja, int& jr, double& dr, double tp[2]) const
{
    /* reduce coordinates to the primitive unit cell */
    auto vr = reduce_coordinates(get_fractional_coordinates(vc));

    for (int ia = 0; ia < num_atoms(); ia++) {
        for (int i0 = -1; i0 <= 1; i0++) {
            for (int i1 = -1; i1 <= 1; i1++) {
                for (int i2 = -1; i2 <= 1; i2++) {
                    /* atom position */
                    vector3d<double> posf = vector3d<double>(i0, i1, i2) + atom(ia).position();

                    /* vector connecting center of atom and reduced point */
                    vector3d<double> vf = vr.first - posf;

                    /* convert to spherical coordinates */
                    auto vs = SHT::spherical_coordinates(get_cartesian_coordinates(vf));

                    if (vs[0] < atom(ia).mt_radius()) {
                        ja    = ia;
                        tp[0] = vs[1]; // theta
                        tp[1] = vs[2]; // phi

                        if (vs[0] < atom(ia).type().radial_grid(0)) {
                            jr = 0;
                            dr = 0.0;
                        } else {
                            for (int ir = 0; ir < atom(ia).num_mt_points() - 1; ir++) {
                                if (vs[0] >= atom(ia).type().radial_grid(ir) &&
                                    vs[0] < atom(ia).type().radial_grid(ir + 1)) {
                                    jr = ir;
                                    dr = vs[0] - atom(ia).type().radial_grid(ir);
                                    break;
                                }
                            }
                        }

                        return true;
                    }
                }
            }
        }
    }
    ja = -1;
    jr = -1;
    return false;
}

void Unit_cell::generate_radial_functions()
{
    PROFILE("sirius::Unit_cell::generate_radial_functions");

    for (int icloc = 0; icloc < (int)spl_num_atom_symmetry_classes().local_size(); icloc++) {
        int ic = spl_num_atom_symmetry_classes(icloc);
        atom_symmetry_class(ic).generate_radial_functions(parameters_.valence_relativity());
    }

    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
        int rank = spl_num_atom_symmetry_classes().local_rank(ic);
        atom_symmetry_class(ic).sync_radial_functions(comm_, rank);
    }

    if (parameters_.control().verbosity_ >= 1) {
        pstdout pout(comm_);

        for (int icloc = 0; icloc < (int)spl_num_atom_symmetry_classes().local_size(); icloc++) {
            int ic = spl_num_atom_symmetry_classes(icloc);
            atom_symmetry_class(ic).write_enu(pout);
        }

        if (comm_.rank() == 0) {
            std::printf("\n");
            std::printf("Linearization energies\n");
        }
    }
    if (parameters_.control().verbosity_ >= 4 && comm_.rank() == 0) {
        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            atom_symmetry_class(ic).dump_lo();
        }
    }
}

void Unit_cell::generate_radial_integrals()
{
    PROFILE("sirius::Unit_cell::generate_radial_integrals");

    for (int icloc = 0; icloc < spl_num_atom_symmetry_classes().local_size(); icloc++) {
        int ic = spl_num_atom_symmetry_classes(icloc);
        atom_symmetry_class(ic).generate_radial_integrals(parameters_.valence_relativity());
    }

    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
        int rank = spl_num_atom_symmetry_classes().local_rank(ic);
        atom_symmetry_class(ic).sync_radial_integrals(comm_, rank);
    }

    for (int ialoc = 0; ialoc < spl_num_atoms_.local_size(); ialoc++) {
        int ia = spl_num_atoms_[ialoc];
        atom(ia).generate_radial_integrals(parameters_.processing_unit(), Communicator::self());
    }

    for (int ia = 0; ia < num_atoms(); ia++) {
        int rank = spl_num_atoms().local_rank(ia);
        atom(ia).sync_radial_integrals(comm_, rank);
    }
}

std::string Unit_cell::chemical_formula()
{
    std::string name;
    for (int iat = 0; iat < num_atom_types(); iat++) {
        name += atom_type(iat).symbol();
        int n = 0;
        for (int ia = 0; ia < num_atoms(); ia++) {
            if (atom(ia).type_id() == atom_type(iat).id())
                n++;
        }
        if (n != 1) {
            std::stringstream s;
            s << n;
            name = (name + s.str());
        }
    }

    return name;
}

Atom_type& Unit_cell::add_atom_type(const std::string label__, const std::string file_name__)
{
    if (atoms_.size()) {
        TERMINATE("Can't add new atom type if atoms are already added");
    }

    int id = next_atom_type_id(label__);
    atom_types_.push_back(std::move(Atom_type(parameters_, id, label__, file_name__)));
    return atom_types_.back();
}

void Unit_cell::add_atom(const std::string label, vector3d<double> position, vector3d<double> vector_field)
{
    if (atom_type_id_map_.count(label) == 0) {
        std::stringstream s;
        s << "atom type with label " << label << " is not found";
        TERMINATE(s);
    }
    if (atom_id_by_position(position) >= 0) {
        std::stringstream s;
        s << "atom with the same position is already in list" << std::endl
          << "  position : " << position[0] << " " << position[1] << " " << position[2];
        TERMINATE(s);
    }

    atoms_.push_back(std::move(Atom(atom_type(label), position, vector_field)));
    atom_type(label).add_atom_id(static_cast<int>(atoms_.size()) - 1);
}

double Unit_cell::min_bond_length() const
{
    double len{1e10};

    for (int ia = 0; ia < num_atoms(); ia++) {
        if (nearest_neighbours_[ia].size() > 1) {
            len = std::min(len, nearest_neighbours_[ia][1].distance);
        }
    }
    return len;
}

void Unit_cell::initialize()
{
    PROFILE("sirius::Unit_cell::initialize");

    /* split number of atom between all MPI ranks */
    spl_num_atoms_ = splindex<splindex_t::block>(num_atoms(), comm_.size(), comm_.rank());

    /* initialize atom types */
    int offs_lo{0};
    for (int iat = 0; iat < num_atom_types(); iat++) {
        atom_type(iat).init(offs_lo);
        max_num_mt_points_        = std::max(max_num_mt_points_, atom_type(iat).num_mt_points());
        max_mt_basis_size_        = std::max(max_mt_basis_size_, atom_type(iat).mt_basis_size());
        max_mt_radial_basis_size_ = std::max(max_mt_radial_basis_size_, atom_type(iat).mt_radial_basis_size());
        max_mt_aw_basis_size_     = std::max(max_mt_aw_basis_size_, atom_type(iat).mt_aw_basis_size());
        max_mt_lo_basis_size_     = std::max(max_mt_lo_basis_size_, atom_type(iat).mt_lo_basis_size());
        lmax_                     = std::max(lmax_, atom_type(iat).indexr().lmax());
        offs_lo += atom_type(iat).mt_lo_basis_size();
    }

    /* find the charges */
    for (int i = 0; i < num_atoms(); i++) {
        total_nuclear_charge_ += atom(i).zn();
        num_core_electrons_ += atom(i).type().num_core_electrons();
        num_valence_electrons_ += atom(i).type().num_valence_electrons();
    }
    num_electrons_ = num_core_electrons_ + num_valence_electrons_;

    /* initialize atoms */
    for (int ia = 0; ia < num_atoms(); ia++) {
        atom(ia).init(mt_lo_basis_size_);
        mt_aw_basis_size_ += atom(ia).mt_aw_basis_size();
        mt_lo_basis_size_ += atom(ia).mt_lo_basis_size();
    }

    init_paw();

    for (int iat = 0; iat < num_atom_types(); iat++) {
        int nat = atom_type(iat).num_atoms();
        if (nat > 0) {
            atom_coord_.push_back(std::move(mdarray<double, 2>(nat, 3, memory_t::host)));
            if (parameters_.processing_unit() == device_t::GPU) {
                atom_coord_.back().allocate(memory_t::device);
            }
        } else {
            atom_coord_.push_back(std::move(mdarray<double, 2>()));
        }
    }
    update();

    //== write_cif();

    //== if (comm().rank() == 0) {
    //==     std::ofstream ofs(std::string("unit_cell.json"), std::ofstream::out | std::ofstream::trunc);
    //==     ofs << serialize().dump(4);
    //== }
}

void Unit_cell::get_symmetry()
{
    PROFILE("sirius::Unit_cell::get_symmetry");

    if (num_atoms() == 0) {
        return;
    }

    if (atom_symmetry_classes_.size() != 0) {
        atom_symmetry_classes_.clear();
        for (int ia = 0; ia < num_atoms(); ia++) {
            atom(ia).set_symmetry_class(nullptr);
        }
    }

    mdarray<double, 2> positions(3, num_atoms());
    mdarray<double, 2> spins(3, num_atoms());
    std::vector<int> types(num_atoms());
    for (int ia = 0; ia < num_atoms(); ia++) {
        auto vp = atom(ia).position();
        auto vf = atom(ia).vector_field();
        for (int x : {0, 1, 2}) {
            positions(x, ia) = vp[x];
            spins(x, ia)     = vf[x];
        }
        types[ia] = atom(ia).type_id();
    }

    symmetry_ = std::unique_ptr<Unit_cell_symmetry>(
        new Unit_cell_symmetry(lattice_vectors_, num_atoms(), types, positions, spins, parameters_.so_correction(),
                               parameters_.spglib_tolerance(), parameters_.use_symmetry()));

    int atom_class_id{-1};
    std::vector<int> asc(num_atoms(), -1);
    for (int i = 0; i < num_atoms(); i++) {
        /* if symmetry class is not assigned to this atom */
        if (asc[i] == -1) {
            /* take next id */
            atom_class_id++;
            atom_symmetry_classes_.push_back(std::move(Atom_symmetry_class(atom_class_id, atoms_[i].type())));

            /* scan all atoms */
            for (int j = 0; j < num_atoms(); j++) {
                bool is_equal = (equivalent_atoms_.size())
                                    ? (equivalent_atoms_[j] == equivalent_atoms_[i])
                                    : (symmetry_->atom_symmetry_class(j) == symmetry_->atom_symmetry_class(i));
                /* assign new class id for all equivalent atoms */
                if (is_equal) {
                    asc[j] = atom_class_id;
                    atom_symmetry_classes_.back().add_atom_id(j);
                }
            }
        }
    }

    for (auto& e : atom_symmetry_classes_) {
        for (int i = 0; i < e.num_atoms(); i++) {
            int ia = e.atom_id(i);
            atoms_[ia].set_symmetry_class(&e);
        }
    }

    assert(num_atom_symmetry_classes() != 0);
}

void Unit_cell::import(Unit_cell_input const &inp__)
{
    if (inp__.exist_) {
        /* first, load all types */
        for (int iat = 0; iat < (int)inp__.labels_.size(); iat++) {
            auto label = inp__.labels_[iat];
            auto fname = inp__.atom_files_.at(label);
            add_atom_type(label, fname);
        }
        /* then load atoms */
        for (int iat = 0; iat < (int)inp__.labels_.size(); iat++) {
            auto label = inp__.labels_[iat];
            auto fname = inp__.atom_files_.at(label);
            for (size_t ia = 0; ia < inp__.coordinates_[iat].size(); ia++) {
                auto v = inp__.coordinates_[iat][ia];
                vector3d<double> p(v[0], v[1], v[2]);
                vector3d<double> f(v[3], v[4], v[5]);
                add_atom(label, p, f);
            }
        }

        set_lattice_vectors(inp__.a0_, inp__.a1_, inp__.a2_);
    }
}

void Unit_cell::update()
{
    PROFILE("sirius::Unit_cell::update");

    auto v0 = lattice_vector(0);
    auto v1 = lattice_vector(1);
    auto v2 = lattice_vector(2);

    double r = std::max(std::max(v0.length(), std::max(v1.length(), v2.length())),
                        parameters_.parameters_input().nn_radius_);

    find_nearest_neighbours(r);

    if (parameters_.full_potential()) {
        /* find new MT radii and initialize radial grid */
        if (parameters_.auto_rmt()) {
            auto rg = get_radial_grid_t(parameters_.settings().radial_grid_);
            std::vector<double> Rmt = find_mt_radii();
            for (int iat = 0; iat < num_atom_types(); iat++) {
                double r0 = atom_type(iat).radial_grid().first();

                atom_type(iat).set_radial_grid(rg.first, atom_type(iat).num_mt_points(), r0, Rmt[iat], rg.second);
            }
        }

        int ia, ja;
        if (check_mt_overlap(ia, ja)) {
            std::stringstream s;
            s << "overlaping muffin-tin spheres for atoms " << ia << "(" << atom(ia).type().symbol() << ")"
              << " and " << ja << "(" << atom(ja).type().symbol() << ")" << std::endl
              << "  radius of atom " << ia << " : " << atom(ia).mt_radius() << std::endl
              << "  radius of atom " << ja << " : " << atom(ja).mt_radius() << std::endl
              << "  distance : " << nearest_neighbours_[ia][1].distance << " " << nearest_neighbours_[ja][1].distance;
            TERMINATE(s);
        }

        min_mt_radius_ = 1e100;
        max_mt_radius_ = 0;
        for (int i = 0; i < num_atom_types(); i++) {
            min_mt_radius_ = std::min(min_mt_radius_, atom_type(i).mt_radius());
            max_mt_radius_ = std::max(max_mt_radius_, atom_type(i).mt_radius());
        }
    }

    get_symmetry();

    spl_num_atom_symmetry_classes_ = splindex<splindex_t::block>(num_atom_symmetry_classes(), comm_.size(), comm_.rank());

    volume_mt_ = 0.0;
    if (parameters_.full_potential()) {
        for (int ia = 0; ia < num_atoms(); ia++) {
            volume_mt_ += fourpi * std::pow(atom(ia).mt_radius(), 3) / 3.0;
        }
    }

    volume_it_ = omega() - volume_mt_;

    for (int iat = 0; iat < num_atom_types(); iat++) {
        int nat = atom_type(iat).num_atoms();
        if (nat > 0) {
            for (int i = 0; i < nat; i++) {
                int ia = atom_type(iat).atom_id(i);
                for (int x: {0, 1, 2}) {
                    atom_coord_[iat](i, x) = atom(ia).position()[x];
                }
            }
            if (parameters_.processing_unit() == device_t::GPU) {
                atom_coord_[iat].copy_to(memory_t::device);
            }
        }
    }
}

void Unit_cell::print_symmetry_info(int verbosity__) const
{
    if (symmetry_ != nullptr) {
        std::printf("\n");
        std::printf("space group number   : %i\n", symmetry_->spacegroup_number());
        std::printf("international symbol : %s\n", symmetry_->international_symbol().c_str());
        std::printf("Hall symbol          : %s\n", symmetry_->hall_symbol().c_str());
        std::printf("number of operations : %i\n", symmetry_->num_mag_sym());
        std::printf("transformation matrix : \n");
        auto tm = symmetry_->transformation_matrix();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                std::printf("%12.6f ", tm(i, j));
            }
            std::printf("\n");
        }
        std::printf("origin shift : \n");
        auto t = symmetry_->origin_shift();
        std::printf("%12.6f %12.6f %12.6f\n", t[0], t[1], t[2]);

        if (verbosity__ >= 2) {
            std::printf("symmetry operations  : \n");
            for (int isym = 0; isym < symmetry_->num_mag_sym(); isym++) {
                auto R = symmetry_->magnetic_group_symmetry(isym).spg_op.R;
                auto t = symmetry_->magnetic_group_symmetry(isym).spg_op.t;
                auto S = symmetry_->magnetic_group_symmetry(isym).spin_rotation;

                std::printf("isym : %i\n", isym);
                std::printf("R : ");
                for (int i = 0; i < 3; i++) {
                    if (i) {
                        std::printf("    ");
                    }
                    for (int j = 0; j < 3; j++) {
                        std::printf("%3i ", R(i, j));
                    }
                    std::printf("\n");
                }
                std::printf("T : ");
                for (int j = 0; j < 3; j++) {
                    std::printf("%8.4f ", t[j]);
                }
                std::printf("\n");
                std::printf("S : ");
                for (int i = 0; i < 3; i++) {
                    if (i) {
                        std::printf("    ");
                    }
                    for (int j = 0; j < 3; j++) {
                        std::printf("%8.4f ", S(i, j));
                    }
                    std::printf("\n");
                }
                std::printf("\n");
            }
        }
    }
}

void Unit_cell::set_lattice_vectors(matrix3d<double> lattice_vectors__)
{
    lattice_vectors_            = lattice_vectors__;
    inverse_lattice_vectors_    = inverse(lattice_vectors_);
    omega_                      = std::abs(lattice_vectors_.det());
    reciprocal_lattice_vectors_ = transpose(inverse(lattice_vectors_)) * twopi;
}

void Unit_cell::set_lattice_vectors(vector3d<double> a0__, vector3d<double> a1__, vector3d<double> a2__)
{
    matrix3d<double> lv;
    for (int x : {0, 1, 2}) {
        lv(x, 0) = a0__[x];
        lv(x, 1) = a1__[x];
        lv(x, 2) = a2__[x];
    }
    set_lattice_vectors(lv);
}

int Unit_cell::atom_id_by_position(vector3d<double> position__)
{
    for (int ia = 0; ia < num_atoms(); ia++) {
        auto vd = atom(ia).position() - position__;
        if (vd.length() < 1e-10) {
            return ia;
        }
    }
    return -1;
}

int Unit_cell::next_atom_type_id(std::string label__)
{
    /* check if the label was already added */
    if (atom_type_id_map_.count(label__) != 0) {
        std::stringstream s;
        s << "atom type with label " << label__ << " is already in list";
        TERMINATE(s);
    }
    /* take text id */
    atom_type_id_map_[label__] = static_cast<int>(atom_types_.size());
    return atom_type_id_map_[label__];
}

void Unit_cell::init_paw()
{
    for (int ia = 0; ia < num_atoms(); ia++) {
        if (atom(ia).type().is_paw()) {
            paw_atom_index_.push_back(ia);
        }
    }

    spl_num_paw_atoms_ = splindex<splindex_t::block>(num_paw_atoms(), comm_.size(), comm_.rank());
}

std::pair<int, std::vector<int>> Unit_cell::num_wf_with_U() const // TODO: remove in future
{
    std::vector<int> offs(this->num_atoms(), -1);
    int counter{0};

    /* we loop over atoms to check which atom has hubbard orbitals and then
       compute the number of hubbard orbitals associated to it */
    for (auto ia = 0; ia < this->num_atoms(); ia++) {
        auto& atom = this->atom(ia);
        if (atom.type().hubbard_correction()) {
            offs[ia] = counter;
            int fact{1};
            /* there is a factor two when the pseudo-potential has no SO but
               we do full non colinear magnetism. Note that we can consider
               now multiple orbitals calculations. The API still does not
               support it */
            if ((this->parameters().num_mag_dims() == 3) && (!atom.type().spin_orbit_coupling())) {
                fact = 2;
            }
            counter += fact * atom.type().hubbard_indexb_wfc().size();
        }
    }
    return std::make_pair(counter, offs);
}

std::pair<int, std::vector<int>> Unit_cell::num_hubbard_wf() const
{
    std::vector<int> offs(this->num_atoms(), -1);
    int counter{0};

    /* we loop over atoms to check which atom has hubbard orbitals and then
       compute the number of hubbard orbitals associated to it */
    for (auto ia = 0; ia < this->num_atoms(); ia++) {
        auto& atom = this->atom(ia);
        if (atom.type().hubbard_correction()) {
            offs[ia] = counter;
            counter += atom.type().indexb_hub().size();
        }
    }
    return std::make_pair(counter, offs);
}

int Unit_cell::num_ps_atomic_wf() const
{
    /* TODO: in spinorbit case this function will work only when pairs of spinor components are present. */
    int N{0};
    /* get the total number of atomic-centered orbitals */
    for (int iat = 0; iat < this->num_atom_types(); iat++) {
        int n{0};
        for (int i = 0; i < this->atom_type(iat).indexr_wfs().size(); i++) {
            /* number of m-components is 2l + 1 */
            n += (2 * std::abs(std::get<1>(atom_type(iat).ps_atomic_wf(i))) + 1);
        }
        N += atom_type(iat).num_atoms() * n;
    }
    return N;
}

} // namespace sirius
