/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file unit_cell.cpp
 *
 *  \brief Contains implementation of sirius::Unit_cell class.
 */

#include <iomanip>
#include "unit_cell.hpp"
#include "symmetry/crystal_symmetry.hpp"
#include "core/ostream_tools.hpp"

namespace sirius {

Unit_cell::Unit_cell(Simulation_parameters const& parameters__, mpi::Communicator const& comm__)
    : parameters_(parameters__)
    , comm_(comm__)
{
}

Unit_cell::~Unit_cell() = default;

std::vector<double>
Unit_cell::find_mt_radii(int auto_rmt__, bool inflate__)
{
    if (nearest_neighbours_.size() == 0) {
        RTE_THROW("array of nearest neighbours is empty");
    }

    std::vector<double> Rmt(num_atom_types(), 1e10);

    if (auto_rmt__ == 1) {
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

    if (auto_rmt__ == 2) {
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
    if (inflate__) {
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

        std::vector<double> Rmt_infl(num_atom_types(), 1e10);

        for (int ia = 0; ia < num_atoms(); ia++) {
            int id1 = atom(ia).type_id();
            if (nearest_neighbours_[ia].size() > 1) {
                int ja      = nearest_neighbours_[ia][1].atom_id;
                int id2     = atom(ja).type_id();
                double dist = nearest_neighbours_[ia][1].distance;

                if (scale_Rmt[id1] && !scale_Rmt[id2]) {
                    Rmt_infl[id1] = std::min(Rmt_infl[id1], std::min(parameters_.rmt_max(), 0.95 * (dist - Rmt[id2])));
                } else {
                    Rmt_infl[id1] = Rmt[id1];
                }
            }
        }
        for (int iat = 0; iat < num_atom_types(); iat++) {
            Rmt[iat] = Rmt_infl[iat];
        }
    }

    for (int i = 0; i < num_atom_types(); i++) {
        if (Rmt[i] < 0.3) {
            std::stringstream s;
            s << "muffin-tin radius for atom type " << i << " (" << atom_type(i).label()
              << ") is too small: " << Rmt[i];
            RTE_THROW(s);
        }
    }

    return Rmt;
}

bool
Unit_cell::check_mt_overlap(int& ia__, int& ja__)
{
    if (num_atoms() != 0 && nearest_neighbours_.size() == 0) {
        RTE_THROW("array of nearest neighbours is empty");
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

void
Unit_cell::print_info(std::ostream& out__, int verbosity__) const
{
    out__ << "lattice vectors" << std::endl;
    for (int i = 0; i < 3; i++) {
        out__ << "  a" << i + 1 << " : ";
        for (int x : {0, 1, 2}) {
            out__ << ffmt(18, 10) << lattice_vectors_(x, i);
        }
        out__ << std::endl;
    }
    out__ << "reciprocal lattice vectors" << std::endl;
    for (int i = 0; i < 3; i++) {
        out__ << "  b" << i + 1 << " : ";
        for (int x : {0, 1, 2}) {
            out__ << ffmt(18, 10) << reciprocal_lattice_vectors_(x, i);
        }
        out__ << std::endl;
    }
    out__ << std::endl
          << "unit cell volume : " << ffmt(18, 8) << omega() << " [a.u.^3]" << std::endl
          << "1/sqrt(omega)    : " << ffmt(18, 8) << 1.0 / sqrt(omega()) << std::endl
          << "MT volume        : " << ffmt(18, 8) << volume_mt() << " (" << ffmt(5, 2) << volume_mt() * 100 / omega()
          << "%)" << std::endl
          << "IT volume        : " << ffmt(18, 8) << volume_it() << " (" << ffmt(5, 2) << volume_it() * 100 / omega()
          << "%)" << std::endl
          << std::endl
          << "number of atom types : " << num_atom_types() << std::endl;
    for (int i = 0; i < num_atom_types(); i++) {
        int id = atom_type(i).id();
        out__ << "type id : " << id << " symbol : " << std::setw(2) << atom_type(i).symbol()
              << " mt_radius : " << ffmt(10, 6) << atom_type(i).mt_radius()
              << " num_atoms : " << atom_type(i).num_atoms() << std::endl;
    }

    out__ << "total number of atoms : " << num_atoms() << std::endl
          << "number of symmetry classes : " << num_atom_symmetry_classes() << std::endl;
    if (!parameters_.full_potential()) {
        out__ << "number of PAW atoms : " << num_paw_atoms() << std::endl;
        if (num_paw_atoms() != 0) {
            out__ << "PAW atoms :";
            for (auto ia : paw_atom_index_) {
                out__ << " " << ia;
            }
            out__ << std::endl;
        }
    }
    if (verbosity__ >= 2) {
        out__ << std::endl
              << "atom id  type id  class id             position                      vector_field" << std::endl
              << hbar(90, '-') << std::endl;
        for (int i = 0; i < num_atoms(); i++) {
            auto pos = atom(i).position();
            auto vf  = atom(i).vector_field();
            out__ << std::setw(6) << i << std::setw(9) << atom(i).type_id() << std::setw(9)
                  << atom(i).symmetry_class_id() << "   ";
            for (int x : {0, 1, 2}) {
                out__ << ffmt(10, 5) << pos[x];
            }
            out__ << "   ";
            for (int x : {0, 1, 2}) {
                out__ << ffmt(10, 5) << vf[x];
            }
            out__ << std::endl;
        }
        out__ << std::endl << "atom id         position (Cartesian, a.u.)" << std::endl << hbar(45, '-') << std::endl;
        for (int i = 0; i < num_atoms(); i++) {
            auto pos = atom(i).position();
            auto vc  = get_cartesian_coordinates(pos);
            out__ << std::setw(6) << i << "   ";
            for (int x : {0, 1, 2}) {
                out__ << ffmt(12, 6) << vc[x];
            }
            out__ << std::endl;
        }
        out__ << std::endl;
        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            out__ << "class id : " << ic << " atom id : ";
            for (int i = 0; i < atom_symmetry_class(ic).num_atoms(); i++) {
                out__ << atom_symmetry_class(ic).atom_id(i) << " ";
            }
            out__ << std::endl;
        }
    }
    out__ << std::endl << "minimum bond length: " << ffmt(12, 6) << min_bond_length() << std::endl;
    if (!parameters_.full_potential()) {
        out__ << std::endl << "total number of pseudo wave-functions: " << this->num_ps_atomic_wf().first << std::endl;
    }
    out__ << std::endl;
}

void
Unit_cell::print_geometry_info(std::ostream& out__, int verbosity__) const
{
    if (verbosity__ >= 1) {
        out__ << std::endl << "lattice vectors" << std::endl;
        for (int i = 0; i < 3; i++) {
            out__ << "  a" << i + 1 << " : ";
            for (int x : {0, 1, 2}) {
                out__ << ffmt(18, 10) << lattice_vectors_(x, i);
            }
            out__ << std::endl;
        }
        out__ << std::endl << "unit cell volume : " << ffmt(18, 8) << omega() << " [a.u.^3]" << std::endl;
    }

    if (verbosity__ >= 2) {
        out__ << std::endl
              << "atom id  type id  class id             position                      vector_field" << std::endl
              << hbar(90, '-') << std::endl;
        for (int i = 0; i < num_atoms(); i++) {
            auto pos = atom(i).position();
            auto vf  = atom(i).vector_field();
            out__ << std::setw(6) << i << std::setw(9) << atom(i).type_id() << std::setw(9)
                  << atom(i).symmetry_class_id() << "   ";
            for (int x : {0, 1, 2}) {
                out__ << ffmt(10, 5) << pos[x];
            }
            out__ << "   ";
            for (int x : {0, 1, 2}) {
                out__ << ffmt(10, 5) << vf[x];
            }
            out__ << std::endl;
        }
        out__ << std::endl << "atom id         position (Cartesian, a.u.)" << std::endl << hbar(45, '-') << std::endl;
        for (int i = 0; i < num_atoms(); i++) {
            auto pos = atom(i).position();
            auto vc  = get_cartesian_coordinates(pos);
            out__ << std::setw(6) << i << "   ";
            for (int x : {0, 1, 2}) {
                out__ << ffmt(12, 6) << vc[x];
            }
            out__ << std::endl;
        }
    }
    if (verbosity__ >= 1) {
        out__ << std::endl << "minimum bond length: " << ffmt(12, 6) << min_bond_length() << std::endl;
    }
}

unit_cell_parameters_descriptor
Unit_cell::unit_cell_parameters()
{
    unit_cell_parameters_descriptor d;

    r3::vector<double> v0(lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0));
    r3::vector<double> v1(lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1));
    r3::vector<double> v2(lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2));

    d.a = v0.length();
    d.b = v1.length();
    d.c = v2.length();

    d.alpha = std::acos(dot(v1, v2) / d.b / d.c) * 180 / pi;
    d.beta  = std::acos(dot(v0, v2) / d.a / d.c) * 180 / pi;
    d.gamma = std::acos(dot(v0, v1) / d.a / d.b) * 180 / pi;

    return d;
}

void
Unit_cell::write_cif()
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

json
Unit_cell::serialize(bool cart_pos__) const
{
    json dict;

    dict["lattice_vectors"]       = {{lattice_vectors_(0, 0), lattice_vectors_(1, 0), lattice_vectors_(2, 0)},
                                     {lattice_vectors_(0, 1), lattice_vectors_(1, 1), lattice_vectors_(2, 1)},
                                     {lattice_vectors_(0, 2), lattice_vectors_(1, 2), lattice_vectors_(2, 2)}};
    dict["lattice_vectors_scale"] = 1.0;
    if (cart_pos__) {
        dict["atom_coordinate_units"] = "au";
    } else {
        dict["atom_coordinate_units"] = "lattice";
    }
    dict["atom_types"] = json::array();
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
                v = dot(lattice_vectors_, v);
            }
            dict["atoms"][atom_type(iat).label()].push_back({v[0], v[1], v[2]});
        }
    }
    return dict;
}

void
Unit_cell::find_nearest_neighbours(double cluster_radius)
{
    PROFILE("sirius::Unit_cell::find_nearest_neighbours");

    auto max_frac_coord = find_translations(cluster_radius, lattice_vectors_);

    nearest_neighbours_.clear();
    nearest_neighbours_.resize(num_atoms());

    #pragma omp parallel for default(shared)
    for (int ia = 0; ia < num_atoms(); ia++) {

        std::vector<nearest_neighbour_descriptor> nn;

        std::vector<std::pair<double, int>> nn_sort;

        for (int i0 = -max_frac_coord[0]; i0 <= max_frac_coord[0]; i0++) {
            for (int i1 = -max_frac_coord[1]; i1 <= max_frac_coord[1]; i1++) {
                for (int i2 = -max_frac_coord[2]; i2 <= max_frac_coord[2]; i2++) {
                    nearest_neighbour_descriptor nnd;
                    nnd.translation[0] = i0;
                    nnd.translation[1] = i1;
                    nnd.translation[2] = i2;

                    for (int ja = 0; ja < num_atoms(); ja++) {
                        auto v1 = atom(ja).position() + r3::vector<int>(nnd.translation) - atom(ia).position();
                        auto rc = get_cartesian_coordinates(v1);

                        nnd.atom_id  = ja;
                        nnd.rc       = rc;
                        nnd.distance = rc.length();

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
}

void
Unit_cell::print_nearest_neighbours(std::ostream& out__) const
{
    out__ << "Nearest neighbors" << std::endl << hbar(17, '-') << std::endl;
    for (int ia = 0; ia < num_atoms(); ia++) {
        out__ << "Central atom: " << atom(ia).type().symbol() << "(" << ia << ")" << std::endl
              << hbar(80, '-') << std::endl;
        out__ << "atom (ia)        D [a.u.]        T                     r_local" << std::endl;
        for (int i = 0; i < (int)nearest_neighbours_[ia].size(); i++) {
            int ja         = nearest_neighbours_[ia][i].atom_id;
            auto ja_symbol = atom(ja).type().symbol();
            auto ja_d      = nearest_neighbours_[ia][i].distance;
            auto T         = nearest_neighbours_[ia][i].translation;
            auto r_loc     = nearest_neighbours_[ia][i].rc;
            out__ << std::setw(4) << ja_symbol << " (" << std::setw(5) << ja << ")" << std::setw(12)
                  << std::setprecision(5) << ja_d << std::setw(5) << T[0] << std::setw(5) << T[1] << std::setw(5)
                  << T[2] << std::setw(13) << std::setprecision(5) << std::fixed << r_loc[0] << std::setw(10)
                  << std::setprecision(5) << std::fixed << r_loc[1] << std::setw(10) << std::setprecision(5)
                  << std::fixed << r_loc[2] << std::endl;
        }
    }
    out__ << std::endl;
}

bool
Unit_cell::is_point_in_mt(r3::vector<double> vc, int& ja, int& jr, double& dr, double tp[2]) const
{
    /* reduce coordinates to the primitive unit cell */
    auto vr = reduce_coordinates(get_fractional_coordinates(vc));

    for (int ia = 0; ia < num_atoms(); ia++) {
        for (int i0 = -1; i0 <= 1; i0++) {
            for (int i1 = -1; i1 <= 1; i1++) {
                for (int i2 = -1; i2 <= 1; i2++) {
                    /* atom position */
                    r3::vector<double> posf = r3::vector<double>(i0, i1, i2) + atom(ia).position();

                    /* vector connecting center of atom and reduced point */
                    r3::vector<double> vf = vr.first - posf;

                    /* convert to spherical coordinates */
                    auto vs = r3::spherical_coordinates(get_cartesian_coordinates(vf));

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

void
Unit_cell::generate_radial_functions(std::ostream& out__)
{
    PROFILE("sirius::Unit_cell::generate_radial_functions");

    for (auto it : spl_num_atom_symmetry_classes()) {
        atom_symmetry_class(it.i).generate_radial_functions(parameters_.valence_relativity());
    }

    for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
        int rank = spl_num_atom_symmetry_classes().location(typename atom_symmetry_class_index_t::global(ic)).ib;
        atom_symmetry_class(ic).sync_radial_functions(comm_, rank);
    }

    if (parameters_.verbosity() >= 2) {
        mpi::pstdout pout(comm_);
        if (comm_.rank() == 0) {
            pout << std::endl << "Linearization energies" << std::endl;
        }

        for (auto it : spl_num_atom_symmetry_classes()) {
            atom_symmetry_class(it.i).write_enu(pout);
        }
        RTE_OUT(out__) << pout.flush(0);
    }
    if (parameters_.verbosity() >= 3) {
        std::stringstream s;
        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            s << "Atom symmetry class : " << ic << std::endl;
            for (int l = 0; l < this->lmax_apw(); l++) {
                for (int o = 0; o < atom_symmetry_class(ic).atom_type().aw_order(l); o++) {
                    s << "l = " << l << ", o = " << o << ", deriv =";
                    for (int m = 0; m <= 2; m++) {
                        s << " " << atom_symmetry_class(ic).aw_surface_deriv(l, o, m);
                    }
                    s << std::endl;
                }
            }
        }
        RTE_OUT(out__) << s.str();
    }
    if (parameters_.cfg().control().save_rf() && comm_.rank() == 0) {
        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            atom_symmetry_class(ic).dump_lo();
        }
    }
}

void
Unit_cell::generate_radial_integrals()
{
    PROFILE("sirius::Unit_cell::generate_radial_integrals");

    try {
        for (auto it : spl_num_atom_symmetry_classes()) {
            atom_symmetry_class(it.i).generate_radial_integrals(parameters_.valence_relativity());
        }

        for (int ic = 0; ic < num_atom_symmetry_classes(); ic++) {
            int rank = spl_num_atom_symmetry_classes().location(typename atom_symmetry_class_index_t::global(ic)).ib;
            atom_symmetry_class(ic).sync_radial_integrals(comm_, rank);
        }
    } catch (std::exception const& e) {
        std::stringstream s;
        s << e.what() << std::endl;
        s << "Error in generating atom_symmetry_class radial integrals";
        RTE_THROW(s);
    }

    try {
        for (auto it : spl_num_atoms_) {
            atom(it.i).generate_radial_integrals(parameters_.processing_unit(), mpi::Communicator::self());
        }

        for (int ia = 0; ia < num_atoms(); ia++) {
            int rank = spl_num_atoms().location(typename atom_index_t::global(ia)).ib;
            atom(ia).sync_radial_integrals(comm_, rank);
        }
    } catch (std::exception const& e) {
        std::stringstream s;
        s << e.what() << std::endl;
        s << "Error in generating atom radial integrals";
        RTE_THROW(s);
    }
}

std::string
Unit_cell::chemical_formula()
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

Atom_type&
Unit_cell::add_atom_type(const std::string label__, const std::string file_name__)
{
    int id = next_atom_type_id(label__);
    atom_types_.push_back(std::make_shared<Atom_type>(parameters_, id, label__, file_name__));
    return *atom_types_.back();
}

void
Unit_cell::add_atom(const std::string label, r3::vector<double> position, r3::vector<double> vector_field)
{
    if (atom_type_id_map_.count(label) == 0) {
        std::stringstream s;
        s << "atom type with label " << label << " is not found";
        RTE_THROW(s);
    }
    if (atom_id_by_position(position) >= 0) {
        std::stringstream s;
        s << "atom with the same position is already in list" << std::endl
          << "  position : " << position[0] << " " << position[1] << " " << position[2];
        RTE_THROW(s);
    }

    atoms_.push_back(std::make_shared<Atom>(atom_type(label), position, vector_field));
    atom_type(label).add_atom_id(static_cast<int>(atoms_.size()) - 1);
}

double
Unit_cell::min_bond_length() const
{
    double len{1e10};

    for (int ia = 0; ia < num_atoms(); ia++) {
        if (nearest_neighbours_[ia].size() > 1) {
            len = std::min(len, nearest_neighbours_[ia][1].distance);
        }
    }
    return len;
}

void
Unit_cell::initialize()
{
    PROFILE("sirius::Unit_cell::initialize");

    /* split number of atom between all MPI ranks */
    spl_num_atoms_ = splindex_block<atom_index_t>(num_atoms(), n_blocks(comm_.size()), block_id(comm_.rank()));

    /* initialize atom types */
    for (int iat = 0; iat < num_atom_types(); iat++) {
        atom_type(iat).init();
    }

    /* initialize atoms */
    for (int ia = 0; ia < num_atoms(); ia++) {
        atom(ia).init();
    }

    init_paw();

    for (int iat = 0; iat < num_atom_types(); iat++) {
        int nat = atom_type(iat).num_atoms();
        if (nat > 0) {
            atom_coord_.push_back(mdarray<double, 2>({nat, 3}));
            if (parameters_.processing_unit() == device_t::GPU) {
                atom_coord_.back().allocate(memory_t::device);
            }
        } else {
            atom_coord_.push_back(mdarray<double, 2>());
        }
    }

    std::vector<int> offs(this->num_atoms(), -1);
    int counter{0};

    /* we loop over atoms to check which atom has hubbard orbitals and then
       compute the number of Hubbard orbitals associated to it */
    for (auto ia = 0; ia < this->num_atoms(); ia++) {
        auto& atom = this->atom(ia);
        if (atom.type().hubbard_correction()) {
            offs[ia] = counter;
            counter += atom.type().indexb_hub().size();
        }
    }
    num_hubbard_wf_ = std::make_pair(counter, offs);

    update();

    //== write_cif();

    //== if (comm().rank() == 0) {
    //==     std::ofstream ofs(std::string("unit_cell.json"), std::ofstream::out | std::ofstream::trunc);
    //==     ofs << serialize().dump(4);
    //== }
}

void
Unit_cell::get_symmetry()
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

    mdarray<double, 2> positions({3, num_atoms()});
    mdarray<double, 2> spins({3, num_atoms()});
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

    symmetry_ = std::unique_ptr<Crystal_symmetry>(new Crystal_symmetry(
            lattice_vectors_, num_atoms(), num_atom_types(), types, positions, spins, parameters_.so_correction(),
            parameters_.spglib_tolerance(), parameters_.use_symmetry()));

    int atom_class_id{-1};
    std::vector<int> asc(num_atoms(), -1);
    for (int i = 0; i < num_atoms(); i++) {
        /* if symmetry class is not assigned to this atom */
        if (asc[i] == -1) {
            /* take next id */
            atom_class_id++;
            atom_symmetry_classes_.push_back(
                    std::shared_ptr<Atom_symmetry_class>(new Atom_symmetry_class(atom_class_id, atom(i).type())));

            /* scan all atoms */
            for (int j = 0; j < num_atoms(); j++) {
                bool is_equal = (equivalent_atoms_.size())
                                        ? (equivalent_atoms_[j] == equivalent_atoms_[i])
                                        : (symmetry_->atom_symmetry_class(j) == symmetry_->atom_symmetry_class(i));
                /* assign new class id for all equivalent atoms */
                if (is_equal) {
                    asc[j] = atom_class_id;
                    atom_symmetry_classes_.back()->add_atom_id(j);
                }
            }
        }
    }

    for (auto e : atom_symmetry_classes_) {
        for (int i = 0; i < e->num_atoms(); i++) {
            int ia = e->atom_id(i);
            atom(ia).set_symmetry_class(e);
        }
    }

    assert(num_atom_symmetry_classes() != 0);
}

void
Unit_cell::import(config_t::unit_cell_t const& inp__)
{
    auto lv = r3::matrix<double>(inp__.lattice_vectors());
    lv *= inp__.lattice_vectors_scale();
    set_lattice_vectors(r3::vector<double>(lv(0, 0), lv(0, 1), lv(0, 2)),
                        r3::vector<double>(lv(1, 0), lv(1, 1), lv(1, 2)),
                        r3::vector<double>(lv(2, 0), lv(2, 1), lv(2, 2)));
    /* here lv are copied from the JSON dictionary as three row vectors; however
       in the code the lattice vectors are stored as three column vectors, so
       transposition is needed here */
    auto ilvT = transpose(inverse(lv));

    auto units = inp__.atom_coordinate_units();

    /* first, load all types */
    for (auto label : inp__.atom_types()) {
        auto fname = inp__.atom_files(label);
        add_atom_type(label, fname);
    }
    for (auto label : inp__.atom_types()) {
        for (auto v : inp__.atoms(label)) {
            r3::vector<double> p(v[0], v[1], v[2]);
            r3::vector<double> f;
            if (v.size() == 6) {
                f = r3::vector<double>(v[3], v[4], v[5]);
            }
            /* convert to atomic units */
            if (units == "A") {
                for (int x : {0, 1, 2}) {
                    p[x] /= bohr_radius;
                }
            }
            /* convert from Cartesian to lattice coordinates */
            if (units == "au" || units == "A") {
                p       = dot(ilvT, p);
                auto rc = reduce_coordinates(p);
                for (int x : {0, 1, 2}) {
                    p[x] = rc.first[x];
                }
            }
            add_atom(label, p, f);
        }
    }
}

void
Unit_cell::update()
{
    PROFILE("sirius::Unit_cell::update");

    auto v0 = lattice_vector(0);
    auto v1 = lattice_vector(1);
    auto v2 = lattice_vector(2);

    double r{0};
    if (parameters_.cfg().parameters().nn_radius() < 0) {
        r = std::max(v0.length(), std::max(v1.length(), v2.length()));
    } else {
        r = parameters_.cfg().parameters().nn_radius();
    }

    find_nearest_neighbours(r);

    if (parameters_.full_potential()) {
        /* find new MT radii and initialize radial grid */
        if (parameters_.auto_rmt()) {
            auto rg  = get_radial_grid_t(parameters_.cfg().settings().radial_grid());
            auto Rmt = find_mt_radii(parameters_.auto_rmt(), true);
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
            RTE_THROW(s);
        }
    }

    get_symmetry();

    spl_num_atom_symmetry_classes_ = splindex_block<atom_symmetry_class_index_t>(
            num_atom_symmetry_classes(), n_blocks(comm_.size()), block_id(comm_.rank()));

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
                for (int x : {0, 1, 2}) {
                    atom_coord_[iat](i, x) = atom(ia).position()[x];
                }
            }
            if (parameters_.processing_unit() == device_t::GPU) {
                atom_coord_[iat].copy_to(memory_t::device);
            }
        }
    }
}

void
Unit_cell::set_lattice_vectors(r3::matrix<double> lattice_vectors__)
{
    lattice_vectors_            = lattice_vectors__;
    inverse_lattice_vectors_    = inverse(lattice_vectors_);
    omega_                      = std::abs(lattice_vectors_.det());
    reciprocal_lattice_vectors_ = transpose(inverse(lattice_vectors_)) * twopi;
}

void
Unit_cell::set_lattice_vectors(r3::vector<double> a0__, r3::vector<double> a1__, r3::vector<double> a2__)
{
    r3::matrix<double> lv;
    for (int x : {0, 1, 2}) {
        lv(x, 0) = a0__[x];
        lv(x, 1) = a1__[x];
        lv(x, 2) = a2__[x];
    }
    set_lattice_vectors(lv);
}

int
Unit_cell::atom_id_by_position(r3::vector<double> position__)
{
    for (int ia = 0; ia < num_atoms(); ia++) {
        auto vd = atom(ia).position() - position__;
        if (vd.length() < 1e-10) {
            return ia;
        }
    }
    return -1;
}

int
Unit_cell::next_atom_type_id(std::string label__)
{
    /* check if the label was already added */
    if (atom_type_id_map_.count(label__) != 0) {
        std::stringstream s;
        s << "atom type with label " << label__ << " is already in list";
        RTE_THROW(s);
    }
    /* take text id */
    atom_type_id_map_[label__] = static_cast<int>(atom_types_.size());
    return atom_type_id_map_[label__];
}

void
Unit_cell::init_paw()
{
    for (int ia = 0; ia < num_atoms(); ia++) {
        if (atom(ia).type().is_paw()) {
            paw_atom_index_.push_back(ia);
        }
    }

    spl_num_paw_atoms_ =
            splindex_block<paw_atom_index_t>(num_paw_atoms(), n_blocks(comm_.size()), block_id(comm_.rank()));
}

std::pair<int, std::vector<int>>
Unit_cell::num_ps_atomic_wf() const
{
    std::vector<int> offs(this->num_atoms(), -1);
    int counter{0};
    for (auto ia = 0; ia < this->num_atoms(); ia++) {
        auto& atom = this->atom(ia);
        offs[ia]   = counter;
        counter += atom.type().indexb_wfs().size();
    }
    return std::make_pair(counter, offs);
}

} // namespace sirius
