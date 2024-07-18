/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
extern "C" {
#include <spglib.h>
}

using namespace sirius;

void
create_supercell(cmd_args const& args__)
{
    r3::matrix<int> scell;
    std::stringstream s(args__.value<std::string>("supercell"));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            s >> scell(j, i);
        }
    }

    std::cout << std::endl;
    std::cout << "supercell vectors (lattice coordinates) : " << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "A" << i << " : ";
        for (int j = 0; j < 3; j++) {
            std::cout << scell(j, i) << " ";
        }
        std::cout << std::endl;
    }

    Simulation_context ctx("sirius.json", mpi::Communicator::self());

    auto scell_lattice_vectors = dot(ctx.unit_cell().lattice_vectors(), r3::matrix<double>(scell));

    std::cout << "supercell vectors (Cartesian coordinates) : " << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "A" << i << " : ";
        for (int x = 0; x < 3; x++) {
            std::cout << scell_lattice_vectors(x, i) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "volume ratio : " << std::abs(r3::matrix<int>(scell).det()) << std::endl;

    Simulation_context ctx_sc(mpi::Communicator::self());

    r3::vector<double> a0, a1, a2;
    for (int x = 0; x < 3; x++) {
        a0[x] = scell_lattice_vectors(x, 0);
        a1[x] = scell_lattice_vectors(x, 1);
        a2[x] = scell_lattice_vectors(x, 2);
    }
    ctx_sc.unit_cell().set_lattice_vectors(a0, a1, a2);

    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        auto label = ctx.unit_cell().atom_type(iat).label();
        ctx_sc.unit_cell().add_atom_type(label, "");
    }

    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        auto label = ctx.unit_cell().atom_type(iat).label();

        for (int i = 0; i < ctx.unit_cell().atom_type(iat).num_atoms(); i++) {
            int ia  = ctx.unit_cell().atom_type(iat).atom_id(i);
            auto va = ctx.unit_cell().atom(ia).position();

            for (int i0 = -10; i0 <= 10; i0++) {
                for (int i1 = -10; i1 <= 10; i1++) {
                    for (int i2 = -10; i2 <= 10; i2++) {
                        r3::vector<double> T(i0, i1, i2);
                        r3::vector<double> vc = ctx.unit_cell().get_cartesian_coordinates(va + T);
                        r3::vector<double> vf = ctx_sc.unit_cell().get_fractional_coordinates(vc);

                        auto vr = reduce_coordinates(vf);
                        for (int x : {0, 1, 2}) {
                            vr.first[x] = round(vr.first[x], 10);
                        }
                        bool add_atom = (ctx_sc.unit_cell().atom_id_by_position(vr.first) == -1);
                        //==if (add_atom && iat == 2)
                        //=={
                        //==    double r = type_wrapper<double>::random();
                        //==    if (r < 0.99) add_atom = false;
                        //==}
                        std::vector<double> u({vr.first[0], vr.first[1], vr.first[2]});
                        if (add_atom) {
                            ctx_sc.unit_cell().add_atom(label, u);
                        }
                    }
                }
            }
        }
    }
    std::printf("number of atoms in the supercell: %i\n", ctx_sc.unit_cell().num_atoms());

    // ctx_sc.unit_cell().get_symmetry();
    // ctx_sc.unit_cell().print_info(4);
    ctx_sc.unit_cell().write_cif();
    json dict;
    dict["unit_cell"] = ctx_sc.unit_cell().serialize();
    if (mpi::Communicator::world().rank() == 0) {
        std::ofstream ofs("unit_cell.json", std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    }
}

void
find_primitive()
{
    Simulation_context ctx("sirius.json", mpi::Communicator::self());

    double lattice[3][3];
    for (int i : {0, 1, 2}) {
        for (int j : {0, 1, 2}) {
            lattice[i][j] = ctx.unit_cell().lattice_vector(j)[i];
        }
    }
    mdarray<double, 2> positions({3, 4 * ctx.unit_cell().num_atoms()});
    mdarray<int, 1> types({4 * ctx.unit_cell().num_atoms()});
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        for (int x : {0, 1, 2}) {
            positions(x, ia) = ctx.unit_cell().atom(ia).position()[x];
            types(ia)        = ctx.unit_cell().atom(ia).type_id();
        }
    }

    std::printf("original number of atoms: %i\n", ctx.unit_cell().num_atoms());

    int nat_new = spg_standardize_cell(lattice, (double(*)[3]) & positions(0, 0), &types[0],
                                       ctx.unit_cell().num_atoms(), 1, 0, ctx.cfg().control().spglib_tolerance());
    std::printf("new number of atoms: %i\n", nat_new);

    Simulation_context ctx_new(mpi::Communicator::self());

    r3::vector<double> a0, a1, a2;
    for (int x = 0; x < 3; x++) {
        a0[x] = lattice[x][0];
        a1[x] = lattice[x][1];
        a2[x] = lattice[x][2];
    }
    ctx_new.unit_cell().set_lattice_vectors(a0, a1, a2);

    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        auto label = ctx.unit_cell().atom_type(iat).label();
        ctx_new.unit_cell().add_atom_type(label, "");
    }

    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        auto label = ctx.unit_cell().atom_type(iat).label();

        for (int i = 0; i < nat_new; i++) {
            if (types[i] == iat) {
                r3::vector<double> p(positions(0, i), positions(1, i), positions(2, i));
                std::vector<double> u({p[0], p[1], p[2]});
                ctx_new.unit_cell().add_atom(label, u);
            }
        }
    }

    json dict;
    dict["unit_cell"] = ctx_new.unit_cell().serialize();
    if (mpi::Communicator::world().rank() == 0) {
        std::ofstream ofs("unit_cell.json", std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    }
}

void
create_qe_input(cmd_args const& args__)
{
    Simulation_context ctx(args__.value<std::string>("input", "sirius.json"), mpi::Communicator::self());

    FILE* fout = fopen("pw.in", "w");
    fprintf(fout, "&control\n"
                  "calculation=\'scf\',\n"
                  "restart_mode=\'from_scratch\',\n"
                  "pseudo_dir = \'./\',\n"
                  "outdir=\'./\',\n"
                  "prefix = \'scf_\',\n"
                  "tstress = false,\n"
                  "tprnfor = false,\n"
                  "verbosity = \'high\',\n"
                  "disk_io = \'none\',\n"
                  "wf_collect = false\n"
                  "/\n");

    fprintf(fout, "&system\nibrav=0, celldm(1)=1, ecutwfc=40, ecutrho = 300,\noccupations = \'smearing\', smearing = "
                  "\'gauss\', degauss = 0.001,\n");
    fprintf(fout, "nat=%i, ntyp=%i\n/\n", ctx.unit_cell().num_atoms(), ctx.unit_cell().num_atom_types());
    fprintf(fout, "&electrons\nconv_thr =  1.0d-11,\nmixing_beta = 0.7,\nelectron_maxstep = 100\n/\n");
    fprintf(fout, "&IONS\n"
                  "ion_dynamics=\'bfgs\',\n"
                  "/\n"
                  "&CELL\n"
                  "cell_dynamics=\'bfgs\',\n"
                  "/\n");

    fprintf(fout, "ATOMIC_SPECIES\n");
    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        fprintf(fout, "%s 0.0 pp.UPF\n", ctx.unit_cell().atom_type(iat).label().c_str());
    }

    fprintf(fout, "CELL_PARAMETERS\n");
    for (int i = 0; i < 3; i++) {
        for (int x = 0; x < 3; x++) {
            fprintf(fout, "%18.8f", ctx.unit_cell().lattice_vector(i)[x]);
        }
        fprintf(fout, "\n");
    }
    fprintf(fout, "ATOMIC_POSITIONS (crystal)\n");
    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        for (int ia = 0; ia < ctx.unit_cell().atom_type(iat).num_atoms(); ia++) {
            int id = ctx.unit_cell().atom_type(iat).atom_id(ia);
            fprintf(fout, "%s  %18.8f %18.8f %18.8f\n", ctx.unit_cell().atom_type(iat).label().c_str(),
                    ctx.unit_cell().atom(id).position()[0], ctx.unit_cell().atom(id).position()[1],
                    ctx.unit_cell().atom(id).position()[2]);
        }
    }
    fprintf(fout, "K_POINTS (automatic)\n2 2 2  0 0 0\n");

    fclose(fout);
}

void
create_exciting_input(cmd_args const& args__)
{
    Simulation_context ctx(args__.value<std::string>("input", "sirius.json"), mpi::Communicator::self());

    FILE* fout = fopen("input.xml", "w");

    fprintf(fout, "<input>\n");
    fprintf(fout, "  <title> converted from SIRIUS json input </title>\n");
    fprintf(fout, "  <structure speciespath=\"./\" autormt=\"false\">\n");
    fprintf(fout, "    <crystal scale=\"1\">\n");
    for (int i = 0; i < 3; i++) {
        auto v = ctx.unit_cell().lattice_vector(i);
        fprintf(fout, "      <basevect> %18.12f %18.12f %18.12f </basevect>\n", v[0], v[1], v[2]);
    }
    fprintf(fout, "    </crystal>\n");
    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        fprintf(fout, "    <species speciesfile=\"%s.xml\" rmt=\"2.0\">\n",
                ctx.unit_cell().atom_type(iat).label().c_str());
        for (int ia = 0; ia < ctx.unit_cell().atom_type(iat).num_atoms(); ia++) {
            int id = ctx.unit_cell().atom_type(iat).atom_id(ia);
            auto v = ctx.unit_cell().atom(id).position();
            fprintf(fout, "      <atom coord=\"%18.12f %18.12f %18.12f\" bfcmt=\"0.0 0.0 0.0\"/>\n", v[0], v[1], v[2]);
        }
        fprintf(fout, "</species>\n");
    }

    fprintf(fout, "  </structure>\n");
    fprintf(fout, "  <groundstate do=\"fromscratch\" ngridk=\"2 2 2\" rgkmax=\"4.0\" gmaxvr=\"16\" maxscl=\"2\"  "
                  "kptgroups=\"1\">\n");
    fprintf(fout, "    <libxc exchange=\"XC_LDA_X\" correlation=\"XC_LDA_C_PZ\"/>\n");
    fprintf(fout, "    <sirius densityinit=\"true\" density=\"true\" vha=\"true\" xc=\"true\" eigenstates=\"true\" "
                  "sfacg=\"true\" cfun=\"true\"/>\n");
    fprintf(fout, "    <spin/>\n");
    fprintf(fout, "  </groundstate>\n");

    fprintf(fout, "</input>\n");
    fclose(fout);
}

void
convert_to_mol(cmd_args& args__)
{
    Simulation_context ctx(args__.value<std::string>("input", "sirius.json"), mpi::Communicator::self());

    json dict;
    dict["unit_cell"]                          = ctx.unit_cell().serialize(true);
    dict["unit_cell"]["atom_coordinate_units"] = "au";
    if (mpi::Communicator::world().rank() == 0) {
        std::ofstream ofs("unit_cell.json", std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    }
}

void
scale_lattice(cmd_args& args__)
{
    Simulation_context ctx(args__.value<std::string>("input", "sirius.json"), mpi::Communicator::self());
    // ctx.unit_cell().get_symmetry();
    // ctx.unit_cell().print_symmetry_info(5);

    double scale = args__.value<double>("scale", 1.0);

    json dict;
    dict["unit_cell"] = ctx.unit_cell().serialize();
    for (auto& x : dict["unit_cell"]["lattice_vectors"]) {
        for (auto& y : x) {
            double v = y.get<double>();
            v *= scale;
            y = v;
        }
    }
    for (auto& coord : dict["unit_cell"]["atoms"]) {
        for (size_t i = 0; i < coord.size(); i++) {
            for (int x : {0, 1, 2}) {
                double v = coord[i][x].get<double>();
                if (v > 0.5) {
                    v -= 1;
                }
                v /= scale;
                coord[i][x] = v;
            }
        }
    }
    if (mpi::Communicator::world().rank() == 0) {
        std::ofstream ofs("unit_cell.json", std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    }
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"input=", "{string} input file name"},
                   {"supercell=", "{string} transformation matrix (9 numbers)"},
                   {"qe", "create input for QE"},
                   {"xml", "create Exciting XML input"},
                   {"find_primitive", "find a primitive cell"},
                   {"cif", "create CIF file"},
                   {"mol", "convert to molecule input file"},
                   {"scale=", "scale lattice"}});

    sirius::initialize(1);
    if (args.exist("supercell")) {
        create_supercell(args);
    }
    if (args.exist("find_primitive")) {
        find_primitive();
    }
    if (args.exist("qe")) {
        create_qe_input(args);
    }
    if (args.exist("xml")) {
        create_exciting_input(args);
    }
    if (args.exist("cif")) {
        Simulation_context ctx(args.value<std::string>("input", "sirius.json"), mpi::Communicator::self());
        ctx.unit_cell().write_cif();
        ctx.unit_cell().find_nearest_neighbours(20);
        std::printf("minimum bond length: %20.12f\n", ctx.unit_cell().min_bond_length());
    }
    if (args.exist("mol")) {
        convert_to_mol(args);
    }
    if (args.exist("scale")) {
        scale_lattice(args);
    }

    sirius::finalize(1);
}
