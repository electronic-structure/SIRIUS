#include <sirius.h>

using namespace sirius;

void create_supercell(cmd_args& args__)
{
    matrix3d<int> scell;
    std::stringstream s(args__.value<std::string>("supercell"));
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) s >> scell(j, i);
    }
    
    std::cout << std::endl;
    std::cout << "supercell vectors (lattice coordinates) : " << std::endl;
    for (int i = 0; i < 3; i++)
    {
        std::cout << "A" << i << " : ";
        for (int j = 0; j < 3; j++) std::cout << scell(j, i) << " ";
        std::cout << std::endl;
    }

    Simulation_context ctx("sirius.json", mpi_comm_self());

    matrix3d<double> scell_lattice_vectors = ctx.unit_cell().lattice_vectors() * matrix3d<double>(scell);

    std::cout << "supercell vectors (Cartesian coordinates) : " << std::endl;
    for (int i = 0; i < 3; i++)
    {
        std::cout << "A" << i << " : ";
        for (int x = 0; x < 3; x++) std::cout << scell_lattice_vectors(x, i) << " ";
        std::cout << std::endl;
    }

    std::cout << "volume ratio : " << std::abs(matrix3d<int>(scell).det()) << std::endl;

    Simulation_context ctx_sc(mpi_comm_self());

    vector3d<double> a0, a1, a2;
    for (int x = 0; x < 3; x++)
    {
        a0[x] = scell_lattice_vectors(x, 0);
        a1[x] = scell_lattice_vectors(x, 1);
        a2[x] = scell_lattice_vectors(x, 2);
    }
    ctx_sc.unit_cell().set_lattice_vectors(a0, a1, a2);

    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++)
    {
        auto label = ctx.unit_cell().atom_type(iat).label();
        ctx_sc.unit_cell().add_atom_type(label, "");
    }
   
    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++)
    {
        auto label = ctx.unit_cell().atom_type(iat).label();

        for (int i = 0; i < ctx.unit_cell().atom_type(iat).num_atoms(); i++)
        {
            int ia = ctx.unit_cell().atom_type(iat).atom_id(i);
            auto va = ctx.unit_cell().atom(ia).position();

            for (int i0 = -10; i0 <= 10; i0++)
            {
                for (int i1 = -10; i1 <= 10; i1++)
                {
                    for (int i2 = -10; i2 <= 10; i2++)
                    {
                        vector3d<double> T(i0, i1, i2);
                        vector3d<double> vc = ctx.unit_cell().get_cartesian_coordinates(va + T);
                        vector3d<double> vf = ctx_sc.unit_cell().get_fractional_coordinates(vc);

                        auto vr = reduce_coordinates(vf);
                        bool add_atom = (ctx_sc.unit_cell().atom_id_by_position(vr.first) == -1);
                        //==if (add_atom && iat == 2)
                        //=={
                        //==    double r = type_wrapper<double>::random();
                        //==    if (r < 0.99) add_atom = false;
                        //==}

                        if (add_atom) ctx_sc.unit_cell().add_atom(label, vr.first);
                    }
                }
            }
        }
    }
    printf("number of atoms in the supercell: %i\n", ctx_sc.unit_cell().num_atoms());
    
    //ctx_sc.unit_cell().get_symmetry();
    ctx_sc.unit_cell().print_info(4);
    ctx_sc.unit_cell().write_cif();
    json dict;
    dict["unit_cell"] = ctx_sc.unit_cell().serialize();
    if (mpi_comm_world().rank() == 0) {
        std::ofstream ofs("unit_cell.json", std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    }
}

void find_primitive()
{
    Simulation_context ctx("sirius.json", mpi_comm_self());

    double lattice[3][3];
    for (int i: {0, 1, 2}) {
        for (int j: {0, 1, 2}) {
            lattice[i][j] = ctx.unit_cell().lattice_vector(j)[i];
        }
    }
    mdarray<double, 2> positions(3, 4 * ctx.unit_cell().num_atoms());
    mdarray<int, 1> types(4 * ctx.unit_cell().num_atoms());
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        for (int x: {0, 1, 2}) {
            positions(x, ia) = ctx.unit_cell().atom(ia).position()[x];
            types(ia) = ctx.unit_cell().atom(ia).type_id();
        }
    }

    printf("original number of atoms: %i\n", ctx.unit_cell().num_atoms());

    int nat_new = spg_find_primitive(lattice, (double(*)[3])&positions(0, 0), &types[0], ctx.unit_cell().num_atoms(), 1e-4);
    printf("new number of atoms: %i\n", nat_new);

    Simulation_context ctx_new(mpi_comm_self());

    vector3d<double> a0, a1, a2;
    for (int x = 0; x < 3; x++) {
        a0[x] = lattice[x][0];
        a1[x] = lattice[x][1];
        a2[x] = lattice[x][2];
    }
    ctx_new.unit_cell().set_lattice_vectors(a0, a1, a2);

    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++)
    {
        auto label = ctx.unit_cell().atom_type(iat).label();
        ctx_new.unit_cell().add_atom_type(label, "");
    }
   
    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        auto label = ctx.unit_cell().atom_type(iat).label();

        for (int i = 0; i < nat_new; i++) {
            if (types[i] == iat) {
                vector3d<double> p(positions(0, i), positions(1, i), positions(2, i));
                ctx_new.unit_cell().add_atom(label, p);
            }
        }
    }
    
    json dict;
    dict["unit_cell"] = ctx_new.unit_cell().serialize();
    if (mpi_comm_world().rank() == 0) {
        std::ofstream ofs("unit_cell.json", std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    }

}

void create_qe_input()
{
    Simulation_context ctx("sirius.json", mpi_comm_self());

    FILE* fout = fopen("pw.in", "w");
    fprintf(fout, "&control\ncalculation=\'scf\',\nrestart_mode=\'from_scratch\',\npseudo_dir = \'./\',\noutdir=\'./\',\nprefix = \'scf_\'\n/\n");
    fprintf(fout, "&system\nibrav=0, celldm(1)=1, ecutwfc=40, ecutrho = 300,\noccupations = \'smearing\', smearing = \'gauss\', degauss = 0.001,\n");
    fprintf(fout, "nat=%i ntyp=%i\n/\n", ctx.unit_cell().num_atoms(), ctx.unit_cell().num_atom_types());
    fprintf(fout, "&electrons\nconv_thr =  1.0d-11,\nmixing_beta = 0.7,\nelectron_maxstep = 100\n/\n");
    fprintf(fout, "ATOMIC_SPECIES\n");
    for (int iat = 0; iat < ctx.unit_cell().num_atom_types(); iat++) {
        fprintf(fout, "%s 0.0 pp.UPF\n", ctx.unit_cell().atom_type(iat).label().c_str());
    }

    fprintf(fout,"CELL_PARAMETERS\n");
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
                    ctx.unit_cell().atom(id).position()[0],
                    ctx.unit_cell().atom(id).position()[1],
                    ctx.unit_cell().atom(id).position()[2]);
        }
    }
    fprintf(fout, "K_POINTS (automatic)\n2 2 2  0 0 0\n");

    fclose(fout);
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--supercell=", "{string} transformation matrix (9 numbers)");
    args.register_key("--qe", "create input for QE");
    args.register_key("--find_primitive", "find a primitive cell");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    if (args.exist("supercell")) {
        create_supercell(args);
    }
    if (args.exist("find_primitive")) {
        find_primitive();
    }
    if (args.exist("qe")) {
        create_qe_input();
    }

    sirius::finalize();
}
