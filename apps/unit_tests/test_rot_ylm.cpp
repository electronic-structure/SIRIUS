#include <sirius.hpp>

/* test rotation of spherical harmonics */

using namespace sirius;

template <typename T>
int run_test_impl(cmd_args& args)
{
    matrix3d<double> lattice;
    lattice(0, 0) = 7;
    lattice(1, 1) = 7;
    lattice(2, 2) = 7;

    int num_atoms = 1;
    mdarray<double, 2> positions(3, num_atoms);
    positions.zero();

    mdarray<double, 2> spins(3, num_atoms);
    spins.zero();

    std::vector<int> types(num_atoms, 0);

    bool const spin_orbit{false};
    bool const use_sym{true};
    double const spg_tol{1e-4};

    Unit_cell_symmetry symmetry(lattice, num_atoms, 1, types, positions, spins, spin_orbit, spg_tol, use_sym);

    for (int iter = 0; iter < 10; iter++) {
        for (int isym = 0; isym < symmetry.size(); isym++) {

            auto ang = symmetry[isym].spg_op.euler_angles;

            int proper_rotation = symmetry[isym].spg_op.proper;

            /* random Cartesian vector */
            vector3d<double> coord(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
            auto scoord = SHT::spherical_coordinates(coord);

            int lmax{10};
            sddk::mdarray<T, 1> ylm(utils::lmmax(lmax));
            /* compute spherical harmonics at original coordinate */
            sf::spherical_harmonics(lmax, scoord[1], scoord[2], &ylm(0));

            /* rotate coordinates with inverse operation */
            auto rotm = inverse(symmetry[isym].spg_op.rotation * double(proper_rotation));

            /* rotated coordinates */
            auto coord2 = dot(rotm, coord);
            auto scoord2 = SHT::spherical_coordinates(coord2);

            sddk::mdarray<T, 1> ylm2(utils::lmmax(lmax));
            /* compute spherical harmonics at rotated coordinates */
            sf::spherical_harmonics(lmax, scoord2[1], scoord2[2], &ylm2(0));

            /* generate rotation matrices; they are block-diagonal in l- index */
            sddk::mdarray<T, 2> ylm_rot_mtrx(utils::lmmax(lmax), utils::lmmax(lmax));
            sht::rotation_matrix(lmax, ang, proper_rotation, ylm_rot_mtrx);

            sddk::mdarray<T, 1> ylm1(utils::lmmax(lmax));
            ylm1.zero();

            /* rotate original sperical harmonics */
            for (int i = 0; i < utils::lmmax(lmax); i++) {
                for (int j = 0; j < utils::lmmax(lmax); j++) {
                    ylm1(i) += ylm_rot_mtrx(j, i) * ylm(j);
                }
            }

            /* compute the difference with the reference */
            double d1{0};
            for (int i = 0; i < utils::lmmax(lmax); i++) {
                d1 += std::abs(ylm1(i) - ylm2(i));
            }
            if (d1 > 1e-10) {
                return 1;
            }
        }
    }
    return 0;
}


int run_test(cmd_args& args)
{
    int result = run_test_impl<double>(args);
    result += run_test_impl<double_complex>(args);
    return result;
}

int main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(true);
    printf("running %-30s : ", argv[0]);
    int result = run_test(args);
    if (result) {
        printf("\x1b[31m" "Failed" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK" "\x1b[0m" "\n");
    }
    sirius::finalize();

    return result;
}
