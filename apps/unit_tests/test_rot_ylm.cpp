#include <sirius.h>

using namespace sirius;

void test_symmetry()
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

    Unit_cell_symmetry symmetry(lattice, num_atoms, positions, spins, types, 1e-4);

    printf("num_sym_op: %i\n", symmetry.num_mag_sym());

    for (int iter = 0; iter < 10; iter++)
    {
        for (int isym = 0; isym < symmetry.num_mag_sym(); isym++)
        {
            printf("\n");
            printf("symmetry operation: %i\n", isym);

            vector3d<double> ang = symmetry.magnetic_group_symmetry(isym).spg_op.euler_angles;

            std::cout << "Euler angles : " << ang[0] / pi << " " << ang[1] / pi << " " << ang[2] / pi << std::endl;

            int proper_rotation = symmetry.magnetic_group_symmetry(isym).spg_op.proper;
            
            vector3d<double> coord(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
            vector3d<double> scoord;
            scoord = SHT::spherical_coordinates(coord);

            int lmax = 10;
            mdarray<double_complex, 1> ylm(Utils::lmmax(lmax));
            mdarray<double, 1> rlm(Utils::lmmax(lmax));
            SHT::spherical_harmonics(lmax, scoord[1], scoord[2], &ylm(0));
            SHT::spherical_harmonics(lmax, scoord[1], scoord[2], &rlm(0));
            
            /* rotate coordinates with inverse operation */
            matrix3d<double> rotm = inverse(symmetry.magnetic_group_symmetry(isym).spg_op.rotation * double(proper_rotation));
            printf("3x3 rotation matrix\n");
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++) printf("%8.4f ", rotm(i, j));
                printf("\n");
            }

            vector3d<double> coord2 = rotm * coord;
            vector3d<double> scoord2;
            scoord2 = SHT::spherical_coordinates(coord2);

            mdarray<double_complex, 1> ylm2(Utils::lmmax(lmax));
            mdarray<double, 1> rlm2(Utils::lmmax(lmax));
            SHT::spherical_harmonics(lmax, scoord2[1], scoord2[2], &ylm2(0));
            SHT::spherical_harmonics(lmax, scoord2[1], scoord2[2], &rlm2(0));

            mdarray<double_complex, 2> ylm_rot_mtrx(Utils::lmmax(lmax), Utils::lmmax(lmax));
            mdarray<double, 2> rlm_rot_mtrx(Utils::lmmax(lmax), Utils::lmmax(lmax));
            utils::timer t0("rotation_matrix_Ylm");
            SHT::rotation_matrix(lmax, ang, proper_rotation, ylm_rot_mtrx);
            t0.stop();
            utils::timer t1("rotation_matrix_Rlm");
            SHT::rotation_matrix(lmax, ang, proper_rotation, rlm_rot_mtrx);
            t1.stop();

            mdarray<double_complex, 1> ylm1(Utils::lmmax(lmax));
            ylm1.zero();

            mdarray<double, 1> rlm1(Utils::lmmax(lmax));
            rlm1.zero();

            for (int i = 0; i < Utils::lmmax(lmax); i++)
            {
                for (int j = 0; j < Utils::lmmax(lmax); j++) ylm1(i) += ylm_rot_mtrx(j, i) * ylm(j);

                for (int j = 0; j < Utils::lmmax(lmax); j++) rlm1(i) += rlm_rot_mtrx(j, i) * rlm(j);
            }

            double d1 = 0;
            double d2 = 0;
            for (int i = 0; i < Utils::lmmax(lmax); i++)
            {
                d1 += std::abs(ylm1(i) - ylm2(i));
                d2 += std::abs(rlm1(i) - rlm2(i));
            }
            printf("diff: %18.12f %18.12f\n", d1, d2);
            if (d1 > 1e-10 || d2 > 1e-10)
            {
                printf("Fail!\n");
                exit(1);
            }
        }
    }
}

int main(int argn, char** argv)
{
    sirius::initialize(1);
    test_symmetry();
    utils::timer::print();
    sirius::finalize();

    return 0;
}
