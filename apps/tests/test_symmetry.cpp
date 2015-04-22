#include <sirius.h>

using namespace sirius;

void test_symmetry()
{
    matrix3d<double> lattice;
    lattice(0, 0) = lattice(1, 1) = lattice(2, 2) = 5;

    int num_atoms = 1;
    mdarray<double, 2> positions(3, num_atoms);
    positions.zero();

    mdarray<double, 2> spins(3, num_atoms);
    spins.zero();
    
    std::vector<int> types(num_atoms, 0);

    Symmetry symmetry(lattice, num_atoms, positions, spins, types, 1e-4);

    printf("num_sym_op: %i\n", symmetry.num_sym_op());

    for (int isym = 0; isym < symmetry.num_sym_op(); isym++)
    {
        printf("\n");
        printf("symmetry operation: %i\n", isym);

        vector3d<double> ang = symmetry.euler_angles(isym);

        std::cout << "Euler angles : " << ang[0] / pi << " " << ang[1] / pi << " " << ang[2] / pi << std::endl;

        int proper_rotation = symmetry.proper_rotation(isym);
        
        vector3d<double> coord(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        vector3d<double> scoord;
        scoord = SHT::spherical_coordinates(coord);

        int lmax = 10;
        mdarray<double_complex, 1> ylm(Utils::lmmax(lmax));
        SHT::spherical_harmonics(lmax, scoord[1], scoord[2], &ylm(0));

        matrix3d<double> rotm = inverse(matrix3d<double>(symmetry.rot_mtrx(isym)));
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
        SHT::spherical_harmonics(lmax, scoord2[1], scoord2[2], &ylm2(0));

        mdarray<double_complex, 2> ylm_rot_mtrx(Utils::lmmax(lmax), Utils::lmmax(lmax));
        SHT::rotation_matrix(lmax, ang, proper_rotation, ylm_rot_mtrx); 


        mdarray<double_complex, 1> ylm1(Utils::lmmax(lmax));
        ylm1.zero();

        for (int i = 0; i < Utils::lmmax(lmax); i++)
        {
            for (int j = 0; j < Utils::lmmax(lmax); j++) ylm1(i) += ylm_rot_mtrx(j, i) * ylm(j);
        }

        double d = 0;
        for (int i = 0; i < Utils::lmmax(lmax); i++) d += std::abs(ylm1(i) - ylm2(i));
        printf("diff: %18.12f\n", d);
        if (d > 1e-10) TERMINATE("Fail!");
    }
}

int main(int argn, char** argv)
{
    Platform::initialize(1);
    test_symmetry();
    Platform::finalize();
}
