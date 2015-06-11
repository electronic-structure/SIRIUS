#include <sirius.h>

using namespace sirius;

int main(int argn, char** argv)
{
    Platform::initialize(1);

    for (int x = 0; x < 10; x++)
    {
        Global g;
        double a1[] = {6, 0, 0};
        double a2[] = {0, 6, 0};
        double a3[] = {0, 0, 6};
    
        g.unit_cell()->set_lattice_vectors(a1, a2, a3);
        g.set_pw_cutoff(9.0);
        g.initialize();

        //int context = linalg<scalapack>::create_blacs_context(MPI_COMM_WORLD);
        //std::cout << "context : " << context << std::endl;
        //linalg<scalapack>::free_blacs_context(context);
    }
    Platform::finalize();
}
