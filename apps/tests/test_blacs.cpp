#include <sirius.h>

using namespace sirius;

class A
{
    public:
    BLACS_grid const& src_;
    BLACS_grid a;
    BLACS_grid b;

    A(BLACS_grid const& src) : src_(src), a(src.comm(), src.comm().size(), 1), b(src.comm(), 1, src.comm().size())
    {
    }
};

void test_blacs()
{
    Communicator comm(MPI_COMM_WORLD);

    std::vector<int> dims = {comm.size()};
    MPI_grid mpi_grid(dims, comm);

    BLACS_grid blacs_grid(mpi_grid.communicator(1 << 1 | 1 << 2), 
                          mpi_grid.dimension_size(1),
                          mpi_grid.dimension_size(2));
    
    A t(blacs_grid);
}

int main(int argn, char** argv)
{
    Platform::initialize(1);

    test_blacs();

    Platform::finalize();
}
