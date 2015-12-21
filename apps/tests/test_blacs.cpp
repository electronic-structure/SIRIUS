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

void test_blacs_grid_order()
{
    Communicator comm(MPI_COMM_WORLD);
    BLACS_grid grid(comm, 3, 2);
    
    if (comm.rank() == 0)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int i = 0; i < 2; i++)
            {
                int rank = grid.cart_rank(j, i);
                printf("%2i ", rank);
            }
            printf("\n");
        }
    }
}

int main(int argn, char** argv)
{
    Platform::initialize(1);

    test_blacs();

    test_blacs_grid_order();

    Platform::finalize();
}
