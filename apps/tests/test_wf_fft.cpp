#include <sirius.hpp>

using namespace sirius;

class Wave_functions_fft
{
  private:
    std::shared_ptr<Gvec_partition> gkvec_fft_;
    sddk::splindex<sddk::splindex_t::block> spl_n_;
    int num_wf_;
  public:
    Wave_functions_fft(std::shared_ptr<Gvec_partition> gkvec_fft__, int num_wf__)
        : gkvec_fft_(gkvec_fft__)
        , num_wf_(num_wf__)
    {
        spl_n_ = 
    }
};

void test_wf_fft()
{
    MPI_grid mpi_grid({2, 2}, sdd::Communicator::world());

    auto gkvec = gkvec_factory(5.0, mpi_grid.communicator());
    auto gkvec_fft = std::make_shared<Gvec_partition>(*gkvec, mpi_grid.communicator(1 << 0), mpi_grid.communicator(1 << 1));

}


int main(int argn, char** argv)
{
    sirius::initialize(1);
    test_wf_fft();
    sirius::finalize();
}
