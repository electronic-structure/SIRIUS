#include <sirius.hpp>

using namespace sirius;

template <typename T>
class Wave_functions_fft
{
  private:
    std::shared_ptr<Gvec_partition> gkvec_fft_;
    sddk::splindex<sddk::splindex_t::block> spl_num_wf_;
    int num_wf_;
    sddk::mdarray<std::complex<T>, 2> data_;
    costa::grid_layout<std::complex<T>> grid_layout_;

    void init_grid_layout()
    {
        std::vector<int> rowsplit(gkvec_fft_->comm_fft().size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < gkvec_fft_->comm_fft().size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + gkvec_fft_->gvec_count_fft(i);
        }

        std::vector<int> colsplit(gkvec_fft_->comm_ortho_fft().size() + 1);
        colsplit[0] = 0;
        for (int i = 0; i < gkvec_fft_->comm_ortho_fft().size(); i++) {
            colsplit[i + 1] = colsplit[i] + spl_num_wf_.local_size(i);
        }

        std::vector<int> owners(gkvec_fft_->gvec().comm().size());
        for (int i = 0; i < gkvec_fft_->gvec().comm().size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->data_.at(memory_t::host);
        localblock.ld = this->data_.ld();
        localblock.row = gkvec_fft_->comm_fft().rank();
        localblock.col = gkvec_fft_->comm_ortho_fft().rank();

        grid_layout_ = costa::custom_layout<std::complex<T>>(gkvec_fft_->comm_fft().size(),
                gkvec_fft_->comm_ortho_fft().size(), rowsplit.data(), colsplit.data(), owners.data(), 1,
                &localblock, 'C');
    }
  public:
    Wave_functions_fft(std::shared_ptr<Gvec_partition> gkvec_fft__, int num_wf__)
        : gkvec_fft_(gkvec_fft__)
        , num_wf_(num_wf__)
    {
        auto& comm_col = gkvec_fft_->comm_ortho_fft();
        spl_num_wf_ = sddk::splindex<sddk::splindex_t::block>(num_wf__, comm_col.size(), comm_col.rank());
        data_ = sddk::mdarray<std::complex<T>, 2>(gkvec_fft_->gvec_count_fft(), spl_num_wf_.local_size());
        init_grid_layout();
    }

    auto& grid_layout()
    {
        return grid_layout_;
    }

    int num_wf_local() const
    {
        return spl_num_wf_.local_size();
    }

    inline std::complex<T>& pw_coeffs(int ig__, int i__)
    {
        return data_(ig__, i__);
    }

    inline T* fft_data_ptr(memory_t mem__, int i__)
    {
        return reinterpret_cast<T*>(data_.at(mem__, 0, i__));
    }
};

class spin
{
  private:
    int idx_;
  public:
    explicit spin(int idx__)
        : idx_(idx__)
    {
        if (!(idx_ == 0 || idx_ == 1)) {
            RTE_THROW("wrong spin index");
        }
    }

    inline int operator()() const
    {
        return idx_;
    }
};

template <typename T>
auto transform_to_fft_layout(experimental::Wave_functions<T>& wf_in__, std::shared_ptr<Gvec_partition> gkvec_fft__,
        int ispn__, int N__, int n__)
{
    Wave_functions_fft<T> wf_fft(gkvec_fft__, n__);

    auto layout_in = wf_in__.grid_layout(ispn__, N__, n__);

    costa::transform(layout_in, wf_fft.grid_layout(), 'N', linalg_const<std::complex<T>>::one(),
            linalg_const<std::complex<T>>::zero(), gkvec_fft__->gvec().comm().mpi_comm());

    return wf_fft;
}
template <typename T>
void transform_from_fft_layout(Wave_functions_fft<T>& wf_fft_in__, experimental::Wave_functions<T>& wf_out__,
        int ispn__, int N__, int n__)
{
    auto layout_out = wf_out__.grid_layout(ispn__, N__, n__);

    costa::transform(wf_fft_in__.grid_layout(), layout_out, 'N', linalg_const<std::complex<T>>::one(),
            linalg_const<std::complex<T>>::zero(), wf_out__.gkvec().comm().mpi_comm());
}

void test_wf_fft()
{
    MPI_grid mpi_grid({2, 3}, sddk::Communicator::world());

    auto gkvec = gkvec_factory(8.0, mpi_grid.communicator());
    std::cout << "num_gvec=" << gkvec->num_gvec() << std::endl;
    auto gkvec_fft = std::make_shared<Gvec_partition>(*gkvec, mpi_grid.communicator(1 << 0), mpi_grid.communicator(1 << 1));

    auto fft_grid = get_min_fft_grid(8.0, gkvec->lattice_vectors());

    experimental::Wave_functions<double> wf(gkvec, 10);
    experimental::Wave_functions<double> wf_ref(gkvec, 10);

    for (int i = 0; i < 10; i++) {
        for (int ig = 0; ig < gkvec->count(); ig++) {
            wf.pw_coeffs(0)(ig, i) = wf_ref.pw_coeffs(0)(ig, i) = utils::random<std::complex<double>>();
        }
    }

    auto wf_fft = transform_to_fft_layout(wf, gkvec_fft, 0, 0, 10);

    auto pu = device_t::CPU;

    auto spfft_pu = pu == device_t::CPU ? SPFFT_PU_HOST : SPFFT_PU_GPU;
    auto spl_z = split_fft_z(fft_grid[2], gkvec_fft->comm_fft());

        /* create spfft buffer for coarse transform */
    auto spfft_grid = std::unique_ptr<spfft::Grid>(new spfft::Grid(
            fft_grid[0], fft_grid[1], fft_grid[2], gkvec_fft->zcol_count_fft(),
            spl_z.local_size(), spfft_pu, -1, gkvec_fft->comm_fft().mpi_comm(), SPFFT_EXCH_DEFAULT));

    const auto fft_type = gkvec->reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

    /* create actual transform object */
    auto spfft_transform = std::make_unique<spfft::Transform>(spfft_grid->create_transform(
        spfft_pu, fft_type, fft_grid[0], fft_grid[1], fft_grid[2],
        spl_z.local_size(), gkvec_fft->gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
        gkvec_fft->gvec_array().at(memory_t::host)));

    for (int i = 0; i < 10; i++) {
        for (int ig = 0; ig < gkvec->count(); ig++) {
            wf.pw_coeffs(0)(ig, i) = 0;
        }
    }

    for (int i = 0; i < wf_fft.num_wf_local(); i++) {
        spfft_transform->backward(wf_fft.fft_data_ptr(memory_t::host, i), spfft_pu);
        spfft_transform->forward(spfft_pu, wf_fft.fft_data_ptr(memory_t::host, i), SPFFT_FULL_SCALING);
    }

    transform_from_fft_layout(wf_fft, wf, 0, 0, 10);

    for (int i = 0; i < 10; i++) {
        for (int ig = 0; ig < gkvec->count(); ig++) {
            if (std::abs(wf.pw_coeffs(0)(ig, i) - wf_ref.pw_coeffs(0)(ig, i)) > 1e-10) {
                std::cout << "Error!" << std::endl;
            }
        }
    }
}


int main(int argn, char** argv)
{
    sirius::initialize(1);
    test_wf_fft();
    sirius::finalize();
}
