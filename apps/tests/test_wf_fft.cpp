#include <sirius.hpp>

using namespace sirius;

namespace sirius {
namespace experimental {

/* PW and LAPW wave-functions
 *
 * Wave_functions wf(gkvec_factory(..), 10);
 *
 *
 * Local coefficients consit of two parts: PW and MT
 * +-------+
 * |       |
 * |  G+k  |   -> swap only PW part
 * |       |
 * +-------+
 * | atom1 |
 * +-------+
 * | atom2 |
 * +-------+
 * | ....  |
 * +-------+
 *
 * wf_fft = remap_to_fft(gkvec_partition, wf, N, n);
 *
 * hpsi_fft = wf_fft_factory(gkvec_partition, n);
 *
 * remap_from_fft(gkvec_partition, wf_fft, wf, N, n)
 *
 * consider Wave_functions_fft class
 *
 *
 * Wave_functions wf(...);
 * memory_guard mem_guard(wf, memory_t::device);
 *
 *
 *
 */

enum class copy_to : unsigned int
{
    none   = 0b0000,
    device = 0b0001,
    host   = 0b0010
};
inline copy_to operator|(copy_to a__, copy_to b__)
{
    return static_cast<copy_to>(static_cast<unsigned int>(a__) | static_cast<unsigned int>(b__));
}

template <typename T>
class device_memory_guard
{
  private:
    T& obj_;
    device_memory_guard(device_memory_guard const&) = delete;
    device_memory_guard& operator=(device_memory_guard const&) = delete;
    memory_t mem_;
    copy_to copy_to_;
  public:
    device_memory_guard(T& obj__, memory_t mem__, copy_to copy_to__)
        : obj_(obj__)
        , mem_(mem__)
        , copy_to_(copy_to__)
    {
        if (is_device_memory(mem_)) {
            obj_.allocate(mem_);
            if (static_cast<unsigned int>(copy_to_) & static_cast<unsigned int>(copy_to::device)) {
                obj_.copy_to(mem_);
            }
        }
    }
    ~device_memory_guard()
    {
        if (is_device_memory(mem_)) {
            if (static_cast<unsigned int>(copy_to_) & static_cast<unsigned int>(copy_to::host)) {
                obj_.copy_to(memory_t::host);
            }
            obj_.deallocate(mem_);
        }
    }
};

template <typename T>
class Wave_functions_base
{
  private:
    int ld_;
    int num_wf_;
    int num_sc_;

    inline void allocate(memory_t mem__)
    {
        for (int is = 0; is < num_sc_; is++) {
            data_[is].allocate(mem__);
        }
    }

    inline void deallocate(memory_t mem__)
    {
        for (int is = 0; is < num_sc_; is++) {
            data_[is].deallocate(mem__);
        }
    }

    inline void copy_to(memory_t mem__)
    {
        for (int is = 0; is < num_sc_; is++) {
            data_[is].copy_to(mem__);
        }
    }

    friend class device_memory_guard<Wave_functions_base<T>>;

  protected:
    std::vector<sddk::mdarray<std::complex<T>, 2>> data_;
  public:
    Wave_functions_base(int ld__, int num_wf__, int num_sc__, memory_t default_mem__)
        : ld_(ld__)
        , num_wf_(num_wf__)
        , num_sc_(num_sc__)
    {
        data_.resize(num_sc_);
        for (int is = 0; is < num_sc_; is++) {
            data_[is] = mdarray<std::complex<T>, 2>(ld_, num_wf_, default_mem__, "Wave_functions_base::data_");
        }
    }

    auto memory_guard(memory_t mem__, experimental::copy_to copy_to__ = experimental::copy_to::none)
    {
        return device_memory_guard(*this, mem__, copy_to__);
    }
};

template <typename T>
class Wave_functions : public Wave_functions_base<T>
{
  private:
    std::shared_ptr<Gvec> gkvec_;
    int num_atoms_{0};
    splindex<splindex_t::block> spl_num_atoms_;
    /// Local size of muffin-tin coefficients for each rank.
    /** Each rank stores local fraction of atoms. Each atom has a set of MT coefficients. */
    block_data_descriptor mt_coeffs_distr_;
    /// Local offset in the block of MT coefficients for current rank.
    /** The size of the vector is equal to the local number of atoms for the current rank. */
    std::vector<int> offset_in_local_mt_coeffs_;
    /// Total numbef of muffin-tin coefficients.
    int num_mt_coeffs_{0};
    static int get_local_num_mt_coeffs(std::vector<int> num_mt_coeffs__, Communicator const& comm__)
    {
        int num_atoms = static_cast<int>(num_mt_coeffs__.size());
        splindex<splindex_t::block> spl_atoms(num_atoms, comm__.size(), comm__.rank());
        auto it_begin = num_mt_coeffs__.begin() + spl_atoms.global_offset();
        auto it_end = it_begin + spl_atoms.local_size();
        return std::accumulate(it_begin, it_end, 0);
    }

  public:
    Wave_functions(std::shared_ptr<Gvec> gkvec__, int num_wf__, int num_sc__, memory_t default_mem__)
        : Wave_functions_base<T>(gkvec__->count(), num_wf__, num_sc__, default_mem__)
        , gkvec_(gkvec__)
    {
    }
    Wave_functions(std::shared_ptr<Gvec> gkvec__, std::vector<int> num_mt_coeffs__, int num_wf__, int num_sc__,
            memory_t default_mem__)
        : Wave_functions_base<T>(gkvec__->count() + get_local_num_mt_coeffs(num_mt_coeffs__, gkvec__->comm()),
                                 num_wf__, num_sc__, default_mem__)
        , gkvec_(gkvec__)
        , num_atoms_(static_cast<int>(num_mt_coeffs__.size()))
        , spl_num_atoms_(splindex<splindex_t::block>(num_atoms_, gkvec__->comm().size(), gkvec__->comm().rank()))
        , num_mt_coeffs_(std::accumulate(num_mt_coeffs__.begin(), num_mt_coeffs__.end(), 0))
    {
        mt_coeffs_distr_ = block_data_descriptor(gkvec_->comm().size());

        for (int ia = 0; ia < num_atoms_; ia++) {
            int rank = spl_num_atoms_.local_rank(ia);
            if (rank == gkvec_->comm().rank()) {
                offset_in_local_mt_coeffs_.push_back(mt_coeffs_distr_.counts[rank]);
            }
            /* increment local number of MT coeffs. for a given rank */
            mt_coeffs_distr_.counts[rank] += num_mt_coeffs__[ia];
        }
        mt_coeffs_distr_.calc_offsets();
    }
    ~Wave_functions()
    {
    }

    inline sddk::mdarray<std::complex<T>, 2>& pw_coeffs(int ispn__)
    {
        return this->data_[ispn__];
    }

    auto grid_layout_pw(int ispn__, int N__, int n__)
    {
        auto& comm = gkvec_->comm();

        std::vector<int> rowsplit(comm.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < comm.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + gkvec_->gvec_count(i);
        }
        std::vector<int> colsplit({0, n__});
        std::vector<int> owners(comm.size());
        for (int i = 0; i < comm.size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->data_[ispn__].at(memory_t::host, 0, N__);
        localblock.ld = this->data_[ispn__].ld();
        localblock.row = comm.rank();
        localblock.col = 0;

        return costa::custom_layout<std::complex<T>>(gkvec_->comm().size(), 1, rowsplit.data(), colsplit.data(),
                owners.data(), 1, &localblock, 'C');
    }

    auto grid_layout_mt(int ispn__, int N__, int n__)
    {
        auto& comm = gkvec_->comm();

        std::vector<int> rowsplit(comm.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < comm.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + mt_coeffs_distr_.counts[i];
        }
        std::vector<int> colsplit({0, n__});
        std::vector<int> owners(comm.size());
        for (int i = 0; i < comm.size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->data_[ispn__].at(memory_t::host, gkvec_->count(), N__);
        localblock.ld = this->data_[ispn__].ld();
        localblock.row = comm.rank();
        localblock.col = 0;

        return costa::custom_layout<std::complex<T>>(gkvec_->comm().size(), 1, rowsplit.data(), colsplit.data(),
                owners.data(), 1, &localblock, 'C');
    }

    Gvec const& gkvec() const
    {
        return *gkvec_;
    }
};

template <typename T>
class Wave_functions_fft : public Wave_functions_base<T>
{
  private:
    std::shared_ptr<Gvec_partition> gkvec_fft_;
    sddk::splindex<sddk::splindex_t::block> spl_num_wf_;

  public:
    Wave_functions_fft(std::shared_ptr<Gvec_partition> gkvec_fft__, int num_wf_max__, memory_t default_mem__)
        : Wave_functions_base<T>(gkvec_fft__->gvec_count_fft(),
                sddk::splindex<sddk::splindex_t::block>(num_wf_max__, gkvec_fft__->comm_ortho_fft().size(),
                    gkvec_fft__->comm_ortho_fft().rank()).local_size(), 1, default_mem__)
        , gkvec_fft_(gkvec_fft__)
    {
    }

    auto grid_layout(int n__)
    {
        auto& comm_row = gkvec_fft_->comm_fft();
        auto& comm_col = gkvec_fft_->comm_ortho_fft();

        spl_num_wf_ = sddk::splindex<sddk::splindex_t::block>(n__, comm_col.size(), comm_col.rank());

        std::vector<int> rowsplit(comm_row.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < comm_row.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + gkvec_fft_->gvec_count_fft(i);
        }

        std::vector<int> colsplit(comm_col.size() + 1);
        colsplit[0] = 0;
        for (int i = 0; i < comm_col.size(); i++) {
            colsplit[i + 1] = colsplit[i] + spl_num_wf_.local_size(i);
        }

        std::vector<int> owners(gkvec_fft_->gvec().comm().size());
        for (int i = 0; i < gkvec_fft_->gvec().comm().size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->data_[0].at(memory_t::host);
        localblock.ld = this->data_[0].ld();
        localblock.row = gkvec_fft_->comm_fft().rank();
        localblock.col = comm_col.rank();

        return costa::custom_layout<std::complex<T>>(comm_row.size(), comm_col.size(), rowsplit.data(),
                colsplit.data(), owners.data(), 1, &localblock, 'C');
    }

    int num_wf_local() const
    {
        return spl_num_wf_.local_size();
    }

    inline std::complex<T>& pw_coeffs(int ig__, int i__)
    {
        return this->data_[0](ig__, i__);
    }

    inline T* pw_coeffs(memory_t mem__, int i__)
    {
        return reinterpret_cast<T*>(this->data_[0].at(mem__, 0, i__));
    }
};

}

}


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

// spin_range(0, 2);
// spin_range(0, 1);
// for (auto s: spins) {
//   pw_coeffs[s()]
//
// }

template <typename T>
void transform_to_fft_layout(experimental::Wave_functions<T>& wf_in__, experimental::Wave_functions_fft<T>& wf_fft_out__,
        std::shared_ptr<Gvec_partition> gkvec_fft__, int ispn__, int N__, int n__)
{

    auto layout_in  = wf_in__.grid_layout_pw(ispn__, N__, n__);
    auto layout_out = wf_fft_out__.grid_layout(n__);

    costa::transform(layout_in, layout_out, 'N', linalg_const<std::complex<T>>::one(),
            linalg_const<std::complex<T>>::zero(), gkvec_fft__->gvec().comm().mpi_comm());
}

template <typename T>
void transform_from_fft_layout(experimental::Wave_functions_fft<T>& wf_fft_in__, experimental::Wave_functions<T>& wf_out__,
        int ispn__, int N__, int n__)
{
    auto layout_in  = wf_fft_in__.grid_layout(n__);
    auto layout_out = wf_out__.grid_layout_pw(ispn__, N__, n__);

    costa::transform(layout_in, layout_out, 'N', linalg_const<std::complex<T>>::one(),
            linalg_const<std::complex<T>>::zero(), wf_out__.gkvec().comm().mpi_comm());
}

void test_wf_fft()
{
    MPI_grid mpi_grid({2, 3}, sddk::Communicator::world());

    /* creation of simple G+k vector set */
    auto gkvec = gkvec_factory(8.0, mpi_grid.communicator());
    std::cout << "num_gvec=" << gkvec->num_gvec() << std::endl;
    /* creation of G+k set for FFTt */
    auto gkvec_fft = std::make_shared<Gvec_partition>(*gkvec, mpi_grid.communicator(1 << 0), mpi_grid.communicator(1 << 1));

    /* get the FFT box boundaries */
    auto fft_grid = get_min_fft_grid(8.0, gkvec->lattice_vectors());

    std::vector<int> num_mt_coeffs({10, 20, 30, 10, 20});

    experimental::Wave_functions<double> wf(gkvec, num_mt_coeffs, 10, 1, memory_t::host);
    experimental::Wave_functions<double> wf_ref(gkvec, 10, 1, memory_t::host);
    experimental::Wave_functions_fft<double> wf_fft(gkvec_fft, 10, memory_t::host);

    for (int i = 0; i < 10; i++) {
        for (int ig = 0; ig < gkvec->count(); ig++) {
            wf.pw_coeffs(0)(ig, i) = wf_ref.pw_coeffs(0)(ig, i) = utils::random<std::complex<double>>();
        }
    }
    auto mg = wf.memory_guard(memory_t::device, experimental::copy_to::device);
    auto mg_fft = wf_fft.memory_guard(memory_t::device);

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

    transform_to_fft_layout(wf, wf_fft, gkvec_fft, 0, 0, 10);

    for (int i = 0; i < 10; i++) {
        for (int ig = 0; ig < gkvec->count(); ig++) {
            wf.pw_coeffs(0)(ig, i) = 0;
        }
    }

    for (int i = 0; i < wf_fft.num_wf_local(); i++) {
        spfft_transform->backward(wf_fft.pw_coeffs(memory_t::host, i), spfft_pu);
        spfft_transform->forward(spfft_pu, wf_fft.pw_coeffs(memory_t::host, i), SPFFT_FULL_SCALING);
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
