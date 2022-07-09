#ifndef __WAVE_FUNCTIONS_HPP__
#define __WAVE_FUNCTIONS_HPP__

//#include <costa/layout.hpp>
//#include <costa/grid2grid/transformer.hpp>
//#include "SDDK/memory.hpp"
//#include "SDDK/gvec.hpp"
//#include "linalg/linalg_base.hpp"

namespace exper {

    class A {

    };

//== /* PW and LAPW wave-functions
//==  *
//==  * Wave_functions wf(gkvec_factory(..), 10);
//==  *
//==  *
//==  * Local coefficients consit of two parts: PW and MT
//==  * +-------+
//==  * |       |
//==  * |  G+k  |   -> swap only PW part
//==  * |       |
//==  * +-------+
//==  * | atom1 |
//==  * +-------+
//==  * | atom2 |
//==  * +-------+
//==  * | ....  |
//==  * +-------+
//==  *
//==  * wf_fft = remap_to_fft(gkvec_partition, wf, N, n);
//==  *
//==  * hpsi_fft = wf_fft_factory(gkvec_partition, n);
//==  *
//==  * remap_from_fft(gkvec_partition, wf_fft, wf, N, n)
//==  *
//==  * consider Wave_functions_fft class
//==  *
//==  *
//==  * Wave_functions wf(...);
//==  * memory_guard mem_guard(wf, memory_t::device);
//==  *
//==  *
//==  *
//==  */
//== 
//== enum class copy_to : unsigned int
//== {
//==     none   = 0b0000,
//==     device = 0b0001,
//==     host   = 0b0010
//== };
//== inline copy_to operator|(copy_to a__, copy_to b__)
//== {
//==     return static_cast<copy_to>(static_cast<unsigned int>(a__) | static_cast<unsigned int>(b__));
//== }
//== 
//== template <typename T>
//== class device_memory_guard
//== {
//==   private:
//==     T& obj_;
//==     device_memory_guard(device_memory_guard const&) = delete;
//==     device_memory_guard& operator=(device_memory_guard const&) = delete;
//==     sddk::memory_t mem_;
//==     copy_to copy_to_;
//==   public:
//==     device_memory_guard(T& obj__, sddk::memory_t mem__, copy_to copy_to__)
//==         : obj_(obj__)
//==         , mem_(mem__)
//==         , copy_to_(copy_to__)
//==     {
//==         if (is_device_memory(mem_)) {
//==             obj_.allocate(mem_);
//==             if (static_cast<unsigned int>(copy_to_) & static_cast<unsigned int>(copy_to::device)) {
//==                 obj_.copy_to(mem_);
//==             }
//==         }
//==     }
//==     device_memory_guard(device_memory_guard&& src__) = default;
//==     ~device_memory_guard()
//==     {
//==         if (is_device_memory(mem_)) {
//==             if (static_cast<unsigned int>(copy_to_) & static_cast<unsigned int>(copy_to::host)) {
//==                 obj_.copy_to(sddk::memory_t::host);
//==             }
//==             obj_.deallocate(mem_);
//==         }
//==     }
//== };
//== 
//== template <typename T>
//== class Wave_functions_base
//== {
//==   private:
//==     int ld_;
//==     int num_wf_;
//==     int num_sc_;
//== 
//==     inline void allocate(sddk::memory_t mem__)
//==     {
//==         for (int is = 0; is < num_sc_; is++) {
//==             data_[is].allocate(mem__);
//==         }
//==     }
//== 
//==     inline void deallocate(sddk::memory_t mem__)
//==     {
//==         for (int is = 0; is < num_sc_; is++) {
//==             data_[is].deallocate(mem__);
//==         }
//==     }
//== 
//==     inline void copy_to(sddk::memory_t mem__)
//==     {
//==         for (int is = 0; is < num_sc_; is++) {
//==             data_[is].copy_to(mem__);
//==         }
//==     }
//== 
//==     friend class device_memory_guard<Wave_functions_base<T>>;
//== 
//==   protected:
//==     std::vector<sddk::mdarray<std::complex<T>, 2>> data_;
//==   public:
//==     Wave_functions_base(int ld__, int num_wf__, int num_sc__, sddk::memory_t default_mem__)
//==         : ld_(ld__)
//==         , num_wf_(num_wf__)
//==         , num_sc_(num_sc__)
//==     {
//==         data_.resize(num_sc_);
//==         for (int is = 0; is < num_sc_; is++) {
//==             data_[is] = sddk::mdarray<std::complex<T>, 2>(ld_, num_wf_, default_mem__, "Wave_functions_base::data_");
//==         }
//==     }
//== 
//==     auto memory_guard(sddk::memory_t mem__, experimental::copy_to copy_to__ = experimental::copy_to::none)
//==     {
//==         return std::move(device_memory_guard<Wave_functions_base<T>>(*this, mem__, copy_to__));
//==     }
//== };
//== 
//== template <typename T>
//== class Wave_functions : public Wave_functions_base<T>
//== {
//==   private:
//==     std::shared_ptr<sddk::Gvec> gkvec_;
//==     int num_atoms_{0};
//==     sddk::splindex<sddk::splindex_t::block> spl_num_atoms_;
//==     /// Local size of muffin-tin coefficients for each rank.
//==     /** Each rank stores local fraction of atoms. Each atom has a set of MT coefficients. */
//==     sddk::block_data_descriptor mt_coeffs_distr_;
//==     /// Local offset in the block of MT coefficients for current rank.
//==     /** The size of the vector is equal to the local number of atoms for the current rank. */
//==     std::vector<int> offset_in_local_mt_coeffs_;
//==     /// Total numbef of muffin-tin coefficients.
//==     int num_mt_coeffs_{0};
//==     static int get_local_num_mt_coeffs(std::vector<int> num_mt_coeffs__, sddk::Communicator const& comm__)
//==     {
//==         int num_atoms = static_cast<int>(num_mt_coeffs__.size());
//==         sddk::splindex<sddk::splindex_t::block> spl_atoms(num_atoms, comm__.size(), comm__.rank());
//==         auto it_begin = num_mt_coeffs__.begin() + spl_atoms.global_offset();
//==         auto it_end = it_begin + spl_atoms.local_size();
//==         return std::accumulate(it_begin, it_end, 0);
//==     }
//== 
//==   public:
//==     Wave_functions(std::shared_ptr<sddk::Gvec> gkvec__, int num_wf__, int num_sc__, sddk::memory_t default_mem__)
//==         : Wave_functions_base<T>(gkvec__->count(), num_wf__, num_sc__, default_mem__)
//==         , gkvec_(gkvec__)
//==     {
//==     }
//==     Wave_functions(std::shared_ptr<sddk::Gvec> gkvec__, std::vector<int> num_mt_coeffs__, int num_wf__, int num_sc__,
//==             sddk::memory_t default_mem__)
//==         : Wave_functions_base<T>(gkvec__->count() + get_local_num_mt_coeffs(num_mt_coeffs__, gkvec__->comm()),
//==                                  num_wf__, num_sc__, default_mem__)
//==         , gkvec_(gkvec__)
//==         , num_atoms_(static_cast<int>(num_mt_coeffs__.size()))
//==         , spl_num_atoms_(sddk::splindex<sddk::splindex_t::block>(num_atoms_, gkvec__->comm().size(), gkvec__->comm().rank()))
//==         , num_mt_coeffs_(std::accumulate(num_mt_coeffs__.begin(), num_mt_coeffs__.end(), 0))
//==     {
//==         mt_coeffs_distr_ = sddk::block_data_descriptor(gkvec_->comm().size());
//== 
//==         for (int ia = 0; ia < num_atoms_; ia++) {
//==             int rank = spl_num_atoms_.local_rank(ia);
//==             if (rank == gkvec_->comm().rank()) {
//==                 offset_in_local_mt_coeffs_.push_back(mt_coeffs_distr_.counts[rank]);
//==             }
//==             /* increment local number of MT coeffs. for a given rank */
//==             mt_coeffs_distr_.counts[rank] += num_mt_coeffs__[ia];
//==         }
//==         mt_coeffs_distr_.calc_offsets();
//==     }
//==     ~Wave_functions()
//==     {
//==     }
//== 
//==     inline sddk::mdarray<std::complex<T>, 2>& pw_coeffs(int ispn__)
//==     {
//==         return this->data_[ispn__];
//==     }
//== 
//==     auto grid_layout_pw(int ispn__, int N__, int n__)
//==     {
//==         auto& comm = gkvec_->comm();
//== 
//==         std::vector<int> rowsplit(comm.size() + 1);
//==         rowsplit[0] = 0;
//==         for (int i = 0; i < comm.size(); i++) {
//==             rowsplit[i + 1] = rowsplit[i] + gkvec_->gvec_count(i);
//==         }
//==         std::vector<int> colsplit({0, n__});
//==         std::vector<int> owners(comm.size());
//==         for (int i = 0; i < comm.size(); i++) {
//==             owners[i] = i;
//==         }
//==         costa::block_t localblock;
//==         localblock.data = this->data_[ispn__].at(sddk::memory_t::host, 0, N__);
//==         localblock.ld = this->data_[ispn__].ld();
//==         localblock.row = comm.rank();
//==         localblock.col = 0;
//== 
//==         return costa::custom_layout<std::complex<T>>(gkvec_->comm().size(), 1, rowsplit.data(), colsplit.data(),
//==                 owners.data(), 1, &localblock, 'C');
//==     }
//== 
//==     auto grid_layout_mt(int ispn__, int N__, int n__)
//==     {
//==         auto& comm = gkvec_->comm();
//== 
//==         std::vector<int> rowsplit(comm.size() + 1);
//==         rowsplit[0] = 0;
//==         for (int i = 0; i < comm.size(); i++) {
//==             rowsplit[i + 1] = rowsplit[i] + mt_coeffs_distr_.counts[i];
//==         }
//==         std::vector<int> colsplit({0, n__});
//==         std::vector<int> owners(comm.size());
//==         for (int i = 0; i < comm.size(); i++) {
//==             owners[i] = i;
//==         }
//==         costa::block_t localblock;
//==         localblock.data = this->data_[ispn__].at(sddk::memory_t::host, gkvec_->count(), N__);
//==         localblock.ld = this->data_[ispn__].ld();
//==         localblock.row = comm.rank();
//==         localblock.col = 0;
//== 
//==         return costa::custom_layout<std::complex<T>>(gkvec_->comm().size(), 1, rowsplit.data(), colsplit.data(),
//==                 owners.data(), 1, &localblock, 'C');
//==     }
//== 
//==     auto const& gkvec() const
//==     {
//==         return *gkvec_;
//==     }
//== };
//== 
//== template <typename T>
//== class Wave_functions_fft : public Wave_functions_base<T>
//== {
//==   private:
//==     std::shared_ptr<sddk::Gvec_partition> gkvec_fft_;
//==     sddk::splindex<sddk::splindex_t::block> spl_num_wf_;
//== 
//==   public:
//==     Wave_functions_fft(std::shared_ptr<sddk::Gvec_partition> gkvec_fft__, int num_wf_max__, sddk::memory_t default_mem__)
//==         : Wave_functions_base<T>(gkvec_fft__->gvec_count_fft(),
//==                 sddk::splindex<sddk::splindex_t::block>(num_wf_max__, gkvec_fft__->comm_ortho_fft().size(),
//==                     gkvec_fft__->comm_ortho_fft().rank()).local_size(), 1, default_mem__)
//==         , gkvec_fft_(gkvec_fft__)
//==     {
//==     }
//== 
//==     auto grid_layout(int n__)
//==     {
//==         auto& comm_row = gkvec_fft_->comm_fft();
//==         auto& comm_col = gkvec_fft_->comm_ortho_fft();
//== 
//==         spl_num_wf_ = sddk::splindex<sddk::splindex_t::block>(n__, comm_col.size(), comm_col.rank());
//== 
//==         std::vector<int> rowsplit(comm_row.size() + 1);
//==         rowsplit[0] = 0;
//==         for (int i = 0; i < comm_row.size(); i++) {
//==             rowsplit[i + 1] = rowsplit[i] + gkvec_fft_->gvec_count_fft(i);
//==         }
//== 
//==         std::vector<int> colsplit(comm_col.size() + 1);
//==         colsplit[0] = 0;
//==         for (int i = 0; i < comm_col.size(); i++) {
//==             colsplit[i + 1] = colsplit[i] + spl_num_wf_.local_size(i);
//==         }
//== 
//==         std::vector<int> owners(gkvec_fft_->gvec().comm().size());
//==         for (int i = 0; i < gkvec_fft_->gvec().comm().size(); i++) {
//==             owners[i] = i;
//==         }
//==         costa::block_t localblock;
//==         localblock.data = this->data_[0].at(sddk::memory_t::host);
//==         localblock.ld = this->data_[0].ld();
//==         localblock.row = gkvec_fft_->comm_fft().rank();
//==         localblock.col = comm_col.rank();
//== 
//==         return costa::custom_layout<std::complex<T>>(comm_row.size(), comm_col.size(), rowsplit.data(),
//==                 colsplit.data(), owners.data(), 1, &localblock, 'C');
//==     }
//== 
//==     int num_wf_local() const
//==     {
//==         return spl_num_wf_.local_size();
//==     }
//== 
//==     inline std::complex<T>& pw_coeffs(int ig__, int i__)
//==     {
//==         return this->data_[0](ig__, i__);
//==     }
//== 
//==     inline T* pw_coeffs(sddk::memory_t mem__, int i__)
//==     {
//==         return reinterpret_cast<T*>(this->data_[0].at(mem__, 0, i__));
//==     }
//== };
//== 
//== template <typename T>
//== void transform_to_fft_layout(exper::Wave_functions<T>& wf_in__, exper::Wave_functions_fft<T>& wf_fft_out__,
//==         std::shared_ptr<sddk::Gvec_partition> gkvec_fft__, int ispn__, int N__, int n__)
//== {
//== 
//==     auto layout_in  = wf_in__.grid_layout_pw(ispn__, N__, n__);
//==     auto layout_out = wf_fft_out__.grid_layout(n__);
//== 
//==     costa::transform(layout_in, layout_out, 'N', sddk::linalg_const<std::complex<T>>::one(),
//==             sddk::linalg_const<std::complex<T>>::zero(), gkvec_fft__->gvec().comm().mpi_comm());
//== }
//== 
//== template <typename T>
//== void transform_from_fft_layout(exper::Wave_functions_fft<T>& wf_fft_in__, exper::Wave_functions<T>& wf_out__,
//==         int ispn__, int N__, int n__)
//== {
//==     auto layout_in  = wf_fft_in__.grid_layout(n__);
//==     auto layout_out = wf_out__.grid_layout_pw(ispn__, N__, n__);
//== 
//==     costa::transform(layout_in, layout_out, 'N', sddk::linalg_const<std::complex<T>>::one(),
//==             sddk::linalg_const<std::complex<T>>::zero(), wf_out__.gkvec().comm().mpi_comm());
//== }

}



#endif // __WAVE_FUNCTIONS_HPP__

