// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file beta_projectors_base.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_base class.
 */

#ifndef __BETA_PROJECTORS_BASE_HPP__
#define __BETA_PROJECTORS_BASE_HPP__

#include "context/simulation_context.hpp"
#include "SDDK/wave_functions.hpp"
#include "memory.hpp"
#include "mpi/communicator.hpp"
#include <spla/context.hpp>

namespace sirius {

#if defined(SIRIUS_GPU)
extern "C" {

void create_beta_gk_gpu_float(int num_atoms, int num_gkvec, int const* beta_desc, std::complex<float> const* beta_gk_t,
                              double const* gkvec, double const* atom_pos, std::complex<float>* beta_gk);

void create_beta_gk_gpu_double(int num_atoms, int num_gkvec, int const* beta_desc,
                               std::complex<double> const* beta_gk_t, double const* gkvec, double const* atom_pos,
                               std::complex<double>* beta_gk);
}
#endif

/// Named index of a descriptor of beta-projectors. The same order is used by the GPU kernel.
struct beta_desc_idx
{
    /// Number of beta-projector functions for this atom.
    static const int nbf = 0;
    /// Offset of beta-projectors in this chunk.
    static const int offset = 1;
    /// Offset of beta-projectors in the array for atom types.
    static const int offset_t = 2;
    /// Global index of atom.
    static const int ia = 3;
};

struct beta_chunk_t
{
    /// Number of beta-projectors in the current chunk.
    int num_beta_;
    /// Number of atoms in the current chunk.
    int num_atoms_;
    /// Offset in the global index of beta projectors.
    int offset_;
    /// Descriptor of block of beta-projectors for an atom.
    sddk::mdarray<int, 2> desc_;
    /// Positions of atoms.
    sddk::mdarray<double, 2> atom_pos_;

    beta_chunk_t() = default;

    beta_chunk_t(const beta_chunk_t& other)
        : desc_{empty_like(other.desc_)}
        , atom_pos_{empty_like(other.atom_pos_)}
    {
        // pass
        num_beta_  = other.num_beta_;
        num_atoms_ = other.num_atoms_;
        offset_    = other.offset_;

        auto_copy(desc_, other.desc_);
        auto_copy(atom_pos_, other.atom_pos_);
    }

    beta_chunk_t& operator=(const beta_chunk_t& other)
    {
        num_beta_  = other.num_beta_;
        num_atoms_ = other.num_atoms_;
        offset_    = other.offset_;

        desc_ = empty_like(other.desc_);
        auto_copy(desc_, other.desc_);

        atom_pos_ = empty_like(other.atom_pos_);
        auto_copy(atom_pos_, other.atom_pos_);
        return *this;
    }
};

/// Stores a chunk of the beta-projector and metadata.
/**
 *  \tparam T Precision type (float or double)
 *
 */
template <typename T>
struct beta_projectors_coeffs_t
{
    using complex_t = std::complex<T>;

    sddk::matrix<complex_t> pw_coeffs_a;
    mpi::Communicator communicator;
    /// the current beta chunk
    beta_chunk_t beta_chunk;
    /// buffer (num_max_beta) for pw_coeffs_a_
    sddk::matrix<complex_t> __pw_coeffs_a_buffer;

    auto actual_spin_index(wf::spin_index s__) const -> wf::spin_index
    {
        return wf::spin_index(0);
    }

    auto ld() const
    {
        return pw_coeffs_a.ld();
    }

    auto num_md() const
    {
        return wf::num_mag_dims(0);
    }

    auto comm() const -> const mpi::Communicator&
    {
        return communicator;
    }

    std::complex<T> const* at(sddk::memory_t mem__, int i__, wf::spin_index s__, wf::band_index b__) const
    {
        return pw_coeffs_a.at(mem__, i__, b__.get());
    }
};

namespace local {

template <class T>
void beta_projectors_generate_cpu(sddk::matrix<std::complex<T>>& pw_coeffs_a,
                                  const sddk::mdarray<std::complex<T>, 3>& pw_coeffs_t, int ichunk__, int j__,
                                  const beta_chunk_t& beta_chunk, const Simulation_context& ctx,
                                  const fft::Gvec& gkvec);

template <class T>
void beta_projectors_generate_gpu(beta_projectors_coeffs_t<T>& out,
                                  const sddk::mdarray<std::complex<double>, 3>& pw_coeffs_t_device,
                                  const sddk::mdarray<std::complex<double>, 3>& pw_coeffs_t_host,
                                  const Simulation_context& ctx, const fft::Gvec& gkvec,
                                  const sddk::mdarray<double, 2>& gkvec_coord_, const beta_chunk_t& beta_chunk,
                                  const std::vector<int>& igk__, int j__);

} // namespace local

/// Generates beta projector PW coefficients and holds GPU memory phase-factor
/// independent coefficients of |> functions for atom types.
template <typename T>
class Beta_projector_generator
{
  public:
    typedef std::complex<T> complex_t;
    typedef sddk::mdarray<complex_t, 3> array_t;

  public:
    Beta_projector_generator(Simulation_context& ctx, const array_t& pw_coeffs_t_host,
                             const sddk::matrix<std::complex<T>>& beta_pw_all, sddk::device_t processing_unit,
                             const std::vector<beta_chunk_t>& beta_chunks, const fft::Gvec& gkvec,
                             const sddk::mdarray<double, 2>& gkvec_coord, int num_gkvec_loc);

    void generate(beta_projectors_coeffs_t<T>& coeffs, int ichunk, int j) const;
    void generate(beta_projectors_coeffs_t<T>& coeffs, int ichunk) const;

    beta_projectors_coeffs_t<T> prepare() const;

    Simulation_context& ctx()
    {
        return ctx_;
    }
    int num_chunks() const
    {
        return beta_chunks_.size();
    }

    const auto& chunks() const
    {
        return beta_chunks_;
    }

    auto device_t() const -> sddk::device_t
    {
        return processing_unit_;
    }

  private:
    Simulation_context& ctx_;
    const array_t& pw_coeffs_t_host_;
    /// precomputed beta coefficients on CPU
    const sddk::matrix<complex_t>& beta_pw_all_atoms_;
    sddk::device_t processing_unit_;
    /// chunk descriptors
    const std::vector<beta_chunk_t>& beta_chunks_;
    /// gkvec
    const fft::Gvec& gkvec_;
    const sddk::mdarray<double, 2>& gkvec_coord_;
    int num_gkvec_loc_;
    /// pw_coeffs_t on device
    array_t pw_coeffs_t_device_;
    int max_num_beta_;
};

template <typename T>
Beta_projector_generator<T>::Beta_projector_generator(Simulation_context& ctx, const array_t& pw_coeffs_t_host,
                                                      const sddk::matrix<std::complex<T>>& beta_pw_all,
                                                      sddk::device_t processing_unit,
                                                      const std::vector<beta_chunk_t>& beta_chunks,
                                                      const fft::Gvec& gkvec,
                                                      const sddk::mdarray<double, 2>& gkvec_coord, int num_gkvec_loc)
    : ctx_(ctx)
    , pw_coeffs_t_host_(pw_coeffs_t_host)
    , beta_pw_all_atoms_(beta_pw_all)
    , processing_unit_(processing_unit)
    , beta_chunks_(beta_chunks)
    , gkvec_(gkvec)
    , gkvec_coord_(gkvec_coord)
    , num_gkvec_loc_(num_gkvec_loc)
{
    if (processing_unit == sddk::device_t::GPU) {
        pw_coeffs_t_device_ = array_t(pw_coeffs_t_host.size(0), pw_coeffs_t_host.size(1), pw_coeffs_t_host.size(2),
                                      sddk::get_memory_pool(sddk::memory_t::device));
        // copy to device
        acc::copyin(pw_coeffs_t_device_.device_data(), pw_coeffs_t_host.host_data(), pw_coeffs_t_host.size());
    }

    int max_num_beta = 0;
    for (auto& e : beta_chunks_) {
        max_num_beta = std::max(max_num_beta, e.num_beta_);
    }
    this->max_num_beta_ = max_num_beta;
}

template <typename T>
beta_projectors_coeffs_t<T>
Beta_projector_generator<T>::prepare() const
{
    beta_projectors_coeffs_t<T> beta_storage;
    beta_storage.communicator = gkvec_.comm().duplicate();

    if (processing_unit_ == sddk::device_t::GPU) {
        beta_storage.__pw_coeffs_a_buffer =
            sddk::matrix<std::complex<T>>(num_gkvec_loc_, max_num_beta_, sddk::get_memory_pool(sddk::memory_t::device));
    }

    return beta_storage;
}

/// Base class for beta-projectors, gradient of beta-projectors and strain derivatives of beta-projectors.
/** \tparam T  Precision of beta-projectors (float or double).
 */
template <typename T>
class Beta_projectors_base
{
  protected:
    Simulation_context& ctx_;

    /// List of G+k vectors.
    fft::Gvec const& gkvec_;

    /// Coordinates of G+k vectors used by GPU kernel.
    sddk::mdarray<double, 2> gkvec_coord_;

    /// Number of different components: 1 for beta-projectors, 3 for gradient, 9 for strain derivatives.
    int N_;

    /// Phase-factor independent coefficients of |beta> functions for atom types.
    sddk::mdarray<std::complex<T>, 3> pw_coeffs_t_;

    bool reallocate_pw_coeffs_t_on_gpu_{true};

    /// Set of beta PW coefficients for a chunk of atoms.
    sddk::matrix<std::complex<T>> pw_coeffs_a_;

    /// Set of beta PW coefficients for all atoms
    sddk::matrix<std::complex<T>> beta_pw_all_atoms_;

    std::vector<beta_chunk_t> beta_chunks_;

    int max_num_beta_;

    /// total number of beta-projectors (=number of columns)
    int num_total_beta_;

    /// Total number of beta-projectors among atom types.
    int num_beta_t_;

    /// Split beta-projectors into chunks.
    void split_in_chunks();

  public:
    Beta_projectors_base(Simulation_context& ctx__, fft::Gvec const& gkvec__, int N__);

    Beta_projector_generator<T> make_generator() const
    {
        return make_generator(ctx_.processing_unit());
    }

    Beta_projector_generator<T> make_generator(sddk::device_t pu) const
    {
        return Beta_projector_generator<T>{ctx_,         pw_coeffs_t_, beta_pw_all_atoms_, pu,
                                           beta_chunks_, gkvec_,       gkvec_coord_,       num_gkvec_loc()};
    }

    Beta_projector_generator<T> make_generator(sddk::memory_t mem) const
    {
        sddk::device_t pu{sddk::device_t::CPU};
        if (sddk::is_device_memory(mem)) {
            pu = sddk::device_t::GPU;
        }
        return make_generator(pu);
    }

    Simulation_context& ctx()
    {
        return ctx_;
    }

    // /// Calculate inner product between beta-projectors and wave-functions.
    // /** The following is matrix computed: <beta|phi>
    //  *
    //  *  \tparam F  Type of the resulting inner product matrix (float, double, complex<float> or complex<double>).
    //  */
    // template <typename F>
    // __attribute__((depcrecated)) std::enable_if_t<std::is_same<T, real_type<F>>::value, la::dmatrix<F>>
    // inner(sddk::memory_t mem__, int chunk__, wf::Wave_functions<T> const& phi__, wf::spin_index ispn__,
    //       wf::band_range br__) const
    // {
    //     throw std::runtime_error("Beta_projector_base::inner has been removed.");
    // }

    inline auto num_gkvec_loc() const
    {
        return gkvec_.count();
    }

    int num_total_beta() const
    {
        return num_total_beta_;
    }

    inline int num_comp() const
    {
        return N_;
    }

    inline auto const& unit_cell() const
    {
        return ctx_.unit_cell();
    }

    std::complex<T>& pw_coeffs_t(int ig__, int n__, int j__)
    {
        return pw_coeffs_t_(ig__, n__, j__);
    }

    sddk::matrix<std::complex<T>> pw_coeffs_t(int j__)
    {
        return sddk::matrix<std::complex<T>>(&pw_coeffs_t_(0, 0, j__), num_gkvec_loc(), num_beta_t());
    }

    // /// Plane wave coefficients of |beta> projectors for a chunk of atoms.
    // __attribute_deprecated__ matrix<std::complex<T>>& pw_coeffs_a()
    // {
    //     throw std::runtime_error("beta_projectors::pw_coeffs_a is not used anymore!");
    //     return pw_coeffs_a_;
    // }

    auto const& pw_coeffs_a() const
    {
        return pw_coeffs_a_;
    }

    inline int num_beta_t() const
    {
        return num_beta_t_;
    }

    inline int num_chunks() const
    {
        return static_cast<int>(beta_chunks_.size());
    }

    // __attribute_deprecated__
    // inline beta_chunk_t const& chunk(int idx__) const
    // {
    //     return beta_chunks_[idx__];
    // }

    inline int max_num_beta() const
    {
        return max_num_beta_;
    }

    int nrows() const
    {
        return gkvec_.num_gvec();
    }

    const mpi::Communicator& comm() const
    {
        return gkvec_.comm();
    }
};

/** The following is matrix computed: <beta|phi>
 *
 *  \tparam F  Type of the resulting inner product matrix (float, double, complex<float> or complex<double>).
 *  \tparam T  precision type
 *
 *  \param  [in]  spla_ctx          Context of the SPLA library
 *  \param  [in]  mem               Location of the input arrays (wfc and beta-projectors)
 *  \param  [in]  host_mem          Host memory type for result allocation (pinned, non-pinned memory)
 *  \param  [in]  result_on_device  Copy result to device if true
 *  \param  [in]  beta_coeffs       Beta-projector coefficient array
 *  \param  [in]  phi               Wave-function
 *  \param  [in]  ispn              Spin index (wfc)
 *  \param  [in]  br                Band range
 *  \return inner product
 */
template <typename F, typename T>
std::enable_if_t<std::is_same<T, real_type<F>>::value, la::dmatrix<F>>
inner_prod_beta(spla::Context& spla_ctx, sddk::memory_t mem__, sddk::memory_t host_mem__, bool result_on_device,
                beta_projectors_coeffs_t<T>& beta_coeffs__, wf::Wave_functions<T> const& phi__, wf::spin_index ispn__,
                wf::band_range br__)
{
    int nbeta = beta_coeffs__.beta_chunk.num_beta_;

    la::dmatrix<F> result(nbeta, br__.size(), get_memory_pool(host_mem__), "<beta|phi>");
    if (result_on_device) {
        result.allocate(get_memory_pool(sddk::memory_t::device));
    }

    wf::inner<F>(spla_ctx, mem__, wf::spin_range(ispn__.get()), beta_coeffs__, wf::band_range(0, nbeta), phi__, br__,
                 result, 0, 0);

    return result;
}

/// computes <beta|beta> and returns result on ctx.processing_unit_memory_t
template <class T>
sddk::matrix<std::complex<T>>
inner_beta(const Beta_projectors_base<T>& beta, const Simulation_context& ctx)
{
    using complex_t     = std::complex<T>;
    auto generator      = beta.make_generator();
    int num_beta_chunks = beta.num_chunks();
    auto bcoeffs_row    = generator.prepare();
    auto bcoeffs_col    = generator.prepare();
    auto mem_t          = ctx.processing_unit_memory_t();

    la::lib_t la{la::lib_t::blas};
    if (sddk::is_device_memory(mem_t)) {
        la = la::lib_t::gpublas;
    }

    int size{beta.num_total_beta()};

    sddk::matrix<complex_t> out(size, size, sddk::get_memory_pool(mem_t));

    complex_t one  = complex_t(1);
    complex_t zero = complex_t(0);

    for (int ichunk = 0; ichunk < num_beta_chunks; ++ichunk) {
        generator.generate(bcoeffs_row, ichunk);

        for (int jchunk = 0; jchunk < num_beta_chunks; ++jchunk) {
            generator.generate(bcoeffs_col, jchunk);
            int m              = bcoeffs_row.beta_chunk.num_beta_;
            int n              = bcoeffs_col.beta_chunk.num_beta_;
            int k              = bcoeffs_col.pw_coeffs_a.size(0);
            int dest_row       = bcoeffs_row.beta_chunk.offset_;
            int dest_col       = bcoeffs_col.beta_chunk.offset_;
            const complex_t* A = bcoeffs_row.pw_coeffs_a.at(mem_t);
            const complex_t* B = bcoeffs_col.pw_coeffs_a.at(mem_t);
            complex_t* C       = out.at(mem_t, dest_row, dest_col);
            la::wrap(la).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a.ld(), B, bcoeffs_col.pw_coeffs_a.ld(),
                              &zero, C, out.ld());
        }
    }

    if (beta.comm().size() > 1) {
        RTE_THROW("this needs to be fixed first");
        beta.comm().allreduce(out.at(sddk::memory_t::host), static_cast<int>(out.size()));
    }

    return out;
}

/// inner product <beta|Op|beta>, return resulting dmatrix<complex> in ctx.processing_unit_memory_t
template <class T, class Op>
sddk::matrix<std::complex<T>>
inner_beta(const Beta_projectors_base<T>& beta, const Simulation_context& ctx, Op&& op)
{
    using complex_t     = std::complex<double>;
    auto generator      = beta.make_generator();
    int num_beta_chunks = beta.num_chunks();
    auto bcoeffs_row    = generator.prepare();
    auto bcoeffs_col    = generator.prepare();
    auto mem_t          = ctx.processing_unit_memory_t();

    la::lib_t la{la::lib_t::blas};
    if (sddk::is_device_memory(mem_t)) {
        la = la::lib_t::gpublas;
    }

    int size{beta.num_total_beta()};

    sddk::matrix<complex_t> out(size, size, mem_t);

    complex_t one  = complex_t(1);
    complex_t zero = complex_t(0);

    for (int ichunk = 0; ichunk < num_beta_chunks; ++ichunk) {
        generator.generate(bcoeffs_row, ichunk);

        for (int jchunk = 0; jchunk < num_beta_chunks; ++jchunk) {
            generator.generate(bcoeffs_col, jchunk);

            int m              = bcoeffs_row.beta_chunk.num_beta_;
            int n              = bcoeffs_col.beta_chunk.num_beta_;
            int k              = bcoeffs_col.pw_coeffs_a.size(0);
            int dest_row       = bcoeffs_row.beta_chunk.offset_;
            int dest_col       = bcoeffs_col.beta_chunk.offset_;
            const complex_t* A = bcoeffs_row.pw_coeffs_a.at(mem_t);
            // apply Op on |b>  (in-place operation)
            auto G = op(bcoeffs_col.pw_coeffs_a);

            const complex_t* B2 = G.at(mem_t);
            complex_t* C        = out.at(mem_t, dest_row, dest_col);
            la::wrap(la).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a.ld(), B2, G.ld(), &zero, C, out.ld());
        }
    }
    if (beta.comm().size() > 1) {
        beta.comm().allreduce(out.at(sddk::memory_t::host), static_cast<int>(out.size()));
    }

    return out;
}

} // namespace sirius

#endif
