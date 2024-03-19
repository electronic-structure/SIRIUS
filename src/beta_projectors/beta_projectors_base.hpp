/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file beta_projectors_base.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_base class.
 */

#ifndef __BETA_PROJECTORS_BASE_HPP__
#define __BETA_PROJECTORS_BASE_HPP__

#include <spla/context.hpp>
#include "core/wf/wave_functions.hpp"
#include "core/mpi/communicator.hpp"
#include "context/simulation_context.hpp"

namespace sirius {

#if defined(SIRIUS_GPU)
extern "C" {

void
create_beta_gk_gpu_float(int num_atoms, int num_gkvec, int const* beta_desc, std::complex<float> const* beta_gk_t,
                         double const* gkvec, double const* atom_pos, std::complex<float>* beta_gk);

void
create_beta_gk_gpu_double(int num_atoms, int num_gkvec, int const* beta_desc, std::complex<double> const* beta_gk_t,
                          double const* gkvec, double const* atom_pos, std::complex<double>* beta_gk);
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

/// Describe chunk of beta-projectors for a block of atoms.
/** Beta-projectors are processed in "chunks" to avoid large memory consumption. */
struct beta_chunk_t
{
    /// Number of beta-projectors in the current chunk.
    int num_beta_;
    /// Number of atoms in the current chunk.
    int num_atoms_;
    /// Offset in the global index of beta projectors.
    int offset_;
    /// Descriptor of block of beta-projectors for an atom.
    mdarray<int, 2> desc_;
    /// Positions of atoms.
    mdarray<double, 2> atom_pos_;
};

/// Stores a chunk of the beta-projector and metadata.
/** A chunk of beta-projectors is stored as a wave-function-like object, so that it can be
 *  used in functions like wf::inner().
 *
 *  \tparam T Precision type (float or double)
 */
template <typename T>
struct beta_projectors_coeffs_t
{
    using complex_t = std::complex<T>;

    /// Communicator that splits G+k vectors.
    mpi::Communicator const& comm_;
    /// Storage for the plane-wave coefficients array.
    matrix<complex_t> pw_coeffs_a_;
    /// Descriptor of the current beta chunk.
    beta_chunk_t const* beta_chunk_{nullptr};

    /// Beta-projectors are treated as non-magnetic.
    auto
    actual_spin_index(wf::spin_index s__) const -> wf::spin_index
    {
        return wf::spin_index(0);
    }

    /// Leading dimension.
    auto
    ld() const
    {
        return static_cast<int>(pw_coeffs_a_.ld());
    }

    auto
    num_md() const
    {
        return wf::num_mag_dims(0);
    }

    auto
    comm() const -> const mpi::Communicator&
    {
        return comm_;
    }

    auto const*
    at(memory_t mem__, int i__, wf::spin_index s__, wf::band_index b__) const
    {
        return pw_coeffs_a_.at(mem__, i__, b__.get());
    }
};

namespace local {

template <class T>
void
beta_projectors_generate_cpu(matrix<std::complex<T>>& pw_coeffs_a, mdarray<std::complex<T>, 3> const& pw_coeffs_t,
                             int ichunk__, int j__, beta_chunk_t const& beta_chunk, Simulation_context const& ctx,
                             fft::Gvec const& gkvec);

template <class T>
void
beta_projectors_generate_gpu(beta_projectors_coeffs_t<T>& out,
                             mdarray<std::complex<double>, 3> const& pw_coeffs_t_device, Simulation_context const& ctx,
                             fft::Gvec const& gkvec, mdarray<double, 2> const& gkvec_coord_,
                             beta_chunk_t const& beta_chunk, std::vector<int> const& igk__, int j__);

} // namespace local

/** Generates beta projector PW coefficients and holds GPU memory phase-factor
 *  independent coefficients of |> functions for atom types.*/
template <typename T>
class Beta_projector_generator
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;
    /// Processing unit.
    device_t pu_;
    /// Beta-projectors for atom-types.
    /** Local array keeps track of GPU memory deallocation */
    mdarray<std::complex<T>, 3> pw_coeffs_t_;
    /// Precomputed beta coefficients.
    mdarray<std::complex<T>, 3> const& pw_coeffs_all_atoms_;
    /// Chunk descriptors.
    std::vector<beta_chunk_t> const& beta_chunks_;
    /// G+k vectors.
    fft::Gvec const& gkvec_;
    /// Coordinates of G+k vectors.
    mdarray<double, 2> const& gkvec_coord_;

  public:
    Beta_projector_generator(Simulation_context& ctx__, device_t pu__, mdarray<std::complex<T>, 3> const& pw_coeffs_t__,
                             mdarray<std::complex<T>, 3> const& pw_coeffs_all_atoms__,
                             std::vector<beta_chunk_t> const& beta_chunks__, fft::Gvec const& gkvec__,
                             mdarray<double, 2> const& gkvec_coord__)
        : ctx_{ctx__}
        , pu_{pu__}
        , pw_coeffs_all_atoms_{pw_coeffs_all_atoms__}
        , beta_chunks_{beta_chunks__}
        , gkvec_{gkvec__}
        , gkvec_coord_{gkvec_coord__}
    {
        std::complex<T>* ptr_h = const_cast<std::complex<T>*>(pw_coeffs_t__.at(memory_t::host));
        std::complex<T>* ptr_d{nullptr};
        if (pu_ == device_t::GPU && pw_coeffs_t__.on_device()) {
            ptr_d = const_cast<std::complex<T>*>(pw_coeffs_t__.at(memory_t::device));
        }
        /* wrap incoming pw_coeffs_t__ in local mdarray */
        pw_coeffs_t_ =
                mdarray<std::complex<T>, 3>({pw_coeffs_t__.size(0), pw_coeffs_t__.size(1), pw_coeffs_t__.size(2)},
                                            ptr_h, ptr_d, mdarray_label("pw_coeffs_t_"));
        /* allocate GPU memeory if needed */
        if (pu_ == device_t::GPU && !ptr_d) {
            pw_coeffs_t_.allocate(memory_t::device).copy_to(memory_t::device);
        }
    }

    auto
    prepare() const
    {
        beta_projectors_coeffs_t<T> beta_storage{gkvec_.comm()};

        int max_num_beta{0};
        for (auto& e : beta_chunks_) {
            max_num_beta = std::max(max_num_beta, e.num_beta_);
        }

        switch (pu_) {
            case device_t::CPU: {
                if (pw_coeffs_all_atoms_.size() == 0) {
                    beta_storage.pw_coeffs_a_ =
                            matrix<std::complex<T>>({gkvec_.count(), max_num_beta}, get_memory_pool(memory_t::host));
                }
                break;
            }
            case device_t::GPU: {
                if (!pw_coeffs_all_atoms_.on_device()) {
                    beta_storage.pw_coeffs_a_ =
                            matrix<std::complex<T>>({gkvec_.count(), max_num_beta}, get_memory_pool(memory_t::device));
                }
                break;
            }
        }
        return beta_storage;
    }

    void
    generate(beta_projectors_coeffs_t<T>& coeffs, int ichunk, int j) const;

    void
    generate(beta_projectors_coeffs_t<T>& coeffs, int ichunk) const
    {
        this->generate(coeffs, ichunk, 0);
    }

    Simulation_context&
    ctx()
    {
        return ctx_;
    }

    int
    num_chunks() const
    {
        return beta_chunks_.size();
    }

    const auto&
    chunks() const
    {
        return beta_chunks_;
    }

    auto
    pu() const -> device_t
    {
        return pu_;
    }
};

/// Base class for beta-projectors, gradient of beta-projectors and strain derivatives of beta-projectors.
/** \tparam T  Precision of beta-projectors (float or double).
 */
template <typename T>
class Beta_projectors_base
{
  protected:
    /// Simulation context.
    Simulation_context& ctx_;

    /// List of G+k vectors.
    fft::Gvec const& gkvec_;

    /// Coordinates of G+k vectors used by GPU kernel.
    mdarray<double, 2> gkvec_coord_;

    /// Number of different components: 1 for beta-projectors, 3 for gradient, 9 for strain derivatives.
    int N_;

    /// Phase-factor independent coefficients of |beta> functions for atom types.
    mdarray<std::complex<T>, 3> pw_coeffs_t_;

    /// Precomputed beta coefficients.
    /** Can hold coefficients of beta projectors, gradients and strain derivatives.
     *  Most used case: store beta-projectors for the entire run.
     */
    mdarray<std::complex<T>, 3> pw_coeffs_all_atoms_;

    /// For large systems beta-projectors are split into "chunks" to reduce memory consumption.
    std::vector<beta_chunk_t> beta_chunks_;

    // int max_num_beta_;

    /// Total number of beta-projectors for the entire unit cell.
    int num_beta_{0};

    /// Total number of beta-projectors among atom types.
    int num_beta_t_{0};

    /// Split beta-projectors into chunks.
    void
    split_in_chunks();

  public:
    Beta_projectors_base(Simulation_context& ctx__, fft::Gvec const& gkvec__, int N__);

    auto
    make_generator(device_t pu__) const -> Beta_projector_generator<T>
    {
        return Beta_projector_generator<T>(ctx_, pu__, pw_coeffs_t_, pw_coeffs_all_atoms_, beta_chunks_, gkvec_,
                                           gkvec_coord_);
    }

    auto
    make_generator() const -> Beta_projector_generator<T>
    {
        return make_generator(ctx_.processing_unit());
    }

    auto
    make_generator(memory_t mem__) const -> Beta_projector_generator<T>
    {
        return make_generator(get_device_t(mem__));
    }

    auto const&
    ctx() const
    {
        return ctx_;
    }

    inline auto
    num_gkvec_loc() const
    {
        return gkvec_.count();
    }

    int
    num_beta() const
    {
        return num_beta_;
    }

    inline int
    num_comp() const
    {
        return N_;
    }

    inline auto const&
    unit_cell() const
    {
        return ctx_.unit_cell();
    }

    auto&
    pw_coeffs_t(int ig__, int n__, int j__)
    {
        return pw_coeffs_t_(ig__, n__, j__);
    }

    auto
    pw_coeffs_t(int j__)
    {
        return matrix<std::complex<T>>({num_gkvec_loc(), num_beta_t()}, &pw_coeffs_t_(0, 0, j__));
    }

    inline int
    num_beta_t() const
    {
        return num_beta_t_;
    }

    inline int
    num_chunks() const
    {
        return static_cast<int>(beta_chunks_.size());
    }

    int
    nrows() const
    {
        return gkvec_.num_gvec();
    }

    const mpi::Communicator&
    comm() const
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
inner_prod_beta(spla::Context& spla_ctx, memory_t mem__, memory_t host_mem__, bool result_on_device,
                beta_projectors_coeffs_t<T>& beta_coeffs__, wf::Wave_functions<T> const& phi__, wf::spin_index ispn__,
                wf::band_range br__)
{
    int nbeta = beta_coeffs__.beta_chunk_->num_beta_;

    la::dmatrix<F> result(nbeta, br__.size(), get_memory_pool(host_mem__), mdarray_label("<beta|phi>"));
    if (result_on_device) {
        result.allocate(get_memory_pool(memory_t::device));
    }

    wf::inner<F>(spla_ctx, mem__, wf::spin_range(ispn__.get()), beta_coeffs__, wf::band_range(0, nbeta), phi__, br__,
                 result, 0, 0);

    return result;
}

/// computes <beta|beta> and returns result on ctx.processing_unit_memory_t
template <class T>
matrix<std::complex<T>>
inner_beta(const Beta_projectors_base<T>& beta, const Simulation_context& ctx)
{
    using complex_t     = std::complex<T>;
    auto generator      = beta.make_generator();
    int num_beta_chunks = beta.num_chunks();
    auto bcoeffs_row    = generator.prepare();
    auto bcoeffs_col    = generator.prepare();
    auto mem_t          = ctx.processing_unit_memory_t();

    la::lib_t la{la::lib_t::blas};
    if (is_device_memory(mem_t)) {
        la = la::lib_t::gpublas;
    }

    int size{beta.num_beta()};

    matrix<complex_t> out({size, size}, get_memory_pool(mem_t));

    complex_t one  = complex_t(1);
    complex_t zero = complex_t(0);

    for (int ichunk = 0; ichunk < num_beta_chunks; ++ichunk) {
        generator.generate(bcoeffs_row, ichunk);

        for (int jchunk = 0; jchunk < num_beta_chunks; ++jchunk) {
            generator.generate(bcoeffs_col, jchunk);
            int m              = bcoeffs_row.beta_chunk_->num_beta_;
            int n              = bcoeffs_col.beta_chunk_->num_beta_;
            int k              = bcoeffs_col.pw_coeffs_a_.size(0);
            int dest_row       = bcoeffs_row.beta_chunk_->offset_;
            int dest_col       = bcoeffs_col.beta_chunk_->offset_;
            const complex_t* A = bcoeffs_row.pw_coeffs_a_.at(mem_t);
            const complex_t* B = bcoeffs_col.pw_coeffs_a_.at(mem_t);
            complex_t* C       = out.at(mem_t, dest_row, dest_col);
            la::wrap(la).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a_.ld(), B,
                              bcoeffs_col.pw_coeffs_a_.ld(), &zero, C, out.ld());
        }
    }

    if (beta.comm().size() > 1) {
        RTE_THROW("this needs to be fixed first");
        beta.comm().allreduce(out.at(memory_t::host), static_cast<int>(out.size()));
    }

    return out;
}

/// inner product <beta|Op|beta>, return resulting dmatrix<complex> in ctx.processing_unit_memory_t
template <class T, class Op>
matrix<std::complex<T>>
inner_beta(const Beta_projectors_base<T>& beta, const Simulation_context& ctx, Op&& op)
{
    using complex_t     = std::complex<double>;
    auto generator      = beta.make_generator();
    int num_beta_chunks = beta.num_chunks();
    auto bcoeffs_row    = generator.prepare();
    auto bcoeffs_col    = generator.prepare();
    auto mem_t          = ctx.processing_unit_memory_t();

    la::lib_t la{la::lib_t::blas};
    if (is_device_memory(mem_t)) {
        la = la::lib_t::gpublas;
    }

    int size{beta.num_beta()};

    matrix<complex_t> out({size, size}, mem_t);

    complex_t one  = complex_t(1);
    complex_t zero = complex_t(0);

    for (int ichunk = 0; ichunk < num_beta_chunks; ++ichunk) {
        generator.generate(bcoeffs_row, ichunk);

        for (int jchunk = 0; jchunk < num_beta_chunks; ++jchunk) {
            generator.generate(bcoeffs_col, jchunk);

            int m        = bcoeffs_row.beta_chunk_->num_beta_; // TODO: take chunks from Beta_projectors_base<T>& beta
            int n        = bcoeffs_col.beta_chunk_->num_beta_;
            int k        = bcoeffs_col.pw_coeffs_a_.size(0);
            int dest_row = bcoeffs_row.beta_chunk_->offset_;
            int dest_col = bcoeffs_col.beta_chunk_->offset_;
            const complex_t* A = bcoeffs_row.pw_coeffs_a_.at(mem_t);
            // apply Op on |b>  (in-place operation)
            auto G = op(bcoeffs_col.pw_coeffs_a_);

            const complex_t* B2 = G.at(mem_t);
            complex_t* C        = out.at(mem_t, dest_row, dest_col);
            la::wrap(la).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a_.ld(), B2, G.ld(), &zero, C,
                              out.ld());
        }
    }
    if (beta.comm().size() > 1) {
        beta.comm().allreduce(out.at(memory_t::host), static_cast<int>(out.size()));
    }

    return out;
}

} // namespace sirius

#endif
