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

namespace sirius {

#if defined(SIRIUS_GPU)
extern "C" void create_beta_gk_gpu_float(int                        num_atoms,
                                         int                        num_gkvec,
                                         int const*                 beta_desc,
                                         std::complex<float> const* beta_gk_t,
                                         double const*              gkvec,
                                         double const*              atom_pos,
                                         std::complex<float>*       beta_gk);

extern "C" void create_beta_gk_gpu_double(int                   num_atoms,
                                          int                   num_gkvec,
                                          int const*            beta_desc,
                                          double_complex const* beta_gk_t,
                                          double const*         gkvec,
                                          double const*         atom_pos,
                                          double_complex*       beta_gk);
#endif

/// Named index of a descriptor of beta-projectors. The same order is used by the GPU kernel.
enum class beta_desc_idx : int
{
    /// Number of beta-projector functions for this atom.
    nbf      = 0,
    /// Offset of beta-projectors in this chunk.
    offset   = 1,
    /// Offset of beta-projectors in the array for atom types.
    offset_t = 2,
    /// Global index of atom.
    ia       = 3
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

        copy_new(desc_, other.desc_);
        copy_new(atom_pos_, other.atom_pos_);
    }

    beta_chunk_t& operator=(const beta_chunk_t& other)
    {
        num_beta_  = other.num_beta_;
        num_atoms_ = other.num_atoms_;
        offset_    = other.offset_;

        desc_ = empty_like(other.desc_);
        copy_new(desc_, other.desc_);

        atom_pos_ = empty_like(other.atom_pos_);
        copy_new(atom_pos_, other.atom_pos_);
        return *this;
    }
};

template <typename T>
struct beta_projectors_coeffs_t
{
    using complex_t = std::complex<T>;

    sddk::matrix<complex_t> pw_coeffs_a;
    sddk::mdarray<complex_t, 1> pw_coeffs_a_g0;
    sddk::Communicator comm;
    /// the current beta chunk
    beta_chunk_t beta_chunk;
    /// buffer (num_max_beta) for pw_coeffs_a_
    sddk::matrix<complex_t> __pw_coeffs_a_buffer;
    /// buffer (num_max_beta) for pw_coeffs_a_g0
    sddk::mdarray<complex_t, 1> __pw_coeffs_a_g0_buffer;
};

namespace local {

template <class T>
void beta_projectors_generate_cpu(sddk::matrix<std::complex<T>>& pw_coeffs_a,
                                  const sddk::mdarray<std::complex<T>, 3>& pw_coeffs_t, int ichunk__, int j__,
                                  const beta_chunk_t& beta_chunk, const Simulation_context& ctx,
                                  const sddk::Gvec& gkvec);

template <class T>
void beta_projectors_generate_gpu(beta_projectors_coeffs_t<T>& out,
                                  const mdarray<double_complex, 3>& pw_coeffs_t_device,
                                  const mdarray<double_complex, 3>& pw_coeffs_t_host, const Simulation_context& ctx,
                                  const Gvec& gkvec, const mdarray<double, 2>& gkvec_coord_,
                                  const beta_chunk_t& beta_chunk, const std::vector<int>& igk__, int j__);

}  // local

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
                             const sddk::matrix<double_complex>& beta_pw_all, sddk::device_t processing_unit,
                             const std::vector<beta_chunk_t>& beta_chunks, const sddk::Gvec& gkvec,
                             const sddk::mdarray<double, 2>& gkvec_coord, int num_gkvec_loc);

    void generate(beta_projectors_coeffs_t<T>& coeffs, int ichunk, int j) const;
    void generate(beta_projectors_coeffs_t<T>& coeffs, int ichunk) const;

    beta_projectors_coeffs_t<T> prepare(sddk::memory_t pm = sddk::memory_t::none) const;

    Simulation_context& ctx() { return ctx_; }
    int num_chunks() const { return beta_chunks_.size(); }

    const auto& chunks() const { return beta_chunks_; }

  private:
    Simulation_context& ctx_;
    const array_t& pw_coeffs_t_host_;
    /// precomputed beta coefficients on CPU
    const sddk::matrix<double_complex>& beta_pw_all_atoms_;
    sddk::device_t processing_unit_;
    /// chunk descriptors
    const std::vector<beta_chunk_t>& beta_chunks_;
    /// gkvec
    const sddk::Gvec& gkvec_;
    const sddk::mdarray<double, 2>& gkvec_coord_;
    // const std::vector<int>& igk_;
    int num_gkvec_loc_;
    /// pw_coeffs_t on device
    array_t pw_coeffs_t_device_;
    int max_num_beta_;
};

template <typename T>
Beta_projector_generator<T>::Beta_projector_generator(Simulation_context& ctx, const array_t& pw_coeffs_t_host,
                                                      const sddk::matrix<double_complex>& beta_pw_all,
                                                      sddk::device_t processing_unit,
                                                      const std::vector<beta_chunk_t>& beta_chunks,
                                                      const sddk::Gvec& gkvec,
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
                                      ctx_.mem_pool(sddk::device_t::GPU));
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
Beta_projector_generator<T>::prepare(sddk::memory_t pm) const
{
    beta_projectors_coeffs_t<T> beta_storage;
    beta_storage.comm = gkvec_.comm().duplicate();

    sddk::device_t pu;
    switch (pm) {
        case sddk::memory_t::none: {
            pu = ctx_.processing_unit();
            break;
        }
        case sddk::memory_t::host: {
            pu = sddk::device_t::CPU;
            break;
        }
        case sddk::memory_t::device: {
            pu = sddk::device_t::GPU;
            break;
        }
        default:
            RTE_THROW("invalid memory_t");
            break;
    }

    if (pu == sddk::device_t::GPU) {
        beta_storage.__pw_coeffs_a_buffer =
            sddk::matrix<double_complex>(num_gkvec_loc_, max_num_beta_, ctx_.mem_pool(sddk::memory_t::device));
        beta_storage.__pw_coeffs_a_g0_buffer =
            sddk::mdarray<double_complex, 1>(max_num_beta_, ctx_.mem_pool(sddk::memory_t::host));
        beta_storage.__pw_coeffs_a_g0_buffer.allocate(ctx_.mem_pool(sddk::memory_t::host));
    }

    return beta_storage;
}


/// Base class for beta-projectors, gradient of beta-projectors and strain derivatives of beta-projectors.
template <typename T>
class Beta_projectors_base
{
  protected:
    Simulation_context& ctx_;

    /// List of G+k vectors.
    sddk::Gvec const& gkvec_;

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
    sddk::matrix<double_complex> beta_pw_all_atoms_;

    sddk::mdarray<std::complex<T>, 1> pw_coeffs_a_g0_;

    std::vector<beta_chunk_t> beta_chunks_;

    int max_num_beta_;

    /// total number of beta-projectors (=number of columns)
    int num_total_beta_;

    /// Total number of beta-projectors among atom types.
    int num_beta_t_;

    /// Split beta-projectors into chunks.
    void split_in_chunks();


  public:
    Beta_projectors_base(Simulation_context& ctx__, sddk::Gvec const& gkvec__, int N__);

    Beta_projector_generator<T> make_generator() const
    {
        return make_generator(ctx_.processing_unit());
    }

    Beta_projector_generator<T> make_generator(sddk::device_t pu) const
    {
        return Beta_projector_generator<T>{ctx_, pw_coeffs_t_, beta_pw_all_atoms_, pu, beta_chunks_, gkvec_,
                                           gkvec_coord_, igk_, num_gkvec_loc()};
    }

    Simulation_context& ctx()
    {
        return ctx_;
    }

    /// Calculate inner product between beta-projectors and wave-functions.
    /** The following is computed: <beta|phi> */
    // template <typename F, typename = std::enable_if_t<std::is_same<T, real_type<F>>::value>>
    // __attribute_deprecated__ matrix<F> inner(int chunk__, Wave_functions<T>& phi__, int ispn__, int idx0__, int n__);

    /// Generate beta-projectors for a chunk of atoms.
    /** Beta-projectors are always generated and stored in the memory of a processing unit.
     *
     *  \param [in] ichunk Index of a chunk of atoms for which beta-projectors are generated.
     *  \param [in] j index of the component (up to 9 components are used for the strain derivative)
     */
    // __attribute__((deprecated)) void generate(int ichunk__, int j__);

    // beta_projectors_coeffs_t<T> prepare(memory_t pm = memory_t::none);

    // __attribute__((deprecated)) void dismiss();

    inline int num_gkvec_loc() const
    {
        //return static_cast<int>(igk_.size());
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

    inline Unit_cell const& unit_cell() const
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

    const sddk::Communicator& comm() const
    {
        return gkvec_.comm();
    }
};

/// inner product <beta|Op|beta>
template <class T, class Op>
sddk::matrix<double_complex>
inner_beta(const Beta_projectors_base<T>& beta, const Simulation_context& ctx, Op&& op)
{
    if (beta.comm().size() == 1) {
        auto generator        = beta.make_generator();
        int num_beta_chunks   = beta.num_chunks();
        auto bcoeffs_row      = generator.prepare();
        auto bcoeffs_col      = generator.prepare();
        auto linalg_t         = ctx.blas_linalg_t();
        auto preferred_memory = ctx.preferred_memory_t();

        int size{beta.num_total_beta()};

        sddk::matrix<double_complex> out(size, size, preferred_memory);

        double_complex one  = double_complex(1);
        double_complex zero = double_complex(0);

        for (int ichunk = 0; ichunk < num_beta_chunks; ++ichunk) {
            generator.generate(bcoeffs_row, ichunk);

            for (int jchunk = 0; jchunk < num_beta_chunks; ++jchunk) {
                generator.generate(bcoeffs_col, jchunk);

                int m                   = bcoeffs_row.beta_chunk.num_beta_;
                int n                   = bcoeffs_col.beta_chunk.num_beta_;
                int k                   = bcoeffs_col.pw_coeffs_a.size(0);
                int dest_row            = bcoeffs_row.beta_chunk.offset_;
                int dest_col            = bcoeffs_col.beta_chunk.offset_;
                const double_complex* A = bcoeffs_row.pw_coeffs_a.at(preferred_memory);
                // const double_complex* B = bcoeffs_col.pw_coeffs_a.at(preferred_memory);
                // apply Op on |b>  (in-place operation)
                auto G = op(bcoeffs_col.pw_coeffs_a);

                const double_complex* B2 = G.at(preferred_memory);
                double_complex* C        = out.at(preferred_memory, dest_row, dest_col);
                // linalg(linalg_t).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a.ld(), B,
                //                       bcoeffs_col.pw_coeffs_a.ld(), &zero, C, out.ld());
                linalg(linalg_t).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a.ld(), B2, G.ld(), &zero, C,
                                      out.ld());
            }
        }
        return out;
    } else {
        throw std::runtime_error("distributed case not yet implemented: " + std::string(__func__) + " in " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

/// inner product <beta|beta>, result in preferred_memory
template <class T>
sddk::matrix<std::complex<T>>
inner_beta(const Beta_projectors_base<T>& beta, const Simulation_context& ctx)
{
    if (beta.comm().size() == 1) {
        auto generator        = beta.make_generator();
        int num_beta_chunks   = beta.num_chunks();
        auto bcoeffs_row      = generator.prepare();
        auto bcoeffs_col      = generator.prepare();
        auto linalg_t         = ctx.blas_linalg_t();
        auto preferred_memory = ctx.preferred_memory_t();

        int size{beta.num_total_beta()};

        sddk::matrix<double_complex> out(size, size, preferred_memory);

        double_complex one  = double_complex(1);
        double_complex zero = double_complex(0);

        for (int ichunk = 0; ichunk < num_beta_chunks; ++ichunk) {
            generator.generate(bcoeffs_row, ichunk);

            for (int jchunk = 0; jchunk < num_beta_chunks; ++jchunk) {
                generator.generate(bcoeffs_col, jchunk);
                int m                   = bcoeffs_row.beta_chunk.num_beta_;
                int n                   = bcoeffs_col.beta_chunk.num_beta_;
                int k                   = bcoeffs_col.pw_coeffs_a.size(0);
                int dest_row            = bcoeffs_row.beta_chunk.offset_;
                int dest_col            = bcoeffs_col.beta_chunk.offset_;
                const double_complex* A = bcoeffs_row.pw_coeffs_a.at(preferred_memory);
                const double_complex* B = bcoeffs_col.pw_coeffs_a.at(preferred_memory);
                double_complex* C       = out.at(preferred_memory, dest_row, dest_col);
                linalg(linalg_t).gemm('C', 'N', m, n, k, &one, A, bcoeffs_row.pw_coeffs_a.ld(), B,
                                      bcoeffs_col.pw_coeffs_a.ld(), &zero, C, out.ld());
            }
        }
        return out;
    } else {
        throw std::runtime_error("distributed case not yet implemented: " + std::string(__func__) + " in " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

// /// TODO: add docstring, standalone inner product, Wave_functions carry communicator
template <class R, class T>
sddk::matrix<R> inner(sddk::linalg_t linalg, sddk::device_t processing_unit, sddk::memory_t preferred_memory,
                      std::function<sddk::memory_pool&(sddk::device_t)> mempool,
                      const beta_projectors_coeffs_t<T>& beta_projector_coeffs, sddk::Wave_functions<T>& phi__,
                      int ispn__, int idx0__, int n__);

/// inner product of beta projectors, mdarray
template <class R, class T>
sddk::matrix<R> inner(sddk::linalg_t linalg, sddk::device_t processing_unit, sddk::memory_t preferred_memory,
                      std::function<sddk::memory_pool&(sddk::device_t)> mempool,
                      const beta_projectors_coeffs_t<T>& beta_projector_coeffs,
                      const sddk::matrix<double_complex>& other, int idx0__, int n__,
                      sddk::memory_t target_memory = sddk::memory_t::none);

} // namespace sirius

#endif
