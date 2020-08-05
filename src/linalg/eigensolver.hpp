// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file eigensolver.hpp
 *
 *  \brief Contains definition of eigensolver factory.
 */

#ifndef __EIGENSOLVER_HPP__
#define __EIGENSOLVER_HPP__

#include "SDDK/memory.hpp"
#include "SDDK/dmatrix.hpp"

using double_complex = std::complex<double>;

/// Type of eigen-value solver.
enum class ev_solver_t
{
    /// LAPACK
    lapack,

    /// ScaLAPACK
    scalapack,

    /// ELPA solver
    elpa,

    /// MAGMA with CPU pointers
    magma,

    /// MAGMA with GPU pointers
    magma_gpu,

    /// PLASMA
    plasma,

    /// CUDA eigen-solver
    cusolver
};

/// Get type of an eigen solver by name (provided as a string).
inline ev_solver_t get_ev_solver_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    static const std::map<std::string, ev_solver_t> map_to_type = {
        {"lapack", ev_solver_t::lapack}, {"scalapack", ev_solver_t::scalapack}, {"elpa1", ev_solver_t::elpa},
        {"elpa2", ev_solver_t::elpa},   {"magma", ev_solver_t::magma},         {"magma_gpu", ev_solver_t::magma_gpu},
        {"plasma", ev_solver_t::plasma}, {"cusolver", ev_solver_t::cusolver}};

    if (map_to_type.count(name__) == 0) {
        std::stringstream s;
        s << "wrong label of eigen-solver : " << name__;
        TERMINATE(s);
    }

    return map_to_type.at(name__);
}

/// Interface to different eigen-solvers.
class Eigensolver
{
  protected:
    /// Type of the eigen-value solver.
    ev_solver_t ev_solver_type_;
    /// Common error message.
    const std::string error_msg_not_implemented = "solver is not implemented";
    /// Memory pool for CPU work buffers.
    sddk::memory_pool mp_h_;
    /// Memory pool for CPU work buffers using pinned memory.
    sddk::memory_pool mp_hp_;
    /// Memory pool for GPU work buffers.
    std::shared_ptr<sddk::memory_pool> mp_d_{nullptr};
    /// True if solver is MPI parallel.
    bool is_parallel_{false};
    /// Type of host memory needed for the solver.
    /** Some solvers, for example MAGMA, require host pilnned memory. */
    sddk::memory_t host_memory_t_{sddk::memory_t::none};
    /// Type of input data memory.
    /** CPU solvers start from host memory, MAGMA can start from host or device memory, cuSolver starts from
     *  devide memoryi. */
    sddk::memory_t data_memory_t_{sddk::memory_t::none};

  public:
    /// Constructor.
    Eigensolver(ev_solver_t type__, sddk::memory_pool* mpd__, bool is_parallel__, sddk::memory_t host_memory_t__,
                sddk::memory_t data_memory_t__)
        : ev_solver_type_(type__)
        , mp_h_(sddk::memory_pool(sddk::memory_t::host))
        , mp_hp_(sddk::memory_pool(sddk::memory_t::host_pinned))
        , is_parallel_(is_parallel__)
        , host_memory_t_(host_memory_t__)
        , data_memory_t_(data_memory_t__)
    {
        if (mpd__) {
            mp_d_ = std::shared_ptr<sddk::memory_pool>(mpd__, [](sddk::memory_pool*){});
        } else {
            mp_d_ = std::shared_ptr<sddk::memory_pool>(new sddk::memory_pool(sddk::memory_t::device));
        }
    }

    /// Destructor.
    virtual ~Eigensolver()
    {
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, sddk::dmatrix<double>& A__, double* eval__, sddk::dmatrix<double>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, sddk::dmatrix<double_complex>& A__, double* eval__,
                      sddk::dmatrix<double_complex>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem of a sub-matrix for N lowest eigen-pairs
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, sddk::dmatrix<double>& A__, double* eval__, sddk::dmatrix<double>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem of a sub-matrix for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, sddk::dmatrix<double_complex>& A__, double* eval__, sddk::dmatrix<double_complex>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, sddk::dmatrix<double>& A__, sddk::dmatrix<double>& B__, double* eval__,
                      sddk::dmatrix<double>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, sddk::dmatrix<double_complex>& A__, sddk::dmatrix<double_complex>& B__,
                      double* eval__, sddk::dmatrix<double_complex>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, sddk::dmatrix<double>& A__, sddk::dmatrix<double>& B__,
                      double* eval__, sddk::dmatrix<double>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, sddk::dmatrix<double_complex>& A__,
                      sddk::dmatrix<double_complex>& B__, double* eval__, sddk::dmatrix<double_complex>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Parallel or sequential solver.
    bool is_parallel() const
    {
        return is_parallel_;
    }

    /// Type of host memory, required by the solver.
    inline sddk::memory_t host_memory_t() const
    {
        return host_memory_t_;
    }

    /// Type of input memory for the solver.
    inline sddk::memory_t data_memory_t() const
    {
        return data_memory_t_;
    }

    /// Type of eigen-solver.
    inline ev_solver_t type() const
    {
        return ev_solver_type_;
    }
};

std::unique_ptr<Eigensolver> Eigensolver_factory(std::string name__, sddk::memory_pool* mpd__);

#endif
