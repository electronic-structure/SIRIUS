/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file linalg_base.hpp
 *
 *  \brief Basic interface to linear algebra functions.
 */

#ifndef __LINALG_BASE_HPP__
#define __LINALG_BASE_HPP__

#include <algorithm>
#include <map>

#include "blas_lapack.h"
#include "scalapack.h"
#include <mpi.h>

namespace sirius {

/// Interface to linear algebra BLAS/LAPACK functions.
namespace la {

template <typename T>
struct constant
{
    static T const&
    one() noexcept
    {
        static const T a = 1;
        return a;
    }

    static T const&
    two() noexcept
    {
        static const T a = 2;
        return a;
    }

    static T const&
    m_one() noexcept
    {
        static const T a = -1;
        return a;
    }

    static T const&
    zero() noexcept
    {
        static const T a = 0;
        return a;
    }
};

/// Type of linear algebra backend library.
enum class lib_t
{
    /// None
    none,
    /// CPU BLAS
    blas,
    /// CPU LAPACK
    lapack,
    /// CPU ScaLAPACK
    scalapack,
    /// GPU BLAS (cuBlas or ROCblas)
    gpublas,
    /// cuBlasXt (cuBlas with CPU pointers and large matrices support)
    cublasxt,
    /// MAGMA with CPU pointers
    magma,
    /// SPLA library. Can take CPU and device pointers
    spla
};

inline auto
get_lib_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    static const std::map<std::string, lib_t> map_to_type = {
            {"blas", lib_t::blas},      {"lapack", lib_t::lapack},   {"scalapack", lib_t::scalapack},
            {"cublas", lib_t::gpublas}, {"gpublas", lib_t::gpublas}, {"cublasxt", lib_t::cublasxt},
            {"magma", lib_t::magma},
    };

    if (map_to_type.count(name__) == 0) {
        std::stringstream s;
        s << "wrong label of linear algebra type: " << name__;
        throw std::runtime_error(s.str());
    }

    return map_to_type.at(name__);
}

inline std::string
to_string(lib_t la__)
{
    switch (la__) {
        case lib_t::none: {
            return "none";
            break;
        }
        case lib_t::blas: {
            return "blas";
            break;
        }
        case lib_t::lapack: {
            return "lapack";
            break;
        }
        case lib_t::scalapack: {
            return "scalapack";
            break;
        }
        case lib_t::gpublas: {
            return "gpublas";
            break;
        }
        case lib_t::cublasxt: {
            return "cublasxt";
            break;
        }
        case lib_t::magma: {
            return "magma";
            break;
        }
        case lib_t::spla: {
            return "spla";
            break;
        }
    }
    return ""; // make compiler happy
}

extern "C" {

ftn_int FORTRAN(ilaenv)(ftn_int* ispec, ftn_char name, ftn_char opts, ftn_int* n1, ftn_int* n2, ftn_int* n3,
                        ftn_int* n4, ftn_len name_len, ftn_len opts_len);

ftn_double FORTRAN(dlamch)(ftn_char cmach, ftn_len cmach_len);

#ifdef SIRIUS_SCALAPACK
int
Csys2blacs_handle(MPI_Comm SysCtxt);

MPI_Comm
Cblacs2sys_handle(int BlacsCtxt);

void
Cblacs_gridinit(int* ConTxt, const char* order, int nprow, int npcol);

void
Cblacs_gridmap(int* ConTxt, int* usermap, int ldup, int nprow0, int npcol0);

void
Cblacs_gridinfo(int ConTxt, int* nprow, int* npcol, int* myrow, int* mycol);

void
Cfree_blacs_system_handle(int ISysCtxt);

void
Cblacs_barrier(int ConTxt, const char* scope);

void
Cblacs_gridexit(int ConTxt);

void FORTRAN(psgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, ftn_single const* aplha,
                     ftn_single const* A, ftn_int* ia, ftn_int* ja, ftn_int const* desca, ftn_single const* B,
                     ftn_int* ib, ftn_int* jb, ftn_int const* descb, ftn_single const* beta, ftn_single* C, ftn_int* ic,
                     ftn_int* jc, ftn_int const* descc, ftn_len transa_len, ftn_len transb_len);

void FORTRAN(pdgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, ftn_double const* aplha,
                     ftn_double const* A, ftn_int* ia, ftn_int* ja, ftn_int const* desca, ftn_double const* B,
                     ftn_int* ib, ftn_int* jb, ftn_int const* descb, ftn_double const* beta, ftn_double* C, ftn_int* ic,
                     ftn_int* jc, ftn_int const* descc, ftn_len transa_len, ftn_len transb_len);

void FORTRAN(pcgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, ftn_complex const* aplha,
                     ftn_complex const* A, ftn_int* ia, ftn_int* ja, ftn_int const* desca, ftn_complex const* B,
                     ftn_int* ib, ftn_int* jb, ftn_int const* descb, ftn_complex const* beta, ftn_complex* C,
                     ftn_int* ic, ftn_int* jc, ftn_int const* descc, ftn_len transa_len, ftn_len transb_len);

void FORTRAN(pzgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k,
                     ftn_double_complex const* aplha, ftn_double_complex const* A, ftn_int* ia, ftn_int* ja,
                     ftn_int const* desca, ftn_double_complex const* B, ftn_int* ib, ftn_int* jb, ftn_int const* descb,
                     ftn_double_complex const* beta, ftn_double_complex* C, ftn_int* ic, ftn_int* jc,
                     ftn_int const* descc, ftn_len transa_len, ftn_len transb_len);

void FORTRAN(descinit)(ftn_int const* desc, ftn_int* m, ftn_int* n, ftn_int* mb, ftn_int* nb, ftn_int* irsrc,
                       ftn_int* icsrc, ftn_int* ictxt, ftn_int* lld, ftn_int* info);

void FORTRAN(pctranc)(ftn_int* m, ftn_int* n, ftn_complex* alpha, ftn_complex* a, ftn_int* ia, ftn_int* ja,
                      ftn_int const* desca, ftn_complex* beta, ftn_complex* c, ftn_int* ic, ftn_int* jc,
                      ftn_int const* descc);

void FORTRAN(pztranc)(ftn_int* m, ftn_int* n, ftn_double_complex* alpha, ftn_double_complex* a, ftn_int* ia,
                      ftn_int* ja, ftn_int const* desca, ftn_double_complex* beta, ftn_double_complex* c, ftn_int* ic,
                      ftn_int* jc, ftn_int const* descc);

void FORTRAN(pztranu)(ftn_int* m, ftn_int* n, ftn_double_complex* alpha, ftn_double_complex* a, ftn_int* ia,
                      ftn_int* ja, ftn_int const* desca, ftn_double_complex* beta, ftn_double_complex* c, ftn_int* ic,
                      ftn_int* jc, ftn_int const* descc);

void FORTRAN(pstran)(ftn_int* m, ftn_int* n, ftn_single* alpha, ftn_single* a, ftn_int* ia, ftn_int* ja,
                     ftn_int const* desca, ftn_single* beta, ftn_single* c, ftn_int* ic, ftn_int* jc,
                     ftn_int const* descc);

void FORTRAN(pdtran)(ftn_int* m, ftn_int* n, ftn_double* alpha, ftn_double* a, ftn_int* ia, ftn_int* ja,
                     ftn_int const* desca, ftn_double* beta, ftn_double* c, ftn_int* ic, ftn_int* jc,
                     ftn_int const* descc);

ftn_int FORTRAN(numroc)(ftn_int* n, ftn_int* nb, ftn_int* iproc, ftn_int* isrcproc, ftn_int* nprocs);

ftn_int FORTRAN(indxl2g)(ftn_int* indxloc, ftn_int* nb, ftn_int* iproc, ftn_int* isrcproc, ftn_int* nprocs);

ftn_len FORTRAN(iceil)(ftn_int* inum, ftn_int* idenom);

void FORTRAN(pzgemr2d)(ftn_int* m, ftn_int* n, ftn_double_complex* a, ftn_int* ia, ftn_int* ja, ftn_int const* desca,
                       ftn_double_complex* b, ftn_int* ib, ftn_int* jb, ftn_int const* descb, ftn_int* gcontext);
#endif
}

/// Base class for linear algebra interface.
class linalg_base
{
  public:
    static ftn_int
    ilaenv(ftn_int ispec, std::string const& name, std::string const& opts, ftn_int n1, ftn_int n2, ftn_int n3,
           ftn_int n4)
    {
        return FORTRAN(ilaenv)(&ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (ftn_len)name.length(),
                               (ftn_len)opts.length());
    }

    static ftn_double
    dlamch(char cmach)
    {
        return FORTRAN(dlamch)(&cmach, (ftn_len)1);
    }

#ifdef SIRIUS_SCALAPACK
    static ftn_int
    numroc(ftn_int n, ftn_int nb, ftn_int iproc, ftn_int isrcproc, ftn_int nprocs)
    {
        return FORTRAN(numroc)(&n, &nb, &iproc, &isrcproc, &nprocs);
    }

    /// Create BLACS handler.
    static int
    create_blacs_handler(MPI_Comm comm)
    {
        return Csys2blacs_handle(comm);
    }

    /// Free BLACS handler.
    static void
    free_blacs_handler(int blacs_handler)
    {
        Cfree_blacs_system_handle(blacs_handler);
    }

    /// Create BLACS context for the grid of MPI ranks
    static void
    gridmap(int* blacs_context, int* map, int ld, int nrow, int ncol)
    {
        Cblacs_gridmap(blacs_context, map, ld, nrow, ncol);
    }

    /// Destroy BLACS context.
    static void
    gridexit(int blacs_context)
    {
        Cblacs_gridexit(blacs_context);
    }

    static void
    gridinfo(int blacs_context, int* nrow, int* ncol, int* irow, int* icol)
    {
        Cblacs_gridinfo(blacs_context, nrow, ncol, irow, icol);
    }

    static void
    descinit(ftn_int* desc, ftn_int m, ftn_int n, ftn_int mb, ftn_int nb, ftn_int irsrc, ftn_int icsrc, ftn_int ictxt,
             ftn_int lld)
    {
        ftn_int info;
        ftn_int lld1 = std::max(1, lld);

        FORTRAN(descinit)(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld1, &info);

        if (info) {
            std::stringstream s;
            s << "error in descinit()" << std::endl
              << "m=" << m << " n=" << n << " mb=" << mb << " nb=" << nb << " irsrc=" << irsrc << " icsrc=" << icsrc
              << " lld=" << lld;
            RTE_THROW(s);
        }
    }

    static int
    pjlaenv(int32_t ictxt, int32_t ispec, const std::string& name, const std::string& opts, int32_t n1, int32_t n2,
            int32_t n3, int32_t n4)
    {
        return FORTRAN(pjlaenv)(&ictxt, &ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int32_t)name.length(),
                                (int32_t)opts.length());
    }

    static int32_t
    indxl2g(int32_t indxloc, int32_t nb, int32_t iproc, int32_t isrcproc, int32_t nprocs)
    {
        return FORTRAN(indxl2g)(&indxloc, &nb, &iproc, &isrcproc, &nprocs);
    }

    static int32_t
    iceil(int32_t inum, int32_t idenom)
    {
        return FORTRAN(iceil)(&inum, &idenom);
    }
#endif
};

} // namespace la

} // namespace sirius

#endif // __LINALG_BASE_HPP__
