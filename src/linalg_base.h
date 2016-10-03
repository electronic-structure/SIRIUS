// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file linalg_base.h
 *
 *  \brief Basic interface to linear algebra functions.
 */

#ifndef __LINALG_BASE_H__
#define __LINALG_BASE_H__

#include "blas_lapack.h"
#include "scalapack.h"

template <typename T>
struct linalg_const
{
    static T const& one()
    {
        static const T a = 1;
        return a;
    }

    static T const& zero()
    {
        static const T a = 0;
        return a;
    }
};

extern "C" {

ftn_int FORTRAN(ilaenv)(ftn_int* ispec, ftn_char name, ftn_char opts, ftn_int* n1, ftn_int* n2, ftn_int* n3,
                        ftn_int* n4, ftn_len name_len, ftn_len opts_len);

ftn_double FORTRAN(dlamch)(ftn_char cmach, ftn_len cmach_len);


#ifdef __SCALAPACK
int Csys2blacs_handle(MPI_Comm SysCtxt);

MPI_Comm Cblacs2sys_handle(int BlacsCtxt);

void Cblacs_gridinit(int* ConTxt, const char* order, int nprow, int npcol);

void Cblacs_gridmap(int* ConTxt, int* usermap, int ldup, int nprow0, int npcol0);

void Cblacs_gridinfo(int ConTxt, int* nprow, int* npcol, int* myrow, int* mycol);

void Cfree_blacs_system_handle(int ISysCtxt);

void Cblacs_barrier(int ConTxt, const char* scope);

void Cblacs_gridexit(int ConTxt);

void FORTRAN(pdgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, ftn_double* aplha,
                     ftn_double* A, ftn_int* ia, ftn_int* ja, ftn_int const* desca,
                     ftn_double* B, ftn_int* ib, ftn_int* jb, ftn_int const* descb,
                     ftn_double* beta,
                     ftn_double* C, ftn_int* ic, ftn_int* jc, ftn_int const* descc,
                     ftn_len transa_len, ftn_len transb_len);

void FORTRAN(pzgemm)(ftn_char transa, ftn_char transb, ftn_int* m, ftn_int* n, ftn_int* k, ftn_double_complex* aplha,
                     ftn_double_complex* A, ftn_int* ia, ftn_int* ja, ftn_int const* desca,
                     ftn_double_complex* B, ftn_int* ib, ftn_int* jb, ftn_int const* descb, ftn_double_complex* beta,
                     ftn_double_complex* C, ftn_int* ic, ftn_int* jc, ftn_int const* descc,
                     ftn_len transa_len, ftn_len transb_len);

void FORTRAN(descinit)(ftn_int const* desc, ftn_int* m, ftn_int* n, ftn_int* mb, ftn_int* nb, ftn_int* irsrc, ftn_int* icsrc,
                       ftn_int* ictxt, ftn_int* lld, ftn_int* info);

void FORTRAN(pztranc)(ftn_int* m, ftn_int* n, ftn_double_complex* alpha,
                      ftn_double_complex* a, ftn_int* ia, ftn_int* ja, ftn_int const* desca,
                      ftn_double_complex* beta, ftn_double_complex* c, ftn_int* ic, ftn_int* jc, ftn_int const* descc);

void FORTRAN(pztranu)(ftn_int* m, ftn_int* n, ftn_double_complex* alpha, ftn_double_complex* a,
                      ftn_int* ia, ftn_int* ja, ftn_int const* desca,
                      ftn_double_complex* beta, ftn_double_complex* c, ftn_int* ic, ftn_int* jc, ftn_int const* descc);

void FORTRAN(pdtran)(ftn_int* m, ftn_int* n, ftn_double* alpha,
                     ftn_double* a, ftn_int* ia, ftn_int* ja, ftn_int const* desca,
                     ftn_double* beta, ftn_double* c, ftn_int* ic, ftn_int* jc, ftn_int const* descc);

ftn_int FORTRAN(numroc)(ftn_int* n, ftn_int* nb, ftn_int* iproc, ftn_int* isrcproc, ftn_int* nprocs);

ftn_int FORTRAN(indxl2g)(ftn_int* indxloc, ftn_int* nb, ftn_int* iproc, ftn_int* isrcproc, ftn_int* nprocs);

ftn_len FORTRAN(iceil)(ftn_int* inum, ftn_int* idenom);

void FORTRAN(pzgemr2d)(ftn_int*            m,
                       ftn_int*            n,
                       ftn_double_complex* a,
                       ftn_int*            ia,
                       ftn_int*            ja,
                       ftn_int const*      desca,
                       ftn_double_complex* b,
                       ftn_int*            ib,
                       ftn_int*            jb,
                       ftn_int const*      descb,
                       ftn_int*            gcontext);
#endif

}

/// Base class for linear algebra interface.
class linalg_base
{
    public:

        static ftn_int ilaenv(ftn_int ispec, std::string const& name, std::string const& opts, ftn_int n1, ftn_int n2,
                              ftn_int n3, ftn_int n4)
        {
            return FORTRAN(ilaenv)(&ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (ftn_len)name.length(),
                                   (ftn_len)opts.length());
        }

        static ftn_double dlamch(char cmach)
        {
            return FORTRAN(dlamch)(&cmach, (ftn_len)1);
        }

        #ifdef __SCALAPACK
        static ftn_int numroc(ftn_int n, ftn_int nb, ftn_int iproc, ftn_int isrcproc, ftn_int nprocs)
        {
            return FORTRAN(numroc)(&n, &nb, &iproc, &isrcproc, &nprocs);
        }

        /// Create BLACS handler.
        static int create_blacs_handler(MPI_Comm comm)
        {
            return Csys2blacs_handle(comm);
        }

        /// Free BLACS handler.
        static void free_blacs_handler(int blacs_handler)
        {
            Cfree_blacs_system_handle(blacs_handler);
        }

        /// Create BLACS context for the grid of MPI ranks
        static void gridmap(int* blacs_context, int* map, int ld, int nrow, int ncol)
        {
            Cblacs_gridmap(blacs_context, map, ld, nrow, ncol);
        }

        /// Destroy BLACS context.
        static void gridexit(int blacs_context)
        {
            Cblacs_gridexit(blacs_context);
        }

        static void gridinfo(int blacs_context, int* nrow, int* ncol, int* irow, int* icol)
        {
            Cblacs_gridinfo(blacs_context, nrow, ncol, irow, icol);
        }

        static void descinit(ftn_int* desc, ftn_int m, ftn_int n, ftn_int mb, ftn_int nb, ftn_int irsrc, ftn_int icsrc,
                             ftn_int ictxt, ftn_int lld)
        {
            ftn_int info;
            ftn_int lld1 = std::max(1, lld);

            FORTRAN(descinit)(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld1, &info);

            if (info)
            {
                printf("error in descinit()\n");
                printf("m=%i n=%i mb=%i nb=%i irsrc=%i icsrc=%i lld=%i\n", m, n, mb, nb, irsrc, icsrc, lld);
                exit(-1);
            }
        }

        static int pjlaenv(int32_t ictxt, int32_t ispec, const std::string& name, const std::string& opts, int32_t n1, int32_t n2,
                           int32_t n3, int32_t n4)
        {
            return FORTRAN(pjlaenv)(&ictxt, &ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int32_t)name.length(),
                                    (int32_t)opts.length());
        }

        static int32_t indxl2g(int32_t indxloc, int32_t nb, int32_t iproc, int32_t isrcproc, int32_t nprocs)
        {
            return FORTRAN(indxl2g)(&indxloc, &nb, &iproc, &isrcproc, &nprocs);
        }

        static int32_t iceil(int32_t inum, int32_t idenom)
        {
            return FORTRAN(iceil)(&inum, &idenom);
        }

        static void pztranc(ftn_int m, ftn_int n, ftn_double_complex alpha, ftn_double_complex* A, ftn_int ia, ftn_int ja,
                            ftn_int const* desca, ftn_double_complex beta, ftn_double_complex* C, ftn_int ic, ftn_int jc,
                            ftn_int const* descc)
        {
            FORTRAN(pztranc)(&m, &n, &alpha, A, &ia, &ja, desca, &beta, C, &ic, &jc, descc);
        }

        static void pztranu(ftn_int m, ftn_int n, ftn_double_complex alpha, ftn_double_complex* A, ftn_int ia, ftn_int ja,
                            ftn_int const* desca, ftn_double_complex beta, ftn_double_complex* C, ftn_int ic, ftn_int jc,
                            ftn_int const* descc)
        {
            FORTRAN(pztranu)(&m, &n, &alpha, A, &ia, &ja, desca, &beta, C, &ic, &jc, descc);
        }

        static void pdtran(ftn_int m, ftn_int n, ftn_double alpha, ftn_double* A, ftn_int ia, ftn_int ja,
                           ftn_int const* desca, ftn_double beta, ftn_double* C, ftn_int ic, ftn_int jc,
                           ftn_int const* descc)
        {
            FORTRAN(pdtran)(&m, &n, &alpha, A, &ia, &ja, desca, &beta, C, &ic, &jc, descc);
        }
        #endif
};

#endif // __LINALG_BASE_H__
