// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
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

#ifndef __LAPACK_H__
#define __LAPACK_H__

/** \file lapack.h
  *
  * \brief Contains LAPACK and ScaLAPACK bindings.
  *
  */

template<linalg_t> 
class linalg;

template<> 
class linalg<lapack>
{
    public:

        static int32_t ilaenv(int32_t ispec, const std::string& name, const std::string& opts, int32_t n1, int32_t n2, 
                              int32_t n3, int32_t n4)
        {
            return FORTRAN(ilaenv)(&ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int32_t)name.length(), 
                                   (int32_t)opts.length());
        }
        
        template <typename T> 
        static int gesv(int32_t n, int32_t nrhs, T* a, int32_t lda, T* b, int32_t ldb);

        template <typename T> 
        static int gtsv(int32_t n, int32_t nrhs, T* dl, T* d, T* du, T* b, int32_t ldb);

        template <typename T>
        static int getrf(int32_t m, int32_t n, T* a, int32_t lda, int32_t* ipiv);
        
        template <typename T>
        static int getri(int32_t n, T* a, int32_t lda, int32_t* ipiv, T* work, int32_t lwork);

        template <typename T>
        static void invert_ge(T* mtrx, int size)
        {
            int32_t nb = std::max(ilaenv(1, "dgetri", "U", size, -1, -1, -1), ilaenv(1, "zgetri", "U", size, -1, -1, -1));
            int32_t lwork = size * nb;
            std::vector<T> work(lwork);
            std::vector<int> ipiv(size);
            int info = getrf(size, size, mtrx, size, &ipiv[0]);
            if (info != 0)
            {
                std::stringstream s;
                s << "getrf returned : " << info;
                error_local(__FILE__, __LINE__, s);
            }

            info = getri(size, mtrx, size, &ipiv[0], &work[0], lwork);
            if (info != 0)
            {
                std::stringstream s;
                s << "getri returned : " << info;
                error_local(__FILE__, __LINE__, s);
            }
        }
};

#ifdef _SCALAPACK_
template<> 
class linalg<scalapack>
{
    private:

        static int cyclic_block_size_;

    public:

        static int cyclic_block_size()
        {
            return cyclic_block_size_;
        }

        static void set_cyclic_block_size(int cyclic_block_size__)
        {
            if (cyclic_block_size_ > 0)
            {
                std::stringstream s;
                s << "cyclic_block_size is already set to " << cyclic_block_size_ << " and can't be changed";
                error_global(__FILE__, __LINE__, s);
            }
            cyclic_block_size_ = cyclic_block_size__;
        }

        /// Create BLACS context
        static int create_blacs_context(MPI_Comm comm)
        {
            return Csys2blacs_handle(comm);
        }

        /// create grid of MPI ranks
        static void gridmap(int* blacs_context, int* map, int ld, int nrow, int ncol)
        {
            Cblacs_gridmap(blacs_context, map, ld, nrow, ncol);
        }

        static void gridinfo(int blacs_context, int* nrow, int* ncol, int* irow, int* icol)
        {
            Cblacs_gridinfo(blacs_context, nrow, ncol, irow, icol);
        }

        static void free_blacs_context(int blacs_context)
        {
            Cfree_blacs_system_handle(blacs_context);
        }

        static void descinit(int32_t* desc, int32_t m, int32_t n, int32_t mb, int32_t nb, int32_t irsrc, int32_t icsrc, int32_t ictxt, int32_t lld)
        {
            int32_t info;
        
            FORTRAN(descinit)(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);
        
            if (info)
            {
                std::stringstream s;
                s << "error in descinit()" << std::endl
                  << "m = " << m << std::endl
                  << "n = " << n << std::endl
                  << "mb = " << mb << std::endl
                  << "nb = " << nb << std::endl
                  << "irsrc = " << irsrc << std::endl
                  << "icsrc = " << icsrc << std::endl
                  << "ictxt = " << ictxt << std::endl
                  << "lld = " << lld;

                error_local(__FILE__, __LINE__, s);
            }
        }

        static int pjlaenv(int32_t ictxt, int32_t ispec, const std::string& name, const std::string& opts, int32_t n1, int32_t n2, 
                           int32_t n3, int32_t n4)
        {
            return FORTRAN(pjlaenv)(&ictxt, &ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int32_t)name.length(),
                                    (int32_t)opts.length());
        }

        static int32_t numroc(int32_t n, int32_t nb, int32_t iproc, int32_t isrcproc, int32_t nprocs)
        {
            return FORTRAN(numroc)(&n, &nb, &iproc, &isrcproc, &nprocs); 
        }

        static int32_t indxl2g(int32_t indxloc, int32_t nb, int32_t iproc, int32_t isrcproc, int32_t nprocs)
        {
            return FORTRAN(indxl2g)(&indxloc, &nb, &iproc, &isrcproc, &nprocs);
        }

        static int32_t iceil(int32_t inum, int32_t idenom)
        {
            return FORTRAN(iceil)(&inum, &idenom);
        }

        static void pztranc(int32_t m, int32_t n, double_complex alpha, double_complex* a, int32_t ia, int32_t ja, int32_t* desca, 
                            double_complex beta, double_complex* c, int32_t ic, int32_t jc, int32_t* descc)
        {
            FORTRAN(pztranc)(&m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
        }
};
#endif

#endif // __LAPACK_H__

