!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

#include "config-f90.h"


module elpa1_auxiliary
  implicit none

  public :: elpa_mult_at_b_real             !< Multiply real matrices A**T * B
  public :: elpa_mult_ah_b_complex          !< Multiply complex matrices A**H * B

  public :: elpa_invert_trm_real            !< Invert real triangular matrix
  public :: elpa_invert_trm_complex         !< Invert complex triangular matrix

  public :: elpa_cholesky_real              !< Cholesky factorization of a real matrix
  public :: elpa_cholesky_complex           !< Cholesky factorization of a complex matrix

  public :: elpa_solve_tridi                !< Solve tridiagonal eigensystem with divide and conquer method

  contains

!> \brief  elpa_cholesky_real: Cholesky factorization of a real symmetric matrix
!> \details
!>
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle is needs to be set.
!>                              On return, the upper triangle contains the Cholesky factor
!>                              and the lower triangle is set to 0.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \param succes                logical, reports success or failure
    subroutine elpa_cholesky_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      use elpa1_compute
      use elpa_utilities
      use elpa_mpi

      implicit none

      integer(kind=ik)              :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=rk)                 :: a(lda,matrixCols)
      ! was
      ! real a(lda, *)

      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
      integer(kind=ik)              :: n, nc, i, info
      integer(kind=ik)              :: lcs, lce, lrs, lre
      integer(kind=ik)              :: tile_size, l_rows_tile, l_cols_tile

      real(kind=rk), allocatable    :: tmp1(:), tmp2(:,:), tmatr(:,:), tmatc(:,:)

      logical, intent(in)           :: wantDebug
      logical, intent(out)          :: success
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("elpa_cholesky_real")
#endif
      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      success = .true.

      ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

      tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
      tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

      l_rows_tile = tile_size/np_rows ! local rows of a tile
      l_cols_tile = tile_size/np_cols ! local cols of a tile

      l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
      l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a

      allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_real: error when allocating tmp1 "//errorMessage
        stop
      endif

      allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_real: error when allocating tmp2 "//errorMessage
        stop
      endif

      tmp1 = 0
      tmp2 = 0

      allocate(tmatr(l_rows,nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_real: error when allocating tmatr "//errorMessage
        stop
      endif

      allocate(tmatc(l_cols,nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_real: error when allocating tmatc "//errorMessage
        stop
      endif

      tmatr = 0
      tmatc = 0

      do n = 1, na, nblk

        ! Calculate first local row and column of the still remaining matrix
        ! on the local processor

        l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
        l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

        l_rowx = local_index(n+nblk, my_prow, np_rows, nblk, +1)
        l_colx = local_index(n+nblk, my_pcol, np_cols, nblk, +1)

        if (n+nblk > na) then

          ! This is the last step, just do a Cholesky-Factorization
          ! of the remaining block

          if (my_prow==prow(n, nblk, np_rows) .and. my_pcol==pcol(n, nblk, np_cols)) then

            call dpotrf('U',na-n+1,a(l_row1,l_col1),lda,info)
            if (info/=0) then
              if (wantDebug) write(error_unit,*) "elpa_cholesky_real: Error in dpotrf"
              success = .false.
              return
            endif

          endif

          exit ! Loop

        endif

        if (my_prow==prow(n, nblk, np_rows)) then

          if (my_pcol==pcol(n, nblk, np_cols)) then

            ! The process owning the upper left remaining block does the
            ! Cholesky-Factorization of this block

            call dpotrf('U',nblk,a(l_row1,l_col1),lda,info)
            if (info/=0) then
              if (wantDebug) write(error_unit,*) "elpa_cholesky_real: Error in dpotrf"
              success = .false.
              return
            endif

            nc = 0
            do i=1,nblk
              tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
              nc = nc+i
            enddo
          endif
#ifdef WITH_MPI
          call MPI_Bcast(tmp1,nblk*(nblk+1)/2,MPI_REAL8,pcol(n, nblk, np_cols),mpi_comm_cols,mpierr)
#endif
          nc = 0
          do i=1,nblk
            tmp2(1:i,i) = tmp1(nc+1:nc+i)
            nc = nc+i
          enddo

          if (l_cols-l_colx+1>0) &
              call dtrsm('L','U','T','N',nblk,l_cols-l_colx+1,1.d0,tmp2,ubound(tmp2,dim=1),a(l_row1,l_colx),lda)

        endif

        do i=1,nblk

          if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:l_cols,i) = a(l_row1+i-1,l_colx:l_cols)
#ifdef WITH_MPI
          if (l_cols-l_colx+1>0) &
              call MPI_Bcast(tmatc(l_colx,i),l_cols-l_colx+1,MPI_REAL8,prow(n, nblk, np_rows),mpi_comm_rows,mpierr)
#endif
        enddo
        ! this has to be checked since it was changed substantially when doing type safe
        call elpa_transpose_vectors_real  (tmatc, ubound(tmatc,dim=1), mpi_comm_cols, &
                                      tmatr, ubound(tmatr,dim=1), mpi_comm_rows, &
                                      n, na, nblk, nblk)

        do i=0,(na-1)/tile_size
          lcs = max(l_colx,i*l_cols_tile+1)
          lce = min(l_cols,(i+1)*l_cols_tile)
          lrs = l_rowx
          lre = min(l_rows,(i+1)*l_rows_tile)
          if (lce<lcs .or. lre<lrs) cycle
          call DGEMM('N','T',lre-lrs+1,lce-lcs+1,nblk,-1.d0, &
                      tmatr(lrs,1),ubound(tmatr,dim=1),tmatc(lcs,1),ubound(tmatc,dim=1), &
                      1.d0,a(lrs,lcs),lda)
        enddo

      enddo

      deallocate(tmp1, tmp2, tmatr, tmatc, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_real: error when deallocating tmp1 "//errorMessage
        stop
      endif

      ! Set the lower triangle to 0, it contains garbage (form the above matrix multiplications)

      do i=1,na
        if (my_pcol==pcol(i, nblk, np_cols)) then
          ! column i is on local processor
          l_col1 = local_index(i  , my_pcol, np_cols, nblk, +1) ! local column number
          l_row1 = local_index(i+1, my_prow, np_rows, nblk, +1) ! first row below diagonal
          a(l_row1:l_rows,l_col1) = 0
        endif
      enddo
#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("elpa_cholesky_real")
#endif

    end subroutine elpa_cholesky_real

!> \brief  elpa_invert_trm_real: Inverts a upper triangular matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle is needs to be set.
!>                              The lower triangle is not referenced.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \param succes                logical, reports success or failure
    subroutine elpa_invert_trm_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)
       use precision
       use elpa1_compute
       use elpa_utilities
       use elpa_mpi

       implicit none

       integer(kind=ik)             :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
       real(kind=rk)                :: a(lda,*)
#else
       real(kind=rk)                :: a(lda,matrixCols)
#endif
       integer(kind=ik)             :: my_prow, my_pcol, np_rows, np_cols, mpierr
       integer(kind=ik)             :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
       integer(kind=ik)             :: n, nc, i, info, ns, nb

       real(kind=rk), allocatable   :: tmp1(:), tmp2(:,:), tmat1(:,:), tmat2(:,:)

       logical, intent(in)          :: wantDebug
       logical, intent(out)         :: success
       integer(kind=ik)             :: istat
       character(200)               :: errorMessage
       call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
       call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
       call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
       call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
       success = .true.

       l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
       l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a

       allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_real: error when allocating tmp1 "//errorMessage
         stop
       endif

       allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_real: error when allocating tmp2 "//errorMessage
         stop
       endif

       tmp1 = 0
       tmp2 = 0

       allocate(tmat1(l_rows,nblk), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_real: error when allocating tmat1 "//errorMessage
         stop
       endif

       allocate(tmat2(nblk,l_cols), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_real: error when allocating tmat2 "//errorMessage
         stop
       endif

       tmat1 = 0
       tmat2 = 0


       ns = ((na-1)/nblk)*nblk + 1

       do n = ns,1,-nblk

         l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
         l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

         nb = nblk
         if (na-n+1 < nblk) nb = na-n+1

         l_rowx = local_index(n+nb, my_prow, np_rows, nblk, +1)
         l_colx = local_index(n+nb, my_pcol, np_cols, nblk, +1)

         if (my_prow==prow(n, nblk, np_rows)) then

           if (my_pcol==pcol(n, nblk, np_cols)) then

             call DTRTRI('U','N',nb,a(l_row1,l_col1),lda,info)
             if (info/=0) then
               if (wantDebug) write(error_unit,*) "elpa_invert_trm_real: Error in DTRTRI"
               success = .false.
               return
             endif

             nc = 0
             do i=1,nb
               tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
               nc = nc+i
             enddo
           endif
#ifdef WITH_MPI
           call MPI_Bcast(tmp1,nb*(nb+1)/2,MPI_REAL8,pcol(n, nblk, np_cols),mpi_comm_cols,mpierr)
#endif
           nc = 0
           do i=1,nb
             tmp2(1:i,i) = tmp1(nc+1:nc+i)
             nc = nc+i
           enddo

           if (l_cols-l_colx+1>0) &
               call DTRMM('L','U','N','N',nb,l_cols-l_colx+1,1.d0,tmp2,ubound(tmp2,dim=1),a(l_row1,l_colx),lda)

           if (l_colx<=l_cols)   tmat2(1:nb,l_colx:l_cols) = a(l_row1:l_row1+nb-1,l_colx:l_cols)
           if (my_pcol==pcol(n, nblk, np_cols)) tmat2(1:nb,l_col1:l_col1+nb-1) = tmp2(1:nb,1:nb) ! tmp2 has the lower left triangle 0

         endif

         if (l_row1>1) then
           if (my_pcol==pcol(n, nblk, np_cols)) then
             tmat1(1:l_row1-1,1:nb) = a(1:l_row1-1,l_col1:l_col1+nb-1)
             a(1:l_row1-1,l_col1:l_col1+nb-1) = 0
           endif

           do i=1,nb
#ifdef WITH_MPI
             call MPI_Bcast(tmat1(1,i),l_row1-1,MPI_REAL8,pcol(n, nblk, np_cols),mpi_comm_cols,mpierr)
#endif
           enddo
         endif
#ifdef WITH_MPI
         if (l_cols-l_col1+1>0) &
            call MPI_Bcast(tmat2(1,l_col1),(l_cols-l_col1+1)*nblk,MPI_REAL8,prow(n, nblk, np_rows),mpi_comm_rows,mpierr)
#endif
         if (l_row1>1 .and. l_cols-l_col1+1>0) &
            call dgemm('N','N',l_row1-1,l_cols-l_col1+1,nb, -1.d0, &
                       tmat1,ubound(tmat1,dim=1),tmat2(1,l_col1),ubound(tmat2,dim=1), &
                       1.d0, a(1,l_col1),lda)

       enddo

       deallocate(tmp1, tmp2, tmat1, tmat2, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_real: error when deallocating tmp1 "//errorMessage
         stop
       endif

    end subroutine elpa_invert_trm_real

!> \brief  elpa_cholesky_complex: Cholesky factorization of a complex hermitian matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle is needs to be set.
!>                              On return, the upper triangle contains the Cholesky factor
!>                              and the lower triangle is set to 0.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \param succes                logical, reports success or failure
    subroutine elpa_cholesky_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)

#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      use elpa1_compute
      use elpa_utilities
      use elpa_mpi

      implicit none

      integer(kind=ik)                 :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      complex(kind=ck)                 :: a(lda,*)
#else
      complex(kind=ck)                 :: a(lda,matrixCols)
#endif
      integer(kind=ik)                 :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)                 :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
      integer(kind=ik)                 :: n, nc, i, info
      integer(kind=ik)                 :: lcs, lce, lrs, lre
      integer(kind=ik)                 :: tile_size, l_rows_tile, l_cols_tile

      complex(kind=ck), allocatable    :: tmp1(:), tmp2(:,:), tmatr(:,:), tmatc(:,:)

      logical, intent(in)              :: wantDebug
      logical, intent(out)             :: success
      integer(kind=ik)                 :: istat
      character(200)                   :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("elpa_cholesky_complex")
#endif
      success = .true.
      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

      tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
      tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

      l_rows_tile = tile_size/np_rows ! local rows of a tile
      l_cols_tile = tile_size/np_cols ! local cols of a tile

      l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
      l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a

      allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_complex: error when allocating tmp1 "//errorMessage
        stop
      endif

      allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_complex: error when allocating tmp2 "//errorMessage
        stop
      endif

      tmp1 = 0
      tmp2 = 0

      allocate(tmatr(l_rows,nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_complex: error when allocating tmatr "//errorMessage
        stop
      endif

      allocate(tmatc(l_cols,nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_complex: error when allocating tmatc "//errorMessage
        stop
      endif

      tmatr = 0
      tmatc = 0

      do n = 1, na, nblk

        ! Calculate first local row and column of the still remaining matrix
        ! on the local processor

        l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
        l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

        l_rowx = local_index(n+nblk, my_prow, np_rows, nblk, +1)
        l_colx = local_index(n+nblk, my_pcol, np_cols, nblk, +1)

        if (n+nblk > na) then

          ! This is the last step, just do a Cholesky-Factorization
          ! of the remaining block

          if (my_prow==prow(n, nblk, np_rows) .and. my_pcol==pcol(n, nblk, np_cols)) then

            call zpotrf('U',na-n+1,a(l_row1,l_col1),lda,info)
            if (info/=0) then
              if (wantDebug) write(error_unit,*) "elpa_cholesky_complex: Error in zpotrf"
              success = .false.
              return
            endif

          endif

          exit ! Loop
        endif

        if (my_prow==prow(n, nblk, np_rows)) then

          if (my_pcol==pcol(n, nblk, np_cols)) then

            ! The process owning the upper left remaining block does the
            ! Cholesky-Factorization of this block

            call zpotrf('U',nblk,a(l_row1,l_col1),lda,info)
            if (info/=0) then
              if (wantDebug) write(error_unit,*) "elpa_cholesky_complex: Error in zpotrf"
              success = .false.
              return
            endif

            nc = 0
            do i=1,nblk
              tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
              nc = nc+i
            enddo
          endif
#ifdef WITH_MPI
          call MPI_Bcast(tmp1,nblk*(nblk+1)/2,MPI_DOUBLE_COMPLEX,pcol(n, nblk, np_cols),mpi_comm_cols,mpierr)
#endif
          nc = 0
          do i=1,nblk
            tmp2(1:i,i) = tmp1(nc+1:nc+i)
            nc = nc+i
          enddo

          if (l_cols-l_colx+1>0) &
                call ztrsm('L','U','C','N',nblk,l_cols-l_colx+1,(1.d0,0.d0),tmp2,ubound(tmp2,dim=1),a(l_row1,l_colx),lda)

        endif

        do i=1,nblk

          if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:l_cols,i) = conjg(a(l_row1+i-1,l_colx:l_cols))
#ifdef WITH_MPI
          if (l_cols-l_colx+1>0) &
                call MPI_Bcast(tmatc(l_colx,i),l_cols-l_colx+1,MPI_DOUBLE_COMPLEX,prow(n, nblk, np_rows),mpi_comm_rows,mpierr)
#endif
        enddo
        ! this has to be checked since it was changed substantially when doing type safe
        call elpa_transpose_vectors_complex  (tmatc, ubound(tmatc,dim=1), mpi_comm_cols, &
                                        tmatr, ubound(tmatr,dim=1), mpi_comm_rows, &
                                        n, na, nblk, nblk)
        do i=0,(na-1)/tile_size
          lcs = max(l_colx,i*l_cols_tile+1)
          lce = min(l_cols,(i+1)*l_cols_tile)
          lrs = l_rowx
          lre = min(l_rows,(i+1)*l_rows_tile)
          if (lce<lcs .or. lre<lrs) cycle
          call ZGEMM('N','C',lre-lrs+1,lce-lcs+1,nblk,(-1.d0,0.d0), &
                        tmatr(lrs,1),ubound(tmatr,dim=1),tmatc(lcs,1),ubound(tmatc,dim=1), &
                        (1.d0,0.d0),a(lrs,lcs),lda)
        enddo

      enddo

      deallocate(tmp1, tmp2, tmatr, tmatc, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_cholesky_complex: error when deallocating tmatr "//errorMessage
        stop
      endif

      ! Set the lower triangle to 0, it contains garbage (form the above matrix multiplications)

      do i=1,na
        if (my_pcol==pcol(i, nblk, np_cols)) then
          ! column i is on local processor
          l_col1 = local_index(i  , my_pcol, np_cols, nblk, +1) ! local column number
          l_row1 = local_index(i+1, my_prow, np_rows, nblk, +1) ! first row below diagonal
          a(l_row1:l_rows,l_col1) = 0
        endif
      enddo
#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("elpa_cholesky_complex")
#endif

    end subroutine elpa_cholesky_complex

!> \brief  elpa_invert_trm_complex: Inverts a complex upper triangular matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle is needs to be set.
!>                              The lower triangle is not referenced.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \param succes                logical, reports success or failure
    subroutine elpa_invert_trm_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)

       use precision
       use elpa1_compute
       use elpa_utilities
       use elpa_mpi

       implicit none

       integer(kind=ik)                 :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
       complex(kind=ck)                 :: a(lda,*)
#else
       complex(kind=ck)                 :: a(lda,matrixCols)
#endif
       integer(kind=ik)                 :: my_prow, my_pcol, np_rows, np_cols, mpierr
       integer(kind=ik)                 :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
       integer(kind=ik)                 :: n, nc, i, info, ns, nb

       complex(kind=ck), allocatable    :: tmp1(:), tmp2(:,:), tmat1(:,:), tmat2(:,:)

       logical, intent(in)              :: wantDebug
       logical, intent(out)             :: success
       integer(kind=ik)                 :: istat
       character(200)                   :: errorMessage
       call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
       call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
       call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
       call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
       success = .true.

       l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
       l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a

       allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_complex: error when allocating tmp1 "//errorMessage
         stop
       endif

       allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_complex: error when allocating tmp2 "//errorMessage
         stop
       endif

       tmp1 = 0
       tmp2 = 0

       allocate(tmat1(l_rows,nblk), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_complex: error when allocating tmat1 "//errorMessage
         stop
       endif

       allocate(tmat2(nblk,l_cols), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_complex: error when allocating tmat2 "//errorMessage
         stop
       endif

       tmat1 = 0
       tmat2 = 0

       ns = ((na-1)/nblk)*nblk + 1

       do n = ns,1,-nblk

         l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
         l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

         nb = nblk
         if (na-n+1 < nblk) nb = na-n+1

         l_rowx = local_index(n+nb, my_prow, np_rows, nblk, +1)
         l_colx = local_index(n+nb, my_pcol, np_cols, nblk, +1)

         if (my_prow==prow(n, nblk, np_rows)) then

           if (my_pcol==pcol(n, nblk, np_cols)) then

             call ZTRTRI('U','N',nb,a(l_row1,l_col1),lda,info)
             if (info/=0) then
               if (wantDebug) write(error_unit,*) "elpa_invert_trm_complex: Error in ZTRTRI"
               success = .false.
               return
             endif

             nc = 0
             do i=1,nb
               tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
               nc = nc+i
             enddo
           endif

#ifdef WITH_MPI
           call MPI_Bcast(tmp1,nb*(nb+1)/2,MPI_DOUBLE_COMPLEX,pcol(n, nblk, np_cols),mpi_comm_cols,mpierr)
#endif
           nc = 0
           do i=1,nb
             tmp2(1:i,i) = tmp1(nc+1:nc+i)
             nc = nc+i
           enddo

           if (l_cols-l_colx+1>0) &
             call ZTRMM('L','U','N','N',nb,l_cols-l_colx+1,(1._ck,0._ck),tmp2,ubound(tmp2,dim=1),a(l_row1,l_colx),lda)

           if (l_colx<=l_cols)   tmat2(1:nb,l_colx:l_cols) = a(l_row1:l_row1+nb-1,l_colx:l_cols)
           if (my_pcol==pcol(n, nblk, np_cols)) tmat2(1:nb,l_col1:l_col1+nb-1) = tmp2(1:nb,1:nb) ! tmp2 has the lower left triangle 0

         endif

         if (l_row1>1) then
           if (my_pcol==pcol(n, nblk, np_cols)) then
             tmat1(1:l_row1-1,1:nb) = a(1:l_row1-1,l_col1:l_col1+nb-1)
             a(1:l_row1-1,l_col1:l_col1+nb-1) = 0
           endif

           do i=1,nb
#ifdef WITH_MPI
             call MPI_Bcast(tmat1(1,i),l_row1-1,MPI_DOUBLE_COMPLEX,pcol(n, nblk, np_cols),mpi_comm_cols,mpierr)
#endif
           enddo
         endif
#ifdef WITH_MPI
         if (l_cols-l_col1+1>0) &
           call MPI_Bcast(tmat2(1,l_col1),(l_cols-l_col1+1)*nblk,MPI_DOUBLE_COMPLEX,prow(n, nblk, np_rows),mpi_comm_rows,mpierr)
#endif
         if (l_row1>1 .and. l_cols-l_col1+1>0) &
           call ZGEMM('N','N',l_row1-1,l_cols-l_col1+1,nb, (-1._ck,0.0_ck), &
                        tmat1,ubound(tmat1,dim=1),tmat2(1,l_col1),ubound(tmat2,dim=1), &
                        (1.0_ck,0.0_ck), a(1,l_col1),lda)

       enddo

       deallocate(tmp1, tmp2, tmat1, tmat2, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"elpa_invert_trm_complex: error when deallocating tmp1 "//errorMessage
         stop
       endif
    end subroutine elpa_invert_trm_complex

!> \brief  elpa_mult_at_b_real: Performs C : = A**T * B
!>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
!>                 B is a (na,ncb) matrix
!>                 C is a (na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details

!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param lda                   leading dimension of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
    subroutine elpa_mult_at_b_real(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)

#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      use elpa1_compute
      use elpa_mpi

      implicit none

      character*1                   :: uplo_a, uplo_c

      integer(kind=ik)              :: na, ncb, lda, ldb, nblk, mpi_comm_rows, mpi_comm_cols, ldc
      real(kind=rk)                 :: a(lda,*), b(ldb,*), c(ldc,*) ! remove assumed size!

      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: l_cols, l_rows, l_rows_np
      integer(kind=ik)              :: np, n, nb, nblk_mult, lrs, lre, lcs, lce
      integer(kind=ik)              :: gcol_min, gcol, goff
      integer(kind=ik)              :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
      integer(kind=ik), allocatable :: lrs_save(:), lre_save(:)

      logical                       :: a_lower, a_upper, c_lower, c_upper

      real(kind=rk), allocatable    :: aux_mat(:,:), aux_bc(:), tmp1(:,:), tmp2(:,:)
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage
#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("elpa_mult_at_b_real")
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

      l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
      l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

      ! Block factor for matrix multiplications, must be a multiple of nblk

      if (na/np_rows<=256) then
         nblk_mult = (31/nblk+1)*nblk
      else
         nblk_mult = (63/nblk+1)*nblk
      endif

      allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating aux_mat "//errorMessage
        stop
      endif

      allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating aux_bc "//errorMessage
        stop
      endif

      allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating lrs_save "//errorMessage
        stop
      endif

      allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating lre_save "//errorMessage
        stop
      endif

      a_lower = .false.
      a_upper = .false.
      c_lower = .false.
      c_upper = .false.

      if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
      if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
      if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
      if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

      ! Build up the result matrix by processor rows

      do np = 0, np_rows-1

        ! In this turn, procs of row np assemble the result

        l_rows_np = local_index(na, np, np_rows, nblk, -1) ! local rows on receiving processors

        nr_done = 0 ! Number of rows done
        aux_mat = 0
        nstor = 0   ! Number of columns stored in aux_mat

        ! Loop over the blocks on row np

        do nb=0,(l_rows_np-1)/nblk

          goff  = nb*np_rows + np ! Global offset in blocks corresponding to nb

          ! Get the processor column which owns this block (A is transposed, so we need the column)
          ! and the offset in blocks within this column.
          ! The corresponding block column in A is then broadcast to all for multiplication with B

          np_bc = MOD(goff,np_cols)
          noff = goff/np_cols
          n_aux_bc = 0

          ! Gather up the complete block column of A on the owner

          do n = 1, min(l_rows_np-nb*nblk,nblk) ! Loop over columns to be broadcast

            gcol = goff*nblk + n ! global column corresponding to n
            if (nstor==0 .and. n==1) gcol_min = gcol

            lrs = 1       ! 1st local row number for broadcast
            lre = l_rows  ! last local row number for broadcast
            if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            if (lrs<=lre) then
              nvals = lre-lrs+1
              if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
              n_aux_bc = n_aux_bc + nvals
            endif

            lrs_save(n) = lrs
            lre_save(n) = lre

          enddo

          ! Broadcast block column
#ifdef WITH_MPI
          call MPI_Bcast(aux_bc,n_aux_bc,MPI_REAL8,np_bc,mpi_comm_cols,mpierr)
#endif
          ! Insert what we got in aux_mat

          n_aux_bc = 0
          do n = 1, min(l_rows_np-nb*nblk,nblk)
            nstor = nstor+1
            lrs = lrs_save(n)
            lre = lre_save(n)
            if (lrs<=lre) then
              nvals = lre-lrs+1
              aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
              n_aux_bc = n_aux_bc + nvals
            endif
          enddo

          ! If we got nblk_mult columns in aux_mat or this is the last block
          ! do the matrix multiplication

          if (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np) then

            lrs = 1       ! 1st local row number for multiply
            lre = l_rows  ! last local row number for multiply
            if (a_lower) lrs = local_index(gcol_min, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            lcs = 1       ! 1st local col number for multiply
            lce = l_cols  ! last local col number for multiply
            if (c_upper) lcs = local_index(gcol_min, my_pcol, np_cols, nblk, +1)
            if (c_lower) lce = MIN(local_index(gcol, my_pcol, np_cols, nblk, -1),l_cols)

            if (lcs<=lce) then
              allocate(tmp1(nstor,lcs:lce),tmp2(nstor,lcs:lce), stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
               print *,"elpa_mult_at_b_real: error when allocating tmp1 "//errorMessage
               stop
              endif

              if (lrs<=lre) then
                call dgemm('T','N',nstor,lce-lcs+1,lre-lrs+1,1.d0,aux_mat(lrs,1),ubound(aux_mat,dim=1), &
                             b(lrs,lcs),ldb,0.d0,tmp1,nstor)
              else
                tmp1 = 0
              endif

              ! Sum up the results and send to processor row np
#ifdef WITH_MPI
              call mpi_reduce(tmp1,tmp2,nstor*(lce-lcs+1),MPI_REAL8,MPI_SUM,np,mpi_comm_rows,mpierr)
#else
              tmp2 = tmp1
#endif
              ! Put the result into C
              if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

              deallocate(tmp1,tmp2, stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
               print *,"elpa_mult_at_b_real: error when deallocating tmp1 "//errorMessage
               stop
              endif

            endif

            nr_done = nr_done+nstor
            nstor=0
            aux_mat(:,:)=0
          endif
        enddo
      enddo

      deallocate(aux_mat, aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_at_b_real: error when deallocating aux_mat "//errorMessage
       stop
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("elpa_mult_at_b_real")
#endif

    end subroutine elpa_mult_at_b_real

!> \brief  elpa_mult_ah_b_complex: Performs C : = A**H * B
!>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
!>                 B is a (na,ncb) matrix
!>                 C is a (na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param lda                   leading dimension of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c

  subroutine elpa_mult_ah_b_complex(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      use elpa1_compute
      use elpa_mpi

      implicit none

      character*1                   :: uplo_a, uplo_c

      integer(kind=ik)              :: na, ncb, lda, ldb, nblk, mpi_comm_rows, mpi_comm_cols, ldc
      complex(kind=ck)              :: a(lda,*), b(ldb,*), c(ldc,*) ! remove assumed size!

      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: l_cols, l_rows, l_rows_np
      integer(kind=ik)              :: np, n, nb, nblk_mult, lrs, lre, lcs, lce
      integer(kind=ik)              :: gcol_min, gcol, goff
      integer(kind=ik)              :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
      integer(kind=ik), allocatable :: lrs_save(:), lre_save(:)

      logical                       :: a_lower, a_upper, c_lower, c_upper

      complex(kind=ck), allocatable :: aux_mat(:,:), aux_bc(:), tmp1(:,:), tmp2(:,:)
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("elpa_mult_ah_b_complex")
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
      l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

      ! Block factor for matrix multiplications, must be a multiple of nblk

      if (na/np_rows<=256) then
        nblk_mult = (31/nblk+1)*nblk
      else
        nblk_mult = (63/nblk+1)*nblk
      endif

      allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating aux_mat "//errorMessage
       stop
      endif

      allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating aux_bc "//errorMessage
       stop
      endif

      allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating lrs_save "//errorMessage
       stop
      endif

      allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating lre_save "//errorMessage
       stop
      endif

      a_lower = .false.
      a_upper = .false.
      c_lower = .false.
      c_upper = .false.

      if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
      if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
      if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
      if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

      ! Build up the result matrix by processor rows

      do np = 0, np_rows-1

        ! In this turn, procs of row np assemble the result

        l_rows_np = local_index(na, np, np_rows, nblk, -1) ! local rows on receiving processors

        nr_done = 0 ! Number of rows done
        aux_mat = 0
        nstor = 0   ! Number of columns stored in aux_mat

        ! Loop over the blocks on row np

        do nb=0,(l_rows_np-1)/nblk

          goff  = nb*np_rows + np ! Global offset in blocks corresponding to nb

          ! Get the processor column which owns this block (A is transposed, so we need the column)
          ! and the offset in blocks within this column.
          ! The corresponding block column in A is then broadcast to all for multiplication with B

          np_bc = MOD(goff,np_cols)
          noff = goff/np_cols
          n_aux_bc = 0

          ! Gather up the complete block column of A on the owner

          do n = 1, min(l_rows_np-nb*nblk,nblk) ! Loop over columns to be broadcast

            gcol = goff*nblk + n ! global column corresponding to n
            if (nstor==0 .and. n==1) gcol_min = gcol

            lrs = 1       ! 1st local row number for broadcast
            lre = l_rows  ! last local row number for broadcast
            if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            if (lrs<=lre) then
              nvals = lre-lrs+1
              if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
              n_aux_bc = n_aux_bc + nvals
            endif

            lrs_save(n) = lrs
            lre_save(n) = lre

          enddo

          ! Broadcast block column
#ifdef WITH_MPI
          call MPI_Bcast(aux_bc,n_aux_bc,MPI_DOUBLE_COMPLEX,np_bc,mpi_comm_cols,mpierr)
#endif
          ! Insert what we got in aux_mat

          n_aux_bc = 0
          do n = 1, min(l_rows_np-nb*nblk,nblk)
            nstor = nstor+1
            lrs = lrs_save(n)
            lre = lre_save(n)
            if (lrs<=lre) then
              nvals = lre-lrs+1
              aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
              n_aux_bc = n_aux_bc + nvals
            endif
          enddo

          ! If we got nblk_mult columns in aux_mat or this is the last block
          ! do the matrix multiplication

          if (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np) then

            lrs = 1       ! 1st local row number for multiply
            lre = l_rows  ! last local row number for multiply
            if (a_lower) lrs = local_index(gcol_min, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            lcs = 1       ! 1st local col number for multiply
            lce = l_cols  ! last local col number for multiply
            if (c_upper) lcs = local_index(gcol_min, my_pcol, np_cols, nblk, +1)
            if (c_lower) lce = MIN(local_index(gcol, my_pcol, np_cols, nblk, -1),l_cols)

            if (lcs<=lce) then
              allocate(tmp1(nstor,lcs:lce),tmp2(nstor,lcs:lce), stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
                print *,"elpa_mult_ah_b_complex: error when allocating tmp1 "//errorMessage
                stop
              endif

              if (lrs<=lre) then
                call zgemm('C','N',nstor,lce-lcs+1,lre-lrs+1,(1.0_ck,0.0_ck),aux_mat(lrs,1),ubound(aux_mat,dim=1), &
                             b(lrs,lcs),ldb,(0.0_ck,0.0_ck),tmp1,nstor)
               else
                 tmp1 = 0
               endif

               ! Sum up the results and send to processor row np
#ifdef WITH_MPI
               call mpi_reduce(tmp1,tmp2,nstor*(lce-lcs+1),MPI_DOUBLE_COMPLEX,MPI_SUM,np,mpi_comm_rows,mpierr)
#else
               tmp2 = tmp1
#endif
               ! Put the result into C
               if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

               deallocate(tmp1,tmp2, stat=istat, errmsg=errorMessage)
               if (istat .ne. 0) then
                 print *,"elpa_mult_ah_b_complex: error when deallocating tmp1 "//errorMessage
                 stop
               endif

            endif

            nr_done = nr_done+nstor
            nstor=0
            aux_mat(:,:)=0
          endif
        enddo
      enddo

      deallocate(aux_mat, aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_ah_b_complex: error when deallocating aux_mat "//errorMessage
        stop
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("elpa_mult_ah_b_complex")
#endif

    end subroutine elpa_mult_ah_b_complex

!> \brief  elpa_solve_tridi: Solve tridiagonal eigensystem with divide and conquer method
!> \details
!>
!> \param na                    Matrix dimension
!> \param nev                   number of eigenvalues/vectors to be computed
!> \param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
!>                              output the eigenvalues in ascending order
!> \param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
!> \param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
!> \param ldq                   leading dimension of matrix q
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param matrixCols            columns of matrix q
!> \param mpi_comm_rows         MPI communicator for rows
!> \param mpi_comm_cols         MPI communicator for columns
!> \param wantDebug             logical, give more debug information if .true.
!> \result success              logical, .true. on success, else .false.

    function elpa_solve_tridi(na, nev, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success)

      use elpa1_compute, solve_tridi_private => solve_tridi
      use precision

      implicit none
      integer(kind=ik)       :: na, nev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=rk)          :: d(na), e(na)
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      real(kind=rk)          :: q(ldq,*)
#else
      real(kind=rk)          :: q(ldq,matrixCols)
#endif
      logical, intent(in)    :: wantDebug
      logical :: success

      success = .false.

      call solve_tridi_private(na, nev, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)

    end function

end module elpa1_auxiliary

