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

module ELPA1_compute
  use elpa_utilities
#ifdef HAVE_DETAILED_TIMINGS
  use timings
#endif
  use elpa_mpi
  implicit none

  PRIVATE ! set default to private

  public :: tridiag_real               ! Transform real symmetric matrix to tridiagonal form
  public :: trans_ev_real              ! Transform eigenvectors of a tridiagonal matrix back

  public :: tridiag_complex            ! Transform complex hermitian matrix to tridiagonal form
  public :: trans_ev_complex           ! Transform eigenvectors of a tridiagonal matrix back

  public :: solve_tridi                ! Solve tridiagonal eigensystem with divide and conquer method

  public :: local_index                ! Get local index of a block cyclic distributed matrix
  public :: least_common_multiple      ! Get least common multiple

  public :: hh_transform_real
  public :: hh_transform_complex

  public :: elpa_reduce_add_vectors_complex, elpa_reduce_add_vectors_real
  public :: elpa_transpose_vectors_complex, elpa_transpose_vectors_real

  contains

#define DATATYPE REAL(kind=rk)
#define BYTESIZE 8
#define REALCASE 1
#include "elpa_transpose_vectors.X90"
#include "elpa_reduce_add_vectors.X90"
#undef DATATYPE
#undef BYTESIZE
#undef REALCASE

    subroutine tridiag_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, d, e, tau)

    !-------------------------------------------------------------------------------
    !  tridiag_real: Reduces a distributed symmetric matrix to tridiagonal form
    !                (like Scalapack Routine PDSYTRD)
    !
    !  Parameters
    !
    !  na          Order of matrix
    !
    !  a(lda,matrixCols)    Distributed matrix which should be reduced.
    !              Distribution is like in Scalapack.
    !              Opposed to PDSYTRD, a(:,:) must be set completely (upper and lower half)
    !              a(:,:) is overwritten on exit with the Householder vectors
    !
    !  lda         Leading dimension of a
    !  matrixCols  local columns of matrix
    !
    !  nblk        blocksize of cyclic distribution, must be the same in both directions!
    !
    !  mpi_comm_rows
    !  mpi_comm_cols
    !              MPI-Communicators for rows/columns
    !
    !  d(na)       Diagonal elements (returned), identical on all processors
    !
    !  e(na)       Off-Diagonal elements (returned), identical on all processors
    !
    !  tau(na)     Factors for the Householder vectors (returned), needed for back transformation
    !
    !-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik)            :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=rk)               :: d(na), e(na), tau(na)
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      real(kind=rk)               :: a(lda,*)
#else
      real(kind=rk)               :: a(lda,matrixCols)
#endif

      integer(kind=ik), parameter :: max_stored_rows = 32

      integer(kind=ik)            :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)            :: totalblocks, max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
      integer(kind=ik)            :: l_cols, l_rows, nstor
      integer(kind=ik)            :: istep, i, j, lcs, lce, lrs, lre
      integer(kind=ik)            :: tile_size, l_rows_tile, l_cols_tile

#ifdef WITH_OPENMP
      integer(kind=ik)            :: my_thread, n_threads, max_threads, n_iter
      integer(kind=ik)            :: omp_get_thread_num, omp_get_num_threads, omp_get_max_threads
#endif

      real(kind=rk)               :: vav, vnorm2, x, aux(2*max_stored_rows), aux1(2), aux2(2), vrl, xf

      real(kind=rk), allocatable  :: tmp(:), vr(:), vc(:), ur(:), uc(:), vur(:,:), uvc(:,:)
#ifdef WITH_OPENMP
      real(kind=rk), allocatable  :: ur_p(:,:), uc_p(:,:)
#endif
      integer(kind=ik)            :: istat
      character(200)              :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("tridiag_real")
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

      tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
      tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

      l_rows_tile = tile_size/np_rows ! local rows of a tile
      l_cols_tile = tile_size/np_cols ! local cols of a tile


      totalblocks = (na-1)/nblk + 1
      max_blocks_row = (totalblocks-1)/np_rows + 1
      max_blocks_col = (totalblocks-1)/np_cols + 1

      max_local_rows = max_blocks_row*nblk
      max_local_cols = max_blocks_col*nblk

      allocate(tmp(MAX(max_local_rows,max_local_cols)), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating tmp "//errorMessage
        stop
      endif

      allocate(vr(max_local_rows+1), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating vr "//errorMessage
        stop
      endif

      allocate(ur(max_local_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating ur "//errorMessage
        stop
      endif

      allocate(vc(max_local_cols), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating vc "//errorMessage
        stop
      endif

      allocate(uc(max_local_cols), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating uc "//errorMessage
        stop
      endif

#ifdef WITH_OPENMP
      max_threads = omp_get_max_threads()

      allocate(ur_p(max_local_rows,0:max_threads-1), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating ur_p "//errorMessage
        stop
      endif

      allocate(uc_p(max_local_cols,0:max_threads-1), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating uc_p "//errorMessage
        stop
      endif

#endif

      tmp = 0
      vr = 0
      ur = 0
      vc = 0
      uc = 0

      allocate(vur(max_local_rows,2*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating vur "//errorMessage
        stop
      endif

      allocate(uvc(max_local_cols,2*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating uvc "//errorMessage
        stop
      endif

      d(:) = 0
      e(:) = 0
      tau(:) = 0

      nstor = 0

      l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
      l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a
      if(my_prow==prow(na, nblk, np_rows) .and. my_pcol==pcol(na, nblk, np_cols)) d(na) = a(l_rows,l_cols)

      do istep=na,3,-1

         ! Calculate number of local rows and columns of the still remaining matrix
         ! on the local processor

         l_rows = local_index(istep-1, my_prow, np_rows, nblk, -1)
         l_cols = local_index(istep-1, my_pcol, np_cols, nblk, -1)

         ! Calculate vector for Householder transformation on all procs
         ! owning column istep

         if(my_pcol==pcol(istep, nblk, np_cols)) then

            ! Get vector to be transformed; distribute last element and norm of
            ! remaining elements to all procs in current column

            vr(1:l_rows) = a(1:l_rows,l_cols+1)
            if(nstor>0 .and. l_rows>0) then
               call DGEMV('N',l_rows,2*nstor,1.d0,vur,ubound(vur,dim=1), &
                          uvc(l_cols+1,1),ubound(uvc,dim=1),1.d0,vr,1)
            endif

            if(my_prow==prow(istep-1, nblk, np_rows)) then
               aux1(1) = dot_product(vr(1:l_rows-1),vr(1:l_rows-1))
               aux1(2) = vr(l_rows)
            else
               aux1(1) = dot_product(vr(1:l_rows),vr(1:l_rows))
               aux1(2) = 0.
            endif

#ifdef WITH_MPI
            call mpi_allreduce(aux1,aux2,2,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
#else
            aux2 = aux1
#endif

            vnorm2 = aux2(1)
            vrl    = aux2(2)

            ! Householder transformation

            call hh_transform_real(vrl, vnorm2, xf, tau(istep))

            ! Scale vr and store Householder vector for back transformation

            vr(1:l_rows) = vr(1:l_rows) * xf
            if(my_prow==prow(istep-1, nblk, np_rows)) then
               vr(l_rows) = 1.
               e(istep-1) = vrl
            endif
            a(1:l_rows,l_cols+1) = vr(1:l_rows) ! store Householder vector for back transformation

         endif

         ! Broadcast the Householder vector (and tau) along columns

         if(my_pcol==pcol(istep, nblk, np_cols)) vr(l_rows+1) = tau(istep)
#ifdef WITH_MPI
         call MPI_Bcast(vr,l_rows+1,MPI_REAL8,pcol(istep, nblk, np_cols),mpi_comm_cols,mpierr)
#endif
         tau(istep) =  vr(l_rows+1)

         ! Transpose Householder vector vr -> vc

         call elpa_transpose_vectors_real  (vr, ubound(vr,dim=1), mpi_comm_rows, &
                                            vc, ubound(vc,dim=1), mpi_comm_cols, &
                                            1, istep-1, 1, nblk)


         ! Calculate u = (A + VU**T + UV**T)*v

         ! For cache efficiency, we use only the upper half of the matrix tiles for this,
         ! thus the result is partly in uc(:) and partly in ur(:)

         uc(1:l_cols) = 0
         ur(1:l_rows) = 0
         if (l_rows>0 .and. l_cols>0) then

#ifdef WITH_OPENMP

#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$OMP PARALLEL PRIVATE(my_thread,n_threads,n_iter,i,lcs,lce,j,lrs,lre)

           my_thread = omp_get_thread_num()
           n_threads = omp_get_num_threads()

           n_iter = 0

           uc_p(1:l_cols,my_thread) = 0.
           ur_p(1:l_rows,my_thread) = 0.
#endif
           do i=0,(istep-2)/tile_size
             lcs = i*l_cols_tile+1
             lce = min(l_cols,(i+1)*l_cols_tile)
             if (lce<lcs) cycle
             do j=0,i
               lrs = j*l_rows_tile+1
               lre = min(l_rows,(j+1)*l_rows_tile)
               if (lre<lrs) cycle
#ifdef WITH_OPENMP
               if (mod(n_iter,n_threads) == my_thread) then
                 call DGEMV('T',lre-lrs+1,lce-lcs+1,1.d0,a(lrs,lcs),lda,vr(lrs),1,1.d0,uc_p(lcs,my_thread),1)
                 if (i/=j) call DGEMV('N',lre-lrs+1,lce-lcs+1,1.d0,a(lrs,lcs),lda,vc(lcs),1,1.d0,ur_p(lrs,my_thread),1)
               endif
               n_iter = n_iter+1
#else
               call DGEMV('T',lre-lrs+1,lce-lcs+1,1.d0,a(lrs,lcs),lda,vr(lrs),1,1.d0,uc(lcs),1)
               if (i/=j) call DGEMV('N',lre-lrs+1,lce-lcs+1,1.d0,a(lrs,lcs),lda,vc(lcs),1,1.d0,ur(lrs),1)

#endif
             enddo
           enddo
#ifdef WITH_OPENMP
!$OMP END PARALLEL
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

           do i=0,max_threads-1
             uc(1:l_cols) = uc(1:l_cols) + uc_p(1:l_cols,i)
             ur(1:l_rows) = ur(1:l_rows) + ur_p(1:l_rows,i)
           enddo
#endif
           if (nstor>0) then
             call DGEMV('T',l_rows,2*nstor,1.d0,vur,ubound(vur,dim=1),vr,1,0.d0,aux,1)
             call DGEMV('N',l_cols,2*nstor,1.d0,uvc,ubound(uvc,dim=1),aux,1,1.d0,uc,1)
           endif

         endif

        ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
        ! on the processors containing the diagonal
        ! This is only necessary if ur has been calculated, i.e. if the
        ! global tile size is smaller than the global remaining matrix

        if (tile_size < istep-1) then
          call elpa_reduce_add_vectors_REAL  (ur, ubound(ur,dim=1), mpi_comm_rows, &
                                        uc, ubound(uc,dim=1), mpi_comm_cols, &
                                        istep-1, 1, nblk)
        endif

        ! Sum up all the uc(:) parts, transpose uc -> ur

        if (l_cols>0) then
          tmp(1:l_cols) = uc(1:l_cols)
#ifdef WITH_MPI
          call mpi_allreduce(tmp,uc,l_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
#else
          uc = tmp
#endif
        endif

        call elpa_transpose_vectors_real  (uc, ubound(uc,dim=1), mpi_comm_cols, &
                                         ur, ubound(ur,dim=1), mpi_comm_rows, &
                                         1, istep-1, 1, nblk)

        ! calculate u**T * v (same as v**T * (A + VU**T + UV**T) * v )

        x = 0
        if (l_cols>0) x = dot_product(vc(1:l_cols),uc(1:l_cols))
#ifdef WITH_MPI
        call mpi_allreduce(x,vav,1,MPI_REAL8,MPI_SUM,mpi_comm_cols,mpierr)
#else
        vav = x
#endif
        ! store u and v in the matrices U and V
        ! these matrices are stored combined in one here

        do j=1,l_rows
          vur(j,2*nstor+1) = tau(istep)*vr(j)
          vur(j,2*nstor+2) = 0.5*tau(istep)*vav*vr(j) - ur(j)
        enddo
        do j=1,l_cols
          uvc(j,2*nstor+1) = 0.5*tau(istep)*vav*vc(j) - uc(j)
          uvc(j,2*nstor+2) = tau(istep)*vc(j)
        enddo

        nstor = nstor+1

        ! If the limit of max_stored_rows is reached, calculate A + VU**T + UV**T

        if (nstor==max_stored_rows .or. istep==3) then

          do i=0,(istep-2)/tile_size
            lcs = i*l_cols_tile+1
            lce = min(l_cols,(i+1)*l_cols_tile)
           lrs = 1
            lre = min(l_rows,(i+1)*l_rows_tile)
            if (lce<lcs .or. lre<lrs) cycle
            call dgemm('N','T',lre-lrs+1,lce-lcs+1,2*nstor,1.d0, &
                       vur(lrs,1),ubound(vur,dim=1),uvc(lcs,1),ubound(uvc,dim=1), &
                       1.d0,a(lrs,lcs),lda)
          enddo

          nstor = 0

        endif

        if (my_prow==prow(istep-1, nblk, np_rows) .and. my_pcol==pcol(istep-1, nblk, np_cols)) then
          if (nstor>0) a(l_rows,l_cols) = a(l_rows,l_cols) &
                        + dot_product(vur(l_rows,1:2*nstor),uvc(l_cols,1:2*nstor))
          d(istep-1) = a(l_rows,l_cols)
        endif

      enddo

      ! Store e(1) and d(1)

      if (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(2, nblk, np_cols)) e(1) = a(1,l_cols) ! use last l_cols value of loop above
      if (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(1, nblk, np_cols)) d(1) = a(1,1)

      deallocate(tmp, vr, ur, vc, uc, vur, uvc, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when deallocating uvc "//errorMessage
        stop
      endif


      ! distribute the arrays d and e to all processors

      allocate(tmp(na),  stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when allocating tmp "//errorMessage
        stop
      endif
#ifdef WITH_MPI
      tmp = d
      call mpi_allreduce(tmp,d,na,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
      tmp = d
      call mpi_allreduce(tmp,d,na,MPI_REAL8,MPI_SUM,mpi_comm_cols,mpierr)
      tmp = e
      call mpi_allreduce(tmp,e,na,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
      tmp = e
      call mpi_allreduce(tmp,e,na,MPI_REAL8,MPI_SUM,mpi_comm_cols,mpierr)
#endif
      deallocate(tmp,  stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_real: error when deallocating tmp "//errorMessage
        stop
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("tridiag_real")
#endif

    end subroutine tridiag_real

    subroutine trans_ev_real(na, nqc, a, lda, tau, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)

    !-------------------------------------------------------------------------------
    !  trans_ev_real: Transforms the eigenvectors of a tridiagonal matrix back
    !                 to the eigenvectors of the original matrix
    !                 (like Scalapack Routine PDORMTR)
    !
    !  Parameters
    !
    !  na          Order of matrix a, number of rows of matrix q
    !
    !  nqc         Number of columns of matrix q
    !
    !  a(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a after tridiag_real)
    !              Distribution is like in Scalapack.
    !
    !  lda         Leading dimension of a
    !  matrixCols  local columns of matrix a and q
    !
    !  tau(na)     Factors of the Householder vectors
    !
    !  q           On input: Eigenvectors of tridiagonal matrix
    !              On output: Transformed eigenvectors
    !              Distribution is like in Scalapack.
    !
    !  ldq         Leading dimension of q
    !
    !  nblk        blocksize of cyclic distribution, must be the same in both directions!
    !
    !  mpi_comm_rows
    !  mpi_comm_cols
    !              MPI-Communicators for rows/columns
    !
    !-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik)           :: na, nqc, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=rk)              :: tau(na)
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      real(kind=rk)              :: a(lda,*), q(ldq,*)
#else
      real(kind=rk)              :: a(lda,matrixCols), q(ldq,matrixCols)
#endif

      integer(kind=ik)           :: max_stored_rows

      integer(kind=ik)           :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)           :: totalblocks, max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
      integer(kind=ik)           :: l_cols, l_rows, l_colh, nstor
      integer(kind=ik)           :: istep, i, n, nc, ic, ics, ice, nb, cur_pcol

      real(kind=rk), allocatable :: tmp1(:), tmp2(:), hvb(:), hvm(:,:)
      real(kind=rk), allocatable :: tmat(:,:), h1(:), h2(:)
      integer(kind=ik)           :: istat
      character(200)             :: errorMessage
#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("trans_ev_real")
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

      totalblocks = (na-1)/nblk + 1
      max_blocks_row = (totalblocks-1)/np_rows + 1
      max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q!

      max_local_rows = max_blocks_row*nblk
      max_local_cols = max_blocks_col*nblk

      max_stored_rows = (63/nblk+1)*nblk

      allocate(tmat(max_stored_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_real: error when allocating tmat "//errorMessage
        stop
      endif

      allocate(h1(max_stored_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_real: error when allocating h1 "//errorMessage
        stop
      endif

      allocate(h2(max_stored_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_real: error when allocating h2 "//errorMessage
        stop
      endif

      allocate(tmp1(max_local_cols*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_real: error when allocating tmp1 "//errorMessage
        stop
      endif

      allocate(tmp2(max_local_cols*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_real: error when allocating tmp2 "//errorMessage
        stop
      endif

      allocate(hvb(max_local_rows*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_real: error when allocating hvn "//errorMessage
        stop
      endif

      allocate(hvm(max_local_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_real: error when allocating hvm "//errorMessage
        stop
      endif

      hvm = 0   ! Must be set to 0 !!!
      hvb = 0   ! Safety only

      l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q

      nstor = 0

      do istep=1,na,nblk

        ics = MAX(istep,3)
        ice = MIN(istep+nblk-1,na)
        if (ice<ics) cycle

        cur_pcol = pcol(istep, nblk, np_cols)

        nb = 0
        do ic=ics,ice

          l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1) ! Column of Householder vector
          l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder vector


          if (my_pcol==cur_pcol) then
            hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)
            if (my_prow==prow(ic-1, nblk, np_rows)) then
              hvb(nb+l_rows) = 1.
            endif
          endif

          nb = nb+l_rows
        enddo

#ifdef WITH_MPI
        if (nb>0) &
            call MPI_Bcast(hvb,nb,MPI_REAL8,cur_pcol,mpi_comm_cols,mpierr)
#endif
        nb = 0
        do ic=ics,ice
          l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder vector
          hvm(1:l_rows,nstor+1) = hvb(nb+1:nb+l_rows)
          nstor = nstor+1
          nb = nb+l_rows
        enddo

        ! Please note: for smaller matix sizes (na/np_rows<=256), a value of 32 for nstor is enough!
        if (nstor+nblk>max_stored_rows .or. istep+nblk>na .or. (na/np_rows<=256 .and. nstor>=32)) then

          ! Calculate scalar products of stored vectors.
          ! This can be done in different ways, we use dsyrk

          tmat = 0
          if (l_rows>0) &
               call dsyrk('U','T',nstor,l_rows,1.d0,hvm,ubound(hvm,dim=1),0.d0,tmat,max_stored_rows)

          nc = 0
          do n=1,nstor-1
            h1(nc+1:nc+n) = tmat(1:n,n+1)
            nc = nc+n
          enddo
#ifdef WITH_MPI
          if (nc>0) call mpi_allreduce(h1,h2,nc,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
#else
          if (nc>0) h2 = h1
#endif
          ! Calculate triangular matrix T

          nc = 0
          tmat(1,1) = tau(ice-nstor+1)
          do n=1,nstor-1
            call dtrmv('L','T','N',n,tmat,max_stored_rows,h2(nc+1),1)
            tmat(n+1,1:n) = -h2(nc+1:nc+n)*tau(ice-nstor+n+1)
            tmat(n+1,n+1) = tau(ice-nstor+n+1)
            nc = nc+n
          enddo

          ! Q = Q - V * T * V**T * Q

          if (l_rows>0) then
            call dgemm('T','N',nstor,l_cols,l_rows,1.d0,hvm,ubound(hvm,dim=1), &
                          q,ldq,0.d0,tmp1,nstor)
          else
            tmp1(1:l_cols*nstor) = 0
          endif
#ifdef WITH_MPI
          call mpi_allreduce(tmp1,tmp2,nstor*l_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
#else
          tmp2 = tmp1
#endif
          if (l_rows>0) then
            call dtrmm('L','L','N','N',nstor,l_cols,1.0d0,tmat,max_stored_rows,tmp2,nstor)
            call dgemm('N','N',l_rows,l_cols,nstor,-1.d0,hvm,ubound(hvm,dim=1), &
                          tmp2,nstor,1.d0,q,ldq)
          endif
          nstor = 0
        endif

      enddo

      deallocate(tmat, h1, h2, tmp1, tmp2, hvb, hvm, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"trans_ev_real: error when deallocating hvm "//errorMessage
        stop
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("trans_ev_real")
#endif

    end subroutine trans_ev_real

#define DATATYPE COMPLEX(kind=ck)
#define BYTESIZE 16
#define COMPLEXCASE 1
#include "elpa_transpose_vectors.X90"
#include "elpa_reduce_add_vectors.X90"
#undef DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE

    subroutine tridiag_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, d, e, tau)

    !-------------------------------------------------------------------------------
    !  tridiag_complex: Reduces a distributed hermitian matrix to tridiagonal form
    !                   (like Scalapack Routine PZHETRD)
    !
    !  Parameters
    !
    !  na          Order of matrix
    !
    !  a(lda,matrixCols)    Distributed matrix which should be reduced.
    !              Distribution is like in Scalapack.
    !              Opposed to PZHETRD, a(:,:) must be set completely (upper and lower half)
    !              a(:,:) is overwritten on exit with the Householder vectors
    !
    !  lda         Leading dimension of a
    !  matrixCols  local columns of matrix a
    !
    !  nblk        blocksize of cyclic distribution, must be the same in both directions!
    !
    !  mpi_comm_rows
    !  mpi_comm_cols
    !              MPI-Communicators for rows/columns
    !
    !  d(na)       Diagonal elements (returned), identical on all processors
    !
    !  e(na)       Off-Diagonal elements (returned), identical on all processors
    !
    !  tau(na)     Factors for the Householder vectors (returned), needed for back transformation
    !
    !-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik)              :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      complex(kind=ck)              :: tau(na)
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      complex(kind=ck)              :: a(lda,*)
#else
      complex(kind=ck)              :: a(lda,matrixCols)
#endif
      real(kind=rk)                 :: d(na), e(na)

      integer(kind=ik), parameter   :: max_stored_rows = 32

      complex(kind=ck), parameter   :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: totalblocks, max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
      integer(kind=ik)              :: l_cols, l_rows, nstor
      integer(kind=ik)              :: istep, i, j, lcs, lce, lrs, lre
      integer(kind=ik)              :: tile_size, l_rows_tile, l_cols_tile

#ifdef WITH_OPENMP
      integer(kind=ik)              :: my_thread, n_threads, max_threads, n_iter
      integer(kind=ik)              :: omp_get_thread_num, omp_get_num_threads, omp_get_max_threads
#endif

      real(kind=rk)                 :: vnorm2
      complex(kind=ck)              :: vav, xc, aux(2*max_stored_rows),  aux1(2), aux2(2), vrl, xf

      complex(kind=ck), allocatable :: tmp(:), vr(:), vc(:), ur(:), uc(:), vur(:,:), uvc(:,:)
#ifdef WITH_OPENMP
      complex(kind=ck), allocatable :: ur_p(:,:), uc_p(:,:)
#endif
      real(kind=rk), allocatable    :: tmpr(:)
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("tridiag_complex")
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

      ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

      tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
      tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

      l_rows_tile = tile_size/np_rows ! local rows of a tile
      l_cols_tile = tile_size/np_cols ! local cols of a tile


      totalblocks = (na-1)/nblk + 1
      max_blocks_row = (totalblocks-1)/np_rows + 1
      max_blocks_col = (totalblocks-1)/np_cols + 1

      max_local_rows = max_blocks_row*nblk
      max_local_cols = max_blocks_col*nblk

      allocate(tmp(MAX(max_local_rows,max_local_cols)), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating tmp "//errorMessage
       stop
      endif

      allocate(vr(max_local_rows+1), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating vr "//errorMessage
       stop
      endif

      allocate(ur(max_local_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating ur "//errorMessage
       stop
      endif

      allocate(vc(max_local_cols), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating vc "//errorMessage
       stop
      endif

      allocate(uc(max_local_cols), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating uc "//errorMessage
       stop
      endif

#ifdef WITH_OPENMP
      max_threads = omp_get_max_threads()

      allocate(ur_p(max_local_rows,0:max_threads-1), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating ur_p "//errorMessage
       stop
      endif

      allocate(uc_p(max_local_cols,0:max_threads-1), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating uc_p "//errorMessage
       stop
      endif
#endif

      tmp = 0
      vr = 0
      ur = 0
      vc = 0
      uc = 0

      allocate(vur(max_local_rows,2*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating vur "//errorMessage
       stop
      endif

      allocate(uvc(max_local_cols,2*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating uvc "//errorMessage
       stop
      endif

      d(:) = 0
      e(:) = 0
      tau(:) = 0

      nstor = 0

      l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
      l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a
      if (my_prow==prow(na, nblk, np_rows) .and. my_pcol==pcol(na, nblk, np_cols)) d(na) = a(l_rows,l_cols)

      do istep=na,3,-1

        ! Calculate number of local rows and columns of the still remaining matrix
        ! on the local processor

        l_rows = local_index(istep-1, my_prow, np_rows, nblk, -1)
        l_cols = local_index(istep-1, my_pcol, np_cols, nblk, -1)

        ! Calculate vector for Householder transformation on all procs
        ! owning column istep

        if (my_pcol==pcol(istep, nblk, np_cols)) then

          ! Get vector to be transformed; distribute last element and norm of
          ! remaining elements to all procs in current column

          vr(1:l_rows) = a(1:l_rows,l_cols+1)
          if (nstor>0 .and. l_rows>0) then
            aux(1:2*nstor) = conjg(uvc(l_cols+1,1:2*nstor))
            call ZGEMV('N',l_rows,2*nstor,CONE,vur,ubound(vur,dim=1), &
                        aux,1,CONE,vr,1)
          endif

          if (my_prow==prow(istep-1, nblk, np_rows)) then
            aux1(1) = dot_product(vr(1:l_rows-1),vr(1:l_rows-1))
            aux1(2) = vr(l_rows)
          else
            aux1(1) = dot_product(vr(1:l_rows),vr(1:l_rows))
            aux1(2) = 0.
          endif
#ifdef WITH_MPI
          call mpi_allreduce(aux1,aux2,2,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)
#else
          aux2 = aux1
#endif
          vnorm2 = aux2(1)
          vrl    = aux2(2)

          ! Householder transformation

          call hh_transform_complex(vrl, vnorm2, xf, tau(istep))

          ! Scale vr and store Householder vector for back transformation

          vr(1:l_rows) = vr(1:l_rows) * xf
          if (my_prow==prow(istep-1, nblk, np_rows)) then
            vr(l_rows) = 1.
            e(istep-1) = vrl
          endif
          a(1:l_rows,l_cols+1) = vr(1:l_rows) ! store Householder vector for back transformation

        endif

        ! Broadcast the Householder vector (and tau) along columns

        if (my_pcol==pcol(istep, nblk, np_cols)) vr(l_rows+1) = tau(istep)
#ifdef WITH_MPI
        call MPI_Bcast(vr,l_rows+1,MPI_DOUBLE_COMPLEX,pcol(istep, nblk, np_cols),mpi_comm_cols,mpierr)
#endif
        tau(istep) =  vr(l_rows+1)

        ! Transpose Householder vector vr -> vc

!        call elpa_transpose_vectors  (vr, 2*ubound(vr,dim=1), mpi_comm_rows, &
!                                      vc, 2*ubound(vc,dim=1), mpi_comm_cols, &
!                                      1, 2*(istep-1), 1, 2*nblk)

        call elpa_transpose_vectors_complex  (vr, ubound(vr,dim=1), mpi_comm_rows, &
                                              vc, ubound(vc,dim=1), mpi_comm_cols, &
                                              1, (istep-1), 1, nblk)
        ! Calculate u = (A + VU**T + UV**T)*v

        ! For cache efficiency, we use only the upper half of the matrix tiles for this,
        ! thus the result is partly in uc(:) and partly in ur(:)

        uc(1:l_cols) = 0
        ur(1:l_rows) = 0
        if (l_rows>0 .and. l_cols>0) then

#ifdef WITH_OPENMP

#ifdef HAVE_DETAILED_TIMINGS
          call timer%start("OpenMP parallel")
#endif

!$OMP PARALLEL PRIVATE(my_thread,n_threads,n_iter,i,lcs,lce,j,lrs,lre)

          my_thread = omp_get_thread_num()
          n_threads = omp_get_num_threads()

          n_iter = 0

          uc_p(1:l_cols,my_thread) = 0.
          ur_p(1:l_rows,my_thread) = 0.
#endif

          do i=0,(istep-2)/tile_size
            lcs = i*l_cols_tile+1
            lce = min(l_cols,(i+1)*l_cols_tile)
            if (lce<lcs) cycle
            do j=0,i
              lrs = j*l_rows_tile+1
              lre = min(l_rows,(j+1)*l_rows_tile)
              if (lre<lrs) cycle
#ifdef WITH_OPENMP
              if (mod(n_iter,n_threads) == my_thread) then
                call ZGEMV('C',lre-lrs+1,lce-lcs+1,CONE,a(lrs,lcs),lda,vr(lrs),1,CONE,uc_p(lcs,my_thread),1)
                if (i/=j) call ZGEMV('N',lre-lrs+1,lce-lcs+1,CONE,a(lrs,lcs),lda,vc(lcs),1,CONE,ur_p(lrs,my_thread),1)
              endif
              n_iter = n_iter+1
#else
              call ZGEMV('C',lre-lrs+1,lce-lcs+1,CONE,a(lrs,lcs),lda,vr(lrs),1,CONE,uc(lcs),1)
              if (i/=j) call ZGEMV('N',lre-lrs+1,lce-lcs+1,CONE,a(lrs,lcs),lda,vc(lcs),1,CONE,ur(lrs),1)
#endif
            enddo
          enddo

#ifdef WITH_OPENMP
!$OMP END PARALLEL
#ifdef HAVE_DETAILED_TIMINGS
          call timer%stop("OpenMP parallel")
#endif

          do i=0,max_threads-1
            uc(1:l_cols) = uc(1:l_cols) + uc_p(1:l_cols,i)
            ur(1:l_rows) = ur(1:l_rows) + ur_p(1:l_rows,i)
          enddo
#endif

          if (nstor>0) then
            call ZGEMV('C',l_rows,2*nstor,CONE,vur,ubound(vur,dim=1),vr,1,CZERO,aux,1)
            call ZGEMV('N',l_cols,2*nstor,CONE,uvc,ubound(uvc,dim=1),aux,1,CONE,uc,1)
          endif

        endif

        ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
        ! on the processors containing the diagonal
        ! This is only necessary if ur has been calculated, i.e. if the
        ! global tile size is smaller than the global remaining matrix

        if (tile_size < istep-1) then
          call elpa_reduce_add_vectors_COMPLEX  (ur, ubound(ur,dim=1), mpi_comm_rows, &
                                          uc, ubound(uc,dim=1), mpi_comm_cols, &
                                          (istep-1), 1, nblk)
        endif

        ! Sum up all the uc(:) parts, transpose uc -> ur

        if (l_cols>0) then
          tmp(1:l_cols) = uc(1:l_cols)
#ifdef WITH_MPI
          call mpi_allreduce(tmp,uc,l_cols,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)
#else
          uc = tmp
#endif
        endif

!        call elpa_transpose_vectors  (uc, 2*ubound(uc,dim=1), mpi_comm_cols, &
!                                      ur, 2*ubound(ur,dim=1), mpi_comm_rows, &
!                                      1, 2*(istep-1), 1, 2*nblk)

        call elpa_transpose_vectors_complex  (uc, ubound(uc,dim=1), mpi_comm_cols, &
                                              ur, ubound(ur,dim=1), mpi_comm_rows, &
                                              1, (istep-1), 1, nblk)



        ! calculate u**T * v (same as v**T * (A + VU**T + UV**T) * v )

        xc = 0
        if (l_cols>0) xc = dot_product(vc(1:l_cols),uc(1:l_cols))
#ifdef WITH_MPI
        call mpi_allreduce(xc,vav,1,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_cols,mpierr)
#else
        vav = xc
#endif
        ! store u and v in the matrices U and V
        ! these matrices are stored combined in one here

        do j=1,l_rows
          vur(j,2*nstor+1) = conjg(tau(istep))*vr(j)
          vur(j,2*nstor+2) = 0.5*conjg(tau(istep))*vav*vr(j) - ur(j)
        enddo
        do j=1,l_cols
          uvc(j,2*nstor+1) = 0.5*conjg(tau(istep))*vav*vc(j) - uc(j)
          uvc(j,2*nstor+2) = conjg(tau(istep))*vc(j)
        enddo

        nstor = nstor+1

        ! If the limit of max_stored_rows is reached, calculate A + VU**T + UV**T

        if (nstor==max_stored_rows .or. istep==3) then

          do i=0,(istep-2)/tile_size
            lcs = i*l_cols_tile+1
            lce = min(l_cols,(i+1)*l_cols_tile)
            lrs = 1
            lre = min(l_rows,(i+1)*l_rows_tile)
            if (lce<lcs .or. lre<lrs) cycle
            call ZGEMM('N','C',lre-lrs+1,lce-lcs+1,2*nstor,CONE, &
                         vur(lrs,1),ubound(vur,dim=1),uvc(lcs,1),ubound(uvc,dim=1), &
                         CONE,a(lrs,lcs),lda)
          enddo

          nstor = 0

        endif

        if (my_prow==prow(istep-1, nblk, np_rows) .and. my_pcol==pcol(istep-1, nblk, np_cols)) then
          if (nstor>0) a(l_rows,l_cols) = a(l_rows,l_cols) &
                          + dot_product(vur(l_rows,1:2*nstor),uvc(l_cols,1:2*nstor))
          d(istep-1) = a(l_rows,l_cols)
        endif

      enddo ! istep

      ! Store e(1) and d(1)

      if (my_pcol==pcol(2, nblk, np_cols)) then
        if (my_prow==prow(1, nblk, np_rows)) then
          ! We use last l_cols value of loop above
          vrl = a(1,l_cols)
          call hh_transform_complex(vrl, 0.d0, xf, tau(2))
          e(1) = vrl
          a(1,l_cols) = 1. ! for consistency only
        endif
#ifdef WITH_MPI
        call mpi_bcast(tau(2),1,MPI_DOUBLE_COMPLEX,prow(1, nblk, np_rows),mpi_comm_rows,mpierr)
#endif
      endif
#ifdef WITH_MPI
      call mpi_bcast(tau(2),1,MPI_DOUBLE_COMPLEX,pcol(2, nblk, np_cols),mpi_comm_cols,mpierr)
#endif

      if (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(1, nblk, np_cols)) d(1) = a(1,1)

      deallocate(tmp, vr, ur, vc, uc, vur, uvc, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when deallocating tmp "//errorMessage
       stop
      endif
      ! distribute the arrays d and e to all processors

      allocate(tmpr(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when allocating tmpr "//errorMessage
       stop
      endif
#ifdef WITH_MPI
      tmpr = d
      call mpi_allreduce(tmpr,d,na,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
      tmpr = d
      call mpi_allreduce(tmpr,d,na,MPI_REAL8,MPI_SUM,mpi_comm_cols,mpierr)
      tmpr = e
      call mpi_allreduce(tmpr,e,na,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
      tmpr = e
      call mpi_allreduce(tmpr,e,na,MPI_REAL8,MPI_SUM,mpi_comm_cols,mpierr)
#endif
      deallocate(tmpr, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"tridiag_complex: error when deallocating tmpr "//errorMessage
       stop
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("tridiag_complex")
#endif

    end subroutine tridiag_complex

    subroutine trans_ev_complex(na, nqc, a, lda, tau, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)

    !-------------------------------------------------------------------------------
    !  trans_ev_complex: Transforms the eigenvectors of a tridiagonal matrix back
    !                    to the eigenvectors of the original matrix
    !                    (like Scalapack Routine PZUNMTR)
    !
    !  Parameters
    !
    !  na          Order of matrix a, number of rows of matrix q
    !
    !  nqc         Number of columns of matrix q
    !
    !  a(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a after tridiag_complex)
    !              Distribution is like in Scalapack.
    !
    !  lda         Leading dimension of a
    !
    !  tau(na)     Factors of the Householder vectors
    !
    !  q           On input: Eigenvectors of tridiagonal matrix
    !              On output: Transformed eigenvectors
    !              Distribution is like in Scalapack.
    !
    !  ldq         Leading dimension of q
    !
    !  nblk        blocksize of cyclic distribution, must be the same in both directions!
    !
    !  mpi_comm_rows
    !  mpi_comm_cols
    !              MPI-Communicators for rows/columns
    !
    !-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik)              ::  na, nqc, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      complex(kind=ck)              ::  tau(na)
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      complex(kind=ck)              :: a(lda,*), q(ldq,*)
#else
      complex(kind=ck)              ::  a(lda,matrixCols), q(ldq,matrixCols)
#endif
      integer(kind=ik)              :: max_stored_rows

      complex(kind=ck), parameter   :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: totalblocks, max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
      integer(kind=ik)              :: l_cols, l_rows, l_colh, nstor
      integer(kind=ik)              :: istep, i, n, nc, ic, ics, ice, nb, cur_pcol

      complex(kind=ck), allocatable :: tmp1(:), tmp2(:), hvb(:), hvm(:,:)
      complex(kind=ck), allocatable :: tmat(:,:), h1(:), h2(:)
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage
#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("trans_ev_complex")
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

      totalblocks = (na-1)/nblk + 1
      max_blocks_row = (totalblocks-1)/np_rows + 1
      max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q!

      max_local_rows = max_blocks_row*nblk
      max_local_cols = max_blocks_col*nblk

      max_stored_rows = (63/nblk+1)*nblk

      allocate(tmat(max_stored_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"trans_ev_complex: error when allocating tmat "//errorMessage
       stop
      endif

      allocate(h1(max_stored_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"trans_ev_complex: error when allocating h1 "//errorMessage
       stop
      endif

      allocate(h2(max_stored_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"trans_ev_complex: error when allocating h2 "//errorMessage
       stop
      endif

      allocate(tmp1(max_local_cols*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"trans_ev_complex: error when allocating tmp1 "//errorMessage
       stop
      endif

      allocate(tmp2(max_local_cols*max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"trans_ev_complex: error when allocating tmp2 "//errorMessage
       stop
      endif

      allocate(hvb(max_local_rows*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"trans_ev_complex: error when allocating hvb "//errorMessage
       stop
      endif

      allocate(hvm(max_local_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"trans_ev_complex: error when allocating hvm "//errorMessage
       stop
      endif

      hvm = 0   ! Must be set to 0 !!!
      hvb = 0   ! Safety only

      l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q

      nstor = 0

      ! In the complex case tau(2) /= 0
      if (my_prow == prow(1, nblk, np_rows)) then
        q(1,1:l_cols) = q(1,1:l_cols)*((1.d0,0.d0)-tau(2))
      endif

      do istep=1,na,nblk

        ics = MAX(istep,3)
        ice = MIN(istep+nblk-1,na)
        if (ice<ics) cycle

        cur_pcol = pcol(istep, nblk, np_cols)

        nb = 0
        do ic=ics,ice

          l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1) ! Column of Householder vector
          l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder vector


          if (my_pcol==cur_pcol) then
            hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)
            if (my_prow==prow(ic-1, nblk, np_rows)) then
              hvb(nb+l_rows) = 1.
            endif
          endif

          nb = nb+l_rows
        enddo

#ifdef WITH_MPI
        if (nb>0) &
           call MPI_Bcast(hvb,nb,MPI_DOUBLE_COMPLEX,cur_pcol,mpi_comm_cols,mpierr)
#endif
        nb = 0
        do ic=ics,ice
          l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder vector
          hvm(1:l_rows,nstor+1) = hvb(nb+1:nb+l_rows)
          nstor = nstor+1
          nb = nb+l_rows
        enddo

        ! Please note: for smaller matix sizes (na/np_rows<=256), a value of 32 for nstor is enough!
        if (nstor+nblk>max_stored_rows .or. istep+nblk>na .or. (na/np_rows<=256 .and. nstor>=32)) then

          ! Calculate scalar products of stored vectors.
          ! This can be done in different ways, we use zherk

          tmat = 0
          if (l_rows>0) &
             call zherk('U','C',nstor,l_rows,CONE,hvm,ubound(hvm,dim=1),CZERO,tmat,max_stored_rows)

          nc = 0
          do n=1,nstor-1
            h1(nc+1:nc+n) = tmat(1:n,n+1)
            nc = nc+n
          enddo
#ifdef WITH_MPI
          if (nc>0) call mpi_allreduce(h1,h2,nc,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)
#else
          if (nc>0) h2=h1
#endif
          ! Calculate triangular matrix T

          nc = 0
          tmat(1,1) = tau(ice-nstor+1)
          do n=1,nstor-1
            call ztrmv('L','C','N',n,tmat,max_stored_rows,h2(nc+1),1)
            tmat(n+1,1:n) = -conjg(h2(nc+1:nc+n))*tau(ice-nstor+n+1)
            tmat(n+1,n+1) = tau(ice-nstor+n+1)
            nc = nc+n
          enddo

          ! Q = Q - V * T * V**T * Q

          if (l_rows>0) then
            call zgemm('C','N',nstor,l_cols,l_rows,CONE,hvm,ubound(hvm,dim=1), &
                        q,ldq,CZERO,tmp1,nstor)
          else
            tmp1(1:l_cols*nstor) = 0
          endif
#ifdef WITH_MPI
          call mpi_allreduce(tmp1,tmp2,nstor*l_cols,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)
#else
          tmp2 = tmp1
#endif
          if (l_rows>0) then
            call ztrmm('L','L','N','N',nstor,l_cols,CONE,tmat,max_stored_rows,tmp2,nstor)
            call zgemm('N','N',l_rows,l_cols,nstor,-CONE,hvm,ubound(hvm,dim=1), &
                        tmp2,nstor,CONE,q,ldq)
          endif
          nstor = 0
        endif

      enddo

      deallocate(tmat, h1, h2, tmp1, tmp2, hvb, hvm, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"trans_ev_complex: error when deallocating hvb "//errorMessage
       stop
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("trans_ev_complex")
#endif

    end subroutine trans_ev_complex

    subroutine solve_tridi( na, nev, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success )
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik)              :: na, nev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=rk)                 :: d(na), e(na)
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      real(kind=rk)                 :: q(ldq,*)
#else
      real(kind=rk)                 :: q(ldq,matrixCols)
#endif

      integer(kind=ik)              :: i, j, n, np, nc, nev1, l_cols, l_rows
      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr

      integer(kind=ik), allocatable :: limits(:), l_col(:), p_col(:), l_col_bc(:), p_col_bc(:)

      logical, intent(in)           :: wantDebug
      logical, intent(out)          :: success
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("solve_tridi")
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      success = .true.

      l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
      l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q

      ! Set Q to 0

      q(1:l_rows, 1:l_cols) = 0.

      ! Get the limits of the subdivisons, each subdivison has as many cols
      ! as fit on the respective processor column

      allocate(limits(0:np_cols), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating limits "//errorMessage
        stop
      endif

      limits(0) = 0
      do np=0,np_cols-1
        nc = local_index(na, np, np_cols, nblk, -1) ! number of columns on proc column np

        ! Check for the case that a column has have zero width.
        ! This is not supported!
        ! Scalapack supports it but delivers no results for these columns,
        ! which is rather annoying
        if (nc==0) then
#ifdef HAVE_DETAILED_TIMINGS
          call timer%stop("solve_tridi")
#endif
          if (wantDebug) write(error_unit,*) 'ELPA1_solve_tridi: ERROR: Problem contains processor column with zero width'
          success = .false.
          return
        endif
        limits(np+1) = limits(np) + nc
      enddo

      ! Subdivide matrix by subtracting rank 1 modifications

      do i=1,np_cols-1
        n = limits(i)
        d(n) = d(n)-abs(e(n))
        d(n+1) = d(n+1)-abs(e(n))
      enddo

      ! Solve sub problems on processsor columns

      nc = limits(my_pcol) ! column after which my problem starts

      if (np_cols>1) then
        nev1 = l_cols ! all eigenvectors are needed
      else
        nev1 = MIN(nev,l_cols)
      endif
      call solve_tridi_col(l_cols, nev1, nc, d(nc+1), e(nc+1), q, ldq, nblk,  &
                        matrixCols, mpi_comm_rows, wantDebug, success)
      if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("solve_tridi")
#endif
        return
      endif
      ! If there is only 1 processor column, we are done

      if (np_cols==1) then
        deallocate(limits, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"solve_tridi: error when deallocating limits "//errorMessage
          stop
        endif

#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("solve_tridi")
#endif
        return
      endif

      ! Set index arrays for Q columns

      ! Dense distribution scheme:

      allocate(l_col(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating l_col "//errorMessage
        stop
      endif

      allocate(p_col(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating p_col "//errorMessage
        stop
      endif

      n = 0
      do np=0,np_cols-1
        nc = local_index(na, np, np_cols, nblk, -1)
        do i=1,nc
          n = n+1
          l_col(n) = i
          p_col(n) = np
        enddo
      enddo

      ! Block cyclic distribution scheme, only nev columns are set:

      allocate(l_col_bc(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating l_col_bc "//errorMessage
        stop
      endif

      allocate(p_col_bc(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when allocating p_col_bc "//errorMessage
        stop
      endif

      p_col_bc(:) = -1
      l_col_bc(:) = -1

      do i = 0, na-1, nblk*np_cols
        do j = 0, np_cols-1
          do n = 1, nblk
            if (i+j*nblk+n <= MIN(nev,na)) then
              p_col_bc(i+j*nblk+n) = j
              l_col_bc(i+j*nblk+n) = i/np_cols + n
             endif
           enddo
         enddo
      enddo

      ! Recursively merge sub problems

      call merge_recursive(0, np_cols, wantDebug, success)
      if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("solve_tridi")
#endif
        return
      endif

      deallocate(limits,l_col,p_col,l_col_bc,p_col_bc, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi: error when deallocating l_col "//errorMessage
        stop
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("solve_tridi")
#endif
      return

      contains
        recursive subroutine merge_recursive(np_off, nprocs, wantDebug, success)
           use precision
           implicit none

           ! noff is always a multiple of nblk_ev
           ! nlen-noff is always > nblk_ev

           integer(kind=ik)     :: np_off, nprocs
           integer(kind=ik)     :: np1, np2, noff, nlen, nmid, n
#ifdef WITH_MPI
           integer(kind=ik)     :: mpi_status(mpi_status_size)
#endif
           logical, intent(in)  :: wantDebug
           logical, intent(out) :: success

           success = .true.

           if (nprocs<=1) then
             ! Safety check only
             if (wantDebug) write(error_unit,*) "ELPA1_merge_recursive: INTERNAL error merge_recursive: nprocs=",nprocs
             success = .false.
             return
           endif
           ! Split problem into 2 subproblems of size np1 / np2

           np1 = nprocs/2
           np2 = nprocs-np1

           if (np1 > 1) call merge_recursive(np_off, np1, wantDebug, success)
           if (.not.(success)) return
           if (np2 > 1) call merge_recursive(np_off+np1, np2, wantDebug, success)
           if (.not.(success)) return

           noff = limits(np_off)
           nmid = limits(np_off+np1) - noff
           nlen = limits(np_off+nprocs) - noff

#ifdef WITH_MPI
           if (my_pcol==np_off) then
             do n=np_off+np1,np_off+nprocs-1
               call mpi_send(d(noff+1),nmid,MPI_REAL8,n,1,mpi_comm_cols,mpierr)
             enddo
           endif
#endif

           if (my_pcol>=np_off+np1 .and. my_pcol<np_off+nprocs) then
#ifdef WITH_MPI
             call mpi_recv(d(noff+1),nmid,MPI_REAL8,np_off,1,mpi_comm_cols,mpi_status,mpierr)
#else
             d(noff+1:noff+1+nmid-1) = d(noff+1:noff+1+nmid-1)
#endif
           endif

           if (my_pcol==np_off+np1) then
             do n=np_off,np_off+np1-1
#ifdef WITH_MPI
               call mpi_send(d(noff+nmid+1),nlen-nmid,MPI_REAL8,n,1,mpi_comm_cols,mpierr)
#endif
             enddo
           endif
           if (my_pcol>=np_off .and. my_pcol<np_off+np1) then
#ifdef WITH_MPI
             call mpi_recv(d(noff+nmid+1),nlen-nmid,MPI_REAL8,np_off+np1,1,mpi_comm_cols,mpi_status,mpierr)
#else
             d(noff+nmid+1:noff+nmid+1+nlen-nmid-1) = d(noff+nmid+1:noff+nmid+1+nlen-nmid-1) 
#endif
           endif
           if (nprocs == np_cols) then

             ! Last merge, result distribution must be block cyclic, noff==0,
             ! p_col_bc is set so that only nev eigenvalues are calculated

             call merge_systems(nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, noff, &
                                 nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, l_col, p_col, &
                                 l_col_bc, p_col_bc, np_off, nprocs, wantDebug, success )
             if (.not.(success)) return
           else
             ! Not last merge, leave dense column distribution

             call merge_systems(nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, noff, &
                                 nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, l_col(noff+1), p_col(noff+1), &
                                 l_col(noff+1), p_col(noff+1), np_off, nprocs, wantDebug, success )
             if (.not.(success)) return
           endif

       end subroutine merge_recursive

    end subroutine solve_tridi

    subroutine solve_tridi_col( na, nev, nqoff, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, wantDebug, success )

   ! Solves the symmetric, tridiagonal eigenvalue problem on one processor column
   ! with the divide and conquer method.
   ! Works best if the number of processor rows is a power of 2!
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik)              :: na, nev, nqoff, ldq, nblk, matrixCols, mpi_comm_rows
      real(kind=rk)                 :: d(na), e(na)
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      real(kind=rk)                 :: q(ldq,*)
#else
      real(kind=rk)                 :: q(ldq,matrixCols)
#endif

      integer(kind=ik), parameter   :: min_submatrix_size = 16 ! Minimum size of the submatrices to be used

      real(kind=rk), allocatable    :: qmat1(:,:), qmat2(:,:)
      integer(kind=ik)              :: i, n, np
      integer(kind=ik)              :: ndiv, noff, nmid, nlen, max_size
      integer(kind=ik)              :: my_prow, np_rows, mpierr

      integer(kind=ik), allocatable :: limits(:), l_col(:), p_col_i(:), p_col_o(:)
      logical, intent(in)           :: wantDebug
      logical, intent(out)          :: success
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("solve_tridi_col")
#endif

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      success = .true.
      ! Calculate the number of subdivisions needed.

      n = na
      ndiv = 1
      do while(2*ndiv<=np_rows .and. n>2*min_submatrix_size)
        n = ((n+3)/4)*2 ! the bigger one of the two halves, we want EVEN boundaries
        ndiv = ndiv*2
      enddo

      ! If there is only 1 processor row and not all eigenvectors are needed
      ! and the matrix size is big enough, then use 2 subdivisions
      ! so that merge_systems is called once and only the needed
      ! eigenvectors are calculated for the final problem.

      if (np_rows==1 .and. nev<na .and. na>2*min_submatrix_size) ndiv = 2

      allocate(limits(0:ndiv), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi_col: error when allocating limits "//errorMessage
        stop
      endif

      limits(0) = 0
      limits(ndiv) = na

      n = ndiv
      do while(n>1)
        n = n/2 ! n is always a power of 2
        do i=0,ndiv-1,2*n
          ! We want to have even boundaries (for cache line alignments)
          limits(i+n) = limits(i) + ((limits(i+2*n)-limits(i)+3)/4)*2
        enddo
      enddo

      ! Calculate the maximum size of a subproblem

      max_size = 0
      do i=1,ndiv
        max_size = MAX(max_size,limits(i)-limits(i-1))
      enddo

      ! Subdivide matrix by subtracting rank 1 modifications

      do i=1,ndiv-1
        n = limits(i)
        d(n) = d(n)-abs(e(n))
        d(n+1) = d(n+1)-abs(e(n))
      enddo

      if (np_rows==1)    then

        ! For 1 processor row there may be 1 or 2 subdivisions

        do n=0,ndiv-1
          noff = limits(n)        ! Start of subproblem
          nlen = limits(n+1)-noff ! Size of subproblem

          call solve_tridi_single(nlen,d(noff+1),e(noff+1), &
                                    q(nqoff+noff+1,noff+1),ubound(q,dim=1), wantDebug, success)
          if (.not.(success)) return
        enddo

      else

        ! Solve sub problems in parallel with solve_tridi_single
        ! There is at maximum 1 subproblem per processor

        allocate(qmat1(max_size,max_size), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"solve_tridi_col: error when allocating qmat1 "//errorMessage
          stop
        endif

        allocate(qmat2(max_size,max_size), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"solve_tridi_col: error when allocating qmat2 "//errorMessage
          stop
        endif

        qmat1 = 0 ! Make sure that all elements are defined

        if (my_prow < ndiv) then

          noff = limits(my_prow)        ! Start of subproblem
          nlen = limits(my_prow+1)-noff ! Size of subproblem

          call solve_tridi_single(nlen,d(noff+1),e(noff+1),qmat1, &
                                    ubound(qmat1,dim=1), wantDebug, success)

          if (.not.(success)) return
        endif

        ! Fill eigenvectors in qmat1 into global matrix q

        do np = 0, ndiv-1

          noff = limits(np)
          nlen = limits(np+1)-noff
#ifdef WITH_MPI
          call MPI_Bcast(d(noff+1),nlen,MPI_REAL8,np,mpi_comm_rows,mpierr)
#endif
          qmat2 = qmat1
#ifdef WITH_MPI
          call MPI_Bcast(qmat2,max_size*max_size,MPI_REAL8,np,mpi_comm_rows,mpierr)
#endif
          do i=1,nlen
            call distribute_global_column(qmat2(1,i), q(1,noff+i), nqoff+noff, nlen, my_prow, np_rows, nblk)
          enddo

        enddo

        deallocate(qmat1, qmat2, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"solve_tridi_col: error when deallocating qmat2 "//errorMessage
          stop
        endif

      endif

      ! Allocate and set index arrays l_col and p_col

      allocate(l_col(na), p_col_i(na),  p_col_o(na), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi_col: error when allocating l_col "//errorMessage
        stop
      endif

      do i=1,na
        l_col(i) = i
        p_col_i(i) = 0
        p_col_o(i) = 0
      enddo

      ! Merge subproblems

      n = 1
      do while(n<ndiv) ! if ndiv==1, the problem was solved by single call to solve_tridi_single

        do i=0,ndiv-1,2*n

          noff = limits(i)
          nmid = limits(i+n) - noff
          nlen = limits(i+2*n) - noff

          if (nlen == na) then
            ! Last merge, set p_col_o=-1 for unneeded (output) eigenvectors
            p_col_o(nev+1:na) = -1
          endif

          call merge_systems(nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, nqoff+noff, nblk, &
                               matrixCols, mpi_comm_rows, mpi_comm_self, l_col(noff+1), p_col_i(noff+1), &
                               l_col(noff+1), p_col_o(noff+1), 0, 1, wantDebug, success)
          if (.not.(success)) return

        enddo

        n = 2*n

      enddo

      deallocate(limits, l_col, p_col_i, p_col_o, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"solve_tridi_col: error when deallocating l_col "//errorMessage
        stop
      endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("solve_tridi_col")
#endif

    end subroutine solve_tridi_col

    subroutine solve_tridi_single(nlen, d, e, q, ldq, wantDebug, success)

   ! Solves the symmetric, tridiagonal eigenvalue problem on a single processor.
   ! Takes precautions if DSTEDC fails or if the eigenvalues are not ordered correctly.
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     use precision
     implicit none

     integer(kind=ik)              :: nlen, ldq
     real(kind=rk)                 :: d(nlen), e(nlen), q(ldq,nlen)

     real(kind=rk), allocatable    :: work(:), qtmp(:), ds(:), es(:)
     real(kind=rk)                 :: dtmp

     integer(kind=ik)              :: i, j, lwork, liwork, info, mpierr
     integer(kind=ik), allocatable :: iwork(:)

     logical, intent(in)           :: wantDebug
     logical, intent(out)          :: success
      integer(kind=ik)             :: istat
      character(200)               :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("solve_tridi_single")
#endif

     success = .true.
     allocate(ds(nlen), es(nlen), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_tridi_single: error when allocating ds "//errorMessage
       stop
     endif

     ! Save d and e for the case that dstedc fails

     ds(:) = d(:)
     es(:) = e(:)

     ! First try dstedc, this is normally faster but it may fail sometimes (why???)

     lwork = 1 + 4*nlen + nlen**2
     liwork =  3 + 5*nlen
     allocate(work(lwork), iwork(liwork), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_tridi_single: error when allocating work "//errorMessage
       stop
     endif
     call dstedc('I',nlen,d,e,q,ldq,work,lwork,iwork,liwork,info)

     if (info /= 0) then

       ! DSTEDC failed, try DSTEQR. The workspace is enough for DSTEQR.

       write(error_unit,'(a,i8,a)') 'Warning: Lapack routine DSTEDC failed, info= ',info,', Trying DSTEQR!'

       d(:) = ds(:)
       e(:) = es(:)
       call dsteqr('I',nlen,d,e,q,ldq,work,info)

       ! If DSTEQR fails also, we don't know what to do further ...

       if (info /= 0) then
         if (wantDebug) &
           write(error_unit,'(a,i8,a)') 'ELPA1_solve_tridi_single: ERROR: Lapack routine DSTEQR failed, info= ',info,', Aborting!'
           success = .false.
           return
         endif
       end if

       deallocate(work,iwork,ds,es, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"solve_tridi_single: error when deallocating ds "//errorMessage
         stop
       endif

      ! Check if eigenvalues are monotonically increasing
      ! This seems to be not always the case  (in the IBM implementation of dstedc ???)

      do i=1,nlen-1
        if (d(i+1)<d(i)) then
          if (abs(d(i+1) - d(i)) / abs(d(i+1) + d(i)) > 1d-14) then
            write(error_unit,'(a,i8,2g25.16)') '***WARNING: Monotony error dste**:',i+1,d(i),d(i+1)
          else
            write(error_unit,'(a,i8,2g25.16)') 'Info: Monotony error dste{dc,qr}:',i+1,d(i),d(i+1)
            write(error_unit,'(a)') 'The eigenvalues from a lapack call are not sorted to machine precision.'
            write(error_unit,'(a)') 'In this extent, this is completely harmless.'
            write(error_unit,'(a)') 'Still, we keep this info message just in case.'
          end if
          allocate(qtmp(nlen), stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"solve_tridi_single: error when allocating qtmp "//errorMessage
            stop
          endif

          dtmp = d(i+1)
          qtmp(1:nlen) = q(1:nlen,i+1)
          do j=i,1,-1
            if (dtmp<d(j)) then
              d(j+1)        = d(j)
              q(1:nlen,j+1) = q(1:nlen,j)
            else
              exit ! Loop
            endif
          enddo
          d(j+1)        = dtmp
          q(1:nlen,j+1) = qtmp(1:nlen)
          deallocate(qtmp, stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"solve_tridi_single: error when deallocating qtmp "//errorMessage
            stop
          endif

       endif
     enddo
#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("solve_tridi_single")
#endif

    end subroutine solve_tridi_single

    subroutine merge_systems( na, nm, d, e, q, ldq, nqoff, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                          l_col, p_col, l_col_out, p_col_out, npc_0, npc_n, wantDebug, success)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik)              :: na, nm, ldq, nqoff, nblk, matrixCols, mpi_comm_rows, &
                                       mpi_comm_cols, npc_0, npc_n
      integer(kind=ik)              :: l_col(na), p_col(na), l_col_out(na), p_col_out(na)
      real(kind=rk)                 :: d(na), e
#ifdef DESPERATELY_WANT_ASSUMED_SIZE
      real(kind=rk)                 :: q(ldq,*)
#else
      real(kind=rk)                 :: q(ldq,matrixCols)
#endif

      integer(kind=ik), parameter   :: max_strip=128

      real(kind=rk)                 :: beta, sig, s, c, t, tau, rho, eps, tol, dlamch, &
                                       dlapy2, qtrans(2,2), dmax, zmax, d1new, d2new
      real(kind=rk)                 :: z(na), d1(na), d2(na), z1(na), delta(na),  &
                                       dbase(na), ddiff(na), ev_scale(na), tmp(na)
      real(kind=rk)                 :: d1u(na), zu(na), d1l(na), zl(na)
      real(kind=rk), allocatable    :: qtmp1(:,:), qtmp2(:,:), ev(:,:)
#ifdef WITH_OPENMP
      real(kind=rk), allocatable    :: z_p(:,:)
#endif

      integer(kind=ik)              :: i, j, na1, na2, l_rows, l_cols, l_rqs, l_rqe, &
                                       l_rqm, ns, info
      integer(kind=ik)              :: l_rnm, nnzu, nnzl, ndef, ncnt, max_local_cols, &
                                       l_cols_qreorg, np, l_idx, nqcols1, nqcols2
      integer(kind=ik)              :: my_proc, n_procs, my_prow, my_pcol, np_rows, &
                                       np_cols, mpierr
#ifdef WITH_MPI
      integer(kind=ik)              :: mpi_status(mpi_status_size)
#endif
      integer(kind=ik)              :: np_next, np_prev, np_rem
      integer(kind=ik)              :: idx(na), idx1(na), idx2(na)
      integer(kind=ik)              :: coltyp(na), idxq1(na), idxq2(na)

      logical, intent(in)           :: wantDebug
      logical, intent(out)          :: success
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage

#ifdef WITH_OPENMP
      integer(kind=ik)              :: max_threads, my_thread
      integer(kind=ik)              :: omp_get_max_threads, omp_get_thread_num


      max_threads = omp_get_max_threads()

      allocate(z_p(na,0:max_threads-1), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"merge_systems: error when allocating z_p "//errorMessage
        stop
      endif
#endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("merge_systems")
#endif
      success = .true.

      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      ! If my processor column isn't in the requested set, do nothing

      if (my_pcol<npc_0 .or. my_pcol>=npc_0+npc_n) then
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("merge_systems")
#endif
        return
      endif
      ! Determine number of "next" and "prev" column for ring sends

      if (my_pcol == npc_0+npc_n-1) then
        np_next = npc_0
      else
        np_next = my_pcol + 1
      endif

      if (my_pcol == npc_0) then
        np_prev = npc_0+npc_n-1
      else
        np_prev = my_pcol - 1
      endif

      call check_monotony(nm,d,'Input1',wantDebug, success)
      if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("merge_systems")
#endif
        return
      endif
      call check_monotony(na-nm,d(nm+1),'Input2',wantDebug, success)
      if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("merge_systems")
#endif
        return
      endif
      ! Get global number of processors and my processor number.
      ! Please note that my_proc does not need to match any real processor number,
      ! it is just used for load balancing some loops.

      n_procs = np_rows*npc_n
      my_proc = my_prow*npc_n + (my_pcol-npc_0) ! Row major


      ! Local limits of the rows of Q

      l_rqs = local_index(nqoff+1 , my_prow, np_rows, nblk, +1) ! First row of Q
      l_rqm = local_index(nqoff+nm, my_prow, np_rows, nblk, -1) ! Last row <= nm
      l_rqe = local_index(nqoff+na, my_prow, np_rows, nblk, -1) ! Last row of Q

      l_rnm  = l_rqm-l_rqs+1 ! Number of local rows <= nm
      l_rows = l_rqe-l_rqs+1 ! Total number of local rows


      ! My number of local columns

      l_cols = COUNT(p_col(1:na)==my_pcol)

      ! Get max number of local columns

      max_local_cols = 0
      do np = npc_0, npc_0+npc_n-1
        max_local_cols = MAX(max_local_cols,COUNT(p_col(1:na)==np))
      enddo

      ! Calculations start here

      beta = abs(e)
      sig  = sign(1.d0,e)

      ! Calculate rank-1 modifier z

      z(:) = 0

      if (MOD((nqoff+nm-1)/nblk,np_rows)==my_prow) then
        ! nm is local on my row
        do i = 1, na
          if (p_col(i)==my_pcol) z(i) = q(l_rqm,l_col(i))
         enddo
      endif

      if (MOD((nqoff+nm)/nblk,np_rows)==my_prow) then
        ! nm+1 is local on my row
        do i = 1, na
          if (p_col(i)==my_pcol) z(i) = z(i) + sig*q(l_rqm+1,l_col(i))
        enddo
      endif

      call global_gather(z, na)

      ! Normalize z so that norm(z) = 1.  Since z is the concatenation of
      ! two normalized vectors, norm2(z) = sqrt(2).

      z = z/sqrt(2.0d0)
      rho = 2.*beta

      ! Calculate index for merging both systems by ascending eigenvalues

      call DLAMRG( nm, na-nm, d, 1, 1, idx )

      ! Calculate the allowable deflation tolerance

      zmax = maxval(abs(z))
      dmax = maxval(abs(d))
      EPS = DLAMCH( 'Epsilon' )
      TOL = 8.*EPS*MAX(dmax,zmax)

      ! If the rank-1 modifier is small enough, no more needs to be done
      ! except to reorganize D and Q

      IF ( RHO*zmax <= TOL ) THEN

        ! Rearrange eigenvalues

        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo

        ! Rearrange eigenvectors

        call resort_ev(idx, na)
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("merge_systems")
#endif

        return
      ENDIF

      ! Merge and deflate system

      na1 = 0
      na2 = 0

      ! COLTYP:
      ! 1 : non-zero in the upper half only;
      ! 2 : dense;
      ! 3 : non-zero in the lower half only;
      ! 4 : deflated.

      coltyp(1:nm) = 1
      coltyp(nm+1:na) = 3

      do i=1,na

        if (rho*abs(z(idx(i))) <= tol) then

          ! Deflate due to small z component.

          na2 = na2+1
          d2(na2)   = d(idx(i))
          idx2(na2) = idx(i)
          coltyp(idx(i)) = 4

        else if (na1>0) then

          ! Check if eigenvalues are close enough to allow deflation.

          S = Z(idx(i))
          C = Z1(na1)

          ! Find sqrt(a**2+b**2) without overflow or
          ! destructive underflow.

          TAU = DLAPY2( C, S )
          T = D1(na1) - D(idx(i))
          C = C / TAU
          S = -S / TAU
          IF ( ABS( T*C*S ) <= TOL ) THEN

            ! Deflation is possible.

            na2 = na2+1

            Z1(na1) = TAU

            d2new = D(idx(i))*C**2 + D1(na1)*S**2
            d1new = D(idx(i))*S**2 + D1(na1)*C**2

            ! D(idx(i)) >= D1(na1) and C**2 + S**2 == 1.0
            ! This means that after the above transformation it must be
            !    D1(na1) <= d1new <= D(idx(i))
            !    D1(na1) <= d2new <= D(idx(i))
            !
            ! D1(na1) may get bigger but it is still smaller than the next D(idx(i+1))
            ! so there is no problem with sorting here.
            ! d2new <= D(idx(i)) which means that it might be smaller than D2(na2-1)
            ! which makes a check (and possibly a resort) necessary.
            !
            ! The above relations may not hold exactly due to numeric differences
            ! so they have to be enforced in order not to get troubles with sorting.


            if (d1new<D1(na1)  ) d1new = D1(na1)
            if (d1new>D(idx(i))) d1new = D(idx(i))

            if (d2new<D1(na1)  ) d2new = D1(na1)
            if (d2new>D(idx(i))) d2new = D(idx(i))

            D1(na1) = d1new

            do j=na2-1,1,-1
              if (d2new<d2(j)) then
                d2(j+1)   = d2(j)
                idx2(j+1) = idx2(j)
              else
                exit ! Loop
              endif
            enddo

            d2(j+1)   = d2new
            idx2(j+1) = idx(i)

            qtrans(1,1) = C; qtrans(1,2) =-S
            qtrans(2,1) = S; qtrans(2,2) = C

            call transform_columns(idx(i), idx1(na1))

            if (coltyp(idx(i))==1 .and. coltyp(idx1(na1))/=1) coltyp(idx1(na1)) = 2
            if (coltyp(idx(i))==3 .and. coltyp(idx1(na1))/=3) coltyp(idx1(na1)) = 2

            coltyp(idx(i)) = 4

          else
            na1 = na1+1
            d1(na1) = d(idx(i))
            z1(na1) = z(idx(i))
            idx1(na1) = idx(i)
          endif
        else
          na1 = na1+1
          d1(na1) = d(idx(i))
          z1(na1) = z(idx(i))
          idx1(na1) = idx(i)
        endif

      enddo
      call check_monotony(na1,d1,'Sorted1', wantDebug, success)
      if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("merge_systems")
#endif
        return
      endif
      call check_monotony(na2,d2,'Sorted2', wantDebug, success)
      if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("merge_systems")
#endif
        return
      endif

      if (na1==1 .or. na1==2) then
        ! if(my_proc==0) print *,'--- Remark solve_tridi: na1==',na1,' proc==',myid

        if (na1==1) then
          d(1) = d1(1) + rho*z1(1)**2 ! solve secular equation
        else ! na1==2
          call DLAED5(1, d1, z1, qtrans(1,1), rho, d(1))
          call DLAED5(2, d1, z1, qtrans(1,2), rho, d(2))

          call transform_columns(idx1(1), idx1(2))
        endif

        ! Add the deflated eigenvalues
        d(na1+1:na) = d2(1:na2)

        ! Calculate arrangement of all eigenvalues  in output

        call DLAMRG( na1, na-na1, d, 1, 1, idx )

        ! Rearrange eigenvalues

        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo

        ! Rearrange eigenvectors

        do i=1,na
          if (idx(i)<=na1) then
            idxq1(i) = idx1(idx(i))
          else
            idxq1(i) = idx2(idx(i)-na1)
          endif
        enddo

        call resort_ev(idxq1, na)

      else if (na1>2) then

        ! Solve secular equation

        z(1:na1) = 1
#ifdef WITH_OPENMP
        z_p(1:na1,:) = 1
#endif
        dbase(1:na1) = 0
        ddiff(1:na1) = 0

        info = 0
#ifdef WITH_OPENMP

#ifdef HAVE_DETAILED_TIMINGS
        call timer%start("OpenMP parallel")
#endif

!$OMP PARALLEL PRIVATE(i,my_thread,delta,s,info,j)
        my_thread = omp_get_thread_num()
!$OMP DO
#endif
        DO i = my_proc+1, na1, n_procs ! work distributed over all processors

          call DLAED4(na1, i, d1, z1, delta, rho, s, info) ! s is not used!

          if (info/=0) then
            ! If DLAED4 fails (may happen especially for LAPACK versions before 3.2)
            ! use the more stable bisection algorithm in solve_secular_equation
            ! print *,'ERROR DLAED4 n=',na1,'i=',i,' Using Bisection'
            call solve_secular_equation(na1, i, d1, z1, delta, rho, s)
          endif

          ! Compute updated z

#ifdef WITH_OPENMP
          do j=1,na1
            if (i/=j)  z_p(j,my_thread) = z_p(j,my_thread)*( delta(j) / (d1(j)-d1(i)) )
          enddo
          z_p(i,my_thread) = z_p(i,my_thread)*delta(i)
#else
          do j=1,na1
            if (i/=j)  z(j) = z(j)*( delta(j) / (d1(j)-d1(i)) )
          enddo
          z(i) = z(i)*delta(i)
#endif
          ! store dbase/ddiff

          if (i<na1) then
            if (abs(delta(i+1)) < abs(delta(i))) then
              dbase(i) = d1(i+1)
              ddiff(i) = delta(i+1)
            else
              dbase(i) = d1(i)
              ddiff(i) = delta(i)
            endif
          else
            dbase(i) = d1(i)
            ddiff(i) = delta(i)
          endif
        enddo
#ifdef WITH_OPENMP
!$OMP END PARALLEL

#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("OpenMP parallel")
#endif

        do i = 0, max_threads-1
          z(1:na1) = z(1:na1)*z_p(1:na1,i)
        enddo
#endif

        call global_product(z, na1)
        z(1:na1) = SIGN( SQRT( -z(1:na1) ), z1(1:na1) )

        call global_gather(dbase, na1)
        call global_gather(ddiff, na1)
        d(1:na1) = dbase(1:na1) - ddiff(1:na1)

        ! Calculate scale factors for eigenvectors

        ev_scale(:) = 0.

#ifdef WITH_OPENMP

#ifdef HAVE_DETAILED_TIMINGS
        call timer%start("OpenMP parallel")
#endif

!$OMP PARALLEL DO PRIVATE(i) SHARED(na1, my_proc, n_procs,  &
!$OMP d1,dbase, ddiff, z, ev_scale) &
!$OMP DEFAULT(NONE)

#endif
        DO i = my_proc+1, na1, n_procs ! work distributed over all processors

          ! tmp(1:na1) = z(1:na1) / delta(1:na1,i)  ! original code
          ! tmp(1:na1) = z(1:na1) / (d1(1:na1)-d(i))! bad results

          ! All we want to calculate is tmp = (d1(1:na1)-dbase(i))+ddiff(i)
          ! in exactly this order, but we want to prevent compiler optimization
!         ev_scale_val = ev_scale(i)
          call add_tmp(d1, dbase, ddiff, z, ev_scale(i), na1,i)
!         ev_scale(i) = ev_scale_val
        enddo
#ifdef WITH_OPENMP
!$OMP END PARALLEL DO

#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("OpenMP parallel")
#endif

#endif

        call global_gather(ev_scale, na1)

        ! Add the deflated eigenvalues
        d(na1+1:na) = d2(1:na2)

        ! Calculate arrangement of all eigenvalues  in output

        call DLAMRG( na1, na-na1, d, 1, 1, idx )

        ! Rearrange eigenvalues

        tmp = d
        do i=1,na
          d(i) = tmp(idx(i))
        enddo
        call check_monotony(na,d,'Output', wantDebug, success)
        if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
          call timer%stop("merge_systems")
#endif
          return
        endif
        ! Eigenvector calculations


        ! Calculate the number of columns in the new local matrix Q
        ! which are updated from non-deflated/deflated eigenvectors.
        ! idxq1/2 stores the global column numbers.

        nqcols1 = 0 ! number of non-deflated eigenvectors
        nqcols2 = 0 ! number of deflated eigenvectors
        DO i = 1, na
          if (p_col_out(i)==my_pcol) then
            if (idx(i)<=na1) then
              nqcols1 = nqcols1+1
              idxq1(nqcols1) = i
            else
              nqcols2 = nqcols2+1
              idxq2(nqcols2) = i
            endif
          endif
        enddo

        allocate(ev(max_local_cols,MIN(max_strip,MAX(1,nqcols1))), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"merge_systems: error when allocating ev "//errorMessage
          stop
        endif

        allocate(qtmp1(MAX(1,l_rows),max_local_cols), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"merge_systems: error when allocating qtmp1 "//errorMessage
          stop
        endif

        allocate(qtmp2(MAX(1,l_rows),MIN(max_strip,MAX(1,nqcols1))), stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"merge_systems: error when allocating qtmp2 "//errorMessage
          stop
        endif

        ! Gather nonzero upper/lower components of old matrix Q
        ! which are needed for multiplication with new eigenvectors

        qtmp1 = 0 ! May contain empty (unset) parts
        qtmp2 = 0 ! Not really needed

        nnzu = 0
        nnzl = 0
        do i = 1, na1
          l_idx = l_col(idx1(i))
          if (p_col(idx1(i))==my_pcol) then
            if (coltyp(idx1(i))==1 .or. coltyp(idx1(i))==2) then
              nnzu = nnzu+1
              qtmp1(1:l_rnm,nnzu) = q(l_rqs:l_rqm,l_idx)
            endif
            if (coltyp(idx1(i))==3 .or. coltyp(idx1(i))==2) then
              nnzl = nnzl+1
              qtmp1(l_rnm+1:l_rows,nnzl) = q(l_rqm+1:l_rqe,l_idx)
            endif
          endif
        enddo

        ! Gather deflated eigenvalues behind nonzero components

        ndef = max(nnzu,nnzl)
        do i = 1, na2
          l_idx = l_col(idx2(i))
          if (p_col(idx2(i))==my_pcol) then
            ndef = ndef+1
            qtmp1(1:l_rows,ndef) = q(l_rqs:l_rqe,l_idx)
          endif
        enddo

        l_cols_qreorg = ndef ! Number of columns in reorganized matrix

        ! Set (output) Q to 0, it will sum up new Q

        DO i = 1, na
          if(p_col_out(i)==my_pcol) q(l_rqs:l_rqe,l_col_out(i)) = 0
        enddo

        np_rem = my_pcol

        do np = 1, npc_n

          ! Do a ring send of qtmp1

          if (np>1) then

            if (np_rem==npc_0) then
              np_rem = npc_0+npc_n-1
            else
              np_rem = np_rem-1
            endif
#ifdef WITH_MPI
            call MPI_Sendrecv_replace(qtmp1, l_rows*max_local_cols, MPI_REAL8, &
                                        np_next, 1111, np_prev, 1111, &
                                        mpi_comm_cols, mpi_status, mpierr)
#endif
          endif

          ! Gather the parts in d1 and z which are fitting to qtmp1.
          ! This also delivers nnzu/nnzl for proc np_rem

          nnzu = 0
          nnzl = 0
          do i=1,na1
            if (p_col(idx1(i))==np_rem) then
              if (coltyp(idx1(i))==1 .or. coltyp(idx1(i))==2) then
                nnzu = nnzu+1
                d1u(nnzu) = d1(i)
                zu (nnzu) = z (i)
              endif
              if (coltyp(idx1(i))==3 .or. coltyp(idx1(i))==2) then
                nnzl = nnzl+1
                d1l(nnzl) = d1(i)
                zl (nnzl) = z (i)
              endif
            endif
          enddo

          ! Set the deflated eigenvectors in Q (comming from proc np_rem)

          ndef = MAX(nnzu,nnzl) ! Remote counter in input matrix
          do i = 1, na
            j = idx(i)
            if (j>na1) then
              if (p_col(idx2(j-na1))==np_rem) then
                ndef = ndef+1
                if (p_col_out(i)==my_pcol) &
                      q(l_rqs:l_rqe,l_col_out(i)) = qtmp1(1:l_rows,ndef)
              endif
            endif
          enddo

          do ns = 0, nqcols1-1, max_strip ! strimining loop

            ncnt = MIN(max_strip,nqcols1-ns) ! number of columns in this strip

            ! Get partial result from (output) Q

            do i = 1, ncnt
              qtmp2(1:l_rows,i) = q(l_rqs:l_rqe,l_col_out(idxq1(i+ns)))
            enddo

            ! Compute eigenvectors of the rank-1 modified matrix.
            ! Parts for multiplying with upper half of Q:

            do i = 1, ncnt
              j = idx(idxq1(i+ns))
              ! Calculate the j-th eigenvector of the deflated system
              ! See above why we are doing it this way!
              tmp(1:nnzu) = d1u(1:nnzu)-dbase(j)
              call v_add_s(tmp,nnzu,ddiff(j))
              ev(1:nnzu,i) = zu(1:nnzu) / tmp(1:nnzu) * ev_scale(j)
            enddo

            ! Multiply old Q with eigenvectors (upper half)

            if (l_rnm>0 .and. ncnt>0 .and. nnzu>0) &
                call dgemm('N','N',l_rnm,ncnt,nnzu,1.d0,qtmp1,ubound(qtmp1,dim=1),ev,ubound(ev,dim=1), &
                           1.d0,qtmp2(1,1),ubound(qtmp2,dim=1))

            ! Compute eigenvectors of the rank-1 modified matrix.
            ! Parts for multiplying with lower half of Q:

            do i = 1, ncnt
              j = idx(idxq1(i+ns))
              ! Calculate the j-th eigenvector of the deflated system
              ! See above why we are doing it this way!
              tmp(1:nnzl) = d1l(1:nnzl)-dbase(j)
              call v_add_s(tmp,nnzl,ddiff(j))
              ev(1:nnzl,i) = zl(1:nnzl) / tmp(1:nnzl) * ev_scale(j)
            enddo

            ! Multiply old Q with eigenvectors (lower half)

             if (l_rows-l_rnm>0 .and. ncnt>0 .and. nnzl>0) &
                call dgemm('N','N',l_rows-l_rnm,ncnt,nnzl,1.d0,qtmp1(l_rnm+1,1),ubound(qtmp1,dim=1),ev,ubound(ev,dim=1), &
                           1.d0,qtmp2(l_rnm+1,1),ubound(qtmp2,dim=1))

             ! Put partial result into (output) Q

             do i = 1, ncnt
               q(l_rqs:l_rqe,l_col_out(idxq1(i+ns))) = qtmp2(1:l_rows,i)
             enddo

           enddo
        enddo

        deallocate(ev, qtmp1, qtmp2, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"merge_systems: error when deallocating ev "//errorMessage
          stop
        endif
      endif

#ifdef WITH_OPENMP
      deallocate(z_p, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"merge_systems: error when deallocating z_p "//errorMessage
        stop
      endif
#endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("merge_systems")
#endif


      return

      contains
        subroutine add_tmp(d1, dbase, ddiff, z, ev_scale_value, na1,i)
          use precision
          implicit none

          integer(kind=ik), intent(in) :: na1, i

          real(kind=rk), intent(in)    :: d1(:), dbase(:), ddiff(:), z(:)
          real(kind=rk), intent(inout) :: ev_scale_value
          real(kind=rk)                :: tmp(1:na1)

               ! tmp(1:na1) = z(1:na1) / delta(1:na1,i)  ! original code
               ! tmp(1:na1) = z(1:na1) / (d1(1:na1)-d(i))! bad results

               ! All we want to calculate is tmp = (d1(1:na1)-dbase(i))+ddiff(i)
               ! in exactly this order, but we want to prevent compiler optimization

          tmp(1:na1) = d1(1:na1) -dbase(i)
          call v_add_s(tmp(1:na1),na1,ddiff(i))

          tmp(1:na1) = z(1:na1) / tmp(1:na1)

          ev_scale_value = 1.0/sqrt(dot_product(tmp(1:na1),tmp(1:na1)))

        end subroutine add_tmp

        subroutine resort_ev(idx_ev, nLength)
          use precision
          implicit none

          integer(kind=ik), intent(in) :: nLength
          integer(kind=ik)             :: idx_ev(nLength)
          integer(kind=ik)             :: i, nc, pc1, pc2, lc1, lc2, l_cols_out

          real(kind=rk), allocatable   :: qtmp(:,:)
          integer(kind=ik)             :: istat
          character(200)               :: errorMessage

          if (l_rows==0) return ! My processor column has no work to do

          ! Resorts eigenvectors so that q_new(:,i) = q_old(:,idx_ev(i))

          l_cols_out = COUNT(p_col_out(1:na)==my_pcol)
          allocate(qtmp(l_rows,l_cols_out), stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"resort_ev: error when allocating qtmp "//errorMessage
            stop
          endif

          nc = 0

          do i=1,na

            pc1 = p_col(idx_ev(i))
            lc1 = l_col(idx_ev(i))
            pc2 = p_col_out(i)

            if (pc2<0) cycle ! This column is not needed in output

            if (pc2==my_pcol) nc = nc+1 ! Counter for output columns

            if (pc1==my_pcol) then
              if (pc2==my_pcol) then
                ! send and recieve column are local
                qtmp(1:l_rows,nc) = q(l_rqs:l_rqe,lc1)
              else
#ifdef WITH_MPI
                call mpi_send(q(l_rqs,lc1),l_rows,MPI_REAL8,pc2,mod(i,4096),mpi_comm_cols,mpierr)
#endif
              endif
            else if (pc2==my_pcol) then
#ifdef WITH_MPI
              call mpi_recv(qtmp(1,nc),l_rows,MPI_REAL8,pc1,mod(i,4096),mpi_comm_cols,mpi_status,mpierr)
#else
              qtmp(1:l_rows,nc) = q(l_rqs:l_rqe,nc)
#endif
            endif
          enddo

          ! Insert qtmp into (output) q

          nc = 0

          do i=1,na

            pc2 = p_col_out(i)
            lc2 = l_col_out(i)

            if (pc2==my_pcol) then
              nc = nc+1
              q(l_rqs:l_rqe,lc2) = qtmp(1:l_rows,nc)
            endif
          enddo

          deallocate(qtmp, stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"resort_ev: error when deallocating qtmp "//errorMessage
            stop
          endif
        end subroutine resort_ev

        subroutine transform_columns(col1, col2)
          use precision
          implicit none

          integer(kind=ik) :: col1, col2
          integer(kind=ik) :: pc1, pc2, lc1, lc2

          if (l_rows==0) return ! My processor column has no work to do

          pc1 = p_col(col1)
          lc1 = l_col(col1)
          pc2 = p_col(col2)
          lc2 = l_col(col2)

          if (pc1==my_pcol) then
            if (pc2==my_pcol) then
              ! both columns are local
              tmp(1:l_rows)      = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + q(l_rqs:l_rqe,lc2)*qtrans(2,1)
              q(l_rqs:l_rqe,lc2) = q(l_rqs:l_rqe,lc1)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
              q(l_rqs:l_rqe,lc1) = tmp(1:l_rows)
            else
#ifdef WITH_MPI
              call mpi_sendrecv(q(l_rqs,lc1),l_rows,MPI_REAL8,pc2,1, &
                                  tmp,l_rows,MPI_REAL8,pc2,1, &
                                  mpi_comm_cols,mpi_status,mpierr)
#else
              tmp(1:l_rows) = q(l_rqs:l_rqe,lc1)
#endif
              q(l_rqs:l_rqe,lc1) = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + tmp(1:l_rows)*qtrans(2,1)
            endif
          else if (pc2==my_pcol) then
#ifdef WITH_MPI
            call mpi_sendrecv(q(l_rqs,lc2),l_rows,MPI_REAL8,pc1,1, &
                               tmp,l_rows,MPI_REAL8,pc1,1, &
                               mpi_comm_cols,mpi_status,mpierr)
#else
            tmp(1:l_rows) = q(l_rqs:l_rqe,lc2)
#endif
            q(l_rqs:l_rqe,lc2) = tmp(1:l_rows)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
          endif

        end subroutine transform_columns

        subroutine global_gather(z, n)

          ! This routine sums up z over all processors.
          ! It should only be used for gathering distributed results,
          ! i.e. z(i) should be nonzero on exactly 1 processor column,
          ! otherways the results may be numerically different on different columns
          use precision
          implicit none

          integer(kind=ik) :: n
          real(kind=rk)    :: z(n)
          real(kind=rk)    :: tmp(n)

          if (npc_n==1 .and. np_rows==1) return ! nothing to do

          ! Do an mpi_allreduce over processor rows
#ifdef WITH_MPI
          call mpi_allreduce(z, tmp, n, MPI_REAL8, MPI_SUM, mpi_comm_rows, mpierr)
#else
          tmp = z
#endif
          ! If only 1 processor column, we are done
          if (npc_n==1) then
            z(:) = tmp(:)
            return
          endif

          ! If all processor columns are involved, we can use mpi_allreduce
          if (npc_n==np_cols) then
#ifdef WITH_MPI
            call mpi_allreduce(tmp, z, n, MPI_REAL8, MPI_SUM, mpi_comm_cols, mpierr)
#else
            tmp = z
#endif
            return
          endif

          ! Do a ring send over processor columns
          z(:) = 0
          do np = 1, npc_n
            z(:) = z(:) + tmp(:)
#ifdef WITH_MPI
            call MPI_Sendrecv_replace(z, n, MPI_REAL8, np_next, 1111, np_prev, 1111, &
                                       mpi_comm_cols, mpi_status, mpierr)
#endif
          enddo

        end subroutine global_gather

        subroutine global_product(z, n)

          ! This routine calculates the global product of z.
          use precision
          implicit none

          integer(kind=ik) :: n
          real(kind=rk)    :: z(n)

          real(kind=rk)    :: tmp(n)

          if (npc_n==1 .and. np_rows==1) return ! nothing to do

          ! Do an mpi_allreduce over processor rows
#ifdef WITH_MPI
          call mpi_allreduce(z, tmp, n, MPI_REAL8, MPI_PROD, mpi_comm_rows, mpierr)
#else
          tmp = z
#endif
          ! If only 1 processor column, we are done
          if (npc_n==1) then
            z(:) = tmp(:)
            return
          endif

          ! If all processor columns are involved, we can use mpi_allreduce
          if (npc_n==np_cols) then
#ifdef WITH_MPI
            call mpi_allreduce(tmp, z, n, MPI_REAL8, MPI_PROD, mpi_comm_cols, mpierr)
#else
            z = tmp
#endif
            return
          endif

          ! We send all vectors to the first proc, do the product there
          ! and redistribute the result.

          if (my_pcol == npc_0) then
            z(1:n) = tmp(1:n)
            do np = npc_0+1, npc_0+npc_n-1
#ifdef WITH_MPI
              call mpi_recv(tmp,n,MPI_REAL8,np,1111,mpi_comm_cols,mpi_status,mpierr)
#else
              tmp(1:n) = z(1:n)
#endif
              z(1:n) = z(1:n)*tmp(1:n)
            enddo
            do np = npc_0+1, npc_0+npc_n-1
#ifdef WITH_MPI
              call mpi_send(z,n,MPI_REAL8,np,1111,mpi_comm_cols,mpierr)
#endif
            enddo
          else
#ifdef WITH_MPI
            call mpi_send(tmp,n,MPI_REAL8,npc_0,1111,mpi_comm_cols,mpierr)
            call mpi_recv(z  ,n,MPI_REAL8,npc_0,1111,mpi_comm_cols,mpi_status,mpierr)
#else
            z(1:n) = tmp(1:n)
#endif
          endif

        end subroutine global_product

        subroutine check_monotony(n,d,text, wantDebug, success)

        ! This is a test routine for checking if the eigenvalues are monotonically increasing.
        ! It is for debug purposes only, an error should never be triggered!
          use precision
          implicit none

          integer(kind=ik)              :: n
          real(kind=rk)                 :: d(n)
          character*(*)                 :: text

          integer(kind=ik)              :: i
          logical, intent(in)           :: wantDebug
          logical, intent(out)          :: success

          success = .true.
          do i=1,n-1
            if (d(i+1)<d(i)) then
              if (wantDebug) write(error_unit,'(a,a,i8,2g25.17)') 'ELPA1_check_monotony: Monotony error on ',text,i,d(i),d(i+1)
              success = .false.
              return
            endif
          enddo
        end subroutine check_monotony

    end subroutine merge_systems

    subroutine v_add_s(v,n,s)
      use precision
      implicit none
      integer(kind=ik) :: n
      real(kind=rk)    :: v(n),s

      v(:) = v(:) + s
    end subroutine v_add_s

    subroutine distribute_global_column(g_col, l_col, noff, nlen, my_prow, np_rows, nblk)
      use precision
      implicit none

      real(kind=rk)     :: g_col(nlen), l_col(*) ! chnage this to proper 2d 1d matching
      integer(kind=ik)  :: noff, nlen, my_prow, np_rows, nblk

      integer(kind=ik)  :: nbs, nbe, jb, g_off, l_off, js, je

      nbs = noff/(nblk*np_rows)
      nbe = (noff+nlen-1)/(nblk*np_rows)

      do jb = nbs, nbe

        g_off = jb*nblk*np_rows + nblk*my_prow
        l_off = jb*nblk

        js = MAX(noff+1-g_off,1)
        je = MIN(noff+nlen-g_off,nblk)

        if (je<js) cycle

        l_col(l_off+js:l_off+je) = g_col(g_off+js-noff:g_off+je-noff)

      enddo

    end subroutine distribute_global_column

    subroutine solve_secular_equation(n, i, d, z, delta, rho, dlam)

    !-------------------------------------------------------------------------------
    ! This routine solves the secular equation of a symmetric rank 1 modified
    ! diagonal matrix:
    !
    !    1. + rho*SUM(z(:)**2/(d(:)-x)) = 0
    !
    ! It does the same as the LAPACK routine DLAED4 but it uses a bisection technique
    ! which is more robust (it always yields a solution) but also slower
    ! than the algorithm used in DLAED4.
    !
    ! The same restictions than in DLAED4 hold, namely:
    !
    !   rho > 0   and   d(i+1) > d(i)
    !
    ! but this routine will not terminate with error if these are not satisfied
    ! (it will normally converge to a pole in this case).
    !
    ! The output in DELTA(j) is always (D(j) - lambda_I), even for the cases
    ! N=1 and N=2 which is not compatible with DLAED4.
    ! Thus this routine shouldn't be used for these cases as a simple replacement
    ! of DLAED4.
    !
    ! The arguments are the same as in DLAED4 (with the exception of the INFO argument):
    !
    !
    !  N      (input) INTEGER
    !         The length of all arrays.
    !
    !  I      (input) INTEGER
    !         The index of the eigenvalue to be computed.  1 <= I <= N.
    !
    !  D      (input) DOUBLE PRECISION array, dimension (N)
    !         The original eigenvalues.  It is assumed that they are in
    !         order, D(I) < D(J)  for I < J.
    !
    !  Z      (input) DOUBLE PRECISION array, dimension (N)
    !         The components of the updating vector.
    !
    !  DELTA  (output) DOUBLE PRECISION array, dimension (N)
    !         DELTA contains (D(j) - lambda_I) in its  j-th component.
    !         See remark above about DLAED4 compatibility!
    !
    !  RHO    (input) DOUBLE PRECISION
    !         The scalar in the symmetric updating formula.
    !
    !  DLAM   (output) DOUBLE PRECISION
    !         The computed lambda_I, the I-th updated eigenvalue.
    !-------------------------------------------------------------------------------

#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik)   :: n, i
      real(kind=rk)      :: d(n), z(n), delta(n), rho, dlam

      integer(kind=ik)   :: iter
      real(kind=rk)      :: a, b, x, y, dshift

      ! In order to obtain sufficient numerical accuracy we have to shift the problem
      ! either by d(i) or d(i+1), whichever is closer to the solution

      ! Upper and lower bound of the shifted solution interval are a and b

#ifdef HAVE_DETAILED_TIMINGS
      call  timer%start("solve_secular_equation")
#endif
      if (i==n) then

       ! Special case: Last eigenvalue
       ! We shift always by d(n), lower bound is d(n),
       ! upper bound is determined by a guess:

       dshift = d(n)
       delta(:) = d(:) - dshift

       a = 0. ! delta(n)
       b = rho*SUM(z(:)**2) + 1. ! rho*SUM(z(:)**2) is the lower bound for the guess

      else

        ! Other eigenvalues: lower bound is d(i), upper bound is d(i+1)
        ! We check the sign of the function in the midpoint of the interval
        ! in order to determine if eigenvalue is more close to d(i) or d(i+1)

        x = 0.5*(d(i)+d(i+1))
        y = 1. + rho*SUM(z(:)**2/(d(:)-x))

        if (y>0) then
          ! solution is next to d(i)
          dshift = d(i)
        else
          ! solution is next to d(i+1)
          dshift = d(i+1)
        endif

        delta(:) = d(:) - dshift
        a = delta(i)
        b = delta(i+1)

      endif

      ! Bisection:

      do iter=1,200

        ! Interval subdivision

        x = 0.5*(a+b)

        if (x==a .or. x==b) exit   ! No further interval subdivisions possible
        if (abs(x) < 1.d-200) exit ! x next to pole

        ! evaluate value at x

        y = 1. + rho*SUM(z(:)**2/(delta(:)-x))

        if (y==0) then
          ! found exact solution
          exit
        elseif (y>0) then
          b = x
        else
          a = x
        endif

      enddo

      ! Solution:

      dlam = x + dshift
      delta(:) = delta(:) - x
#ifdef HAVE_DETAILED_TIMINGS
      call  timer%stop("solve_secular_equation")
#endif

    end subroutine solve_secular_equation

    !-------------------------------------------------------------------------------

    integer function local_index(idx, my_proc, num_procs, nblk, iflag)

    !-------------------------------------------------------------------------------
    !  local_index: returns the local index for a given global index
    !               If the global index has no local index on the
    !               processor my_proc behaviour is defined by iflag
    !
    !  Parameters
    !
    !  idx         Global index
    !
    !  my_proc     Processor row/column for which to calculate the local index
    !
    !  num_procs   Total number of processors along row/column
    !
    !  nblk        Blocksize
    !
    !  iflag       Controls the behaviour if idx is not on local processor
    !              iflag< 0 : Return last local index before that row/col
    !              iflag==0 : Return 0
    !              iflag> 0 : Return next local index after that row/col
    !-------------------------------------------------------------------------------
      use precision
      implicit none

      integer(kind=ik) :: idx, my_proc, num_procs, nblk, iflag

      integer(kind=ik) :: iblk

      iblk = (idx-1)/nblk  ! global block number, 0 based

      if (mod(iblk,num_procs) == my_proc) then

        ! block is local, always return local row/col number

        local_index = (iblk/num_procs)*nblk + mod(idx-1,nblk) + 1

      else

        ! non local block

        if (iflag == 0) then

          local_index = 0

        else

          local_index = (iblk/num_procs)*nblk

          if (mod(iblk,num_procs) > my_proc) local_index = local_index + nblk

          if (iflag>0) local_index = local_index + 1
        endif
      endif

    end function local_index
    integer function least_common_multiple(a, b)

      ! Returns the least common multiple of a and b
      ! There may be more efficient ways to do this, we use the most simple approach
      use precision
      implicit none
      integer(kind=ik), intent(in) :: a, b

      do least_common_multiple = a, a*(b-1), a
        if(mod(least_common_multiple,b)==0) exit
      enddo
      ! if the loop is left regularly, least_common_multiple = a*b

    end function

    subroutine hh_transform_real(alpha, xnorm_sq, xf, tau)
      ! Similar to LAPACK routine DLARFP, but uses ||x||**2 instead of x(:)
      ! and returns the factor xf by which x has to be scaled.
      ! It also hasn't the special handling for numbers < 1.d-300 or > 1.d150
      ! since this would be expensive for the parallel implementation.
      use precision
      implicit none
      real(kind=rk), intent(inout) :: alpha
      real(kind=rk), intent(in)    :: xnorm_sq
      real(kind=rk), intent(out)   :: xf, tau

      real(kind=rk)                :: BETA

      if ( XNORM_SQ==0. ) then

        if ( ALPHA>=0. ) then
          TAU = 0.
        else
          TAU = 2.
          ALPHA = -ALPHA
        endif
        XF = 0.

      else

        BETA = SIGN( SQRT( ALPHA**2 + XNORM_SQ ), ALPHA )
        ALPHA = ALPHA + BETA
        IF ( BETA<0 ) THEN
          BETA = -BETA
          TAU = -ALPHA / BETA
        ELSE
          ALPHA = XNORM_SQ / ALPHA
          TAU = ALPHA / BETA
          ALPHA = -ALPHA
       END IF
       XF = 1./ALPHA
       ALPHA = BETA
     endif

    end subroutine

    subroutine hh_transform_complex(alpha, xnorm_sq, xf, tau)

      ! Similar to LAPACK routine ZLARFP, but uses ||x||**2 instead of x(:)
      ! and returns the factor xf by which x has to be scaled.
      ! It also hasn't the special handling for numbers < 1.d-300 or > 1.d150
      ! since this would be expensive for the parallel implementation.
      use precision
      implicit none
      complex(kind=ck), intent(inout) :: alpha
      real(kind=rk), intent(in)       :: xnorm_sq
      complex(kind=ck), intent(out)   :: xf, tau

      real*8 ALPHR, ALPHI, BETA

      ALPHR = DBLE( ALPHA )
      ALPHI = DIMAG( ALPHA )

      if ( XNORM_SQ==0. .AND. ALPHI==0. ) then

        if ( ALPHR>=0. ) then
          TAU = 0.
        else
          TAU = 2.
          ALPHA = -ALPHA
        endif
        XF = 0.

      else

        BETA = SIGN( SQRT( ALPHR**2 + ALPHI**2 + XNORM_SQ ), ALPHR )
        ALPHA = ALPHA + BETA
        IF ( BETA<0 ) THEN
          BETA = -BETA
          TAU = -ALPHA / BETA
        ELSE
          ALPHR = ALPHI * (ALPHI/DBLE( ALPHA ))
          ALPHR = ALPHR + XNORM_SQ/DBLE( ALPHA )
          TAU = DCMPLX( ALPHR/BETA, -ALPHI/BETA )
          ALPHA = DCMPLX( -ALPHR, ALPHI )
        END IF
        XF = 1./ALPHA
        ALPHA = BETA
      endif

    end subroutine

end module ELPA1_compute
