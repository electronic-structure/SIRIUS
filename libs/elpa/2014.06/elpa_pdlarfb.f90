!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium, 
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG), 
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
!
!    More information can be found here:
!    http://elpa.rzg.mpg.de/
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
module elpa_pdlarfb

    use elpa1
    use qr_utils_mod
 
    implicit none

    PRIVATE

    public :: qr_pdlarfb_1dcomm
    public :: qr_pdlarft_pdlarfb_1dcomm
    public :: qr_pdlarft_set_merge_1dcomm
    public :: qr_pdlarft_tree_merge_1dcomm
    public :: qr_pdlarfl_1dcomm
    public :: qr_pdlarfl2_tmatrix_1dcomm
    public :: qr_tmerge_pdlarfb_1dcomm
    
    include 'mpif.h'

contains

subroutine qr_pdlarfb_1dcomm(m,mb,n,k,a,lda,v,ldv,tau,t,ldt,baseidx,idx,rev,mpicomm,work,lwork)
    
    use qr_utils_mod

    implicit none
 
    ! input variables (local)
    integer lda,ldv,ldt,lwork
    double precision a(lda,*),v(ldv,*),tau(*),t(ldt,*),work(k,*)

    ! input variables (global)
    integer m,mb,n,k,baseidx,idx,rev,mpicomm
 
    ! output variables (global)

    ! derived input variables from QR_PQRPARAM

    ! local scalars
    integer localsize,offset,baseoffset
    integer mpirank,mpiprocs,mpierr

        if (idx .le. 1) return

    if (n .le. 0) return ! nothing to do

    if (k .eq. 1) then
        call qr_pdlarfl_1dcomm(v,1,baseidx,a,lda,tau(1), &
                                work,lwork,m,n,idx,mb,rev,mpicomm)
        return
    else if (k .eq. 2) then
        call qr_pdlarfl2_tmatrix_1dcomm(v,ldv,baseidx,a,lda,t,ldt, &
                                 work,lwork,m,n,idx,mb,rev,mpicomm)
        return
    end if

    if (lwork .eq. -1) then
        work(1,1) = DBLE(2*k*n)
        return
    end if
 
    !print *,'updating trailing matrix with k=',k

    call MPI_Comm_rank(mpicomm,mpirank,mpierr)
    call MPI_Comm_size(mpicomm,mpiprocs,mpierr)

    ! use baseidx as idx here, otherwise the upper triangle part will be lost
    ! during the calculation, especially in the reversed case
    call local_size_offset_1d(m,mb,baseidx,baseidx,rev,mpirank,mpiprocs, &
                                localsize,baseoffset,offset)

    ! Z' = Y' * A
    if (localsize .gt. 0) then
        call dgemm("Trans","Notrans",k,n,localsize,1.0d0,v(baseoffset,1),ldv,a(offset,1),lda,0.0d0,work(1,1),k)
    else
        work(1:k,1:n) = 0.0d0
    end if

    ! data exchange
    call mpi_allreduce(work(1,1),work(1,n+1),k*n,mpi_real8,mpi_sum,mpicomm,mpierr)
    
    call qr_pdlarfb_kernel_local(localsize,n,k,a(offset,1),lda,v(baseoffset,1),ldv,t,ldt,work(1,n+1),k)
end subroutine qr_pdlarfb_1dcomm 

! generalized pdlarfl2 version
! TODO: include T merge here (seperate by "old" and "new" index)
subroutine qr_pdlarft_pdlarfb_1dcomm(m,mb,n,oldk,k,v,ldv,tau,t,ldt,a,lda,baseidx,rev,mpicomm,work,lwork)
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer ldv,ldt,lda,lwork
    double precision v(ldv,*),tau(*),t(ldt,*),work(k,*),a(lda,*)

    ! input variables (global)
    integer m,mb,n,k,oldk,baseidx,rev,mpicomm
 
    ! output variables (global)

    ! derived input variables from QR_PQRPARAM

    ! local scalars
    integer localsize,offset,baseoffset
    integer mpirank,mpiprocs,mpierr
    integer icol

    integer sendoffset,recvoffset,sendsize

    sendoffset = 1
    sendsize = k*(k+n+oldk)
    recvoffset = sendoffset+(k+n+oldk)

    if (lwork .eq. -1) then
        work(1,1) = DBLE(2*(k*k+k*n+oldk))
        return
    end if

    call MPI_Comm_rank(mpicomm,mpirank,mpierr)
    call MPI_Comm_size(mpicomm,mpiprocs,mpierr)

    call local_size_offset_1d(m,mb,baseidx,baseidx,rev,mpirank,mpiprocs, &
                                localsize,baseoffset,offset)

    if (localsize .gt. 0) then
            ! calculate inner product of householdervectors
            call dsyrk("Upper","Trans",k,localsize,1.0d0,v(baseoffset,1),ldv,0.0d0,work(1,1),k)

            ! calculate matrix matrix product of householder vectors and target matrix 
            ! Z' = Y' * A
            call dgemm("Trans","Notrans",k,n,localsize,1.0d0,v(baseoffset,1),ldv,a(offset,1),lda,0.0d0,work(1,k+1),k)

            ! TODO: reserved for T merge parts
            work(1:k,n+k+1:n+k+oldk) = 0.0d0
    else
        work(1:k,1:(n+k+oldk)) = 0.0d0
    end if

    ! exchange data
    call mpi_allreduce(work(1,sendoffset),work(1,recvoffset),sendsize,mpi_real8,mpi_sum,mpicomm,mpierr)

        ! generate T matrix (pdlarft)
        t(1:k,1:k) = 0.0d0 ! DEBUG: clear buffer first

        ! T1 = tau1
        ! | tauk  Tk-1' * (-tauk * Y(:,1,k+1:n) * Y(:,k))' |
        ! | 0           Tk-1                           |
        t(k,k) = tau(k)
        do icol=k-1,1,-1
            t(icol,icol+1:k) = -tau(icol)*work(icol,recvoffset+icol:recvoffset+k-1)
            call dtrmv("Upper","Trans","Nonunit",k-icol,t(icol+1,icol+1),ldt,t(icol,icol+1),ldt)
            t(icol,icol) = tau(icol)
        end do

        ! TODO: elmroth and gustavson
 
        ! update matrix (pdlarfb)
        ! Z' = T * Z'
        call dtrmm("Left","Upper","Notrans","Nonunit",k,n,1.0d0,t,ldt,work(1,recvoffset+k),k)

        ! A = A - Y * V'
        call dgemm("Notrans","Notrans",localsize,n,k,-1.0d0,v(baseoffset,1),ldv,work(1,recvoffset+k),k,1.0d0,a(offset,1),lda)

end subroutine qr_pdlarft_pdlarfb_1dcomm

subroutine qr_pdlarft_set_merge_1dcomm(m,mb,n,blocksize,v,ldv,t,ldt,baseidx,rev,mpicomm,work,lwork)
    use qr_utils_mod

    implicit none
 
    ! input variables (local)
    integer ldv,ldt,lwork
    double precision v(ldv,*),t(ldt,*),work(n,*)

    ! input variables (global)
    integer m,mb,n,blocksize,baseidx,rev,mpicomm
 
    ! output variables (global)

    ! derived input variables from QR_PQRPARAM

    ! local scalars
    integer localsize,offset,baseoffset
    integer mpirank,mpiprocs,mpierr

    if (lwork .eq. -1) then
        work(1,1) = DBLE(2*n*n)
        return
    end if
 
    call MPI_Comm_rank(mpicomm,mpirank,mpierr)
    call MPI_Comm_size(mpicomm,mpiprocs,mpierr)

    call local_size_offset_1d(m,mb,baseidx,baseidx,rev,mpirank,mpiprocs, &
                                localsize,baseoffset,offset)

    if (localsize .gt. 0) then
        call dsyrk("Upper","Trans",n,localsize,1.0d0,v(baseoffset,1),ldv,0.0d0,work(1,1),n)
    else
        work(1:n,1:n) = 0.0d0
    end if
 
    call mpi_allreduce(work(1,1),work(1,n+1),n*n,mpi_real8,mpi_sum,mpicomm,mpierr)

        ! skip Y4'*Y4 part
        offset = mod(n,blocksize)
        if (offset .eq. 0) offset=blocksize
        call qr_tmerge_set_kernel(n,blocksize,t,ldt,work(1,n+1+offset),n)

end subroutine qr_pdlarft_set_merge_1dcomm

subroutine qr_pdlarft_tree_merge_1dcomm(m,mb,n,blocksize,treeorder,v,ldv,t,ldt,baseidx,rev,mpicomm,work,lwork)
    use qr_utils_mod

    implicit none
 
    ! input variables (local)
    integer ldv,ldt,lwork
    double precision v(ldv,*),t(ldt,*),work(n,*)

    ! input variables (global)
    integer m,mb,n,blocksize,treeorder,baseidx,rev,mpicomm
 
    ! output variables (global)

    ! derived input variables from QR_PQRPARAM

    ! local scalars
    integer localsize,offset,baseoffset
    integer mpirank,mpiprocs,mpierr

    if (lwork .eq. -1) then
        work(1,1) = DBLE(2*n*n)
        return
    end if

    if (n .le. blocksize) return ! nothing to do
 
    call MPI_Comm_rank(mpicomm,mpirank,mpierr)
    call MPI_Comm_size(mpicomm,mpiprocs,mpierr)

    call local_size_offset_1d(m,mb,baseidx,baseidx,rev,mpirank,mpiprocs, &
                                localsize,baseoffset,offset)

    if (localsize .gt. 0) then
        call dsyrk("Upper","Trans",n,localsize,1.0d0,v(baseoffset,1),ldv,0.0d0,work(1,1),n)
    else
        work(1:n,1:n) = 0.0d0
    end if
 
    call mpi_allreduce(work(1,1),work(1,n+1),n*n,mpi_real8,mpi_sum,mpicomm,mpierr)

        ! skip Y4'*Y4 part
        offset = mod(n,blocksize)
        if (offset .eq. 0) offset=blocksize
        call qr_tmerge_tree_kernel(n,blocksize,treeorder,t,ldt,work(1,n+1+offset),n)

end subroutine qr_pdlarft_tree_merge_1dcomm

! apply householder vector to the left 
! - assume unitary matrix
! - assume right positions for v
subroutine qr_pdlarfl_1dcomm(v,incv,baseidx,a,lda,tau,work,lwork,m,n,idx,mb,rev,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none
 
    ! input variables (local)
    integer incv,lda,lwork,baseidx
    double precision v(*),a(lda,*),work(*)

    ! input variables (global)
    integer m,n,mb,rev,idx,mpicomm
    double precision tau
 
    ! output variables (global)
 
    ! local scalars
    integer mpierr,mpirank,mpiprocs
    integer sendsize,recvsize,icol
    integer local_size,local_offset
    integer v_local_offset

    ! external functions
    double precision ddot
    external dgemv,dger,ddot
  
    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    sendsize = n
    recvsize = sendsize

    if (lwork .eq. -1) then
        work(1) = DBLE(sendsize + recvsize)
        return
    end if
 
    if (n .le. 0) return

        if (idx .le. 1) return
 
    call local_size_offset_1d(m,mb,baseidx,idx,rev,mpirank,mpiprocs, &
                              local_size,v_local_offset,local_offset)
 
    !print *,'hl ref',local_size,n

    v_local_offset = v_local_offset * incv

    if (local_size > 0) then
        
        do icol=1,n
            work(icol) = dot_product(v(v_local_offset:v_local_offset+local_size-1),a(local_offset:local_offset+local_size-1,icol))

        end do
    else
        work(1:n) = 0.0d0
    end if
 
    call mpi_allreduce(work, work(sendsize+1), sendsize, mpi_real8, mpi_sum, mpicomm, mpierr)

    if (local_size > 0) then

         do icol=1,n
               a(local_offset:local_offset+local_size-1,icol) = a(local_offset:local_offset+local_size-1,icol) &
                                                                - tau*work(sendsize+icol)*v(v_local_offset:v_local_offset+ &
                                                                           local_size-1)
         enddo
    end if

end subroutine qr_pdlarfl_1dcomm

subroutine qr_pdlarfl2_tmatrix_1dcomm(v,ldv,baseidx,a,lda,t,ldt,work,lwork,m,n,idx,mb,rev,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none
 
    ! input variables (local)
    integer ldv,lda,lwork,baseidx,ldt
    double precision v(ldv,*),a(lda,*),work(*),t(ldt,*)

    ! input variables (global)
    integer m,n,mb,rev,idx,mpicomm
 
    ! output variables (global)
 
    ! local scalars
    integer mpierr,mpirank,mpiprocs,mpirank_top1,mpirank_top2
    integer dgemv1_offset,dgemv2_offset
    integer sendsize, recvsize
    integer local_size1,local_offset1
    integer local_size2,local_offset2
    integer local_size_dger,local_offset_dger
    integer v1_local_offset,v2_local_offset
    integer v_local_offset_dger
    double precision hvdot
    integer irow,icol,v1col,v2col

    ! external functions
    double precision ddot
    external dgemv,dger,ddot,daxpy

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)
 
    sendsize = 2*n
    recvsize = sendsize

    if (lwork .eq. -1) then
        work(1) = sendsize + recvsize
        return
    end if
 
    dgemv1_offset = 1
    dgemv2_offset = dgemv1_offset + n

        ! in 2x2 matrix case only one householder vector was generated
        if (idx .le. 2) then
            call qr_pdlarfl_1dcomm(v(1,2),1,baseidx,a,lda,t(2,2), &
                                    work,lwork,m,n,idx,mb,rev,mpicomm)
            return
        end if

        call local_size_offset_1d(m,mb,baseidx,idx,rev,mpirank,mpiprocs, &
                                  local_size1,v1_local_offset,local_offset1)
        call local_size_offset_1d(m,mb,baseidx,idx-1,rev,mpirank,mpiprocs, &
                                  local_size2,v2_local_offset,local_offset2)

        v1_local_offset = v1_local_offset * 1
        v2_local_offset = v2_local_offset * 1

        v1col = 2
        v2col = 1

        ! keep buffers clean in case that local_size1/local_size2 are zero
        work(1:sendsize) = 0.0d0

        call dgemv("Trans",local_size1,n,1.0d0,a(local_offset1,1),lda,v(v1_local_offset,v1col),1,0.0d0,work(dgemv1_offset),1)
        call dgemv("Trans",local_size2,n,t(v2col,v2col),a(local_offset2,1),lda,v(v2_local_offset,v2col),1,0.0d0, &
                   work(dgemv2_offset),1)

        call mpi_allreduce(work, work(sendsize+1), sendsize, mpi_real8, mpi_sum, mpicomm, mpierr)
  
        ! update second vector
        call daxpy(n,t(1,2),work(sendsize+dgemv1_offset),1,work(sendsize+dgemv2_offset),1)

        call local_size_offset_1d(m,mb,baseidx,idx-2,rev,mpirank,mpiprocs, &
                                  local_size_dger,v_local_offset_dger,local_offset_dger)

        ! get ranks of processes with topelements
        mpirank_top1 = MOD((idx-1)/mb,mpiprocs)
        mpirank_top2 = MOD((idx-2)/mb,mpiprocs)

        if (mpirank_top1 .eq. mpirank) local_offset1 = local_size1
        if (mpirank_top2 .eq. mpirank) then
            local_offset2 = local_size2
            v2_local_offset = local_size2
        end if

    ! use hvdot as temporary variable
    hvdot = t(v1col,v1col)
    do icol=1,n
        ! make use of "1" entries in householder vectors
        if (mpirank_top1 .eq. mpirank) then
            a(local_offset1,icol) = a(local_offset1,icol) &
                                    - work(sendsize+dgemv1_offset+icol-1)*hvdot
        end if

        if (mpirank_top2 .eq. mpirank) then
            a(local_offset2,icol) = a(local_offset2,icol) & 
                                    - v(v2_local_offset,v1col)*work(sendsize+dgemv1_offset+icol-1)*hvdot &
                                    - work(sendsize+dgemv2_offset+icol-1)
        end if

        do irow=1,local_size_dger
            a(local_offset_dger+irow-1,icol) = a(local_offset_dger+irow-1,icol) &
                                    - work(sendsize+dgemv1_offset+icol-1)*v(v_local_offset_dger+irow-1,v1col)*hvdot & 
                                    - work(sendsize+dgemv2_offset+icol-1)*v(v_local_offset_dger+irow-1,v2col)
        end do
    end do

end subroutine qr_pdlarfl2_tmatrix_1dcomm

! generalized pdlarfl2 version
! TODO: include T merge here (seperate by "old" and "new" index)
subroutine qr_tmerge_pdlarfb_1dcomm(m,mb,n,oldk,k,v,ldv,t,ldt,a,lda,baseidx,rev,updatemode,mpicomm,work,lwork)
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer ldv,ldt,lda,lwork
    double precision v(ldv,*),t(ldt,*),work(*),a(lda,*)

    ! input variables (global)
    integer m,mb,n,k,oldk,baseidx,rev,updatemode,mpicomm
 
    ! output variables (global)

    ! derived input variables from QR_PQRPARAM

    ! local scalars
    integer localsize,offset,baseoffset
    integer mpirank,mpiprocs,mpierr

    integer sendoffset,recvoffset,sendsize
    integer updateoffset,updatelda,updatesize
    integer mergeoffset,mergelda,mergesize
    integer tgenoffset,tgenlda,tgensize

        if (updatemode .eq. ichar('I')) then
            updatelda = oldk+k
        else
            updatelda = k
        end if

        updatesize = updatelda*n

        mergelda = k
        mergesize = mergelda*oldk

        tgenlda = 0
        tgensize = 0

        sendsize = updatesize + mergesize + tgensize

    if (lwork .eq. -1) then
        work(1) = DBLE(2*sendsize)
        return
    end if

    call MPI_Comm_rank(mpicomm,mpirank,mpierr)
    call MPI_Comm_size(mpicomm,mpiprocs,mpierr)
 
    ! use baseidx as idx here, otherwise the upper triangle part will be lost
    ! during the calculation, especially in the reversed case
    call local_size_offset_1d(m,mb,baseidx,baseidx,rev,mpirank,mpiprocs, &
                                localsize,baseoffset,offset)

    sendoffset = 1

        if (oldk .gt. 0) then
            updateoffset = 0
            mergeoffset = updateoffset + updatesize
            tgenoffset = mergeoffset + mergesize
            
            sendsize = updatesize + mergesize + tgensize

            !print *,'sendsize',sendsize,updatesize,mergesize,tgensize
            !print *,'merging nr of rotations', oldk+k
 
            if (localsize .gt. 0) then
                ! calculate matrix matrix product of householder vectors and target matrix 

                if (updatemode .eq. ichar('I')) then
                    ! Z' = (Y1,Y2)' * A
                    call dgemm("Trans","Notrans",k+oldk,n,localsize,1.0d0,v(baseoffset,1),ldv,a(offset,1),lda,0.0d0, &
                               work(sendoffset+updateoffset),updatelda)
                else
                    ! Z' = Y1' * A
                    call dgemm("Trans","Notrans",k,n,localsize,1.0d0,v(baseoffset,1),ldv,a(offset,1),lda,0.0d0, &
                               work(sendoffset+updateoffset),updatelda)
                end if

                ! calculate parts needed for T merge
                call dgemm("Trans","Notrans",k,oldk,localsize,1.0d0,v(baseoffset,1),ldv,v(baseoffset,k+1),ldv,0.0d0, &
                           work(sendoffset+mergeoffset),mergelda)

            else
                ! cleanup buffer
                work(sendoffset:sendoffset+sendsize-1) = 0.0d0
            end if
        else
            ! do not calculate parts for T merge as there is nothing to merge

            updateoffset = 0
            
            tgenoffset = updateoffset + updatesize
            
            sendsize = updatesize + tgensize
 
            if (localsize .gt. 0) then
                ! calculate matrix matrix product of householder vectors and target matrix 
                ! Z' = (Y1)' * A
                call dgemm("Trans","Notrans",k,n,localsize,1.0d0,v(baseoffset,1),ldv,a(offset,1),lda,0.0d0, &
                           work(sendoffset+updateoffset),updatelda)

            else
                ! cleanup buffer
                work(sendoffset:sendoffset+sendsize-1) = 0.0d0
            end if

        end if

    recvoffset = sendoffset + sendsize

    if (sendsize .le. 0) return ! nothing to do

    ! exchange data
    call mpi_allreduce(work(sendoffset),work(recvoffset),sendsize,mpi_real8,mpi_sum,mpicomm,mpierr)
 
    updateoffset = recvoffset+updateoffset
    mergeoffset = recvoffset+mergeoffset
    tgenoffset = recvoffset+tgenoffset

        if (oldk .gt. 0) then
            call qr_pdlarft_merge_kernel_local(oldk,k,t,ldt,work(mergeoffset),mergelda)

            if (localsize .gt. 0) then
                if (updatemode .eq. ichar('I')) then

                    ! update matrix (pdlarfb) with complete T
                    call qr_pdlarfb_kernel_local(localsize,n,k+oldk,a(offset,1),lda,v(baseoffset,1),ldv,t(1,1),ldt, &
                                                 work(updateoffset),updatelda)
                else
                    ! update matrix (pdlarfb) with small T (same as update with no old T TODO)
                    call qr_pdlarfb_kernel_local(localsize,n,k,a(offset,1),lda,v(baseoffset,1),ldv,t(1,1),ldt, &
                                                 work(updateoffset),updatelda)
                end if
            end if
        else
            if (localsize .gt. 0) then
                ! update matrix (pdlarfb) with small T
                call qr_pdlarfb_kernel_local(localsize,n,k,a(offset,1),lda,v(baseoffset,1),ldv,t(1,1),ldt, &
                                             work(updateoffset),updatelda)
            end if
        end if

end subroutine qr_tmerge_pdlarfb_1dcomm

end module elpa_pdlarfb
