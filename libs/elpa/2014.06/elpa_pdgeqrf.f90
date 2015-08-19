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
module elpa_pdgeqrf

    use elpa1
    use elpa_pdlarfb
    use qr_utils_mod

    implicit none

    PRIVATE

    public :: qr_pdgeqrf_2dcomm
    public :: qr_pqrparam_init
    public :: qr_pdlarfg2_1dcomm_check

    include 'mpif.h'

contains

subroutine qr_pdgeqrf_2dcomm(a,lda,v,ldv,tau,t,ldt,work,lwork,m,n,mb,nb,rowidx,colidx,rev,trans,PQRPARAM, &
                             mpicomm_rows,mpicomm_cols,blockheuristic)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! parameter setup
    INTEGER     gmode_,rank_,eps_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),work(*),t(ldt,*)

    ! input variables (global)
    integer m,n,mb,nb,rowidx,colidx,rev,trans,mpicomm_cols,mpicomm_rows
    integer PQRPARAM(*)

    ! output variables (global)
    double precision blockheuristic(*)

    ! input variables derived from PQRPARAM
    integer updatemode,tmerge,size2d

    ! local scalars
    integer mpierr,mpirank_cols,broadcast_size,mpirank_rows
    integer mpirank_cols_qr,mpiprocs_cols
    integer lcols_temp,lcols,icol,lastcol
    integer baseoffset,offset,idx,voffset
    integer update_voffset,update_tauoffset
    integer update_lcols
    integer work_offset

    double precision dbroadcast_size(1),dtmat_bcast_size(1)
    double precision pdgeqrf_size(1),pdlarft_size(1),pdlarfb_size(1),tmerge_pdlarfb_size(1)
    integer temptau_offset,temptau_size,broadcast_offset,tmat_bcast_size
    integer remaining_cols
    integer total_cols
    integer incremental_update_size ! needed for incremental update mode

    size2d = PQRPARAM(1)
    updatemode = PQRPARAM(2)
    tmerge = PQRPARAM(3)

    ! copy value before we are going to filter it
    total_cols = n

    call mpi_comm_rank(mpicomm_cols,mpirank_cols,mpierr)
    call mpi_comm_rank(mpicomm_rows,mpirank_rows,mpierr)
    call mpi_comm_size(mpicomm_cols,mpiprocs_cols,mpierr)


    call qr_pdgeqrf_1dcomm(a,lda,v,ldv,tau,t,ldt,pdgeqrf_size(1),-1,m,total_cols,mb,rowidx,rowidx,rev,trans, &
                           PQRPARAM(4),mpicomm_rows,blockheuristic)
    call qr_pdgeqrf_pack_unpack(v,ldv,dbroadcast_size(1),-1,m,total_cols,mb,rowidx,rowidx,rev,0,mpicomm_rows)
    call qr_pdgeqrf_pack_unpack_tmatrix(tau,t,ldt,dtmat_bcast_size(1),-1,total_cols,0)
    pdlarft_size(1) = 0.0d0
    call qr_pdlarfb_1dcomm(m,mb,total_cols,total_cols,a,lda,v,ldv,tau,t,ldt,rowidx,rowidx,rev,mpicomm_rows, &
                           pdlarfb_size(1),-1)
    call qr_tmerge_pdlarfb_1dcomm(m,mb,total_cols,total_cols,total_cols,v,ldv,t,ldt,a,lda,rowidx,rev,updatemode, &
                                  mpicomm_rows,tmerge_pdlarfb_size(1),-1)


    temptau_offset = 1
    temptau_size = total_cols
    broadcast_offset = temptau_offset + temptau_size
    broadcast_size = dbroadcast_size(1) + dtmat_bcast_size(1)
    work_offset = broadcast_offset + broadcast_size

    if (lwork .eq. -1) then
        work(1) = (DBLE(temptau_size) + DBLE(broadcast_size) + max(pdgeqrf_size(1),pdlarft_size(1),pdlarfb_size(1), &
                   tmerge_pdlarfb_size(1)))
        return
    end if

    lastcol = colidx-total_cols+1
    voffset = total_cols

    incremental_update_size = 0

    ! clear v buffer: just ensure that there is no junk in the upper triangle
    ! part, otherwise pdlarfb gets some problems
    ! pdlarfl(2) do not have these problems as they are working more on a vector
    ! basis
    v(1:ldv,1:total_cols) = 0.0d0

    icol = colidx

    remaining_cols = total_cols

    !print *,'start decomposition',m,rowidx,colidx

    do while (remaining_cols .gt. 0)

        ! determine rank of process column with next qr block
        mpirank_cols_qr = MOD((icol-1)/nb,mpiprocs_cols)

        ! lcols can't be larger than than nb
        ! exception: there is only one process column

        ! however, we might not start at the first local column.
        ! therefore assume a matrix of size (1xlcols) starting at (1,icol)
        ! determine the real amount of local columns
        lcols_temp = min(nb,(icol-lastcol+1))

        ! blocking parameter
        lcols_temp = max(min(lcols_temp,size2d),1)

        ! determine size from last decomposition column
        !  to first decomposition column
        call local_size_offset_1d(icol,nb,icol-lcols_temp+1,icol-lcols_temp+1,0, &
                                      mpirank_cols_qr,mpiprocs_cols, &
                                      lcols,baseoffset,offset)

        voffset = remaining_cols - lcols + 1

        idx = rowidx - colidx + icol

        if (mpirank_cols .eq. mpirank_cols_qr) then
            ! qr decomposition part

            tau(offset:offset+lcols-1) = 0.0d0

            call qr_pdgeqrf_1dcomm(a(1,offset),lda,v(1,voffset),ldv,tau(offset),t(voffset,voffset),ldt, &
                                   work(work_offset),lwork,m,lcols,mb,rowidx,idx,rev,trans,PQRPARAM(4), &
                                   mpicomm_rows,blockheuristic)

            ! pack broadcast buffer (v + tau)
            call qr_pdgeqrf_pack_unpack(v(1,voffset),ldv,work(broadcast_offset),lwork,m,lcols,mb,rowidx,&
                                        idx,rev,0,mpicomm_rows)

            ! determine broadcast size
            call qr_pdgeqrf_pack_unpack(v(1,voffset),ldv,dbroadcast_size(1),-1,m,lcols,mb,rowidx,idx,rev,&
                                        0,mpicomm_rows)
            broadcast_size = dbroadcast_size(1)

            !if (mpirank_rows .eq. 0) then
            ! pack tmatrix into broadcast buffer and calculate new size
            call qr_pdgeqrf_pack_unpack_tmatrix(tau(offset),t(voffset,voffset),ldt, &
                                                work(broadcast_offset+broadcast_size),lwork,lcols,0)
            call qr_pdgeqrf_pack_unpack_tmatrix(tau(offset),t(voffset,voffset),ldt,dtmat_bcast_size(1),-1,lcols,0)
            broadcast_size = broadcast_size + dtmat_bcast_size(1)
            !end if

            ! initiate broadcast (send part)
            call MPI_Bcast(work(broadcast_offset),broadcast_size,mpi_real8, &
                           mpirank_cols_qr,mpicomm_cols,mpierr)

            ! copy tau parts into temporary tau buffer
            work(temptau_offset+voffset-1:temptau_offset+(voffset-1)+lcols-1) = tau(offset:offset+lcols-1)

            !print *,'generated tau:', tau(offset)
        else
            ! vector exchange part

            ! determine broadcast size
            call qr_pdgeqrf_pack_unpack(v(1,voffset),ldv,dbroadcast_size(1),-1,m,lcols,mb,rowidx,idx,rev,1,mpicomm_rows)
            broadcast_size = dbroadcast_size(1)

            call qr_pdgeqrf_pack_unpack_tmatrix(work(temptau_offset+voffset-1),t(voffset,voffset),ldt, &
                                                dtmat_bcast_size(1),-1,lcols,0)
            tmat_bcast_size = dtmat_bcast_size(1)

            !print *,'broadcast_size (nonqr)',broadcast_size
            broadcast_size = dbroadcast_size(1) + dtmat_bcast_size(1)

            ! initiate broadcast (recv part)
            call MPI_Bcast(work(broadcast_offset),broadcast_size,mpi_real8, &
                           mpirank_cols_qr,mpicomm_cols,mpierr)

            ! last n*n elements in buffer are (still empty) T matrix elements
            ! fetch from first process in each column

            ! unpack broadcast buffer (v + tau)
            call qr_pdgeqrf_pack_unpack(v(1,voffset),ldv,work(broadcast_offset),lwork,m,lcols,mb,rowidx,idx,rev,1,mpicomm_rows)

            ! now send t matrix to other processes in our process column
            broadcast_size = dbroadcast_size(1)
            tmat_bcast_size = dtmat_bcast_size(1)

            ! t matrix should now be available on all processes => unpack
            call qr_pdgeqrf_pack_unpack_tmatrix(work(temptau_offset+voffset-1),t(voffset,voffset),ldt, &
                                                work(broadcast_offset+broadcast_size),lwork,lcols,1)
        end if

        remaining_cols = remaining_cols - lcols

        ! apply householder vectors to whole trailing matrix parts (if any)

        update_voffset = voffset
        update_tauoffset = icol
        update_lcols = lcols
        incremental_update_size = incremental_update_size + lcols

        icol = icol - lcols
        ! count colums from first column of global block to current index
        call local_size_offset_1d(icol,nb,colidx-n+1,colidx-n+1,0, &
                                      mpirank_cols,mpiprocs_cols, &
                                      lcols,baseoffset,offset)

        if (lcols .gt. 0) then

            !print *,'updating trailing matrix'

			if (updatemode .eq. ichar('I')) then
				print *,'pdgeqrf_2dcomm: incremental update not yet implemented! rev=1'
			else if (updatemode .eq. ichar('F')) then
				! full update no merging
				call qr_pdlarfb_1dcomm(m,mb,lcols,update_lcols,a(1,offset),lda,v(1,update_voffset),ldv, &
							work(temptau_offset+update_voffset-1),                          &
                                                        t(update_voffset,update_voffset),ldt, &
							rowidx,idx,1,mpicomm_rows,work(work_offset),lwork)
			else
				! full update + merging default
				call qr_tmerge_pdlarfb_1dcomm(m,mb,lcols,n-(update_voffset+update_lcols-1),update_lcols, &
                                                              v(1,update_voffset),ldv, &
							      t(update_voffset,update_voffset),ldt, &
							      a(1,offset),lda,rowidx,1,updatemode,mpicomm_rows, &
                                                              work(work_offset),lwork)
			end if
        else
			if (updatemode .eq. ichar('I')) then
				print *,'sole merging of (incremental) T matrix', mpirank_cols,  &
                                        n-(update_voffset+incremental_update_size-1)
				call qr_tmerge_pdlarfb_1dcomm(m,mb,0,n-(update_voffset+incremental_update_size-1),   &
                                                              incremental_update_size,v(1,update_voffset),ldv, &
							      t(update_voffset,update_voffset),ldt, &
							      a,lda,rowidx,1,updatemode,mpicomm_rows,work(work_offset),lwork)

				! reset for upcoming incremental updates
				incremental_update_size = 0
			else if (updatemode .eq. ichar('M')) then
				! final merge
				call qr_tmerge_pdlarfb_1dcomm(m,mb,0,n-(update_voffset+update_lcols-1),update_lcols, &
                                                              v(1,update_voffset),ldv, &
							      t(update_voffset,update_voffset),ldt, &
							      a,lda,rowidx,1,updatemode,mpicomm_rows,work(work_offset),lwork)
			else
				! full updatemode - nothing to update
			end if

			! reset for upcoming incremental updates
			incremental_update_size = 0
        end if
    end do

    if ((tmerge .gt. 0) .and. (updatemode .eq. ichar('F'))) then
        ! finally merge all small T parts
        call qr_pdlarft_tree_merge_1dcomm(m,mb,n,size2d,tmerge,v,ldv,t,ldt,rowidx,rev,mpicomm_rows,work,lwork)
    end if

    !print *,'stop decomposition',rowidx,colidx

end subroutine qr_pdgeqrf_2dcomm

subroutine qr_pdgeqrf_1dcomm(a,lda,v,ldv,tau,t,ldt,work,lwork,m,n,mb,baseidx,rowidx,rev,trans,PQRPARAM,mpicomm,blockheuristic)
    use ELPA1

    implicit none

    ! parameter setup
    INTEGER     gmode_,rank_,eps_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),t(ldt,*),work(*)

    ! input variables (global)
    integer m,n,mb,baseidx,rowidx,rev,trans,mpicomm
    integer PQRPARAM(*)

    ! derived input variables

    ! derived further input variables from QR_PQRPARAM
    integer size1d,updatemode,tmerge

    ! output variables (global)
    double precision blockheuristic(*)

    ! local scalars
    integer nr_blocks,remainder,current_block,aoffset,idx,updatesize
    double precision pdgeqr2_size(1),pdlarfb_size(1),tmerge_tree_size(1)

    size1d = max(min(PQRPARAM(1),n),1)
    updatemode = PQRPARAM(2)
    tmerge = PQRPARAM(3)

    if (lwork .eq. -1) then
        call qr_pdgeqr2_1dcomm(a,lda,v,ldv,tau,t,ldt,pdgeqr2_size,-1, &
                                m,size1d,mb,baseidx,baseidx,rev,trans,PQRPARAM(4),mpicomm,blockheuristic)

        ! reserve more space for incremental mode
        call qr_tmerge_pdlarfb_1dcomm(m,mb,n,n,n,v,ldv,t,ldt, &
                                       a,lda,baseidx,rev,updatemode,mpicomm,pdlarfb_size,-1)

        call qr_pdlarft_tree_merge_1dcomm(m,mb,n,size1d,tmerge,v,ldv,t,ldt,baseidx,rev,mpicomm,tmerge_tree_size,-1)

        work(1) = max(pdlarfb_size(1),pdgeqr2_size(1),tmerge_tree_size(1))
        return
    end if

        nr_blocks = n / size1d
        remainder = n - nr_blocks*size1d

        current_block = 0
        do while (current_block .lt. nr_blocks)
            idx = rowidx-current_block*size1d
            updatesize = n-(current_block+1)*size1d
            aoffset = 1+updatesize

            call qr_pdgeqr2_1dcomm(a(1,aoffset),lda,v(1,aoffset),ldv,tau(aoffset),t(aoffset,aoffset),ldt,work,lwork, &
                                    m,size1d,mb,baseidx,idx,1,trans,PQRPARAM(4),mpicomm,blockheuristic)

            if (updatemode .eq. ichar('M')) then
                ! full update + merging
                call qr_tmerge_pdlarfb_1dcomm(m,mb,updatesize,current_block*size1d,size1d, &
                                               v(1,aoffset),ldv,t(aoffset,aoffset),ldt, &
                                               a,lda,baseidx,1,ichar('F'),mpicomm,work,lwork)
            else if (updatemode .eq. ichar('I')) then
                if (updatesize .ge. size1d) then
                    ! incremental update + merging
                    call qr_tmerge_pdlarfb_1dcomm(m,mb,size1d,current_block*size1d,size1d, &
                                                   v(1,aoffset),ldv,t(aoffset,aoffset),ldt, &
                                                   a(1,aoffset-size1d),lda,baseidx,1,updatemode,mpicomm,work,lwork)

                else ! only remainder left
                    ! incremental update + merging
                    call qr_tmerge_pdlarfb_1dcomm(m,mb,remainder,current_block*size1d,size1d, &
                                                   v(1,aoffset),ldv,t(aoffset,aoffset),ldt, &
                                                   a(1,1),lda,baseidx,1,updatemode,mpicomm,work,lwork)
                end if
            else ! full update no merging is default
                ! full update no merging
                call qr_pdlarfb_1dcomm(m,mb,updatesize,size1d,a,lda,v(1,aoffset),ldv, &
                                        tau(aoffset),t(aoffset,aoffset),ldt,baseidx,idx,1,mpicomm,work,lwork)
            end if

            ! move on to next block
            current_block = current_block+1
        end do

        if (remainder .gt. 0) then
            aoffset = 1
            idx = rowidx-size1d*nr_blocks
            call qr_pdgeqr2_1dcomm(a(1,aoffset),lda,v,ldv,tau,t,ldt,work,lwork, &
                                    m,remainder,mb,baseidx,idx,1,trans,PQRPARAM(4),mpicomm,blockheuristic)

            if ((updatemode .eq. ichar('I')) .or. (updatemode .eq. ichar('M'))) then
                ! final merging
                call qr_tmerge_pdlarfb_1dcomm(m,mb,0,size1d*nr_blocks,remainder, &
                                               v,ldv,t,ldt, &
                                               a,lda,baseidx,1,updatemode,mpicomm,work,lwork) ! updatemode argument does not matter
            end if
        end if

    if ((tmerge .gt. 0) .and. (updatemode .eq. ichar('F'))) then
        ! finally merge all small T parts
        call qr_pdlarft_tree_merge_1dcomm(m,mb,n,size1d,tmerge,v,ldv,t,ldt,baseidx,rev,mpicomm,work,lwork)
    end if

end subroutine qr_pdgeqrf_1dcomm

! local a and tau are assumed to be positioned at the right column from a local
! perspective
! TODO: if local amount of data turns to zero the algorithm might produce wrong
! results (probably due to old buffer contents)
subroutine qr_pdgeqr2_1dcomm(a,lda,v,ldv,tau,t,ldt,work,lwork,m,n,mb,baseidx,rowidx,rev,trans,PQRPARAM,mpicomm,blockheuristic)
    use ELPA1

    implicit none

    ! parameter setup
    INTEGER     gmode_,rank_,eps_,upmode1_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3, upmode1_=4)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),t(ldt,*),work(*)

    ! input variables (global)
    integer m,n,mb,baseidx,rowidx,rev,trans,mpicomm
    integer PQRPARAM(*)

    ! output variables (global)
    double precision blockheuristic(*)

    ! derived further input variables from QR_PQRPARAM
    integer maxrank,hgmode,updatemode

    ! local scalars
    integer icol,incx,idx
    double precision pdlarfg_size(1),pdlarf_size(1),total_size
    double precision pdlarfg2_size(1),pdlarfgk_size(1),pdlarfl2_size(1)
    double precision pdlarft_size(1),pdlarfb_size(1),pdlarft_pdlarfb_size(1),tmerge_pdlarfb_size(1)
    integer mpirank,mpiprocs,mpierr
    integer rank,lastcol,actualrank,nextrank
    integer update_cols,decomposition_cols
    integer current_column

    maxrank = min(PQRPARAM(1),n)
    updatemode = PQRPARAM(2)
    hgmode = PQRPARAM(4)

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    if (trans .eq. 1) then
        incx = lda
    else
        incx = 1
    end if

    if (lwork .eq. -1) then
        call qr_pdlarfg_1dcomm(a,incx,tau(1),pdlarfg_size(1),-1,n,rowidx,mb,hgmode,rev,mpicomm)
        call qr_pdlarfl_1dcomm(v,1,baseidx,a,lda,tau(1),pdlarf_size(1),-1,m,n,rowidx,mb,rev,mpicomm)
        call qr_pdlarfg2_1dcomm_ref(a,lda,tau,t,ldt,v,ldv,baseidx,pdlarfg2_size(1),-1,m,rowidx,mb,PQRPARAM,rev,mpicomm,actualrank)
        call qr_pdlarfgk_1dcomm(a,lda,tau,t,ldt,v,ldv,baseidx,pdlarfgk_size(1),-1,m,n,rowidx,mb,PQRPARAM,rev,mpicomm,actualrank)
        call qr_pdlarfl2_tmatrix_1dcomm(v,ldv,baseidx,a,lda,t,ldt,pdlarfl2_size(1),-1,m,n,rowidx,mb,rev,mpicomm)
        pdlarft_size(1) = 0.0d0
        call qr_pdlarfb_1dcomm(m,mb,n,n,a,lda,v,ldv,tau,t,ldt,baseidx,rowidx,1,mpicomm,pdlarfb_size(1),-1)
        pdlarft_pdlarfb_size(1) = 0.0d0
        call qr_tmerge_pdlarfb_1dcomm(m,mb,n,n,n,v,ldv,t,ldt,a,lda,rowidx,rev,updatemode,mpicomm,tmerge_pdlarfb_size(1),-1)

        total_size = max(pdlarfg_size(1),pdlarf_size(1),pdlarfg2_size(1),pdlarfgk_size(1),pdlarfl2_size(1),pdlarft_size(1), &
                         pdlarfb_size(1),pdlarft_pdlarfb_size(1),tmerge_pdlarfb_size(1))

        work(1) = total_size
        return
    end if

        icol = 1
        lastcol = min(rowidx,n)
        decomposition_cols = lastcol
        update_cols = n
        do while (decomposition_cols .gt. 0) ! local qr block
            icol = lastcol-decomposition_cols+1
            idx = rowidx-icol+1

            ! get possible rank size
            ! limited by number of columns and remaining rows
            rank = min(n-icol+1,maxrank,idx)

            current_column = n-icol+1-rank+1

            if (rank .eq. 1) then

                call qr_pdlarfg_1dcomm(a(1,current_column),incx, &
                                        tau(current_column),work,lwork, &
                                        m,idx,mb,hgmode,1,mpicomm)

                v(1:ldv,current_column) = 0.0d0
                call qr_pdlarfg_copy_1dcomm(a(1,current_column),incx, &
                                             v(1,current_column),1, &
                                             m,baseidx,idx,mb,1,mpicomm)

                ! initialize t matrix part
                t(current_column,current_column) = tau(current_column)

                actualrank = 1

            else if (rank .eq. 2) then
                call qr_pdlarfg2_1dcomm_ref(a(1,current_column),lda,tau(current_column), &
                                             t(current_column,current_column),ldt,v(1,current_column),ldv, &
                                            baseidx,work,lwork,m,idx,mb,PQRPARAM,1,mpicomm,actualrank)

            else
                call qr_pdlarfgk_1dcomm(a(1,current_column),lda,tau(current_column), &
                                         t(current_column,current_column),ldt,v(1,current_column),ldv, &
                                         baseidx,work,lwork,m,rank,idx,mb,PQRPARAM,1,mpicomm,actualrank)

            end if

            blockheuristic(actualrank) = blockheuristic(actualrank) + 1

            ! the blocked decomposition versions already updated their non
            ! decomposed parts using their information after communication
            update_cols = decomposition_cols - rank
            decomposition_cols = decomposition_cols - actualrank

            ! needed for incremental update
            nextrank = min(n-(lastcol-decomposition_cols+1)+1,maxrank,rowidx-(lastcol-decomposition_cols+1)+1)

            if (current_column .gt. 1) then
                idx = rowidx-icol+1

                if (updatemode .eq. ichar('I')) then
                    ! incremental update + merging
                    call qr_tmerge_pdlarfb_1dcomm(m,mb,nextrank-(rank-actualrank),n-(current_column+rank-1),actualrank, &
                                                  v(1,current_column+(rank-actualrank)),ldv, &
                                                  t(current_column+(rank-actualrank),current_column+(rank-actualrank)),ldt, &
                                                  a(1,current_column-nextrank+(rank-actualrank)),lda,baseidx,rev,updatemode,&
                                                  mpicomm,work,lwork)
                else
                    ! full update + merging
                    call qr_tmerge_pdlarfb_1dcomm(m,mb,update_cols,n-(current_column+rank-1),actualrank, &
                                                  v(1,current_column+(rank-actualrank)),ldv, &
                                                  t(current_column+(rank-actualrank),current_column+(rank-actualrank)),ldt, &
                                                  a(1,1),lda,baseidx,rev,updatemode,mpicomm,work,lwork)
                end if
            else
                call qr_tmerge_pdlarfb_1dcomm(m,mb,0,n-(current_column+rank-1),actualrank,v(1,current_column+(rank-actualrank)), &
                                              ldv, &
                                              t(current_column+(rank-actualrank),current_column+(rank-actualrank)),ldt, &
                                              a,lda,baseidx,rev,updatemode,mpicomm,work,lwork)
            end if

        end do
end subroutine qr_pdgeqr2_1dcomm

! incx == 1: column major
! incx != 1: row major
subroutine qr_pdlarfg_1dcomm(x,incx,tau,work,lwork,n,idx,nb,hgmode,rev,mpi_comm)
    use ELPA1
    use qr_utils_mod

    implicit none
 
    ! parameter setup
    INTEGER     gmode_,rank_,eps_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3)

    ! input variables (local)
    integer incx,lwork,hgmode
    double precision x(*),work(*)

    ! input variables (global)
    integer mpi_comm,nb,idx,n,rev

    ! output variables (global)
    double precision tau

    ! local scalars
    integer mpierr,mpirank,mpiprocs,mpirank_top
    integer sendsize,recvsize
    integer local_size,local_offset,baseoffset
    integer topidx,top,iproc
    double precision alpha,xnorm,dot,xf

    ! external functions
    double precision ddot,dlapy2,dnrm2
    external ddot,dscal,dlapy2,dnrm2

    ! intrinsic
    intrinsic sign

	if (idx .le. 1) then
		tau = 0.0d0
		return
	end if

    call MPI_Comm_rank(mpi_comm, mpirank, mpierr)
    call MPI_Comm_size(mpi_comm, mpiprocs, mpierr)

    ! calculate expected work size and store in work(1)
    if (hgmode .eq. ichar('s')) then
        ! allreduce (MPI_SUM)
        sendsize = 2
        recvsize = sendsize
    else if (hgmode .eq. ichar('x')) then
        ! alltoall
        sendsize = mpiprocs*2
        recvsize = sendsize
    else if (hgmode .eq. ichar('g')) then
        ! allgather
        sendsize = 2
        recvsize = mpiprocs*sendsize
    else
        ! no exchange at all (benchmarking)
        sendsize = 2
        recvsize = sendsize
    end if

    if (lwork .eq. -1) then
        work(1) = DBLE(sendsize + recvsize)
        return
    end if

    ! Processor id for global index of top element
    mpirank_top = MOD((idx-1)/nb,mpiprocs)
    if (mpirank .eq. mpirank_top) then
        topidx = local_index(idx,mpirank_top,mpiprocs,nb,0)
        top = 1+(topidx-1)*incx
    end if

	call local_size_offset_1d(n,nb,idx,idx-1,rev,mpirank,mpiprocs, &
							  local_size,baseoffset,local_offset)

    local_offset = local_offset * incx

    ! calculate and exchange information
    if (hgmode .eq. ichar('s')) then
        if (mpirank .eq. mpirank_top) then
            alpha = x(top)
        else
            alpha = 0.0d0
        end if

        dot = ddot(local_size, &
                   x(local_offset), incx, &
                   x(local_offset), incx)

        work(1) = alpha
        work(2) = dot

        call mpi_allreduce(work(1),work(sendsize+1), &
                           sendsize,mpi_real8,mpi_sum, &
                           mpi_comm,mpierr)

        alpha = work(sendsize+1)
        xnorm = sqrt(work(sendsize+2))
    else if (hgmode .eq. ichar('x')) then
        if (mpirank .eq. mpirank_top) then
            alpha = x(top)
        else
            alpha = 0.0d0
        end if

        xnorm = dnrm2(local_size, x(local_offset), incx)

        do iproc=0,mpiprocs-1
            work(2*iproc+1) = alpha
            work(2*iproc+2) = xnorm
        end do

        call mpi_alltoall(work(1),2,mpi_real8, &
                          work(sendsize+1),2,mpi_real8, &
                          mpi_comm,mpierr)

        ! extract alpha value
        alpha = work(sendsize+1+mpirank_top*2)

        ! copy norm parts of buffer to beginning
        do iproc=0,mpiprocs-1
            work(iproc+1) = work(sendsize+1+2*iproc+1)
        end do

        xnorm = dnrm2(mpiprocs, work(1), 1)
    else if (hgmode .eq. ichar('g')) then
        if (mpirank .eq. mpirank_top) then
            alpha = x(top)
        else
            alpha = 0.0d0
        end if

        xnorm = dnrm2(local_size, x(local_offset), incx)
        work(1) = alpha
        work(2) = xnorm

        ! allgather
        call mpi_allgather(work(1),sendsize,mpi_real8, &
                          work(sendsize+1),sendsize,mpi_real8, &
                          mpi_comm,mpierr)

        ! extract alpha value
        alpha = work(sendsize+1+mpirank_top*2)
 
        ! copy norm parts of buffer to beginning
        do iproc=0,mpiprocs-1
            work(iproc+1) = work(sendsize+1+2*iproc+1)
        end do

        xnorm = dnrm2(mpiprocs, work(1), 1)
    else
        ! dnrm2
        xnorm = dnrm2(local_size, x(local_offset), incx)

        if (mpirank .eq. mpirank_top) then
            alpha = x(top)
        else
            alpha = 0.0d0
        end if

        ! no exchange at all (benchmarking)

        xnorm = 0.0d0
    end if

    !print *,'ref hg:', idx,xnorm,alpha
    !print *,x(1:n)

    ! calculate householder information
    if (xnorm .eq. 0.0d0) then
        ! H = I

        tau = 0.0d0
    else
        ! General case

        call hh_transform_real(alpha,xnorm**2,xf,tau)
        if (mpirank .eq. mpirank_top) then
            x(top) = alpha
        end if

        call dscal(local_size, xf, &
                   x(local_offset), incx)

        ! TODO: reimplement norm rescale method of
        ! original PDLARFG using mpi?

    end if

    ! useful for debugging
    !print *,'hg:mpirank,idx,beta,alpha:',mpirank,idx,beta,alpha,1.0d0/(beta+alpha),tau
    !print *,x(1:n)

end subroutine qr_pdlarfg_1dcomm

subroutine qr_pdlarfg2_1dcomm_ref(a,lda,tau,t,ldt,v,ldv,baseidx,work,lwork,m,idx,mb,PQRPARAM,rev,mpicomm,actualk)
    implicit none

    ! parameter setup
    INTEGER     gmode_,rank_,eps_,upmode1_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3, upmode1_=4)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),work(*),t(ldt,*)

    ! input variables (global)
    integer m,idx,baseidx,mb,rev,mpicomm
    integer PQRPARAM(*)

    ! output variables (global)
    integer actualk

    ! derived input variables from QR_PQRPARAM
    integer eps

    ! local scalars
    double precision dseedwork_size(1)
    integer seedwork_size,seed_size
    integer seedwork_offset,seed_offset
    logical accurate

    call qr_pdlarfg2_1dcomm_seed(a,lda,dseedwork_size(1),-1,work,m,mb,idx,rev,mpicomm)
    seedwork_size = dseedwork_size(1)
    seed_size = seedwork_size

    if (lwork .eq. -1) then
        work(1) = seedwork_size + seed_size
        return
    end if

    seedwork_offset = 1
    seed_offset = seedwork_offset + seedwork_size

    eps = PQRPARAM(3)

    ! check for border cases (only a 2x2 matrix left)
	if (idx .le. 1) then
		tau(1:2) = 0.0d0
		t(1:2,1:2) = 0.0d0
		return
	end if

    call qr_pdlarfg2_1dcomm_seed(a,lda,work(seedwork_offset),lwork,work(seed_offset),m,mb,idx,rev,mpicomm)

        if (eps .gt. 0) then
            accurate = qr_pdlarfg2_1dcomm_check(work(seed_offset),eps)
        else
            accurate = .true.
        end if

        call qr_pdlarfg2_1dcomm_vector(a(1,2),1,tau(2),work(seed_offset), &
                                        m,mb,idx,0,1,mpicomm)

        call qr_pdlarfg_copy_1dcomm(a(1,2),1, &
                                     v(1,2),1, &
                                     m,baseidx,idx,mb,1,mpicomm)

        call qr_pdlarfg2_1dcomm_update(v(1,2),1,baseidx,a(1,1),lda,work(seed_offset),m,idx,mb,rev,mpicomm)

        ! check for 2x2 matrix case => only one householder vector will be
        ! generated
        if (idx .gt. 2) then
            if (accurate .eqv. .true.) then
                call qr_pdlarfg2_1dcomm_vector(a(1,1),1,tau(1),work(seed_offset), &
                                                m,mb,idx-1,1,1,mpicomm)

                call qr_pdlarfg_copy_1dcomm(a(1,1),1, &
                                             v(1,1),1, &
                                             m,baseidx,idx-1,mb,1,mpicomm)

                ! generate fuse element
                call qr_pdlarfg2_1dcomm_finalize_tmatrix(work(seed_offset),tau,t,ldt)

                actualk = 2
            else
                t(1,1) = 0.0d0
                t(1,2) = 0.0d0
                t(2,2) = tau(2)

                actualk = 1
            end if
        else
            t(1,1) = 0.0d0
            t(1,2) = 0.0d0
            t(2,2) = tau(2)

            ! no more vectors to create

            tau(1) = 0.0d0

            actualk = 2

            !print *,'rank2: no more data'
        end if

end subroutine qr_pdlarfg2_1dcomm_ref

subroutine qr_pdlarfg2_1dcomm_seed(a,lda,work,lwork,seed,n,nb,idx,rev,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer lda,lwork
    double precision a(lda,*),work(*),seed(*)

    ! input variables (global)
    integer n,nb,idx,rev,mpicomm

    ! output variables (global)

    ! external functions
    double precision ddot
    external ddot

    ! local scalars
    double precision top11,top21,top12,top22
    double precision dot11,dot12,dot22
    integer mpirank,mpiprocs,mpierr
    integer mpirank_top11,mpirank_top21
    integer top11_offset,top21_offset
    integer baseoffset
    integer local_offset1,local_size1
    integer local_offset2,local_size2

    if (lwork .eq. -1) then
        work(1) = DBLE(8)
        return
    end if

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

        call local_size_offset_1d(n,nb,idx,idx-1,rev,mpirank,mpiprocs, &
                              local_size1,baseoffset,local_offset1)

        call local_size_offset_1d(n,nb,idx,idx-2,rev,mpirank,mpiprocs, &
                              local_size2,baseoffset,local_offset2)

        mpirank_top11 = MOD((idx-1)/nb,mpiprocs)
        mpirank_top21 = MOD((idx-2)/nb,mpiprocs)

        top11_offset = local_index(idx,mpirank_top11,mpiprocs,nb,0)
        top21_offset = local_index(idx-1,mpirank_top21,mpiprocs,nb,0)

        if (mpirank_top11 .eq. mpirank) then
            top11 = a(top11_offset,2)
            top12 = a(top11_offset,1)
        else
            top11 = 0.0d0
            top12 = 0.0d0
        end if

        if (mpirank_top21 .eq. mpirank) then
            top21 = a(top21_offset,2)
            top22 = a(top21_offset,1)
        else
            top21 = 0.0d0
            top22 = 0.0d0
        end if

        ! calculate 3 dot products
        dot11 = ddot(local_size1,a(local_offset1,2),1,a(local_offset1,2),1)
        dot12 = ddot(local_size1,a(local_offset1,2),1,a(local_offset1,1),1)
        dot22 = ddot(local_size2,a(local_offset2,1),1,a(local_offset2,1),1)

    ! store results in work buffer
    work(1) = top11
    work(2) = dot11
    work(3) = top12
    work(4) = dot12
    work(5) = top21
    work(6) = top22
    work(7) = dot22
    work(8) = 0.0d0 ! fill up buffer

    ! exchange partial results
    call mpi_allreduce(work, seed, 8, mpi_real8, mpi_sum, &
                       mpicomm, mpierr)
end subroutine qr_pdlarfg2_1dcomm_seed

logical function qr_pdlarfg2_1dcomm_check(seed,eps)
    implicit none

    ! input variables
    double precision seed(*)
    integer eps

    ! local scalars
    double precision epsd,first,second,first_second,estimate
    logical accurate
    double precision dot11,dot12,dot22
    double precision top11,top12,top21,top22

    EPSD = EPS

    top11 = seed(1)
    dot11 = seed(2)
    top12 = seed(3)
    dot12 = seed(4)

    top21 = seed(5)
    top22 = seed(6)
    dot22 = seed(7)

    ! reconstruct the whole inner products
    ! (including squares of the top elements)
    first = dot11 + top11*top11
    second = dot22 + top22*top22 + top12*top12
    first_second = dot12 + top11*top12

    ! zero Householder vector (zero norm) case
    if (first*second .eq. 0.0d0) then
       qr_pdlarfg2_1dcomm_check = .false.
       return
    end if

    estimate = abs((first_second*first_second)/(first*second))

    !print *,'estimate:',estimate

    ! if accurate the following check holds
    accurate = (estimate .LE. (epsd/(1.0d0+epsd)))

    qr_pdlarfg2_1dcomm_check = accurate
end function qr_pdlarfg2_1dcomm_check

! id=0: first vector
! id=1: second vector
subroutine qr_pdlarfg2_1dcomm_vector(x,incx,tau,seed,n,nb,idx,id,rev,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer incx
    double precision x(*),seed(*),tau

    ! input variables (global)
    integer n,nb,idx,id,rev,mpicomm

    ! output variables (global)

    ! external functions
    double precision dlapy2
    external dlapy2,dscal

    ! local scalars
    integer mpirank,mpirank_top,mpiprocs,mpierr
    double precision alpha,dot,beta,xnorm
    integer local_size,baseoffset,local_offset,top,topidx

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    call local_size_offset_1d(n,nb,idx,idx-1,rev,mpirank,mpiprocs, &
                                  local_size,baseoffset,local_offset)

    local_offset = local_offset * incx

    ! Processor id for global index of top element
    mpirank_top = MOD((idx-1)/nb,mpiprocs)
    if (mpirank .eq. mpirank_top) then
        topidx = local_index(idx,mpirank_top,mpiprocs,nb,0)
        top = 1+(topidx-1)*incx
    end if

    alpha = seed(id*5+1)
    dot = seed(id*5+2)

    xnorm = sqrt(dot)

    if (xnorm .eq. 0.0d0) then
        ! H = I

        tau = 0.0d0
    else
        ! General case

        beta = sign(dlapy2(alpha, xnorm), alpha)
        tau = (beta+alpha) / beta

        !print *,'hg2',tau,xnorm,alpha

        call dscal(local_size, 1.0d0/(beta+alpha), &
                   x(local_offset), incx)

        ! TODO: reimplement norm rescale method of
        ! original PDLARFG using mpi?

        if (mpirank .eq. mpirank_top) then
            x(top) = -beta
        end if

        seed(8) = beta
    end if
end subroutine qr_pdlarfg2_1dcomm_vector

subroutine qr_pdlarfg2_1dcomm_update(v,incv,baseidx,a,lda,seed,n,idx,nb,rev,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer incv,lda
    double precision v(*),a(lda,*),seed(*)

    ! input variables (global)
    integer n,baseidx,idx,nb,rev,mpicomm

    ! output variables (global)

    ! external functions
    external daxpy

    ! local scalars
    integer mpirank,mpiprocs,mpierr
    integer local_size,local_offset,baseoffset
    double precision z,coeff,beta
    double precision dot11,dot12,dot22
    double precision top11,top12,top21,top22

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)


    ! seed should be updated by previous householder generation
    ! Update inner product of this column and next column vector
    top11 = seed(1)
    dot11 = seed(2)
    top12 = seed(3)
    dot12 = seed(4)

    top21 = seed(5)
    top22 = seed(6)
    dot22 = seed(7)
    beta = seed(8)

    call local_size_offset_1d(n,nb,baseidx,idx,rev,mpirank,mpiprocs, &
                              local_size,baseoffset,local_offset)
    baseoffset = baseoffset * incv

    ! zero Householder vector (zero norm) case
    if (beta .eq. 0.0d0) then
        return
    end if
    z = (dot12 + top11 * top12) / beta + top12

    !print *,'hg2 update:',baseidx,idx,mpirank,local_size

    call daxpy(local_size, -z, v(baseoffset),1, a(local_offset,1),1)

    ! prepare a full dot22 for update
    dot22 = dot22 + top22*top22

    ! calculate coefficient
    COEFF = z / (top11 + beta)

    ! update inner product of next vector
    dot22 = dot22 - coeff * (2*dot12 - coeff*dot11)

    ! update dot12 value to represent update with first vector
    ! (needed for T matrix)
    dot12 = dot12 - COEFF * dot11

    ! update top element of next vector
    top22 = top22 - coeff * top21
    seed(6) = top22

    ! restore separated dot22 for vector generation
    seed(7) = dot22  - top22*top22

    !------------------------------------------------------
    ! prepare elements for T matrix
    seed(4) = dot12

    ! prepare dot matrix for fuse element of T matrix
    ! replace top11 value with -beta1
    seed(1) = beta
end subroutine qr_pdlarfg2_1dcomm_update

! run this function after second vector
subroutine qr_pdlarfg2_1dcomm_finalize_tmatrix(seed,tau,t,ldt)
    implicit none

    integer ldt
    double precision seed(*),t(ldt,*),tau(*)
    double precision dot12,beta1,top21,beta2

    beta1 = seed(1)
    dot12 = seed(4)
    top21 = seed(5)
    beta2 = seed(8)

    !print *,'beta1 beta2',beta1,beta2

    dot12 = dot12 / beta2 + top21
    dot12 = -(dot12 / beta1)

    t(1,1) = tau(1)
    t(1,2) = dot12
    t(2,2) = tau(2)
end subroutine qr_pdlarfg2_1dcomm_finalize_tmatrix

subroutine qr_pdlarfgk_1dcomm(a,lda,tau,t,ldt,v,ldv,baseidx,work,lwork,m,k,idx,mb,PQRPARAM,rev,mpicomm,actualk)

    implicit none

    ! parameter setup

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),work(*),t(ldt,*)

    ! input variables (global)
    integer m,k,idx,baseidx,mb,rev,mpicomm
    integer PQRPARAM(*)

    ! output variables (global)
    integer actualk

    ! local scalars
    integer ivector
    double precision pdlarfg_size(1),pdlarf_size(1)
    double precision pdlarfgk_1dcomm_seed_size(1),pdlarfgk_1dcomm_check_size(1)
    double precision pdlarfgk_1dcomm_update_size(1)
    integer seedC_size,seedC_offset
    integer seedD_size,seedD_offset
    integer work_offset

    seedC_size = k*k
    seedC_offset = 1
    seedD_size = k*k
    seedD_offset = seedC_offset + seedC_size
    work_offset = seedD_offset + seedD_size

    if (lwork .eq. -1) then
        call qr_pdlarfg_1dcomm(a,1,tau(1),pdlarfg_size(1),-1,m,baseidx,mb,PQRPARAM(4),rev,mpicomm)
        call qr_pdlarfl_1dcomm(v,1,baseidx,a,lda,tau(1),pdlarf_size(1),-1,m,k,baseidx,mb,rev,mpicomm)
        call qr_pdlarfgk_1dcomm_seed(a,lda,baseidx,pdlarfgk_1dcomm_seed_size(1),-1,work,work,m,k,mb,mpicomm)
        !call qr_pdlarfgk_1dcomm_check(work,work,k,PQRPARAM,pdlarfgk_1dcomm_check_size(1),-1,actualk)
        call qr_pdlarfgk_1dcomm_check_improved(work,work,k,PQRPARAM,pdlarfgk_1dcomm_check_size(1),-1,actualk)
        call qr_pdlarfgk_1dcomm_update(a,lda,baseidx,pdlarfgk_1dcomm_update_size(1),-1,work,work,k,k,1,work,m,mb,rev,mpicomm)
        work(1) = max(pdlarfg_size(1),pdlarf_size(1),pdlarfgk_1dcomm_seed_size(1),pdlarfgk_1dcomm_check_size(1), &
                      pdlarfgk_1dcomm_update_size(1)) + DBLE(seedC_size + seedD_size);
        return
    end if

        call qr_pdlarfgk_1dcomm_seed(a(1,1),lda,idx,work(work_offset),lwork,work(seedC_offset),work(seedD_offset),m,k,mb,mpicomm)
        !call qr_pdlarfgk_1dcomm_check(work(seedC_offset),work(seedD_offset),k,PQRPARAM,work(work_offset),lwork,actualk)
        call qr_pdlarfgk_1dcomm_check_improved(work(seedC_offset),work(seedD_offset),k,PQRPARAM,work(work_offset),lwork,actualk)

        !print *,'possible rank:', actualk

        ! override useful for debugging
        !actualk = 1
        !actualk = k
        !actualk= min(actualk,2)
        do ivector=1,actualk
            call qr_pdlarfgk_1dcomm_vector(a(1,k-ivector+1),1,idx,tau(k-ivector+1), &
                                            work(seedC_offset),work(seedD_offset),k, &
                                            ivector,m,mb,rev,mpicomm)

            call qr_pdlarfgk_1dcomm_update(a(1,1),lda,idx,work(work_offset),lwork,work(seedC_offset), &
                                            work(seedD_offset),k,actualk,ivector,tau, &
                                            m,mb,rev,mpicomm)

            call qr_pdlarfg_copy_1dcomm(a(1,k-ivector+1),1, &
                                         v(1,k-ivector+1),1, &
                                         m,baseidx,idx-ivector+1,mb,1,mpicomm)
        end do

        ! generate final T matrix and convert preliminary tau values into real ones
        call qr_pdlarfgk_1dcomm_generateT(work(seedC_offset),work(seedD_offset),k,actualk,tau,t,ldt)

end subroutine qr_pdlarfgk_1dcomm

subroutine qr_pdlarfgk_1dcomm_seed(a,lda,baseidx,work,lwork,seedC,seedD,m,k,mb,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! parameter setup

    ! input variables (local)
    integer lda,lwork
    double precision a(lda,*), work(*)

    ! input variables (global)
    integer m,k,baseidx,mb,mpicomm
    double precision seedC(k,*),seedD(k,*)

    ! output variables (global)

    ! derived input variables from QR_PQRPARAM

    ! local scalars
    integer mpierr,mpirank,mpiprocs,mpirank_top
    integer icol,irow,lidx,remsize
    integer remaining_rank

    integer C_size,D_size,sendoffset,recvoffset,sendrecv_size
    integer localoffset,localsize,baseoffset

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    C_size = k*k
    D_size = k*k
    sendoffset = 1
    sendrecv_size = C_size+D_size
    recvoffset = sendoffset + sendrecv_size

    if (lwork .eq. -1) then
        work(1) = DBLE(2*sendrecv_size)
        return
    end if

    ! clear buffer
    work(sendoffset:sendoffset+sendrecv_size-1)=0.0d0

    ! collect C part
    do icol=1,k

        remaining_rank = k
        do while (remaining_rank .gt. 0)
            irow = k - remaining_rank + 1
            lidx = baseidx - remaining_rank + 1

            ! determine chunk where the current top element is located
            mpirank_top = MOD((lidx-1)/mb,mpiprocs)

            ! limit max number of remaining elements of this chunk to the block
            ! distribution parameter
            remsize = min(remaining_rank,mb)

            ! determine the number of needed elements in this chunk
            call local_size_offset_1d(lidx+remsize-1,mb, &
                                      lidx,lidx,0, &
                                      mpirank_top,mpiprocs, &
                                      localsize,baseoffset,localoffset)

            !print *,'local rank',localsize,localoffset

            if (mpirank .eq. mpirank_top) then
                ! copy elements to buffer
                work(sendoffset+(icol-1)*k+irow-1:sendoffset+(icol-1)*k+irow-1+localsize-1) &
                            = a(localoffset:localoffset+remsize-1,icol)
            end if

            ! jump to next chunk
            remaining_rank = remaining_rank - localsize
        end do
    end do

    ! collect D part
	call local_size_offset_1d(m,mb,baseidx-k,baseidx-k,1, &
							  mpirank,mpiprocs, &
							  localsize,baseoffset,localoffset)

    !print *,'localsize',localsize,localoffset
    if (localsize > 0) then
        call dsyrk("Upper", "Trans", k, localsize, &
                   1.0d0, a(localoffset,1), lda, &
                   0.0d0, work(sendoffset+C_size), k)
    else
        work(sendoffset+C_size:sendoffset+C_size+k*k-1) = 0.0d0
    end if

    ! TODO: store symmetric part more efficiently

    ! allreduce operation on results
    call mpi_allreduce(work(sendoffset),work(recvoffset),sendrecv_size, &
                       mpi_real8,mpi_sum,mpicomm,mpierr)

    ! unpack result from buffer into seedC and seedD
    seedC(1:k,1:k) = 0.0d0
    do icol=1,k
        seedC(1:k,icol) = work(recvoffset+(icol-1)*k:recvoffset+icol*k-1)
    end do

    seedD(1:k,1:k) = 0.0d0
    do icol=1,k
        seedD(1:k,icol) = work(recvoffset+C_size+(icol-1)*k:recvoffset+C_size+icol*k-1)
    end do
end subroutine qr_pdlarfgk_1dcomm_seed

! k is assumed to be larger than two
subroutine qr_pdlarfgk_1dcomm_check_improved(seedC,seedD,k,PQRPARAM,work,lwork,possiblerank)
    implicit none

    ! input variables (global)
    integer k,lwork
    integer PQRPARAM(*)
    double precision seedC(k,*),seedD(k,*),work(k,*)

    ! output variables (global)
    integer possiblerank

    ! derived input variables from QR_PQRPARAM
    integer eps

    ! local variables
    integer i,j,l
    double precision sum_squares,diagonal_square,relative_error,epsd,diagonal_root
    double precision dreverse_matrix_work(1)

    ! external functions
    double precision ddot,dlapy2,dnrm2
    external ddot,dscal,dlapy2,dnrm2

    if (lwork .eq. -1) then
        call reverse_matrix_local(1,k,k,work,k,dreverse_matrix_work,-1)
        work(1,1) = DBLE(k*k) + dreverse_matrix_work(1)
        return
    end if

    eps = PQRPARAM(3)

    if (eps .eq. 0) then
        possiblerank = k
        return
    end if

    epsd = DBLE(eps)

    ! build complete inner product from seedC and seedD
    ! copy seedD to work
    work(:,1:k) = seedD(:,1:k)

    ! add inner products of seedC to work
    call dsyrk("Upper", "Trans", k, k, &
               1.0d0, seedC(1,1), k, &
               1.0d0, work, k)

	! TODO: optimize this part!
	call reverse_matrix_local(0,k,k,work(1,1),k,work(1,k+1),lwork-2*k)
	call reverse_matrix_local(1,k,k,work(1,1),k,work(1,k+1),lwork-2*k)

    ! transpose matrix
	do i=1,k
	  do j=i+1,k
	    work(i,j) = work(j,i)
	  end do
	end do


    ! do cholesky decomposition
    i = 0
    do while ((i .lt. k))
      i = i + 1

      diagonal_square = abs(work(i,i))
      diagonal_root  = sqrt(diagonal_square)

      ! zero Householder vector (zero norm) case
      if ((abs(diagonal_square) .eq. 0.0d0) .or. (abs(diagonal_root) .eq. 0.0d0)) then
        possiblerank = max(i-1,1)
        return
      end if

      ! check if relative error is bounded for each Householder vector
      ! Householder i is stable iff Househoulder i-1 is "stable" and the accuracy criterion
      ! holds.
      ! first Householder vector is considered as "stable".

      do j=i+1,k
          work(i,j) = work(i,j) / diagonal_root
          do l=i+1,j
              work(l,j) = work(l,j) - work(i,j) * work(i,l)
          end do
      end do
      !print *,'cholesky step done'

      ! build sum of squares
      if(i .eq. 1) then
        sum_squares = 0.0d0
      else
        sum_squares = ddot(i-1,work(1,i),1,work(1,i),1)
      end if
      !relative_error = sum_squares / diagonal_square
      !print *,'error ',i,sum_squares,diagonal_square,relative_error

      if (sum_squares .ge. (epsd * diagonal_square)) then
        possiblerank = max(i-1,1)
        return
      end if
    end do

    possiblerank = i
    !print *,'possible rank', possiblerank
end subroutine qr_pdlarfgk_1dcomm_check_improved

! TODO: zero Householder vector (zero norm) case
! - check alpha values as well (from seedC)
subroutine qr_pdlarfgk_1dcomm_check(seedC,seedD,k,PQRPARAM,work,lwork,possiblerank)
    use qr_utils_mod

    implicit none

    ! parameter setup

    ! input variables (local)

    ! input variables (global)
    integer k,lwork
    integer PQRPARAM(*)
    double precision seedC(k,*),seedD(k,*),work(k,*)

    ! output variables (global)
    integer possiblerank

    ! derived input variables from QR_PQRPARAM
    integer eps

    ! local scalars
    integer icol,isqr,iprod
    double precision epsd,sum_sqr,sum_products,diff,temp,ortho,ortho_sum
    double precision dreverse_matrix_work(1)

    if (lwork .eq. -1) then
        call reverse_matrix_local(1,k,k,work,k,dreverse_matrix_work,-1)
        work(1,1) = DBLE(k*k) + dreverse_matrix_work(1)
        return
    end if

    eps = PQRPARAM(3)

    if (eps .eq. 0) then
        possiblerank = k
        return
    end if

    epsd = DBLE(eps)


    ! copy seedD to work
    work(:,1:k) = seedD(:,1:k)

    ! add inner products of seedC to work
    call dsyrk("Upper", "Trans", k, k, &
               1.0d0, seedC(1,1), k, &
               1.0d0, work, k)

	! TODO: optimize this part!
	call reverse_matrix_local(0,k,k,work(1,1),k,work(1,k+1),lwork-2*k)
	call reverse_matrix_local(1,k,k,work(1,1),k,work(1,k+1),lwork-2*k)

	! transpose matrix
	do icol=1,k
		do isqr=icol+1,k
			work(icol,isqr) = work(isqr,icol)
		end do
	end do

    ! work contains now the full inner product of the global (sub-)matrix
    do icol=1,k
        ! zero Householder vector (zero norm) case
        if (abs(work(icol,icol)) .eq. 0.0d0) then
            !print *,'too small ', icol, work(icol,icol)
            possiblerank = max(icol,1)
            return
        end if

        sum_sqr = 0.0d0
        do isqr=1,icol-1
            sum_products = 0.0d0
            do iprod=1,isqr-1
                sum_products = sum_products + work(iprod,isqr)*work(iprod,icol)
            end do

            !print *,'divisor',icol,isqr,work(isqr,isqr)
            temp = (work(isqr,icol) - sum_products)/work(isqr,isqr)
            work(isqr,icol) = temp
            sum_sqr = sum_sqr + temp*temp
        end do

        ! calculate diagonal value
        diff = work(icol,icol) - sum_sqr
        if (diff .lt. 0.0d0) then
            ! we definitely have a problem now
            possiblerank = icol-1 ! only decompose to previous column (including)
            return
        end if
        work(icol,icol) = sqrt(diff)

        ! calculate orthogonality
        ortho = 0.0d0
        do isqr=1,icol-1
            ortho_sum = 0.0d0
            do iprod=isqr,icol-1
                temp = work(isqr,iprod)*work(isqr,iprod)
                !print *,'ortho ', work(iprod,iprod)
                temp = temp / (work(iprod,iprod)*work(iprod,iprod))
                ortho_sum = ortho_sum + temp
            end do
            ortho = ortho + ortho_sum * (work(isqr,icol)*work(isqr,icol))
        end do

        ! ---------------- with division by zero ----------------------- !

        !ortho = ortho / diff;

        ! if current estimate is not accurate enough, the following check holds
        !if (ortho .gt. epsd) then
        !    possiblerank = icol-1 ! only decompose to previous column (including)
        !    return
        !end if

        ! ---------------- without division by zero ----------------------- !

        ! if current estimate is not accurate enough, the following check holds
        if (ortho .gt. epsd * diff) then
            possiblerank = icol-1 ! only decompose to previous column (including)
            return
        end if
    end do

    ! if we get to this point, the accuracy condition holds for the whole block
    possiblerank = k
end subroutine qr_pdlarfgk_1dcomm_check

!sidx: seed idx
!k: max rank used during seed phase
!rank: actual rank (k >= rank)
subroutine qr_pdlarfgk_1dcomm_vector(x,incx,baseidx,tau,seedC,seedD,k,sidx,n,nb,rev,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer incx
    double precision x(*),tau

    ! input variables (global)
    integer n,nb,baseidx,rev,mpicomm,k,sidx
    double precision seedC(k,*),seedD(k,*)

    ! output variables (global)

    ! external functions
    double precision dlapy2,dnrm2
    external dlapy2,dscal,dnrm2

    ! local scalars
    integer mpirank,mpirank_top,mpiprocs,mpierr
    double precision alpha,dot,beta,xnorm
    integer local_size,baseoffset,local_offset,top,topidx
    integer lidx

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

	lidx = baseidx-sidx+1
	call local_size_offset_1d(n,nb,baseidx,lidx-1,rev,mpirank,mpiprocs, &
							  local_size,baseoffset,local_offset)

    local_offset = local_offset * incx

    ! Processor id for global index of top element
    mpirank_top = MOD((lidx-1)/nb,mpiprocs)
    if (mpirank .eq. mpirank_top) then
        topidx = local_index((lidx),mpirank_top,mpiprocs,nb,0)
        top = 1+(topidx-1)*incx
    end if

	alpha = seedC(k-sidx+1,k-sidx+1)
	dot = seedD(k-sidx+1,k-sidx+1)
	! assemble actual norm from both seed parts
	xnorm = dlapy2(sqrt(dot), dnrm2(k-sidx,seedC(1,k-sidx+1),1))

    if (xnorm .eq. 0.0d0) then
        tau = 0.0d0
    else
        ! General case

        beta = sign(dlapy2(alpha, xnorm), alpha)
        ! store a preliminary version of beta in tau
        tau = beta

        ! update global part
        call dscal(local_size, 1.0d0/(beta+alpha), &
                   x(local_offset), incx)

        ! do not update local part here due to
        ! dependency of c vector during update process

        ! TODO: reimplement norm rescale method of
        ! original PDLARFG using mpi?

        if (mpirank .eq. mpirank_top) then
            x(top) = -beta
        end if
    end if

end subroutine qr_pdlarfgk_1dcomm_vector

!k: original max rank used during seed function
!rank: possible rank as from check function
! TODO: if rank is less than k, reduce buffersize in such a way
! that only the required entries for the next pdlarfg steps are
! computed
subroutine qr_pdlarfgk_1dcomm_update(a,lda,baseidx,work,lwork,seedC,seedD,k,rank,sidx,tau,n,nb,rev,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! parameter setup
    INTEGER     gmode_,rank_,eps_,upmode1_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3, upmode1_=4)

    ! input variables (local)
    integer lda,lwork
    double precision a(lda,*),work(*)

    ! input variables (global)
    integer k,rank,sidx,n,baseidx,nb,rev,mpicomm
    double precision beta

    ! output variables (global)
    double precision seedC(k,*),seedD(k,*),tau(*)

    ! derived input variables from QR_PQRPARAM

    ! local scalars
    double precision alpha
    integer coffset,zoffset,yoffset,voffset,buffersize
    integer mpirank,mpierr,mpiprocs,mpirank_top
    integer localsize,baseoffset,localoffset,topidx
    integer lidx

    if (lwork .eq. -1) then
        ! buffer for c,z,y,v
        work(1) = 4*k
        return
    end if

    ! nothing to update anymore
    if (sidx .gt. rank) return

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

	lidx = baseidx-sidx
	if (lidx .lt. 1) return

    call local_size_offset_1d(n,nb,baseidx,lidx,rev,mpirank,mpiprocs, &
                              localsize,baseoffset,localoffset)

    coffset = 1
    zoffset = coffset + k
    yoffset = zoffset + k
    voffset = yoffset + k
    buffersize = k - sidx

    ! finalize tau values
	alpha = seedC(k-sidx+1,k-sidx+1)
	beta = tau(k-sidx+1)

    ! zero Householder vector (zero norm) case
    !print *,'k update: alpha,beta',alpha,beta
    if ((beta .eq. 0.0d0) .or. (alpha .eq. 0.0d0))  then
        tau(k-sidx+1) = 0.0d0
        seedC(k,k-sidx+1) = 0.0d0
        return
    end if

    tau(k-sidx+1) = (beta+alpha) / beta

    ! ---------------------------------------
        ! calculate c vector (extra vector or encode in seedC/seedD?
        work(coffset:coffset+buffersize-1) = seedD(1:buffersize,k-sidx+1)
        call dgemv("Trans", buffersize+1, buffersize, &
               1.0d0,seedC(1,1),k,seedC(1,k-sidx+1),1, &
               1.0d0,work(coffset),1)

        ! calculate z using tau,seedD,seedC and c vector
        work(zoffset:zoffset+buffersize-1) = seedC(k-sidx+1,1:buffersize)
        call daxpy(buffersize, 1.0d0/beta, work(coffset), 1, work(zoffset), 1)

        ! update A1(local copy) and generate part of householder vectors for use
        call daxpy(buffersize, -1.0d0, work(zoffset),1,seedC(k-sidx+1,1),k)
        call dscal(buffersize, 1.0d0/(alpha+beta), seedC(1,k-sidx+1),1)
        call dger(buffersize, buffersize, -1.0d0, seedC(1,k-sidx+1),1, work(zoffset), 1, seedC(1,1), k)

        ! update A global (householder vector already generated by pdlarfgk)
        mpirank_top = MOD(lidx/nb,mpiprocs)
        if (mpirank .eq. mpirank_top) then
            ! handle first row separately
            topidx = local_index(lidx+1,mpirank_top,mpiprocs,nb,0)
            call daxpy(buffersize,-1.0d0,work(zoffset),1,a(topidx,1),lda)
        end if

        call dger(localsize, buffersize,-1.0d0, &
              a(localoffset,k-sidx+1),1,work(zoffset),1, &
              a(localoffset,1),lda)

        ! update D (symmetric) => two buffer vectors of size rank
        ! generate y vector
        work(yoffset:yoffset+buffersize-1) = 0.d0
        call daxpy(buffersize,1.0d0/(alpha+beta),work(zoffset),1,work(yoffset),1)

        ! generate v vector
        work(voffset:voffset+buffersize-1) = seedD(1:buffersize,k-sidx+1)
        call daxpy(buffersize, -0.5d0*seedD(k-sidx+1,k-sidx+1), work(yoffset), 1, work(voffset),1)

        ! symmetric update of D using y and v
        call dsyr2("Upper", buffersize,-1.0d0, &
                   work(yoffset),1,work(voffset),1, &
                   seedD(1,1), k)

    ! prepare T matrix inner products
    ! D_k(1:k,k+1:n) = D_(k-1)(1:k,k+1:n) - D_(k-1)(1:k,k) * y'
    ! store coefficient 1.0d0/(alpha+beta) in C diagonal elements
	call dger(k-sidx,sidx,-1.0d0,work(yoffset),1,seedD(k-sidx+1,k-sidx+1),k,seedD(1,k-sidx+1),k)
	seedC(k,k-sidx+1) = 1.0d0/(alpha+beta)

end subroutine qr_pdlarfgk_1dcomm_update


subroutine qr_pdlarfgk_1dcomm_generateT(seedC,seedD,k,actualk,tau,t,ldt)
    implicit none

    integer k,actualk,ldt
    double precision seedC(k,*),seedD(k,*),tau(*),t(ldt,*)

    integer irow,icol
    double precision column_coefficient

        !print *,'reversed on the fly T generation NYI'

        do icol=1,actualk-1
            ! calculate inner product of householder vector parts in seedC
            ! (actually calculating more than necessary, if actualk < k)
            ! => a lot of junk from row 1 to row k-actualk
            call dtrmv('Upper','Trans','Unit',k-icol,seedC(1,1),k,seedC(1,k-icol+1),1)

            ! add scaled D parts to current column of C (will become later T rows)
            column_coefficient = seedC(k,k-icol+1)
            do irow=k-actualk+1,k-1
                seedC(irow,k-icol+1) = ( seedC(irow,k-icol+1) ) +  ( seedD(irow,k-icol+1) * column_coefficient * seedC(k,irow) )
            end do
        end do

        call qr_dlarft_kernel(actualk,tau(k-actualk+1),seedC(k-actualk+1,k-actualk+2),k,t(k-actualk+1,k-actualk+1),ldt)

end subroutine qr_pdlarfgk_1dcomm_generateT

!direction=0: pack into work buffer
!direction=1: unpack from work buffer
subroutine qr_pdgeqrf_pack_unpack(v,ldv,work,lwork,m,n,mb,baseidx,rowidx,rev,direction,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer ldv,lwork
    double precision v(ldv,*), work(*)

    ! input variables (global)
    integer m,n,mb,baseidx,rowidx,rev,direction,mpicomm

    ! output variables (global)

    ! local scalars
    integer mpierr,mpirank,mpiprocs
    integer buffersize,icol
    integer local_size,baseoffset,offset

    ! external functions

    call mpi_comm_rank(mpicomm,mpirank,mpierr)
    call mpi_comm_size(mpicomm,mpiprocs,mpierr)

    call local_size_offset_1d(m,mb,baseidx,rowidx,rev,mpirank,mpiprocs, &
                                  local_size,baseoffset,offset)

    !print *,'pack/unpack',local_size,baseoffset,offset

    ! rough approximate for buffer size
    if (lwork .eq. -1) then
        buffersize = local_size * n ! vector elements
        work(1) = DBLE(buffersize)
        return
    end if

    if (direction .eq. 0) then
        ! copy v part to buffer (including zeros)
        do icol=1,n
            work(1+local_size*(icol-1):local_size*icol) = v(baseoffset:baseoffset+local_size-1,icol)
        end do
    else
        ! copy v part from buffer (including zeros)
        do icol=1,n
            v(baseoffset:baseoffset+local_size-1,icol) = work(1+local_size*(icol-1):local_size*icol)
        end do
    end if

    return

end subroutine qr_pdgeqrf_pack_unpack

!direction=0: pack into work buffer
!direction=1: unpack from work buffer
subroutine qr_pdgeqrf_pack_unpack_tmatrix(tau,t,ldt,work,lwork,n,direction)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer ldt,lwork
    double precision work(*), t(ldt,*),tau(*)

    ! input variables (global)
    integer n,direction

    ! output variables (global)

    ! local scalars
    integer icol

    ! external functions

    if (lwork .eq. -1) then
        work(1) = DBLE(n*n)
        return
    end if

    if (direction .eq. 0) then
        ! append t matrix to buffer (including zeros)
        do icol=1,n
            work(1+(icol-1)*n:icol*n) = t(1:n,icol)
        end do
    else
        ! append t matrix from buffer (including zeros)
        do icol=1,n
            t(1:n,icol) = work(1+(icol-1)*n:icol*n)
            tau(icol) = t(icol,icol)
        end do
    end if

end subroutine qr_pdgeqrf_pack_unpack_tmatrix


! TODO: encode following functionality
!   - Direction? BOTTOM UP or TOP DOWN ("Up", "Down")
!        => influences all related kernels (including DLARFT / DLARFB)
!   - rank-k parameter (k=1,2,...,b)
!        => influences possible update strategies
!        => parameterize the function itself? (FUNCPTR, FUNCARG)
!   - Norm mode? Allreduce, Allgather, AlltoAll, "AllHouse", (ALLNULL = benchmarking local kernels)
!   - subblocking
!         (maximum block size bounded by data distribution along rows)
!   - blocking method (householder vectors only or compact WY?)
!   - update strategy of trailing parts (incremental, complete)
!        - difference for subblocks and normal blocks? (UPDATE and UPDATESUB)
!        o "Incremental"
!        o "Full"
!   - final T generation (recursive: subblock wise, block wise, end) (TMERGE)
!        ' (implicitly given by / influences update strategies?)
!        => alternative: during update: iterate over sub t parts
!           => advantage: smaller (cache aware T parts)
!           => disadvantage: more memory write backs
!                (number of T parts * matrix elements)
!   - partial/sub T generation (TGEN)
!        o add vectors right after creation (Vector)
!        o add set of vectors (Set)
!   - bcast strategy of householder vectors to other process columns
!        (influences T matrix generation and trailing update
!         in other process columns)
!        o no broadcast (NONE = benchmarking?,
!            or not needed due to 1D process grid)
!        o after every housegen (VECTOR)
!        o after every subblk   (SUBBLOCK)
!        o after full local column block decomposition (BLOCK)
!  LOOP Housegen -> BCAST -> GENT/EXTENDT -> LOOP HouseLeft

!subroutine qr_pqrparam_init(PQRPARAM, DIRECTION, RANK, NORMMODE, &
!                             SUBBLK, UPDATE, TGEN, BCAST)

! gmode: control communication pattern of dlarfg
! maxrank: control max number of householder vectors per communication
! eps: error threshold (integer)
! update*: control update pattern in pdgeqr2_1dcomm ('incremental','full','merge')
!               merging = full update with tmatrix merging
! tmerge*: 0: do not merge, 1: incremental merge, >1: recursive merge
!               only matters if update* == full
subroutine qr_pqrparam_init(pqrparam,size2d,update2d,tmerge2d,size1d,update1d,tmerge1d,maxrank,update,eps,hgmode)

    implicit none

    ! input
    CHARACTER   update2d,update1d,update,hgmode
    INTEGER     size2d,size1d,maxrank,eps,tmerge2d,tmerge1d

    ! output
    INTEGER     PQRPARAM(*)

    PQRPARAM(1) = size2d
    PQRPARAM(2) = ichar(update2d)
    PQRPARAM(3) = tmerge2d
    ! TODO: broadcast T yes/no

    PQRPARAM(4) = size1d
    PQRPARAM(5) = ichar(update1d)
    PQRPARAM(6) = tmerge1d

    PQRPARAM(7) = maxrank
    PQRPARAM(8) = ichar(update)
    PQRPARAM(9) = eps
    PQRPARAM(10) = ichar(hgmode)

end subroutine qr_pqrparam_init


subroutine qr_pdlarfg_copy_1dcomm(x,incx,v,incv,n,baseidx,idx,nb,rev,mpicomm)
    use ELPA1
    use qr_utils_mod

    implicit none

    ! input variables (local)
    integer incx,incv
    double precision x(*), v(*)

    ! input variables (global)
    integer baseidx,idx,rev,nb,n
    integer mpicomm

    ! output variables (global)

    ! local scalars
    integer mpierr,mpiprocs
    integer mpirank,mpirank_top
    integer irow,x_offset
    integer v_offset,local_size


    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    call local_size_offset_1d(n,nb,baseidx,idx,rev,mpirank,mpiprocs, &
                              local_size,v_offset,x_offset)
    v_offset = v_offset * incv

    !print *,'copy:',mpirank,baseidx,v_offset,x_offset,local_size

    ! copy elements
    do irow=1,local_size
        v((irow-1)*incv+v_offset) = x((irow-1)*incx+x_offset)
    end do

    ! replace top element to build an unitary vector
    mpirank_top = MOD((idx-1)/nb,mpiprocs)
    if (mpirank .eq. mpirank_top) then
        v(local_size*incv) = 1.0d0
    end if

end subroutine qr_pdlarfg_copy_1dcomm

end module elpa_pdgeqrf
