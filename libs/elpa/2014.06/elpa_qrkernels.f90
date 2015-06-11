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
! calculates A = A - Y*T'*Z (rev=0)
! calculates A = A - Y*T*Z (rev=1)
! T upper triangle matrix
! assuming zero entries in matrix in upper kxk block
subroutine qr_pdlarfb_kernel_local(m,n,k,a,lda,v,ldv,t,ldt,z,ldz)
    implicit none
 
    ! input variables (local)
    integer lda,ldv,ldt,ldz
    double precision a(lda,*),v(ldv,*),t(ldt,*),z(ldz,*)

    ! input variables (global)
    integer m,n,k

    ! local variables
    double precision t11
    double precision t12,t22,sum1,sum2
    double precision t13,t23,t33,sum3
    double precision sum4,t44
    double precision y1,y2,y3,y4
    double precision a1
    integer icol,irow,v1col,v2col,v3col
  
    ! reference implementation
    if (k .eq. 1) then
        t11 = t(1,1)
        do icol=1,n
            sum1 = z(1,icol)
            a(1:m,icol) = a(1:m,icol) - t11*sum1*v(1:m,1)
        enddo
        return
    else if (k .eq. 2) then
            v1col = 2
            v2col = 1
            t22 = t(1,1)
            t12 = t(1,2)
            t11 = t(2,2)

        do icol=1,n
            sum1 = t11 * z(v1col,icol)
            sum2 = t12 * z(v1col,icol) + t22 * z(v2col,icol)

            do irow=1,m
                a(irow,icol) = a(irow,icol) - v(irow,v1col) * sum1 - v(irow,v2col) * sum2
            end do
        end do
    else if (k .eq. 3) then
            v1col = 3
            v2col = 2
            v3col = 1

            t33 = t(1,1)

            t23 = t(1,2)
            t22 = t(2,2)

            t13 = t(1,3)
            t12 = t(2,3)
            t11 = t(3,3)
 
        do icol=1,n
            ! misusing variables for fetch of z parts
            y1 = z(v1col,icol)
            y2 = z(v2col,icol)
            y3 = z(v3col,icol)

            sum1 = t11 * y1!+ 0   * y2!+ 0   * y3
            sum2 = t12 * y1 + t22 * y2!+ 0   * y3
            sum3 = t13 * y1 + t23 * y2 + t33 * y3

            do irow=1,m
                a(irow,icol) = a(irow,icol) - v(irow,v1col) * sum1 - v(irow,v2col) * sum2 - v(irow,v3col) * sum3
            end do
        end do
    else if (k .eq. 4) then
            do icol=1,n
                ! misusing variables for fetch of z parts
                y1 = z(1,icol)
                y2 = z(2,icol)
                y3 = z(3,icol)
                y4 = z(4,icol)

                ! dtrmv like - starting from main diagonal and working
                ! upwards
                t11 = t(1,1)
                t22 = t(2,2)
                t33 = t(3,3)
                t44 = t(4,4)
                
                sum1 = t11 * y1
                sum2 = t22 * y2
                sum3 = t33 * y3
                sum4 = t44 * y4
 
                t11 = t(1,2)
                t22 = t(2,3)
                t33 = t(3,4)
 
                sum1 = sum1 + t11 * y2
                sum2 = sum2 + t22 * y3
                sum3 = sum3 + t33 * y4
  
                t11 = t(1,3)
                t22 = t(2,4)
 
                sum1 = sum1 + t11 * y3
                sum2 = sum2 + t22 * y4
  
                t11 = t(1,4)
                sum1 = sum1 + t11 * y4
 
                ! one column of V is calculated 
                ! time to calculate A - Y * V
                do irow=1,m ! TODO: loop unrolling
                    y1 = v(irow,1)
                    y2 = v(irow,2)
                    y3 = v(irow,3)
                    y4 = v(irow,4)

                    a1 = a(irow,icol)

                    a1 = a1 - y1*sum1
                    a1 = a1 - y2*sum2 
                    a1 = a1 - y3*sum3
                    a1 = a1 - y4*sum4

                    a(irow,icol) = a1
                end do
            end do
    else
        ! reference implementation
            ! V' = T * Z'
            call dtrmm("Left","Upper","Notrans","Nonunit",k,n,1.0d0,t,ldt,z,ldz)
            ! A = A - Y * V'
            call dgemm("Notrans","Notrans",m,n,k,-1.0d0,v,ldv,z,ldz,1.0d0,a,lda)
    end if

end subroutine
subroutine qr_pdlarft_merge_kernel_local(oldk,k,t,ldt,yty,ldy)
    implicit none

    ! input variables (local)
    integer ldt,ldy
    double precision t(ldt,*),yty(ldy,*)

    ! input variables (global)
    integer k,oldk
 
    ! output variables (global)

    ! local scalars
    integer icol,leftk,rightk

    ! local scalars for optimized versions
    integer irow
    double precision t11
    double precision yty1,yty2,yty3,yty4,yty5,yty6,yty7,yty8
    double precision reg01,reg02,reg03,reg04,reg05,reg06,reg07,reg08
    double precision final01,final02,final03,final04,final05,final06,final07,final08

    if (oldk .eq. 0) return ! nothing to be done

        leftk = k
        rightk = oldk
     
    ! optimized implementations:
    if (leftk .eq. 1) then
        do icol=1,rightk
            ! multiply inner products with right t matrix
            ! (dtrmv like)
            yty1 = yty(1,1)
            t11 = t(leftk+1,leftk+icol)

            reg01 = yty1 * t11

            do irow=2,icol
                yty1 = yty(1,irow)
                t11 = t(leftk+irow,leftk+icol)

                reg01 = reg01 + yty1 * t11
            end do

            ! multiply intermediate results with left t matrix and store in final t
            ! matrix
            t11 = -t(1,1)
            final01 = t11 * reg01
            t(1,leftk+icol) = final01
        end do

        !print *,'efficient tmerge - leftk=1'
    else if (leftk .eq. 2) then
        do icol=1,rightk
            ! multiply inner products with right t matrix
            ! (dtrmv like)
            yty1 = yty(1,1)
            yty2 = yty(2,1)

            t11  = t(leftk+1,leftk+icol)

            reg01 = yty1 * t11
            reg02 = yty2 * t11

            do irow=2,icol
                yty1 = yty(1,irow)
                yty2 = yty(2,irow)
                t11 = t(leftk+irow,leftk+icol)

                reg01 = reg01 + yty1 * t11
                reg02 = reg02 + yty2 * t11
            end do

            ! multiply intermediate results with left t matrix and store in final t
            ! matrix
            yty1 = -t(1,1)
            yty2 = -t(1,2)
            yty3 = -t(2,2)

            final01 = reg02 * yty2
            final02 = reg02 * yty3

            final01 = final01 + reg01 * yty1

            t(1,leftk+icol) = final01
            t(2,leftk+icol) = final02
        end do
 
        !print *,'efficient tmerge - leftk=2'
    else if (leftk .eq. 4) then
        do icol=1,rightk
            ! multiply inner products with right t matrix
            ! (dtrmv like)
            yty1 = yty(1,1)
            yty2 = yty(2,1)
            yty3 = yty(3,1)
            yty4 = yty(4,1)

            t11  = t(leftk+1,leftk+icol)

            reg01 = yty1 * t11
            reg02 = yty2 * t11
            reg03 = yty3 * t11
            reg04 = yty4 * t11

            do irow=2,icol
                yty1 = yty(1,irow)
                yty2 = yty(2,irow)
                yty3 = yty(3,irow)
                yty4 = yty(4,irow)

                t11 = t(leftk+irow,leftk+icol)

                reg01 = reg01 + yty1 * t11
                reg02 = reg02 + yty2 * t11
                reg03 = reg03 + yty3 * t11
                reg04 = reg04 + yty4 * t11
            end do

            ! multiply intermediate results with left t matrix and store in final t
            ! matrix (start from diagonal and move upwards)
            yty1 = -t(1,1)
            yty2 = -t(2,2)
            yty3 = -t(3,3)
            yty4 = -t(4,4)

            ! main diagonal
            final01 = reg01 * yty1
            final02 = reg02 * yty2
            final03 = reg03 * yty3
            final04 = reg04 * yty4

            ! above main diagonal
            yty1 = -t(1,2)
            yty2 = -t(2,3)
            yty3 = -t(3,4)

            final01 = final01 + reg02 * yty1
            final02 = final02 + reg03 * yty2
            final03 = final03 + reg04 * yty3

            ! above first side diagonal
            yty1 = -t(1,3)
            yty2 = -t(2,4)

            final01 = final01 + reg03 * yty1
            final02 = final02 + reg04 * yty2

            ! above second side diagonal
            yty1 = -t(1,4)

            final01 = final01 + reg04 * yty1

            ! write back to final matrix
            t(1,leftk+icol) = final01
            t(2,leftk+icol) = final02
            t(3,leftk+icol) = final03
            t(4,leftk+icol) = final04
        end do
 
        !print *,'efficient tmerge - leftk=4'
    else if (leftk .eq. 8) then
        do icol=1,rightk
            ! multiply inner products with right t matrix
            ! (dtrmv like)
            yty1 = yty(1,1)
            yty2 = yty(2,1)
            yty3 = yty(3,1)
            yty4 = yty(4,1)
            yty5 = yty(5,1)
            yty6 = yty(6,1)
            yty7 = yty(7,1)
            yty8 = yty(8,1)

            t11  = t(leftk+1,leftk+icol)

            reg01 = yty1 * t11
            reg02 = yty2 * t11
            reg03 = yty3 * t11
            reg04 = yty4 * t11
            reg05 = yty5 * t11
            reg06 = yty6 * t11
            reg07 = yty7 * t11
            reg08 = yty8 * t11

            do irow=2,icol
                yty1 = yty(1,irow)
                yty2 = yty(2,irow)
                yty3 = yty(3,irow)
                yty4 = yty(4,irow)
                yty5 = yty(5,irow)
                yty6 = yty(6,irow)
                yty7 = yty(7,irow)
                yty8 = yty(8,irow)

                t11 = t(leftk+irow,leftk+icol)

                reg01 = reg01 + yty1 * t11
                reg02 = reg02 + yty2 * t11
                reg03 = reg03 + yty3 * t11
                reg04 = reg04 + yty4 * t11
                reg05 = reg05 + yty5 * t11
                reg06 = reg06 + yty6 * t11
                reg07 = reg07 + yty7 * t11
                reg08 = reg08 + yty8 * t11
            end do

            ! multiply intermediate results with left t matrix and store in final t
            ! matrix (start from diagonal and move upwards)
            yty1 = -t(1,1)
            yty2 = -t(2,2)
            yty3 = -t(3,3)
            yty4 = -t(4,4)
            yty5 = -t(5,5)
            yty6 = -t(6,6)
            yty7 = -t(7,7)
            yty8 = -t(8,8)

            ! main diagonal
            final01 = reg01 * yty1
            final02 = reg02 * yty2
            final03 = reg03 * yty3
            final04 = reg04 * yty4
            final05 = reg05 * yty5
            final06 = reg06 * yty6
            final07 = reg07 * yty7
            final08 = reg08 * yty8

            ! above main diagonal
            yty1 = -t(1,2)
            yty2 = -t(2,3)
            yty3 = -t(3,4)
            yty4 = -t(4,5)
            yty5 = -t(5,6)
            yty6 = -t(6,7)
            yty7 = -t(7,8)

            final01 = final01 + reg02 * yty1
            final02 = final02 + reg03 * yty2
            final03 = final03 + reg04 * yty3
            final04 = final04 + reg05 * yty4
            final05 = final05 + reg06 * yty5
            final06 = final06 + reg07 * yty6
            final07 = final07 + reg08 * yty7

            ! above first side diagonal
            yty1 = -t(1,3)
            yty2 = -t(2,4)
            yty3 = -t(3,5)
            yty4 = -t(4,6)
            yty5 = -t(5,7)
            yty6 = -t(6,8)

            final01 = final01 + reg03 * yty1
            final02 = final02 + reg04 * yty2
            final03 = final03 + reg05 * yty3
            final04 = final04 + reg06 * yty4
            final05 = final05 + reg07 * yty5
            final06 = final06 + reg08 * yty6

            !above second side diagonal

            yty1 = -t(1,4)
            yty2 = -t(2,5)
            yty3 = -t(3,6)
            yty4 = -t(4,7)
            yty5 = -t(5,8)

            final01 = final01 + reg04 * yty1
            final02 = final02 + reg05 * yty2
            final03 = final03 + reg06 * yty3
            final04 = final04 + reg07 * yty4
            final05 = final05 + reg08 * yty5

            ! i think you got the idea by now
 
            yty1 = -t(1,5)
            yty2 = -t(2,6)
            yty3 = -t(3,7)
            yty4 = -t(4,8)

            final01 = final01 + reg05 * yty1
            final02 = final02 + reg06 * yty2
            final03 = final03 + reg07 * yty3
            final04 = final04 + reg08 * yty4

            ! .....

            yty1 = -t(1,6)
            yty2 = -t(2,7)
            yty3 = -t(3,8)

            final01 = final01 + reg06 * yty1
            final02 = final02 + reg07 * yty2
            final03 = final03 + reg08 * yty3

            ! .....

            yty1 = -t(1,7)
            yty2 = -t(2,8)

            final01 = final01 + reg07 * yty1
            final02 = final02 + reg08 * yty2
 
            ! .....

            yty1 = -t(1,8)

            final01 = final01 + reg08 * yty1

            ! write back to final matrix
            t(1,leftk+icol) = final01
            t(2,leftk+icol) = final02
            t(3,leftk+icol) = final03
            t(4,leftk+icol) = final04
            t(5,leftk+icol) = final05
            t(6,leftk+icol) = final06
            t(7,leftk+icol) = final07
            t(8,leftk+icol) = final08
        end do

        !print *,'efficient tmerge - leftk=8'
    else
        ! reference implementation
        do icol=1,rightk
            t(1:leftk,leftk+icol) = yty(1:leftk,icol)
        end do
            
        ! -T1 * Y1'*Y2
        call dtrmm("Left","Upper","Notrans","Nonunit",leftk,rightk,-1.0d0,t(1,1),ldt,t(1,leftk+1),ldt)
        ! (-T1 * Y1'*Y2) * T2
        call dtrmm("Right","Upper","Notrans","Nonunit",leftk,rightk,1.0d0,t(leftk+1,leftk+1),ldt,t(1,leftk+1),ldt)
    end if

end subroutine
! yty structure
! Y1'*Y2   Y1'*Y3  Y1'*Y4 ...
!    0     Y2'*Y3  Y2'*Y4 ...
!    0        0    Y3'*Y4 ...
!    0        0       0   ...
subroutine qr_tmerge_set_kernel(k,blocksize,t,ldt,yty,ldy)
    implicit none
 
    ! input variables (local)
    integer ldt,ldy
    double precision t(ldt,*),yty(ldy,*)

    ! input variables (global)
    integer k,blocksize
 
    ! output variables (global)

    ! local scalars
    integer nr_blocks,current_block
    integer remainder,oldk
    integer yty_column,toffset
  
    if (k .le. blocksize) return ! nothing to merge

    nr_blocks = k / blocksize
    remainder = k - nr_blocks*blocksize

        ! work in "negative" direction:
        ! start with latest T matrix part and add older ones
        toffset = 1
        yty_column = 1
 
        if (remainder .gt. 0) then
            call qr_pdlarft_merge_kernel_local(blocksize,remainder,t(toffset,toffset),ldt,yty(1,yty_column),ldy)
            current_block = 1
            oldk = remainder+blocksize
            yty_column =  yty_column + blocksize
        else
            call qr_pdlarft_merge_kernel_local(blocksize,blocksize,t(toffset,toffset),ldt,yty(1,yty_column),ldy)
            current_block = 2
            oldk = 2*blocksize
            yty_column = yty_column + blocksize
        end if
 
        do while (current_block .lt. nr_blocks)
            call qr_pdlarft_merge_kernel_local(blocksize,oldk,t(toffset,toffset),ldt,yty(toffset,yty_column),ldy)

            current_block = current_block + 1
            oldk = oldk + blocksize
            yty_column = yty_column + blocksize
        end do

end subroutine
! yty structure
! Y1'*Y2   Y1'*Y3  Y1'*Y4 ...
!    0     Y2'*Y3  Y2'*Y4 ...
!    0        0    Y3'*Y4 ...
!    0        0       0   ...

subroutine qr_tmerge_tree_kernel(k,blocksize,treeorder,t,ldt,yty,ldy)
    implicit none
 
    ! input variables (local)
    integer ldt,ldy
    double precision t(ldt,*),yty(ldy,*)

    ! input variables (global)
    integer k,blocksize,treeorder
 
    ! output variables (global)

    ! local scalars
    integer temp_blocksize,nr_sets,current_set,setsize,nr_blocks
    integer remainder,max_treeorder,remaining_size
    integer toffset,yty_column
    integer toffset_start,yty_column_start
    integer yty_end,total_remainder,yty_remainder

    if (treeorder .eq. 0) return ! no merging

    if (treeorder .eq. 1) then
        call qr_tmerge_set_kernel(k,blocksize,t,ldt,yty,ldy)
        return
    end if
  
    nr_blocks = k / blocksize
    max_treeorder = min(nr_blocks,treeorder)

    if (max_treeorder .eq. 1) then
        call qr_tmerge_set_kernel(k,blocksize,t,ldt,yty,ldy)
        return
    end if
 
        ! work in "negative" direction: from latest set to oldest set
        ! implementation differs from rev=0 version due to issues with
        ! calculating the remainder parts
        ! compared to the rev=0 version we split remainder parts directly from
        ! parts which can be easily merged in a recursive way

        yty_end = (k / blocksize) * blocksize
        if (yty_end .eq. k) then
            yty_end = yty_end - blocksize
        end if

        !print *,'tree',yty_end,k,blocksize

        yty_column_start = 1
        toffset_start = 1

        ! is there a remainder block?
        nr_blocks = k / blocksize
        remainder = k - nr_blocks * blocksize
        if (remainder .eq. 0) then
            !print *,'no initial remainder'

            ! set offsets to the very beginning as there is no remainder part
            yty_column_start = 1
            toffset_start = 1
            total_remainder = 0
            remaining_size = k
            yty_remainder = 0
        else
            !print *,'starting with initial remainder'
            ! select submatrix and make remainder block public
            yty_column_start = 1 + blocksize
            toffset_start = 1 + remainder
            total_remainder = remainder
            remaining_size = k - remainder
            yty_remainder = 1
        end if
 
        ! from now on it is a clean set of blocks with sizes of multiple of
        ! blocksize

        temp_blocksize = blocksize

        !-------------------------------
        do while (remaining_size .gt. 0)
            nr_blocks = remaining_size / temp_blocksize
            max_treeorder = min(nr_blocks,treeorder)

            if (max_treeorder .eq. 1) then
                remainder = 0
                nr_sets = 0
                setsize = 0

                if (yty_remainder .gt. 0) then
                    yty_column = yty_remainder
                    !print *,'final merging with remainder',temp_blocksize,k,remaining_size,yty_column
                    call qr_tmerge_set_kernel(k,temp_blocksize,t,ldt,yty(1,yty_column),ldy)
                else
                    !print *,'no remainder - no merging needed',temp_blocksize,k,remaining_size
                endif
  
                remaining_size = 0
             
                return ! done
            else
                nr_sets = nr_blocks / max_treeorder
                setsize = max_treeorder*temp_blocksize
                remainder = remaining_size - nr_sets*setsize
            end if
  
            if (remainder .gt. 0) then
                if (remainder .gt. temp_blocksize) then
                    toffset = toffset_start
                    yty_column = yty_column_start
 
                    !print *,'set merging', toffset, yty_column,remainder
                    call qr_tmerge_set_kernel(remainder,temp_blocksize,t(toffset,toffset),ldt,yty(toffset,yty_column),ldy)

                    if (total_remainder .gt. 0) then
                        ! merge with existing global remainder part
                        !print *,'single+set merging',yty_remainder,total_remainder,remainder

                        call qr_pdlarft_merge_kernel_local(remainder,total_remainder,t(1,1),ldt,yty(1,yty_remainder),ldy)
      
                        yty_remainder = yty_remainder + remainder
                        toffset_start = toffset_start + remainder

                        !print *,'single+set merging (new offsets)',yty_remainder,yty_column_start,toffset_start

                        yty_column_start = yty_column_start + remainder
                    else
                        ! create new remainder part
                        !print *,'new remainder+set',yty_remainder
                        yty_remainder = yty_column_start + remainder - temp_blocksize 
                        yty_column_start = yty_column_start + remainder
                        toffset_start = toffset_start + remainder
                        !print *,'new remainder+set (new offsets)',yty_remainder,yty_column_start,toffset_start
                    end if

                else
                    if (total_remainder .gt. 0) then
                        ! merge with existing global remainder part
                        !print *,'single merging',yty_remainder,total_remainder,remainder

                        call qr_pdlarft_merge_kernel_local(remainder,total_remainder,t(1,1),ldt,yty(1,yty_remainder),ldy)
      
                        yty_remainder = yty_remainder + remainder
                        toffset_start = toffset_start + remainder

                        !print *,'single merging (new offsets)',yty_remainder,yty_column_start,toffset_start

                        yty_column_start = yty_column_start + remainder
                    else
                        ! create new remainder part
                        !print *,'new remainder',yty_remainder
                        yty_remainder = yty_column_start
                        yty_column_start = yty_column_start + temp_blocksize
                        toffset_start = toffset_start + remainder
                        !print *,'new remainder (new offsets)',yty_remainder,yty_column_start,toffset_start
                    end if
                end if
 
                total_remainder = total_remainder + remainder
                remaining_size = remaining_size - remainder
            end if

            current_set = 0
            do while (current_set .lt. nr_sets)
                toffset = toffset_start + current_set * setsize
                yty_column = yty_column_start + current_set * setsize

                !print *,'recursive merging', toffset, yty_column,setsize

                call qr_tmerge_set_kernel(setsize,temp_blocksize,t(toffset,toffset),ldt,yty(toffset,yty_column),ldy)
                
                current_set = current_set +  1
            end do

            !print *,'increasing blocksize', temp_blocksize, setsize
            yty_column_start = yty_column_start + (setsize - temp_blocksize)
            temp_blocksize = setsize
        end do
end subroutine
! yty should not contain the inner products vi'*vi
subroutine qr_dlarft_kernel(n,tau,yty,ldy,t,ldt)
    implicit none

    ! input variables
    integer n,ldy,ldt
    double precision tau(*),yty(ldy,*)
    
    ! output variables
    double precision t(ldt,*)

    ! local variables
    integer icol
 
    ! DEBUG: clear buffer first
    !t(1:n,1:n) = 0.0d0

        ! T1 = tau1
        ! | tauk  Tk-1' * (-tauk * Y(:,1,k+1:n) * Y(:,k))' |
        ! | 0           Tk-1                           |
        t(n,n) = tau(n)
        do icol=n-1,1,-1
            t(icol,icol+1:n) = -tau(icol)*yty(icol,icol:n-1)
            call dtrmv("Upper","Trans","Nonunit",n-icol,t(icol+1,icol+1),ldt,t(icol,icol+1),ldt)
            t(icol,icol) = tau(icol)
        end do
end subroutine
