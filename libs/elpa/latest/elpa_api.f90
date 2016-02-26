subroutine elpa_cholesky_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1_compute
use elpa1
implicit none
integer na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
complex*16 a(lda,matrixCols)
logical success
    
  call cholesky_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false., success)

end subroutine

subroutine elpa_cholesky_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1_compute
use elpa1
implicit none
integer na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
real*8 a(lda,matrixCols)
logical success
    
  call cholesky_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false., success)

end subroutine


subroutine elpa_invert_trm_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1_compute
use elpa1
implicit none
integer na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
complex*16 a(lda,matrixCols)
logical success

  call invert_trm_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false., success)

end subroutine

subroutine elpa_invert_trm_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1_compute
use elpa1
implicit none
integer na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
real*8 a(lda,matrixCols)
logical success

  call invert_trm_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false., success)

end subroutine

subroutine elpa_mult_ah_b_complex(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)
use elpa1_compute
use elpa1
implicit none
character*1 uplo_a, uplo_c
integer na, ncb, lda, ldb, nblk, mpi_comm_rows, mpi_comm_cols, ldc
complex*16 a(lda,*), b(ldb,*), c(ldc,*)

  call mult_ah_b_complex(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)

end subroutine

subroutine elpa_mult_at_b_real(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)
use elpa1_compute
use elpa1
implicit none
character*1 uplo_a, uplo_c
integer na, ncb, lda, ldb, nblk, mpi_comm_rows, mpi_comm_cols, ldc
real(8) a(lda,*), b(ldb,*), c(ldc,*)

  call mult_at_b_real(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)

end subroutine


subroutine elpa_solve_evp_complex(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1
implicit none
integer, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
complex*16 :: a(lda,matrixCols), q(ldq,matrixCols)
real*8 :: ev(na)
logical success

  success = solve_evp_complex(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)

end subroutine

subroutine elpa_solve_evp_real(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1
implicit none
integer, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
real*8 :: a(lda,matrixCols), q(ldq,matrixCols)
real*8 :: ev(na)
logical success

  success = solve_evp_real(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)

end subroutine


subroutine elpa_solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
use elpa2
implicit none
integer, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
complex*16, intent(inout) :: a(lda,matrixCols), q(ldq,matrixCols)
real*8, intent(inout) :: ev(na)
logical success

  success = solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)

end subroutine

subroutine elpa_solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
use elpa2
implicit none
integer, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
real*8, intent(inout) :: a(lda,matrixCols), q(ldq,matrixCols)
real*8, intent(inout) :: ev(na)
logical success

  success = solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)

end subroutine


