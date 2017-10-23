subroutine elpa_cholesky_complex_wrapper(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1_auxiliary
implicit none
integer na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
complex*16 a(lda,matrixCols)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_cholesky_complex_double(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false.)
#else
  call elpa_cholesky_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false., success)
#endif

end subroutine

subroutine elpa_cholesky_real_wrapper(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1_auxiliary
implicit none
integer na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
real*8 a(lda,matrixCols)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_cholesky_real_double(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false.)
#else
  call elpa_cholesky_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false., success)
#endif
end subroutine


subroutine elpa_invert_trm_complex_wrapper(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1_auxiliary
implicit none
integer na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
complex*16 a(lda,matrixCols)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_invert_trm_complex_double(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false.)
#else
  call elpa_invert_trm_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false., success)
#endif

end subroutine

subroutine elpa_invert_trm_real_wrapper(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
use elpa1_auxiliary
implicit none
integer na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
real*8 a(lda,matrixCols)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_invert_trm_real_double(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false.)
#else
  call elpa_invert_trm_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, .false., success)
#endif

end subroutine

subroutine elpa_mult_ah_b_complex_wrapper(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols,&
                                         &nblk, mpi_comm_rows, mpi_comm_cols, c, ldc, ldcCols)
use elpa1_auxiliary
implicit none
character*1 uplo_a, uplo_c
integer na, ncb, lda, ldaCols, ldb, ldbCols, nblk, mpi_comm_rows, mpi_comm_cols, ldc, ldcCols
complex*16 a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_mult_ah_b_complex_double(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, nblk,&
                                         &mpi_comm_rows, mpi_comm_cols, c, ldc, ldcCols)
#else
  call elpa_mult_ah_b_complex(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)
#endif

end subroutine

subroutine elpa_mult_at_b_real_wrapper(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols,&
                                         &nblk, mpi_comm_rows, mpi_comm_cols, c, ldc, ldcCols)
use elpa1_auxiliary
implicit none
character*1 uplo_a, uplo_c
integer na, ncb, lda, ldaCols, ldb, ldbCols, nblk, mpi_comm_rows, mpi_comm_cols, ldc, ldcCols
real(8) a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_mult_at_b_real_double(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, nblk,&
                                      &mpi_comm_rows, mpi_comm_cols, c, ldc, ldcCols)
#else
  call elpa_mult_at_b_real(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)
#endif

end subroutine


subroutine elpa_solve_evp_complex(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
use elpa1
implicit none
integer, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
complex*16 :: a(lda,matrixCols), q(ldq,matrixCols)
real*8 :: ev(na)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_solve_evp_complex_1stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols,&
                                                &mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#else
  success = solve_evp_complex_1stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#endif

if (.not.success) then
  write(*,*)'elpa_solve_evp_complex: error'
  stop
endif

end subroutine

subroutine elpa_solve_evp_real(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
use elpa1
implicit none
integer, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
real*8 :: a(lda,matrixCols), q(ldq,matrixCols)
real*8 :: ev(na)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_solve_evp_real_1stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols,&
                                             &mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#else
  success = solve_evp_real_1stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#endif

if (.not.success) then
  write(*,*)'elpa_solve_evp_real: error'
  stop
endif

end subroutine

subroutine elpa_solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
use elpa2
implicit none
integer, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
complex*16, intent(inout) :: a(lda,matrixCols), q(ldq,matrixCols)
real*8, intent(inout) :: ev(na)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_solve_evp_complex_2stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols,&
                                                &mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#else
  success = solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#endif

if (.not.success) then
  write(*,*)'elpa_solve_evp_complex_2stage: error'
  stop
endif

end subroutine

subroutine elpa_solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
use elpa2
implicit none
integer, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
real*8, intent(inout) :: a(lda,matrixCols), q(ldq,matrixCols)
real*8, intent(inout) :: ev(na)
logical success

#ifdef __EXTERNAL_ELPA_LIB
  success = elpa_solve_evp_real_2stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols,&
                                             &mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#else
  success = solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#endif

if (.not.success) then
  write(*,*)'elpa_solve_evp_real_2stage: error'
  stop
endif

end subroutine

