! This file is part of SIRIUS electronic structure library.
! Copyright (c), ETH Zurich. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause

program test_core_kinetic_density_api
  use sirius
  use mpi
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  implicit none
  type(sirius_context_handler) :: ctx
  integer, parameter :: nr = 16
  real(8) :: r(nr), tau(nr), invalid(nr)
  character(len=15), parameter :: labels(2) = [character(len=15) :: 'ps_core_tau', 'ae_paw_core_tau']
  integer :: i, j, ierr, failures, total, rank

  call sirius_initialize(call_mpi_init=.true.)
  call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
  call sirius_create_context(MPI_COMM_WORLD, ctx)
  call sirius_import_parameters(ctx, &
    '{"parameters":{"electronic_structure_method":"pseudopotential"},"control":{"verbosity":0}}')
  call sirius_add_atom_type(ctx, 'He', zn=2)
  do i = 1, nr
    r(i) = 4.d0 * real(i-1,8) / real(nr-1,8)
  end do
  call sirius_set_atom_type_radial_grid(ctx, 'He', nr, r)
  tau = exp(-r)
  failures = 0
  do j = 1, size(labels)
    call sirius_add_atom_type_radial_function(ctx, 'He', trim(labels(j)), tau, nr, error_code=ierr)
    if (ierr /= 0) failures = failures + 1
    call sirius_add_atom_type_radial_function(ctx, 'He', trim(labels(j)), tau, nr-1, error_code=ierr)
    if (ierr == 0) failures = failures + 1
    call sirius_add_atom_type_radial_function(ctx, 'He', trim(labels(j)), tau, 0, error_code=ierr)
    if (ierr == 0) failures = failures + 1
    invalid = tau
    invalid(3) = -1.d0
    call sirius_add_atom_type_radial_function(ctx, 'He', trim(labels(j)), invalid, nr, error_code=ierr)
    if (ierr == 0) failures = failures + 1
    invalid(3) = ieee_value(0.d0, ieee_quiet_nan)
    call sirius_add_atom_type_radial_function(ctx, 'He', trim(labels(j)), invalid, nr, error_code=ierr)
    if (ierr == 0) failures = failures + 1
    call sirius_add_atom_type_radial_function(ctx, 'He', trim(labels(j)), 0.d0*tau, nr, error_code=ierr)
    if (ierr /= 0) failures = failures + 1
  end do
  call sirius_free_handler(ctx)
  call MPI_ALLREDUCE(failures, total, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (rank == 0) print *, 'Core kinetic-density API failures:', total
  call sirius_finalize(call_mpi_fin=.true.)
  if (total /= 0) stop 1
end program
