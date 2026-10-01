! This file is part of SIRIUS electronic structure library.
! Copyright (c), ETH Zurich. All rights reserved.
! SPDX-License-Identifier: BSD-3-Clause

module external_xc_test
  use iso_c_binding
  use mpi
  implicit none
  type, bind(C) :: callback_state
    integer(C_INT) :: calls, failures, fail_next
    real(C_DOUBLE) :: energy
  end type
contains
  subroutine evaluate_xc(user_data, nr, ns, dims, z_offset, context_comm, fft_comm, &
                         lattice, weight, rho, gradient, tau, energy, vrho, vgradient, vtau, status) bind(C)
    type(C_PTR), value :: user_data
    integer(C_INT), value :: nr, ns, z_offset, context_comm, fft_comm
    integer(C_INT), intent(in) :: dims(3)
    real(C_DOUBLE), value :: weight
    real(C_DOUBLE), intent(in) :: lattice(3,3), rho(nr,ns), gradient(nr,3,ns), tau(nr,ns)
    real(C_DOUBLE), intent(inout) :: energy, vrho(nr,ns), vgradient(nr,3,ns), vtau(nr,ns)
    integer(C_INT), intent(inout) :: status
    type(callback_state), pointer :: state
    real(C_DOUBLE) :: moment, local_energy, volume
    integer :: ierr, rank, points, expected_points

    call C_F_POINTER(user_data, state)
    state%calls = state%calls + 1
    volume = lattice(1,1) * lattice(2,2) * lattice(3,3)
    if (abs(volume - 720.d0) > 1.d-12) state%failures = state%failures + 1
    if (abs(weight * product(dims) - volume) > 1.d-10) state%failures = state%failures + 1
    if (z_offset < 0 .or. z_offset + nr/(dims(1)*dims(2)) > dims(3)) &
      state%failures = state%failures + 1
    if (ns /= 1 .and. ns /= 2) state%failures = state%failures + 1
    if (any(vrho /= 0) .or. any(vgradient /= 0) .or. any(vtau /= 0) .or. status /= 0) &
      state%failures = state%failures + 1
    call MPI_ALLREDUCE(nr, points, 1, MPI_INTEGER, MPI_SUM, fft_comm, ierr)
    expected_points = product(dims)
    if (points /= expected_points) state%failures = state%failures + 1

    moment = weight * sum(tau)
    call MPI_ALLREDUCE(MPI_IN_PLACE, moment, 1, MPI_DOUBLE_PRECISION, MPI_SUM, fft_comm, ierr)
    local_energy = weight * (sum(0.7d0*rho**2 + 0.3d0*tau**2 + 0.2d0*rho*tau) + &
                            0.1d0*sum(gradient**2))
    call MPI_ALLREDUCE(local_energy, energy, 1, MPI_DOUBLE_PRECISION, MPI_SUM, fft_comm, ierr)
    energy = energy + 0.2d0 * moment**2 / volume
    vrho = 1.4d0*rho + 0.2d0*tau
    vgradient = 0.2d0*gradient
    vtau = 0.6d0*tau + 0.2d0*rho + 0.4d0*moment/volume
    state%energy = energy
    call MPI_COMM_RANK(context_comm, rank, ierr)
    if (state%fail_next /= 0 .and. rank == 0) status = 7
  end subroutine

  subroutine evaluate_points(user_data, nr, ns, nb, comm, coordinates, weights, descriptor_weights, &
                             block_sizes, centres, rho, gradient, tau, energy, drho, dgradient, dtau, status) bind(C)
    type(C_PTR), value :: user_data
    integer(C_INT), value :: nr, ns, nb, comm
    integer(C_INT), intent(in) :: block_sizes(nb)
    real(C_DOUBLE), intent(in) :: coordinates(3,nr), weights(nr), descriptor_weights(nr), centres(3,nb)
    real(C_DOUBLE), intent(in) :: rho(nr,ns), gradient(nr,3,ns), tau(nr,ns)
    real(C_DOUBLE), intent(inout) :: energy, drho(nr,ns), dgradient(nr,3,ns), dtau(nr,ns)
    integer(C_INT), intent(inout) :: status
    type(callback_state), pointer :: state
    real(C_DOUBLE) :: moment, local_energy
    integer :: ierr, rank, s, p

    call C_F_POINTER(user_data, state)
    state%calls = state%calls + 1
    call MPI_COMM_RANK(comm, rank, ierr)
    if (sum(block_sizes) /= nr) state%failures = state%failures + 1
    if (rank == 0) then
      if (nr /= 64 .or. nb /= 1) state%failures = state%failures + 1
      if (maxval(abs(coordinates(:,1) - [0.296d0, 0.261d0, 0.410d0])) > 1.d-12) &
        state%failures = state%failures + 1
      if (maxval(abs(centres(:,1) - [1.6d0, 2.7d0, 4.d0])) > 1.d-12) state%failures = state%failures + 1
    else
      if (nr /= 0 .or. nb /= 0) state%failures = state%failures + 1
    endif
    if (any(drho /= 0) .or. any(dgradient /= 0) .or. any(dtau /= 0) .or. status /= 0) &
      state%failures = state%failures + 1
    moment = 0
    local_energy = 0
    do s = 1, ns
      moment = moment + sum(descriptor_weights * tau(:,s))
      local_energy = local_energy + sum(weights * (0.7d0*rho(:,s)**2 + 0.3d0*tau(:,s)**2 + &
                                                   0.2d0*rho(:,s)*tau(:,s) + 0.1d0*sum(gradient(:,:,s)**2,dim=2)))
    enddo
    call MPI_ALLREDUCE(MPI_IN_PLACE, moment, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
    call MPI_ALLREDUCE(local_energy, energy, 1, MPI_DOUBLE_PRECISION, MPI_SUM, comm, ierr)
    energy = energy + 0.2d0*moment**2/720.d0
    do s = 1, ns
      drho(:,s) = weights*(1.4d0*rho(:,s) + 0.2d0*tau(:,s))
      dtau(:,s) = weights*(0.6d0*tau(:,s) + 0.2d0*rho(:,s)) + descriptor_weights*0.4d0*moment/720.d0
      do p = 1, 3
        dgradient(:,p,s) = weights*0.2d0*gradient(:,p,s)
      enddo
    enddo
    state%energy = energy
    if (state%fail_next /= 0 .and. rank == 0) status = 7
  end subroutine
end module

program test_external_xc_api
  use sirius
  use external_xc_test
  implicit none
  type(sirius_context_handler) :: ctx
  type(sirius_kpoint_set_handler) :: ks
  type(sirius_ground_state_handler) :: gs
  type(callback_state), target :: state
  integer, parameter :: n = 400
  real(8) :: r(n), v(n), rho(n), orbital(n), lattice(3,3), dion(0,0), kpoint(3,1), energy
  real(8), allocatable :: coordinates(:,:), weights(:), descriptor_weights(:), centres(:,:)
  integer, allocatable :: blocks(:)
  integer :: i, ierr, mag, failures, all_failures, mode, rank, nr, nb, x, y, z
  logical :: converged

  call sirius_initialize(call_mpi_init=.true.)
  call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
  failures = 0
  do mode = 0, 1
  do mag = 0, 1
    state = callback_state(0, 0, 0, 0.d0)
    call sirius_create_context(MPI_COMM_WORLD, ctx)
    call sirius_import_parameters(ctx, &
      '{"parameters":{"electronic_structure_method":"pseudopotential","use_symmetry":false,' // &
      '"pw_cutoff":8.0,"gk_cutoff":2.6,"num_bands":4,"xc_functionals":[],"smearing_width":0.01},' // &
      '"control":{"mpi_grid_dims":[1,1],"processing_unit":"CPU","verbosity":0},' // &
      '"mixer":{"type":"linear","beta":0.4,"use_hartree":false},' // &
      '"iterative_solver":{"min_tolerance":1e-11,"num_steps":80}}')
    call sirius_set_parameters(ctx, num_mag_dims=mag)
    if (mode == 0) call sirius_set_xc_callback(ctx, C_FUNLOC(evaluate_xc), C_LOC(state))
    lattice = 0
    lattice(1,1) = 8
    lattice(2,2) = 9
    lattice(3,3) = 10
    call sirius_set_lattice_vectors(ctx, lattice(:,1), lattice(:,2), lattice(:,3))
    call sirius_add_atom_type(ctx, 'He', zn=2)
    do i = 1, n
      r(i) = 20.d0 * (real(i-1,8)/real(n-1,8))**2
    end do
    v = -2.d0 / sqrt(1.d0+r**2)
    rho = 4.d0 * exp(-r**2) * r
    orbital = exp(-r)
    call sirius_set_atom_type_radial_grid(ctx, 'He', n, r)
    call sirius_add_atom_type_radial_function(ctx, 'He', 'vloc', v, n)
    call sirius_add_atom_type_radial_function(ctx, 'He', 'ps_rho_total', rho, n)
    call sirius_add_atom_type_radial_function(ctx, 'He', 'ps_core_tau', -orbital, n, error_code=ierr)
    if (ierr == 0) failures = failures + 1
    call sirius_add_atom_type_radial_function(ctx, 'He', 'ps_core_tau', 0.d0*orbital, n, error_code=ierr)
    if (ierr /= 0) failures = failures + 1
    call sirius_add_atom_type_radial_function(ctx, 'He', 'ps_atomic_wf', orbital, n, n=1, l=0)
    call sirius_set_atom_type_dion(ctx, 'He', 0, dion)
    call sirius_add_atom(ctx, 'He', [0.2d0, 0.3d0, 0.4d0])
    if (mode == 1) then
      nr = 0
      nb = 0
      if (rank == 0) then
        nr = 64
        nb = 1
      endif
      allocate(coordinates(3,nr), weights(nr), descriptor_weights(nr), centres(3,nb), blocks(nb))
      if (rank == 0) then
        i = 0
        do z = 0, 3
          do y = 0, 3
            do x = 0, 3
              i = i + 1
              coordinates(:,i) = [8*(0.037d0 + x/4.d0), 9*(0.029d0 + y/4.d0), 10*(0.041d0 + z/4.d0)]
            enddo
          enddo
        enddo
        centres(:,1) = [1.6d0, 2.7d0, 4.d0]
        blocks = 64
      endif
      weights = 720.d0/64
      descriptor_weights = weights
      if (rank == 0) weights(1) = 0
      call sirius_set_xc_point_callback(ctx, C_FUNLOC(evaluate_points), C_LOC(state), nr, nb, coordinates, &
                                        weights, descriptor_weights, blocks, centres)
      ! Registration copies the quadrature. A bad replacement must not change it.
      coordinates = 0
      if (rank == 0) blocks(1) = 63
      call sirius_set_xc_point_callback(ctx, C_FUNLOC(evaluate_points), C_LOC(state), nr, nb, coordinates, &
                                        weights, descriptor_weights, blocks, centres, error_code=ierr)
      if (ierr == 0) failures = failures + 1
      deallocate(coordinates, weights, descriptor_weights, centres, blocks)
    endif
    call sirius_initialize_context(ctx)
    call sirius_set_xc_callback(ctx, C_NULL_FUNPTR, C_NULL_PTR, error_code=ierr)
    if (ierr == 0) failures = failures + 1
    kpoint(:,1) = [0.07d0, 0.03d0, 0.02d0]
    call sirius_create_kset(ctx, 1, kpoint, [1.d0], .true., ks)
    call sirius_create_ground_state(ks, gs)
    converged = .false.
    call sirius_find_ground_state(gs, density_tol=1.d-8, energy_tol=1.d-9, iter_solver_tol=1.d-8, &
                                 max_niter=100, save_state=.false., converged=converged)
    if (.not. converged .or. state%calls < 2) failures = failures + 1
    call sirius_get_energy(gs, 'exc', energy)
    if (abs(energy - state%energy) > 1.d-12) failures = failures + 1
    failures = failures + state%failures
    state%fail_next = 1
    call sirius_generate_effective_potential(gs, error_code=ierr)
    if (ierr == 0) failures = failures + 1
    call sirius_free_handler(gs)
    call sirius_free_handler(ks)
    call sirius_free_handler(ctx)
  end do
  end do
  call MPI_ALLREDUCE(failures, all_failures, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
  call sirius_finalize()
  if (all_failures /= 0) stop 1
end program
