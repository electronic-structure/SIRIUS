program test_fortran_api
use sirius
type(C_PTR) :: handler
type(C_PTR) :: kset
type(C_PTR) :: dft
integer i
real(8) :: lat_vec(3,3), pos(3), forces(3, 5), stress(3,3)
integer comm

! initialize the library
call sirius_initialize(call_mpi_init=.true.)

! create simulation context using a specified communicator
comm = MPI_COMM_WORLD
write(*,*)'comm=',comm
call sirius_create_context(comm, handler)

call sirius_import_parameters(handler, &
    '{"parameters" : {"electronic_structure_method" : "pseudopotential"},&
      "control" : {"verbosity" : 1, "verification" : 0}}')

! atomic units are used everywhere
! plane-wave cutoffs are provided in a.u.^-1
call sirius_set_parameters(handler, pw_cutoff=20.d0, gk_cutoff=7.d0)

lat_vec = 0.d0
do i = 1, 3
  lat_vec(i,i) = 7.260327248
enddo
! disturb the lattice a little bit
lat_vec(1,3) = 0.001

call sirius_set_lattice_vectors(handler, lat_vec(:, 1), lat_vec(:, 2), lat_vec(:, 3))

call sirius_add_atom_type(handler, "Sr", fname="Sr.json")
call sirius_add_atom_type(handler, "V", fname="V.json")
call sirius_add_atom_type(handler, "O", fname="O.json")

! atomic coordinates are provided in fractional units of the lattice vectors
call sirius_add_atom(handler, "Sr", (/0.5d0, 0.5d0, 0.5d0/))
call sirius_add_atom(handler, "V", (/0.d0, 0.d0, 0.d0/))
call sirius_add_atom(handler, "O", (/0.5d0, 0.d0, 0.d0/))
call sirius_add_atom(handler, "O", (/0.d0, 0.5d0, 0.d0/))
! disturb the coordinate a little bit to get non-zero forces
call sirius_add_atom(handler, "O", (/0.01d0, 0.01d0, 0.5d0/))

! exchange and correlation parts
call sirius_add_xc_functional(handler, "XC_LDA_X")
call sirius_add_xc_functional(handler, "XC_LDA_C_VWN")

! initialize the simulation handler
call sirius_initialize_context(handler)

! create [2,2,2] k-grid with [0,0,0] offset and use symmetry to get irreducible set of k-points
call sirius_create_kset_from_grid(handler, (/2, 2, 2/), (/0, 0, 0/), .true., kset)

call sirius_create_ground_state(kset, dft)
call sirius_find_ground_state(dft)

call sirius_get_forces(dft, "total", forces)
call sirius_get_stress_tensor(dft, "total", stress)

write(*,*)'Forces:'
do i = 1, 5
  write(*,*)'atom=',i, ' force=',forces(:,i)
enddo
write(*,*)'Stress:'
do i = 1, 3
  write(*,*)stress(i,:)
enddo


call sirius_free_handler(dft)
call sirius_free_handler(kset)
call sirius_free_handler(handler)

call sirius_finalize

end program
