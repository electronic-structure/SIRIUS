program test_fortran_api
use mpi
use sirius
implicit none
type(sirius_context_handler) :: handler
type(sirius_kpoint_set_handler) :: kset
type(sirius_ground_state_handler) :: dft
logical :: stat
integer i,j,k,l
character(100) key
real(8) :: lat_vec(3,3), pos(3)
integer lmax
integer nr
real(8), allocatable :: rgrid(:), ae_rho(:)
real(8) rmin, rmax, t

call sirius_initialize(call_mpi_init=.true.)

call sirius_create_context(MPI_COMM_WORLD, handler)

call sirius_import_parameters(handler, &
    '{"parameters" : {"electronic_structure_method" : "full_potential_lapwlo"},&
      "control" : {"verbosity" : 3, "verification" : 0}}')

lmax = 8
call sirius_set_parameters(handler, lmax_apw=lmax, lmax_rho=lmax, lmax_pot=lmax, auto_rmt=0)

lat_vec=0.d0
do i = 1, 3
  lat_vec(i,i) = 12.d0
enddo

call sirius_set_lattice_vectors(handler, lat_vec(:, 1), lat_vec(:, 2), lat_vec(:, 3))

call sirius_add_atom_type(handler, "H", zn=1)
call sirius_add_atom_type_aw_descriptor(handler, "H", 1, 0, -0.2d0, 0, .true.) 
call sirius_add_atom_type_aw_descriptor(handler, "H", 1, 0, -0.2d0, 1, .true.)
do l = 1, lmax
  call sirius_add_atom_type_aw_descriptor(handler, "H", -1, l, 0.15d0, 0, .false.)
enddo

call sirius_add_atom_type_lo_descriptor(handler, "H", 1, 1, 0, -0.2d0, 0, .true.)
call sirius_add_atom_type_lo_descriptor(handler, "H", 1, 1, 0, -0.2d0, 1, .true.)

call sirius_add_atom_type_lo_descriptor(handler, "H", 2, 1, 0, -0.15d0, 0, .false.)
call sirius_add_atom_type_lo_descriptor(handler, "H", 2, 1, 0, -0.15d0, 1, .false.)
call sirius_add_atom_type_lo_descriptor(handler, "H", 2, 2, 0,  0.15d0, 0, .true.)

rmin=1d-6
rmax=2.5
nr=1500
allocate(rgrid(3 * nr))
do i = 1, 3 * nr
  t = dble(i - 1) / dble(nr -1)
  rgrid(i) = rmin + (rmax-rmin) * t**2 
enddo
call sirius_set_atom_type_radial_grid(handler, "H", nr, rgrid(1:nr))
call sirius_set_atom_type_radial_grid_inf(handler, "H", 3 * nr, rgrid)

allocate(ae_rho(3 * nr))
do i = 1, 3 * nr
  ae_rho(i) = exp(-rgrid(i))
enddo

call sirius_add_atom(handler, "H", (/0.d0, 0.d0, 0.d0/))

call sirius_add_atom_type_radial_function(handler, "H", "ae_rho", ae_rho, 3 * nr)

call sirius_add_xc_functional(handler, "XC_LDA_X")
call sirius_add_xc_functional(handler, "XC_LDA_C_VWN")

stat = .true.
call sirius_context_initialized(handler, stat)
if (stat) then
    stop 'error'
endif

call sirius_option_get_length('control', i)
write(*,*)'length of control:', i

do j=1,i
  call sirius_option_get_name_and_type('control', j, key, k)
  write(*,*)j,trim(adjustl(key)),k
enddo

call sirius_initialize_context(handler)
call sirius_print_info(handler)

call sirius_create_kset_from_grid(handler, (/2, 2, 2/), (/0, 0, 0/), .true., kset)

call sirius_create_ground_state(kset, dft)
call sirius_find_ground_state(dft)


call sirius_free_handler(dft)
call sirius_free_handler(kset)
call sirius_free_handler(handler)

call sirius_finalize

end program
