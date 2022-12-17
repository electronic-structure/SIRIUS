program test_fortran_api
use mpi
use sirius
implicit none
type(sirius_context_handler) :: handler
type(sirius_kpoint_set_handler) :: kset
type(sirius_ground_state_handler) :: dft
logical :: stat
integer, target :: i,j,k,l,n,ctype, enum_size
character(100) ,target :: key, section, desc, usage
character(100), target :: str_val
real(8) :: lat_vec(3,3), pos(3)
integer lmax
integer nr
real(8), allocatable :: rgrid(:), ae_rho(:)
real(8) , target :: rmin, rmax, t, d
real(8), target, allocatable :: d_array(:)
type(c_ptr) :: iptr

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

str_val = "XC_LDA_X"
call sirius_option_set(handler, "parameters", "xc_functionals", SIRIUS_STRING_TYPE, C_LOC(str_val), len(trim(str_val)), .true.)
str_val = "XC_LDA_C_VWN"
call sirius_option_set(handler, "parameters", "xc_functionals", SIRIUS_STRING_TYPE, C_LOC(str_val), len(trim(str_val)), .true.)
!call sirius_add_xc_functional(handler, "XC_LDA_X")
!call sirius_add_xc_functional(handler, "XC_LDA_C_VWN")

stat = .true.
call sirius_context_initialized(handler, stat)
if (stat) then
    stop 'error'
endif

call sirius_option_get_number_of_sections(n)
write(*,*)'number of sections : ', n
do i = 1, n
  call sirius_option_get_section_name(i, section, len(section))
  call sirius_option_get_section_length(trim(adjustl(section)), l)
  write(*,'("section : ",I2," [",A,"],  length : ",I2)')i,trim(adjustl(section)),l
  do j = 1, l
    call sirius_option_get_info(trim(adjustl(section)), j, key, len(key), ctype, l, enum_size,&
        &desc, len(desc), usage, len(usage))
    write(*,'(" key : ", I2," [",A,"], type : ",I2,", lenght : ",I2)')j,trim(adjustl(key)),ctype,l
    write(*,*)trim(adjustl(desc))
    write(*,*)trim(adjustl(usage))
    if (ctype == SIRIUS_INTEGER_TYPE) then
      call sirius_option_get(trim(adjustl(section)), trim(adjustl(key)), ctype, C_LOC(k))
      write(*,*)'default value : ', k
    endif
    if (ctype == SIRIUS_NUMBER_TYPE) then
      call sirius_option_get(trim(adjustl(section)), trim(adjustl(key)), ctype, C_LOC(d))
      write(*,*)'default value : ', d
    endif
    if (ctype == SIRIUS_NUMBER_ARRAY_TYPE) then
      allocate(d_array(l))
      call sirius_option_get(trim(adjustl(section)), trim(adjustl(key)), ctype, C_LOC(d_array), l)
      write(*,*)'default value : ', d_array
      deallocate(d_array)
    endif
    if (ctype == SIRIUS_STRING_TYPE) then
      call sirius_option_get(trim(adjustl(section)), trim(adjustl(key)), ctype, C_LOC(str_val), len(str_val))
      write(*,*)'default value  : ', trim(str_val)
      if (enum_size .ne. 0) then
          do k = 1, enum_size
            call sirius_option_get(trim(adjustl(section)), trim(adjustl(key)), ctype, C_LOC(str_val), len(str_val), k)
            write(*,*)'possible value: ', trim(str_val)
          enddo
      endif
    endif
  enddo
enddo


call sirius_option_get("control", "output", SIRIUS_STRING_TYPE, C_LOC(str_val), len(str_val))
write(*,*)trim(adjustl(str_val))
str_val = "file:output.txt"
call sirius_option_set(handler, "control", "output", SIRIUS_STRING_TYPE, C_LOC(str_val), len(trim(adjustl(str_val))))


call sirius_initialize_context(handler)
!call sirius_print_info(handler)
!
!call sirius_create_kset_from_grid(handler, (/2, 2, 2/), (/0, 0, 0/), .true., kset)
!
!call sirius_create_ground_state(kset, dft)
!call sirius_find_ground_state(dft)


!call sirius_free_handler(dft)
!call sirius_free_handler(kset)
call sirius_free_handler(handler)

call sirius_finalize

end program
