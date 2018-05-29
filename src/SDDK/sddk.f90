module sddk
    use, intrinsic :: ISO_C_BINDING

    interface

        !subroutine sddk_delete_object(object_id)&
        !    &bind(C, name="sddk_delete_object")
        !    integer,                 intent(in)  :: object_id
        !end subroutine

        subroutine sddk_delete_object(handler)&
            &bind(C, name="sddk_delete_object")
            use, intrinsic :: ISO_C_BINDING
            type(C_PTR),             intent(inout)  :: handler
        end subroutine

        !subroutine sddk_create_fft_grid(dims, fft_grid_id)&
        !    &bind(C, name="sddk_create_fft_grid")
        !    integer,                 intent(in)  :: dims(3)
        !    integer,                 intent(out) :: fft_grid_id
        !end subroutine

        subroutine sddk_create_fft(initial_dims, fcomm, handler)&
            &bind(C, name="sddk_create_fft")
            use, intrinsic :: ISO_C_BINDING
            integer(C_INT),          intent(in)  :: initial_dims
            integer(C_INT),          intent(in)  :: fcomm
            type(C_PTR),             intent(out) :: handler
        end subroutine

        subroutine sddk_create_gkvec(vk, b1, b2, b3, gmax, reduce_gvec, comm, handler)&
            &bind(C, name="sddk_create_gkvec")
            use, intrinsic :: ISO_C_BINDING
            real(C_DOUBLE),          intent(in)  :: vk(3)
            real(C_DOUBLE),          intent(in)  :: b1(3)
            real(C_DOUBLE),          intent(in)  :: b2(3)
            real(C_DOUBLE),          intent(in)  :: b3(3)
            real(C_DOUBLE),          intent(in)  :: gmax
            logical(C_BOOL),         intent(in)  :: reduce_gvec
            integer(C_INT),          intent(in)  :: comm
            type(C_PTR),             intent(out) :: handler
        end subroutine

        subroutine sddk_create_gvec(b1, b2, b3, gmax, reduce_gvec, comm, handler)&
            &bind(C, name="sddk_create_gvec")
            use, intrinsic :: ISO_C_BINDING
            real(C_DOUBLE),          intent(in)  :: b1(3)
            real(C_DOUBLE),          intent(in)  :: b2(3)
            real(C_DOUBLE),          intent(in)  :: b3(3)
            real(C_DOUBLE),          intent(in)  :: gmax
            logical(C_BOOL),         intent(in)  :: reduce_gvec
            integer(C_INT),          intent(in)  :: comm
            type(C_PTR),             intent(out) :: handler
        end subroutine

        subroutine sddk_create_gvec_partition(gvec_handler, fft_comm, comm_ortho_fft, handler)&
            &bind(C, name="sddk_create_gvec_partition")
            use, intrinsic :: ISO_C_BINDING
            type(C_PTR),             intent(in)  :: gvec_handler
            integer(C_INT),          intent(in)  :: fft_comm
            integer(C_INT),          intent(in)  :: comm_ortho_fft
            type(C_PTR),             intent(out) :: handler
        end subroutine

        subroutine sddk_get_num_gvec(gvec_id, num_gvec)&
            &bind(C, name="sddk_get_num_gvec")
            integer,                 intent(in)  :: gvec_id
            integer,                 intent(out) :: num_gvec
        end subroutine

        subroutine sddk_get_gvec_count(gvec_id, rank, gvec_count)&
            &bind(C, name="sddk_get_gvec_count")
            integer,                 intent(in)  :: gvec_id
            integer,                 intent(in)  :: rank
            integer,                 intent(out) :: gvec_count
        end subroutine

        subroutine sddk_create_wave_functions(gvec_id, num_wf, wf_id)&
            &bind(C, name="sddk_create_wave_functions")
            integer,                 intent(in)  :: gvec_id
            integer,                 intent(in)  :: num_wf
            integer,                 intent(out) :: wf_id
        end subroutine

        subroutine sddk_get_gvec_offset(gvec_id, rank, gvec_offset)&
            &bind(C, name="sddk_get_gvec_offset")
            integer,                 intent(in)  :: gvec_id
            integer,                 intent(in)  :: rank
            integer,                 intent(out) :: gvec_offset
        end subroutine

        subroutine sddk_get_gvec_count_fft(gvec_id, gvec_count)&
            &bind(C, name="sddk_get_gvec_count_fft")
            integer,                 intent(in)  :: gvec_id
            integer,                 intent(out) :: gvec_count
        end subroutine

        subroutine sddk_get_gvec_offset_fft(gvec_id, gvec_offset)&
            &bind(C, name="sddk_get_gvec_offset_fft")
            integer,                 intent(in)  :: gvec_id
            integer,                 intent(out) :: gvec_offset
        end subroutine

        subroutine sddk_fft(fft_id, dir, dat)&
            &bind(C, name="sddk_fft")
            integer,                 intent(in)  :: fft_id
            integer,                 intent(in)  :: dir
            complex(8),              intent(inout)  :: dat
        end subroutine

        subroutine sddk_fft_prepare(fft_id, gvec_id)&
            &bind(C, name="sddk_fft_prepare")
            integer,                 intent(in)  :: fft_id
            integer,                 intent(in)  :: gvec_id
        end subroutine

        subroutine sddk_fft_dismiss(fft_id)&
            &bind(C, name="sddk_fft_dismiss")
            integer,                 intent(in)  :: fft_id
        end subroutine

        subroutine sddk_print_timers()&
            &bind(C, name="sddk_print_timers")
        end subroutine

        subroutine sddk_get_num_wave_functions(wf_id, num_wf)&
            &bind(C, name="sddk_get_num_wave_functions")
            integer,                 intent(in)  :: wf_id
            integer,                 intent(out) :: num_wf
        end subroutine

        subroutine sddk_get_num_wave_functions_local(wf_id, num_wf)&
            &bind(C, name="sddk_get_num_wave_functions_local")
            integer,                 intent(in)  :: wf_id
            integer,                 intent(out) :: num_wf
        end subroutine

        subroutine sddk_get_wave_functions_prime_ld(wf_id, ld)&
            &bind(C, name="sddk_get_wave_functions_prime_ld")
            integer,                 intent(in)  :: wf_id
            integer,                 intent(out) :: ld
        end subroutine

        subroutine sddk_get_wave_functions_extra_ld(wf_id, ld)&
            &bind(C, name="sddk_get_wave_functions_extra_ld")
            integer,                 intent(in)  :: wf_id
            integer,                 intent(out) :: ld
        end subroutine

        subroutine sddk_remap_wave_functions_forward(wf_id, n, idx0)&
            &bind(C, name="sddk_remap_wave_functions_forward")
            integer,                 intent(in)  :: wf_id
            integer,                 intent(in)  :: n
            integer,                 intent(in)  :: idx0
        end subroutine

        subroutine sddk_remap_wave_functions_backward(wf_id, n, idx0)&
            &bind(C, name="sddk_remap_wave_functions_backward")
            integer,                 intent(in)  :: wf_id
            integer,                 intent(in)  :: n
            integer,                 intent(in)  :: idx0
        end subroutine

    end interface

contains

    function c_str(f_string) result(c_string)
        implicit none
        character(len=*), intent(in)  :: f_string
        character(len=1, kind=C_CHAR) :: c_string(len_trim(f_string) + 1)
        integer i
        do i = 1, len_trim(f_string)
            c_string(i) = f_string(i:i)
        end do
        c_string(len_trim(f_string) + 1) = C_NULL_CHAR
    end function c_str

    function c_logical(val) result(c_val)
        implicit none
        logical, intent(in) :: val
        logical(C_BOOL)     :: c_val
        c_val = val
    end function c_logical

    subroutine sddk_get_wave_functions_prime_ptr(wf_id, wf_ptr)
        implicit none
        integer,             intent(in)  :: wf_id
        complex(8), pointer, intent(out) :: wf_ptr(:, :)
        type(C_PTR) :: wf_c_ptr
        integer nwf, ld
        interface
            subroutine sddk_get_wave_functions_prime_ptr_aux(wf_id, wf_c_ptr)&
                &bind(C, name="sddk_get_wave_functions_prime_ptr")
                use, intrinsic :: ISO_C_BINDING
                integer,                 intent(in)  :: wf_id
                type(C_PTR),             intent(out) :: wf_c_ptr
            end subroutine
        end interface
        call sddk_get_wave_functions_prime_ptr_aux(wf_id, wf_c_ptr)
        call sddk_get_num_wave_functions(wf_id, nwf)
        call sddk_get_wave_functions_prime_ld(wf_id, ld)
        call C_F_POINTER(wf_c_ptr, wf_ptr, (/ld, nwf/))
    end subroutine

    subroutine sddk_get_wave_functions_extra_ptr(wf_id, wf_ptr)
        implicit none
        integer,             intent(in)  :: wf_id
        complex(8), pointer, intent(out) :: wf_ptr(:, :)
        type(C_PTR) :: wf_c_ptr
        integer nwf, ld
        interface
            subroutine sddk_get_wave_functions_extra_ptr_aux(wf_id, wf_c_ptr)&
                &bind(C, name="sddk_get_wave_functions_extra_ptr")
                use, intrinsic :: ISO_C_BINDING
                integer,                 intent(in)  :: wf_id
                type(C_PTR),             intent(out) :: wf_c_ptr
            end subroutine
        end interface
        call sddk_get_wave_functions_extra_ptr_aux(wf_id, wf_c_ptr)
        call sddk_get_num_wave_functions_local(wf_id, nwf)
        call sddk_get_wave_functions_extra_ld(wf_id, ld)
        call C_F_POINTER(wf_c_ptr, wf_ptr, (/ld, nwf/))
    end subroutine

end module
