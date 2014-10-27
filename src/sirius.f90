module sirius

interface
    subroutine sirius_density_initialize_v2(rhoir, rhomt) bind(C, name="sirius_density_initialize_v2")
        real(8),           dimension(:),       intent(in) :: rhoir
        real(8), optional, dimension(:, :, :), intent(in) :: rhomt
    end subroutine
end interface

end module
