add_executable(sirius.scf sirius.scf.cpp)
target_link_libraries(sirius.scf PRIVATE sirius sirius::filesystem)
install(TARGETS sirius.scf RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set_property(TARGET sirius.scf PROPERTY POSITION_INDEPENDENT_CODE OFF)

# Register verification tests as well
set(SIRIUS_SCF_LABELS
    "gpu_serial"
    "gpu_band_parallel"
    "gpu_k_point_parallel"
    "cpu_serial"
    "cpu_band_parallel")

set(SIRIUS_SCF_FLAGS
    "--control.processing_unit=gpu --control.std_evp_solver_name=cusolver --control.gen_evp_solver_name=cusolver"
    "--control.processing_unit=gpu --control.mpi_grid_dims=2:2 --control.std_evp_solver_name=scalapack --control.gen_evp_solver_name=scalapack"
    "--control.processing_unit=gpu --control.std_evp_solver_name=cusolver --control.gen_evp_solver_name=cusolver"
    "--control.processing_unit=cpu"
    "--control.processing_unit=cpu --control.mpi_grid_dims=2:2 --control.std_evp_solver_name=scalapack --control.gen_evp_solver_name=scalapack")

# todo: Add OMP_NUM_THREADS + srun / mpiexec flags here too?

if(BUILD_TESTING)
    file(GLOB dirs LIST_DIRECTORIES true "${CMAKE_SOURCE_DIR}/verification/test*")

    foreach(full_path ${dirs})
        get_filename_component(test_name "${full_path}" NAME)
        set(test_name "sirius.scf_${test_name}")

        # Now iterate over the different setups (cmake 3.17 has ZIP_LISTS)
        list(LENGTH SIRIUS_SCF_LABELS len)
        math(EXPR len "${len}-1")

        foreach(idx RANGE ${len})
            list(GET SIRIUS_SCF_LABELS ${idx} sirius_scf_label)
            list(GET SIRIUS_SCF_FLAGS ${idx} sirius_scf_flag)
            set(full_test_name "${test_name}_${sirius_scf_label}")
            add_test(NAME "${full_test_name}" COMMAND $<TARGET_FILE:sirius.scf> --test_against=output_ref.json "${sirius_scf_flag}" WORKING_DIRECTORY "${full_path}")
            set_tests_properties("${full_test_name}" PROPERTIES LABELS "integration_test ${sirius_scf_label}")
        endforeach()
    endforeach()
endif()