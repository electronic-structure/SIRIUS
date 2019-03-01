
if (BUILD_HIPBLAS_TESTS)
    add_subdirectory(ext/googletest)
    add_executable(test_hipblas tests/main.cpp tests/gemv_test.cpp tests/gemm_test.cpp tests/trmm_test.cpp tests/ger_test.cpp tests/axpy_test.cpp)
    target_link_libraries(run_tests ${ROCM_LIBRARIES} hipblas_port gtest_main)
endif()


# set(DEFINITIONS_PROP "$<TARGET_PROPERTY:run_tests,COMPILE_DEFINITIONS>")
# set(DEFINITIONS_GENERATOR "$<$<BOOL:${DEFINITIONS_PROP}>:-D$<JOIN:${DEFINITIONS_PROP}, -D>>")
# set(INCLUDE_DIR_PROP "$<TARGET_PROPERTY:run_tests,INCLUDE_DIRECTORIES>")
# set(INCLUDE_DIR_GENERATOR "$<$<BOOL:${INCLUDE_DIR_PROP}>:-I$<JOIN:${INCLUDE_DIR_PROP}, -I>>")

rocm_hip_add_library(hipblas_port SHARED rocblas_port_gemv.hip.cpp hipblas_port.hip.cpp rocblas_port_gemm.hip.cpp rocblas_port_trmm.hip.cpp rocblas_port_ger.hip.cpp rocblas_port_axpy.hip.cpp
    FLAGS "-Wno-macro-redefined -std=c++14" INCLUDE_DIRS ${ROCM_INCLUDE_DIRS})

if (BUILD_HIPBLAS_TESTS)
    add_subdirectory(ext/googletest)
    add_executable(test_hipblas tests/main.cpp tests/gemv_test.cpp tests/gemm_test.cpp tests/trmm_test.cpp tests/ger_test.cpp tests/axpy_test.cpp)
    target_link_libraries(run_tests ${ROCM_LIBRARIES} hipblas_port gtest_main)
endif()

