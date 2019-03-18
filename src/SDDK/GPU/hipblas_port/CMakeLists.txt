
if(NOT USE_ROCM)
    message(FATAL_ERROR "CMake file must not be included without ROCm enabled!")
endif()

rocm_hip_add_library(hipblas_port SHARED
    rocblas_port_gemv.hip.cpp hipblas_port.hip.cpp rocblas_port_gemm.hip.cpp 
    rocblas_port_trmm.hip.cpp rocblas_port_ger.hip.cpp rocblas_port_axpy.hip.cpp
    FLAGS "-Wno-macro-redefined -std=c++14" INCLUDE_DIRS ${ROCM_INCLUDE_DIRS})

option(BUILD_HIPBLAS_TESTS "Build tests for custom implementation of blas functions in ROCm" OFF)
if (BUILD_HIPBLAS_TESTS)
    # download google test
	set(BUILD_GMOCK OFF)
	set(INSTALL_GTEST OFF)
	include(FetchContent) # requires CMake 3.11
	FetchContent_Declare(
	  googletest
	  GIT_REPOSITORY https://github.com/google/googletest.git
	  GIT_TAG        release-1.8.1
	)
	FetchContent_GetProperties(googletest)
	if(NOT googletest_POPULATED)
	  FetchContent_Populate(googletest)
	  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
	endif()


    add_executable(test_hipblas_port tests/main.cpp tests/gemv_test.cpp tests/gemm_test.cpp tests/trmm_test.cpp tests/ger_test.cpp tests/axpy_test.cpp)
    target_link_libraries(test_hipblas_port ${ROCM_LIBRARIES} hipblas_port gtest_main)
	target_include_directories(test_hipblas_port PRIVATE ${CMAKE_CURRENT_LIST_DIR})
endif()

