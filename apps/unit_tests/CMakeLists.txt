# FILE(GLOB _tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
set(_tests "test_init;test_sht;test_fft_correctness;test_fft_real;test_spline;test_rot_ylm;test_linalg;test_wf_ortho;test_serialize")

foreach(_test ${_tests})
  string(REGEX REPLACE ".cpp" "" target ${_test})
  add_executable(${target} "${_test}")
  SIRIUS_SETUP_TARGET(${target})
  install(TARGETS ${target} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endforeach()
