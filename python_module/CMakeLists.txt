find_package(Python3 COMPONENTS Development REQUIRED)
set(CMAKE_CXX_STANDARD 17)
find_package(mpi4py REQUIRED)

# TODO: for some reason PYBIND11 is not happy here; explicit export of PYBIND11_CPP17 is needed for Intel compiler
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  add_definitions("-DPYBIND11_CPP17")
endif()

set(pb11_src_dir "${PROJECT_SOURCE_DIR}/python_module/pybind11")
check_git_submodule(pybind11 "${pb11_src_dir}")
if(NOT pybind11_avail)
  # attempt to find system installation of pybind11
  find_package(pybind11 REQUIRED)
  if (NOT pybind11_FOUND)
    message(FATAL_ERROR "ERROR: pybind11 cannot be found!")
  endif()
else()
  add_subdirectory(pybind11)
endif()

set(libname py_sirius)
pybind11_add_module(${libname} py_sirius.cpp py_sirius_operators.cpp py_sirius_r3.cpp)

add_dependencies(${libname} sirius_cxx)
target_link_libraries(${libname} PRIVATE mpi4py::mpi4py)
if(USE_MKL)
  # workaround for `Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.`
  target_link_libraries(${libname} PRIVATE ${MKL_DEF_LIBRARY})
endif()

# figure out correct python site-packages directory for installation
file(WRITE "${CMAKE_BINARY_DIR}/install-prefix" "${CMAKE_INSTALL_PREFIX}")
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "${PROJECT_SOURCE_DIR}/python_module/cmake_helpers/where.py"
  INPUT_FILE "${CMAKE_BINARY_DIR}/install-prefix"
  OUTPUT_VARIABLE SIRIUS_PYTHON_LIB_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# set relative RPATH
if(isSystemDir STREQUAL "-1")
  file(RELATIVE_PATH relDir ${SIRIUS_PYTHON_LIB_PATH}/sirius
    ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
  set_target_properties(${libname} PROPERTIES INSTALL_RPATH "${basePoint};${basePoint}/${relDir}")
endif()

target_link_libraries(${libname} PRIVATE sirius_cxx)
# collect python files in module dir
# install to cmake prefix
install(DIRECTORY sirius
  DESTINATION
  ${SIRIUS_PYTHON_LIB_PATH}
  FILES_MATCHING REGEX
  ".*py"
)
install(TARGETS ${libname}
  LIBRARY
  DESTINATION ${SIRIUS_PYTHON_LIB_PATH}/sirius)
install(
  PROGRAMS
  ${CMAKE_SOURCE_DIR}/python_module/apps/neugebaur_cg.py
  ${CMAKE_SOURCE_DIR}/python_module/apps/marzari_cg.py
  ${CMAKE_SOURCE_DIR}/python_module/apps/nlcg.py
  ${CMAKE_SOURCE_DIR}/python_module/apps/nlcg
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
