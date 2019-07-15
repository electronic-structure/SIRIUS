if(CREATE_PYTHON_MODULE)
  set(CMAKE_CXX_STANDARD 14)
  find_package(mpi4py REQUIRED)

  set(PYBIND11_CPP14 1)

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
  pybind11_add_module(${libname} py_sirius.cpp)

#  if(USE_CUDA)
    add_dependencies(${libname} sirius)
#  endif()
  target_include_directories(${libname} PRIVATE ${MPI4PY_INCLUDE_DIR})
  if(USE_MKL)
    # workaround for `Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.`
    target_link_libraries(${libname} PRIVATE ${MKL_DEF_LIBRARY})
  endif()

  SIRIUS_SETUP_TARGET(${libname})
  # collect python files in module dir
  # install to cmake prefix

  if(NOT PYTHON2)
    install(TARGETS ${libname}
      LIBRARY
      DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/sirius)
    install(DIRECTORY sirius
      DESTINATION
      ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/
      FILES_MATCHING REGEX
      ".*py"
      )
  else()
    # minimal support for Python 2.7, full functionality is provided for Python >3.5 only
    install(TARGETS ${libname}
      LIBRARY
      DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/sirius)
    install(DIRECTORY sirius_minimal/
      DESTINATION
      ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/sirius
      FILES_MATCHING REGEX
      ".*py")
  endif()

endif(CREATE_PYTHON_MODULE)
