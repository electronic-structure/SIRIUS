if(CREATE_PYTHON_MODULE)
  set(CMAKE_CXX_STANDARD 14)

  set(pb11_src_dir "${PROJECT_SOURCE_DIR}/python_module/pybind11")
  check_git_submodule(pybind11 "${pb11_src_dir}")
  if(NOT pybind11_avail)
    message(FATAL_ERROR "The git submodule for pybind11 is not available, required for python support")
  endif()

  add_subdirectory(pybind11)

  set(libname py_sirius)
  pybind11_add_module(${libname} py_sirius.cpp)
  add_dependencies(${libname} sirius)
  message("in python_module, h5libs: ${HDF5_C_LIBRARIES}")
  SIRIUS_SETUP_TARGET(${libname})

  # collect python files in module dir
  file(GLOB_RECURSE _pyfiles "sirius/*.py")
  # install to cmake prefix
  install(TARGETS ${libname}
    LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/sirius)
  install(DIRECTORY sirius
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/
    FILES_MATCHING REGEX
    ".*py"
    )

endif(CREATE_PYTHON_MODULE)
