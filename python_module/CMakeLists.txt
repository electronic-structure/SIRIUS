if(CREATE_PYTHON_MODULE)
  set(CMAKE_CXX_STANDARD 14)

  find_package(pybind11 REQUIRED)
  include_directories(BEFORE SYSTEM ${pybind11_INCLUDE_DIRS})

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
  install(FILES ${_pyfiles}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/sirius)

endif(CREATE_PYTHON_MODULE)
