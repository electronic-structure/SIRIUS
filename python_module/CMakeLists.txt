if(CREATE_PYTHON_MODULE)
  set(CMAKE_CXX_STANDARD 14)
  find_package(mpi4py REQUIRED)

  # TODO: for some reason PYBIND11 is not happy here; explicit export of PYBIND11_CPP14 is needed for Intel compiler
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    add_definitions("-DPYBIND11_CPP14")
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
  pybind11_add_module(${libname} py_sirius.cpp)

  add_dependencies(${libname} sirius)
  target_link_libraries(${libname} PRIVATE mpi4py::mpi4py)
  if(USE_MKL)
    # workaround for `Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.`
    target_link_libraries(${libname} PRIVATE ${MKL_DEF_LIBRARY})
  endif()

  set(PYTHON_INSTALL_SITE_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)

  # set relative RPATH
  if(isSystemDir STREQUAL "-1")
    file(RELATIVE_PATH relDir ${PYTHON_INSTALL_SITE_DIR}/sirius
      ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set_target_properties(${libname} PROPERTIES INSTALL_RPATH "${basePoint};${basePoint}/${relDir}")
  endif()

  target_link_libraries(${libname} PRIVATE sirius)
  # collect python files in module dir
  # install to cmake prefix
  if(NOT PYTHON2)
    install(DIRECTORY sirius
      DESTINATION
      ${PYTHON_INSTALL_SITE_DIR}
      FILES_MATCHING REGEX
      ".*py"
      )
    install(TARGETS ${libname}
      LIBRARY
      DESTINATION ${PYTHON_INSTALL_SITE_DIR}/sirius)
    install(
      PROGRAMS
      ${CMAKE_SOURCE_DIR}/python_module/apps/neugebaur_cg.py
      ${CMAKE_SOURCE_DIR}/python_module/apps/marzari_cg.py
      ${CMAKE_SOURCE_DIR}/python_module/apps/nlcg.py
      ${CMAKE_SOURCE_DIR}/python_module/apps/nlcg
      DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
      )
  else()
    # minimal support for Python 2.7, full functionality is provided for Python >3.5 only
    install(DIRECTORY sirius_minimal/
      DESTINATION
      ${PYTHON_INSTALL_SITE_DIR}sirius
      FILES_MATCHING REGEX
      ".*py")
    install(TARGETS ${libname}
      LIBRARY
      DESTINATION ${PYTHON_INSTALL_SITE_DIR}/sirius)
  endif()

endif(CREATE_PYTHON_MODULE)
