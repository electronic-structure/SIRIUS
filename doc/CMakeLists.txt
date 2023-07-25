find_package(Doxygen)

if(DOXYGEN_FOUND AND SIRIUS_BUILD_DOCS)
	set(DOXYGEN_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg")
  # configure doxygen.cfg.in and adapt paths
	configure_file(doxygen.cfg.in ${DOXYGEN_CONFIG_FILE} @ONLY IMMEDIATE)
	set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
	# Add a target and build doc with all targets
	add_custom_target(doc ALL
	  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG_FILE}
	  WORKING_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
  install(DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html/ DESTINATION ${CMAKE_INSTALL_PREFIX}/doc)
endif()
