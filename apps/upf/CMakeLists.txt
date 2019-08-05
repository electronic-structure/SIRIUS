install(
  PROGRAMS
  ${CMAKE_SOURCE_DIR}/apps/upf/upf_to_json
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

install(
  FILES
  ${CMAKE_SOURCE_DIR}/apps/upf/upf_to_json.py
  ${CMAKE_SOURCE_DIR}/apps/upf/upf2_to_json.py
  ${CMAKE_SOURCE_DIR}/apps/upf/upf1_to_json.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
