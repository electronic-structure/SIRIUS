add_executable(sirius.nlcg sirius.nlcg.cpp)
target_link_libraries(sirius.nlcg PRIVATE sirius_cxx)
target_link_libraries(sirius.nlcg PRIVATE nlcglib::nlcglib)
install(TARGETS sirius.nlcg RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")


# add_executable(sirius.test.nlcg sirius.test.nlcg.cpp)
# find_package(nlcglib REQUIRED)
# target_link_libraries(sirius.test.nlcg PRIVATE sirius)
# set_property(TARGET sirius.test.nlcg PROPERTY POSITION_INDEPENDENT_CODE OFF)
# install(TARGETS sirius.test.nlcg RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
