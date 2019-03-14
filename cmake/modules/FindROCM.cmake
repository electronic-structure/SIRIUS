# - Find the ROCM library
#
# Usage:
#   find_package(ROCM [REQUIRED] [QUIET] COMPONENTS [components ...] )
#
# Compnents available:
#  - hipblas
#  - hipsparse
#  - rocfft
#  - rocblas
#  - rocsparse
#
# Commands made available:
#   rocm_hip_add_library(<name> <sources> [STATIC | SHARED] [FLAGS] <flags> [OUTPUT_DIR] <dir> [INCLUDE_DIRS] <dirs ...>)
#    --- Compiles source files into an imported library with hipcc. No global defitions or include directories are taken into account.
#
# The following variables can be set for compilation:
#   ROCM_HIPCC_FLAGS ----------------- Flags passed on to hipcc compiler
#   ROCM_HIPCC_FLAGS_DEBUG ----------- Flags passed on to hipcc compiler in DEBUG mode
#   ROCM_HIPCC_FLAGS_RELEASE --------- Flags passed on to hipcc compiler in RELEASE mode
#   ROCM_HIPCC_FLAGS_RELWITHDEBINFO -- Flags passed on to hipcc compiler in RELWITHDEBINFO mode
#   ROCM_HIPCC_FLAGS_MINSIZEREL ------ Flags passed on to hipcc compiler in MINSIZEREL mode
#
# The following variables can be set to specify a search location
#   ROCM_ROOT ------------ if set, the libraries are exclusively searched under this path
#   <COMPONENT>_ROOT ------ if set, search for component specific libraries at given path. Takes precedence over ROCM_ROOT
#
# The following variables are generated:
#   ROCM_FOUND ------------------- true if ROCM is found on the system
#   ROCM_LIBRARIES --------------- full path to ROCM
#   ROCM_INCLUDE_DIRS ------------ ROCM include directories
#   ROCM_DEFINITIONS ------------- ROCM definitions
#   ROCM_HCC_EXECUTABLE ---------- ROCM HCC compiler
#   ROCM_HCC-CONFIG_EXECUTABLE --- ROCM HCC config
#   ROCM_HIPCC_EXECUTABLE -------- HIPCC compiler
#   ROCM_HIPCONFIG_EXECUTABLE ---- hip config
#   ROCM_HIPIFY-PERL_EXECUTABLE -- hipify
#   ROCM_HIP_PLATFORM ------------ Platform identifier: "hcc" or "nvcc"
#


set(ROCM_HIPCC_FLAGS "" CACHE STRING "Flags for HIPCC Compiler")
set(ROCM_HIPCC_FLAGS_DEBUG "-g" CACHE STRING "Debug flags for HIPCC Compiler")
set(ROCM_HIPCC_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "Release flags for HIPCC Compiler")
set(ROCM_HIPCC_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" CACHE STRING "Release with debug flags for HIPCC Compiler")
set(ROCM_HIPCC_FLAGS_MINSIZEREL "-Os -DNDEBUG" CACHE STRING "Minimum size flags for HIPCC Compiler")

#If environment variable ROCM_ROOT is specified
if(NOT ROCM_ROOT AND ENV{ROCM_ROOT})
    file(TO_CMAKE_PATH "$ENV{ROCM_ROOT}" ROCM_ROOT)
    set(ROCM_ROOT "${ROCM_ROOT}" CACHE PATH "Root directory for ROCM installation.")
endif()

set(ROCM_FOUND FALSE)
set(ROCM_LIBRARIES)
set(ROCM_INCLUDE_DIRS)
set(ROCM_DEFINITIONS)
unset(ROCM_HCC_EXECUTABLE)
unset(ROCM_HCC-CONFIG_EXECUTABLE)
unset(ROCM_HIPCC_EXECUTABLE)
unset(ROCM_HIPCONFIG_EXECUTABLE)
unset(ROCM_HIPFIY-PERL-EXECUTABLE)
unset(ROCM_HIP_PLATFORM)

include(FindPackageHandleStandardArgs)


# Finds libraries and include path for rocm modules
# IN:
#   - module_name: name of a module (e.g. hcc)
#   - following arguments: name of libraries required
# OUT:
#   - ROCM_LIBRARIES: Appends to list of libraries
#   - ROCM_INCLUDE_DIRS: Appends to include dirs
function(find_rcm_module module_name)
    # convert module name to upper case for consistent variable naming
    string(TOUPPER ${module_name} MODULE_NAME_UPPER)


    if(DEFINED ${MODULE_NAME_UPPER}_ROOT)
	set(ROOT_DIR ${${MODULE_NAME_UPPER}_ROOT})
    elseif(DEFINED ROCM_ROOT)
	set(ROOT_DIR ${ROCM_ROOT})
    endif()

    # get abosolute path to avoid issues with tilde
    if(ROOT_DIR)
        get_filename_component(ROOT_DIR ${ROOT_DIR} ABSOLUTE)
    endif()

    # remove module name from input arguments
    set(LIBRARY_NAMES ${ARGV})
    list(REMOVE_AT LIBRARY_NAMES 0)

    if(${ROCM_FIND_REQUIRED})
	set(ROCM_${MODULE_NAME_UPPER}_FIND_REQUIRED TRUE)
    else()
	set(ROCM_${MODULE_NAME_UPPER}_FIND_REQUIRED FALSE)
    endif()
    if(${ROCM_FIND_QUIETLY})
	set(ROCM_${MODULE_NAME_UPPER}_FIND_QUIETLY TRUE)
    else()
	set(ROCM_${MODULE_NAME_UPPER}_FIND_QUIETLY FALSE)
    endif()

    set(ROCM_LIBRARIES_${MODULE_NAME_UPPER})

    if(ROOT_DIR)
        # find libraries
        foreach(library_name IN LISTS LIBRARY_NAMES)
            find_library(
                ROCM_LIBRARIES_${library_name}
                NAMES ${library_name}
                PATHS ${ROOT_DIR}
                PATH_SUFFIXES "lib" "${module_name}/lib"
                NO_DEFAULT_PATH
            )
	    find_package_handle_standard_args(ROCM_${MODULE_NAME_UPPER} FAIL_MESSAGE
                "For ROCM module ${module_name}, library ${library_name} could not be found. Please specify ROCM_ROOT or ${MODULE_NAME_UPPER}_ROOT." 
                REQUIRED_VARS ROCM_LIBRARIES_${library_name})
	    if(ROCM_LIBRARIES_${library_name})
		list(APPEND ROCM_LIBRARIES_${MODULE_NAME_UPPER} ${ROCM_LIBRARIES_${library_name}})
		mark_as_advanced(ROCM_LIBRARIES_${library_name})
	    endif()
        endforeach()

        # find include directory
        find_path(
            ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER}
            NAMES ${module_name}/include
	    PATHS ${ROOT_DIR} ${ROOT_DIR}/..
            NO_DEFAULT_PATH
        )
        # set include directory for module if found
        if(ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER})
            set(ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER} ${ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER}}/${module_name}/include)
        endif()

    else()

        foreach(library_name IN LISTS LIBRARY_NAMES)
            find_library(
                ROCM_LIBRARIES_${library_name}
                NAMES ${library_name}
                PATHS /opt/rocm
                PATH_SUFFIXES "lib" "lib64" "${module_name}/lib" "rocm/${module_name}/lib"
            )
	    find_package_handle_standard_args(ROCM_${MODULE_NAME_UPPER} FAIL_MESSAGE
                "For ROCM module ${module_name}, library ${library_name} could not be found. Please specify ROCM_ROOT or ${MODULE_NAME_UPPER}_ROOT." 
                REQUIRED_VARS ROCM_LIBRARIES_${library_name})
	    if(ROCM_LIBRARIES_${library_name})
		list(APPEND ROCM_LIBRARIES_${MODULE_NAME_UPPER} ${ROCM_LIBRARIES_${library_name}})
		mark_as_advanced(ROCM_LIBRARIES_${library_name})
	    endif()
        endforeach()

        # find include directory
        find_path(
            ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER}
            NAMES ${module_name}/include
            PATHS /opt/rocm/
        )
        # set include directory for module if found
        if(ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER})
            set(ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER} ${ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER}}/${module_name}/include)
        endif()
    endif()


    # check if all required parts found
    find_package_handle_standard_args(ROCM_${MODULE_NAME_UPPER} FAIL_MESSAGE
        "ROCM module ${module_name} could not be found. Please specify ROCM_ROOT or ${MODULE_NAME_UPPER}_ROOT." 
        REQUIRED_VARS ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER})
    if(ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER})
	mark_as_advanced(ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER})
    endif()

    # set global variables
    if(ROCM_LIBRARIES_${MODULE_NAME_UPPER})
        set(ROCM_LIBRARIES ${ROCM_LIBRARIES} ${ROCM_LIBRARIES_${MODULE_NAME_UPPER}} PARENT_SCOPE)
    endif()
    if(ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER})
        set(ROCM_INCLUDE_DIRS ${ROCM_INCLUDE_DIRS} ${ROCM_INCLUDE_DIRS_${MODULE_NAME_UPPER}} PARENT_SCOPE)
    endif()

endfunction()


# Finds executables of rocm modules
# IN:
#   - module_name: name of a module (e.g. hcc)
#   - executable_name: name of the executable (e.g. hcc)
# OUT:
#   - ROCM_${executable_name}_EXECUTABLE: Path to executable
function(find_rocm_executable module_name executable_name)
    string(TOUPPER ${module_name} MODULE_NAME_UPPER)
    string(TOUPPER ${executable_name} EXECUTABLE_NAME_UPPER)
    unset(ROCM_${EXECUTABLE_NAME_UPPER}_EXECUTABLE PARENT_SCOPE)

    if(DEFINED ${MODULE_NAME_UPPER}_ROOT)
	set(ROOT_DIR ${${MODULE_NAME_UPPER}_ROOT})
    elseif(DEFINED ROCM_ROOT)
	set(ROOT_DIR ${ROCM_ROOT})
    endif()

    # get abosolute path to avoid issues with tilde
    if(ROOT_DIR)
        get_filename_component(ROOT_DIR ${ROOT_DIR} ABSOLUTE)
    endif()

    if(ROOT_DIR)
            find_file(
                ROCM_${EXECUTABLE_NAME_UPPER}_EXECUTABLE
                NAMES ${executable_name}
		PATHS ${ROOT_DIR}
		PATH_SUFFIXES "bin" "${module_name}/bin"
                NO_DEFAULT_PATH
            )
    else()
            find_file(
                ROCM_${EXECUTABLE_NAME_UPPER}_EXECUTABLE
                NAMES ${executable_name}
                PATHS "/opt/rocm"
		PATH_SUFFIXES "bin" "${module_name}/bin"
            )
    endif()
    set(ROCM_${EXECUTABLE_NAME_UPPER} ROCM_${EXECUTABLE_NAME_UPPER} PARENT_SCOPE)

    if(${ROCM_FIND_REQUIRED})
	set(ROCM_${MODULE_NAME_UPPER}_${EXECUTABLE_NAME_UPPER}_FIND_REQUIRED TRUE)
    else()
	set(ROCM_${MODULE_NAME_UPPER}_${EXECUTABLE_NAME_UPPER}_FIND_REQUIRED FALSE)
    endif()
    if(${ROCM_FIND_QUIETLY})
	set(ROCM_${MODULE_NAME_UPPER}_${EXECUTABLE_NAME_UPPER}_FIND_QUIETLY TRUE)
    else()
	set(ROCM_${MODULE_NAME_UPPER}_${EXECUTABLE_NAME_UPPER}_FIND_QUIETLY FALSE)
    endif()
    find_package_handle_standard_args(ROCM FAIL_MESSAGE
	"ROCM_${MODULE_NAME_UPPER}_${EXECUTABLE_NAME_UPPER} ${executable_name} executable could not be found. Please specify ROCM_ROOT or ${MODULE_NAME_UPPER}_ROOT."
        REQUIRED_VARS ROCM_${EXECUTABLE_NAME_UPPER}_EXECUTABLE)
    if(ROCM_${EXECUTABLE_NAME_UPPER}_EXECUTABLE)
	set(ROCM_${EXECUTABLE_NAME_UPPER}_EXECUTABLE ${ROCM_${EXECUTABLE_NAME_UPPER}_EXECUTABLE} PARENT_SCOPE)
	mark_as_advanced(ROCM_${EXECUTABLE_NAME_UPPER}_EXECUTABLE)
    endif()
endfunction()



# find compilers
find_rocm_executable(hcc hcc)
find_rocm_executable(hip hipcc)

if(ROCM_HIPCC_EXECUTABLE AND ROCM_HCC_EXECUTABLE)
    set(ROCM_FOUND TRUE)
else()
    set(ROCM_FOUND FALSE)
    return()
endif()


# find other executables and libraries
find_rocm_executable(hcc hcc-config)
find_rocm_executable(hip hipconfig)
find_rocm_executable(hip hipify-perl)
find_rcm_module(hcc LTO OptRemarks mcwamp mcwamp_cpu mcwamp_hsa hc_am)
find_rcm_module(hip hip_hcc)
find_rcm_module(rocm hsa-runtime64)


# parse hip config
execute_process(COMMAND ${ROCM_HIPCONFIG_EXECUTABLE} -P OUTPUT_VARIABLE ROCM_HIP_PLATFORM RESULT_VARIABLE RESULT_VALUE)
if(NOT ${RESULT_VALUE} EQUAL 0)
    message(FATAL_ERROR "Error parsing platform identifier from hipconfig! Code: ${RESULT_VALUE}")
endif()
if(NOT ROCM_HIP_PLATFORM)
    message(FATAL_ERROR "Empty platform identifier from hipconfig!")
endif()

# set definitions
if("${ROCM_HIP_PLATFORM}" STREQUAL "hcc")
    set(ROCM_DEFINITIONS -D__HIP_PLATFORM_HCC__)
elseif("${ROCM_HIP_PLATFORM}" STREQUAL "nvcc")
    set(ROCM_DEFINITIONS -D__HIP_PLATFORM_NVCC__)
else()
    message(FATAL_ERROR "Could not parse platform identifier from hipconfig! Value: ${ROCM_HIP_PLATFORM}")
endif()

# find libraries for each specified components
foreach(module_name IN LISTS ROCM_FIND_COMPONENTS)
    # set required libaries for each module
    if("${module_name}" STREQUAL "hipblas")
        find_rcm_module(hipblas hipblas)
    elseif("${module_name}" STREQUAL "hipsparse")
        find_rcm_module(hipsparse hipsparse)
    elseif("${module_name}" STREQUAL "rocblas")
        find_rcm_module(rocblas rocblas)
    elseif("${module_name}" STREQUAL "rocsparse")
        find_rcm_module(rocsparse rocsparse)
    elseif("${module_name}" STREQUAL "rocfft")
        find_rcm_module(rocfft rocfft rocfft-device)
    else()
        message(FATAL_ERROR "Unrecognized component \"${module_name}\" in FindROCM module!")
    endif()
endforeach()


# Generates library compiled with hipcc
# Usage:
#   rocm_hip_add_library(<name> <sources> [STATIC | SHARED] [FLAGS] <flags> [OUTPUT_DIR] <dir> [INCLUDE_DIRS] <dirs ...>)
macro(rocm_hip_add_library)
    cmake_parse_arguments(
        HIP_LIB
        "SHARED;STATIC"
        "OUTPUT_DIR"
        "FLAGS;INCLUDE_DIRS"
        ${ARGN}
    )
    # allow either STATIC or SHARED
    if(HIP_LIB_SHARED AND HIP_LIB_STATIC)
        message(FATAL_ERROR "rocm_hip_add_library: library cannot by both static and shared!")
    endif()

    # default to SHARED
    if(NOT (HIP_LIB_SHARED OR HIP_LIB_STATIC))
        set(HIP_LIB_SHARED TRUE)
    endif()

    # default to current binary output directory
    if(NOT HIP_LIB_OUTPUT_DIR)
	set(HIP_LIB_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    # parse positional arguments
    list(LENGTH HIP_LIB_UNPARSED_ARGUMENTS NARGS)
    if(${NARGS} LESS 2)
        message(FATAL_ERROR "rocm_hip_add_library: Not enough arguments!")
    endif()
    list(GET HIP_LIB_UNPARSED_ARGUMENTS 0 HIP_LIB_NAME)
    list(REMOVE_AT HIP_LIB_UNPARSED_ARGUMENTS 0)
    set(HIP_LIB_SOURCES ${HIP_LIB_UNPARSED_ARGUMENTS})

    # generate include flags
    set(_ROCM_FULL_PATH_INCLUDE_FLAGS)
    foreach(_rocm_iternal_dir IN LISTS HIP_LIB_INCLUDE_DIRS)
	if(NOT IS_ABSOLUTE ${_rocm_iternal_dir})
	    get_filename_component(_rocm_iternal_dir ${_rocm_iternal_dir} ABSOLUTE)
	endif()
	list(APPEND _ROCM_FULL_PATH_INCLUDE_FLAGS -I${_rocm_iternal_dir})
    endforeach()

    # generate full path to source files
    unset(_ROCM_SOURCES)
    foreach(source IN LISTS HIP_LIB_SOURCES)
	if(NOT IS_ABSOLUTE ${source})
	    get_filename_component(source ${source} ABSOLUTE)
	endif()
	set(_ROCM_SOURCES ${_ROCM_SOURCES} ${source})
    endforeach()
    get_filename_component(HIP_LIB_OUTPUT_DIR ${HIP_LIB_OUTPUT_DIR} ABSOLUTE)

    # generate flags to use
    set(_ROCM_STD_FLAGS ${HIP_LIB_FLAGS} ${ROCM_HIPCC_FLAGS})
    list(FILTER _ROCM_STD_FLAGS INCLUDE REGEX -std=)
    set(_ROCM_FLAGS ${HIP_LIB_FLAGS})
    if(CMAKE_CXX_STANDARD AND NOT _ROCM_STD_FLAGS)
	list(APPEND _ROCM_FLAGS -std=c++${CMAKE_CXX_STANDARD})
    endif()
    if(CMAKE_BUILD_TYPE)
	string(TOUPPER ${CMAKE_BUILD_TYPE} _ROCM_BUILD_TYPE_UPPER)
	list(APPEND _ROCM_FLAGS ${ROCM_HIPCC_FLAGS_${_ROCM_BUILD_TYPE_UPPER}})
    endif()

    if(NOT ROCM_HIPCC_EXECUTABLE)
	    message(FATAL_ERROR "HIPCC executable not found!")
    endif()

    # create imported shared library
    if(HIP_LIB_SHARED)
        set(_ROCM_FLAGS ${_ROCM_FLAGS} -fPIC)
    endif()

    # compile all files to .o
    set(_ROCM_OBJS)
    set(_ROCM_OBJ_TARGETS)
    foreach(_rocm_file IN LISTS _ROCM_SOURCES)

	# create output directory for .o file
	get_filename_component(_ROCM_CURRENT_DIR ${_rocm_file} DIRECTORY)
	file(RELATIVE_PATH _ROCM_CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ${_ROCM_CURRENT_DIR})
	set(_ROCM_OBJ_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${HIP_LIB_NAME}.dir/${_ROCM_CURRENT_DIR}")
	file(MAKE_DIRECTORY ${_ROCM_OBJ_OUT_DIR})

	# set .o name and path
	get_filename_component(_ROCM_FILE_NAME_ONLY ${_rocm_file} NAME)
	set(_ROCM_OBJ_FILE ${_ROCM_OBJ_OUT_DIR}/${_ROCM_FILE_NAME_ONLY}.o)
	list(APPEND _ROCM_OBJS ${_ROCM_OBJ_FILE})
	list(APPEND _ROCM_OBJ_TARGETS HIP_TARGET_${_ROCM_FILE_NAME_ONLY})

	# compile .o file
	add_custom_target(HIP_TARGET_${_ROCM_FILE_NAME_ONLY} COMMAND ${ROCM_HIPCC_EXECUTABLE} -c ${_rocm_file} -o ${_ROCM_OBJ_FILE} ${_ROCM_FLAGS} ${_ROCM_FULL_PATH_INCLUDE_FLAGS}
	    WORKING_DIRECTORY ${_ROCM_OBJ_OUT_DIR} SOURCES ${_rocm_file})

    endforeach()

    # compile shared library
    if(HIP_LIB_SHARED)
	add_custom_target(HIP_TARGET_${HIP_LIB_NAME} COMMAND ${ROCM_HIPCC_EXECUTABLE} ${_ROCM_OBJS} -fPIC --shared -o ${HIP_LIB_OUTPUT_DIR}/lib${HIP_LIB_NAME}.so
	    ${_ROCM_FLAGS} ${_ROCM_FULL_PATH_INCLUDE_FLAGS}
	    WORKING_DIRECTORY ${HIP_LIB_OUTPUT_DIR})

	add_library(${HIP_LIB_NAME} INTERFACE)
	target_link_libraries(${HIP_LIB_NAME} INTERFACE ${HIP_LIB_OUTPUT_DIR}/lib${HIP_LIB_NAME}.so)

	# add depencies
	add_dependencies(${HIP_LIB_NAME} HIP_TARGET_${HIP_LIB_NAME})
	foreach(_rocm_target IN LISTS _ROCM_OBJ_TARGETS)
	    add_dependencies(HIP_TARGET_${HIP_LIB_NAME} ${_rocm_target})
	endforeach()
    endif()

    # static library
    if(HIP_LIB_STATIC)
        # create library from object files
        add_library(${HIP_LIB_NAME} ${_ROCM_OBJS})
        set_target_properties(${HIP_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
        set_source_files_properties(
            ${_ROCM_OBJS}
            PROPERTIES
            EXTERNAL_OBJECT true
            GENERATED true
            )
	# add dependencies
	foreach(_rocm_target IN LISTS _ROCM_OBJ_TARGETS)
	    add_dependencies(${HIP_LIB_NAME} ${_rocm_target})
	endforeach()
    endif()

endmacro()

