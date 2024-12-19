
cmake_minimum_required(VERSION 3.28)

project("HeCBench-roofline")


IF (NOT DEFINED HECBENCH_SRC_DIR)
    set(HECBENCH_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
ENDIF()

IF (NOT DEFINED OFFLOAD_ARCH)
    set(OFFLOAD_ARCH "sm_86")
ENDIF()

message("HECBENCH_SRC_DIR [${HECBENCH_SRC_DIR}]")

# we assume you're using CLANG to build everything
# we assume building for NVIDIA hardware for now
option(BUILD_ALL "Build ALL the codes" OFF)
option(BUILD_OMP "Build OMP variants of target codes" OFF)
option(BUILD_CUDA "Build CUDA variants of target codes" ON)

IF(BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    #find_package(CUDA REQUIRED)
ENDIF()

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children LIST_DIRECTORIES true ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${child})
        IF ((BUILD_CUDA) AND (${child} MATCHES "\-cuda"))
            LIST(APPEND dirlist ${child})
        ENDIF()
        IF ((BUILD_OMP) AND (${child} MATCHES "\-omp"))
            LIST(APPEND dirlist ${child})
        ENDIF()
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


# this is the build include list, by default we just build what's in this list
# unless otherwise instructed
IF(NOT BUILD_ALL)
    message("Only building specific targets!")
    set(TARGET_NAMES atomicReduction attention attentionMultiHead axhelm babelstream backprop)

    SET(TARGET_DIRS "")

    FOREACH(TARGET_NAME ${TARGET_NAMES})
        #message("${TARGET_NAME}")
        IF (BUILD_CUDA)
            LIST(APPEND TARGET_DIRS "${HECBENCH_SRC_DIR}/${TARGET_NAME}-cuda")
        ENDIF()
        IF (BUILD_OMP)
            LIST(APPEND TARGET_DIRS "${HECBENCH_SRC_DIR}/${TARGET_NAME}-omp")
        ENDIF()
    ENDFOREACH()
ELSE()
    message("Building ALL targets!")
    # go into the source directory and gather all the program names
    #message("HECBENCH_SRC_DIR [${HECBENCH_SRC_DIR}]")
    SUBDIRLIST(TARGET_DIRS ${HECBENCH_SRC_DIR})

    SET(TARGET_NAMES "")
ENDIF()


set(IGNORE_TARGETS adv-cuda amgmk-cuda aop-cuda aes-cuda asmooth-cuda allreduce-cuda atomicSystemWide-cuda atomicIntrinsics-cuda atomicPerf-cuda)
set(TARGETS "")

FOREACH(TARGET_DIR ${TARGET_DIRS})
    GET_FILENAME_COMPONENT(TARGET_NAME ${TARGET_DIR} NAME)

    # skip undesired builds
    IF(${TARGET_NAME} IN_LIST IGNORE_TARGETS)
        continue()
    ENDIF()

    #message("${TARGET_NAME}")
    #message("${TARGET_DIR}")
    FILE(GLOB SOURCES_CU LIST_DIRECTORIES false "${TARGET_DIR}/*.cu")
    FILE(GLOB SOURCES_H LIST_DIRECTORIES false "${TARGET_DIR}/*.h")
    FILE(GLOB SOURCES_CPP LIST_DIRECTORIES false "${TARGET_DIR}/*.cpp")
    FILE(GLOB SOURCES_HPP LIST_DIRECTORIES false "${TARGET_DIR}/*.hpp")
    FILE(GLOB SOURCES_C LIST_DIRECTORIES false "${TARGET_DIR}/*.c")
    set(SOURCES ${SOURCES_CU} ${SOURCES_H} ${SOURCES_CPP} ${SOURCES_HPP} ${SOURCES_C})

    # get the subdirectories of this directory
    set(SUBDIRS "")
    set(HAS_ZIP false)

    FILE(GLOB_RECURSE DIR_ELEMS LIST_DIRECTORIES true "${TARGET_DIR}/*")
    FOREACH(elem ${DIR_ELEMS})
        IF (IS_DIRECTORY ${elem})
            LIST(APPEND SUBDIRS ${elem})
        # if we find a zip file or gz file, this is a complicated code, so let's skip building it
        ELSEIF ((${elem} MATCHES ".tar.gz") OR (${elem} MATCHES ".zip") )
            set(HAS_ZIP true)
            break()
        ENDIF()
    ENDFOREACH()
    LIST(LENGTH SUBDIRS NUM_SUBDIRS)

    # if it has a zip file, let's skip this target
    IF(HAS_ZIP)
        continue()
    ENDIF()
    #message("subdirs ${SUBDIRS}")

    # if SOURCES didn't pick anything up, the program probably is large and complex
    # so we're going to omit those files for now. If there are other subdirectories,
    # we will also skip the program for now.
    LIST(LENGTH SOURCES NUM_SOURCES)
    IF ((NUM_SOURCES GREATER 0) AND (NUM_SUBDIRS EQUAL 0))
        #message("sources: [${SOURCES}]")
        add_executable(${TARGET_NAME} ${SOURCES})
        target_include_directories(${TARGET_NAME} PRIVATE "${TARGET_DIR}")
        LIST(APPEND TARGETS ${TARGET_NAME})
    ENDIF()

ENDFOREACH()

LIST(LENGTH TARGETS NUM_TARGETS)
message("TARGETS: [${TARGETS}]")
message("Building [${NUM_TARGETS}] targets!")
set_target_properties(${TARGETS} PROPERTIES LINKER_LANGUAGE CXX)
#set_target_properties(${TARGETS} PROPERTIES COMPILE_FLAGS "-Wall -x cu --offload-arch=${OFFLOAD_ARCH}")

# set target-specific properties
#set_target_properties(axhelm-cuda PROPERTIES COMPILE_FLAGS "-Ddfloat=float -Ddlong=int")
#set_target_properties(axhelm-cuda PROPERTIES LINK_FLAGS "BlasLapack/libBlasLapack.a -lgfortran")
#file(GLOB_RECURSE CFILES "${HECBENCH_SRC_DIR}/amgmk-cuda/*.c")
#set_source_files_properties(${CFILES} PROPERTIES LANGUAGE CUDA)

# these flags/args get applied to all targets
#add_compile_options("-Wall -O3 -std=c++14 -x cu --offload-arch=${OFFLOAD_ARCH} --arch=${OFFLOAD_ARCH}")

