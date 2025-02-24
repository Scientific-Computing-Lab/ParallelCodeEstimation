
cmake_minimum_required(VERSION 3.28)

project("HeCBench-roofline")

enable_language(CUDA)
enable_language(Fortran) # need this for axhelm
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message("Host System Processor: [${CMAKE_HOST_SYSTEM_PROCESSOR}]")
message("Requested device arch: [${CUDA_ARCH}]")

#option(OMP_INCLUDE_DIR "OpenMP Include Directory (contains omp.h)" "")


IF (NOT DEFINED HECBENCH_SRC_DIR)
    set(HECBENCH_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
ENDIF()

message("\tHECBENCH_SRC_DIR [${HECBENCH_SRC_DIR}]")

# we assume you're using CLANG to build everything
# we assume building for NVIDIA hardware for now
option(BUILD_ALL "Build ALL the codes" OFF)
option(BUILD_OMP "Build OMP variants of target codes" OFF)
option(BUILD_CUDA "Build CUDA variants of target codes" ON)

#IF(BUILD_CUDA)
    #set(CMAKE_CUDA_ARCHITECTURES "70")
    #set(CMAKE_CUDA_ARCHITECTURES_NATIVE "70-real")

    # we're requiring CUDA as it is expected for building OMP codes anyways
    #message("more vars: [${CMAKE_CUDA_ARCHITECTURES_NATIVE_CLAMP}]")
    #find_package(CUDA REQUIRED)


    #IF(CUDA_HAS_FP16)
    #    message("\tCUDA has FP16 support")
    #ELSE()
    #    message("\tThis version of CUDA does NOT have FP16 support")
    #ENDIF()
#ENDIF()

# we will be using clang++ OMP offloading 
#IF(BUILD_OMP)
#    find_package(OpenMP REQUIRED)
#    #find_program(clang REQUIRED)
#    #find_program(clang-offload-packager-18 REQUIRED)
#ENDIF()

# the typical build pipeline for OMP target codes is:
# clang++ -std=c++14 -O3 --offload-arch=sm_86 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-link -fopenmp-offload-mandatory haccmk.cpp -o haccmk
# LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH ncu -f -o deleteme --set roofline -c 10 ./haccmk 1000
# we can't cuobjdump this program as LLVM did the fat binary packing differently from nvidia
# we can use regular objdump to read the omp_offloading_entries section
# which contains the names of the omp offloaded kernels

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


MACRO(SET_SOURCES_CUDA_LANGUAGE target_name)
    get_target_property(SRC_FILES_TO_CHANGE ${target_name} SOURCES)
    set_source_files_properties(${SRC_FILES_TO_CHANGE} TARGET_DIRECTORY ${target_name} PROPERTIES LANGUAGE CUDA)
ENDMACRO()

MACRO(SET_SOURCES_TO_LANGUAGE target_name target_lang)
    get_target_property(SRC_FILES_TO_CHANGE ${target_name} SOURCES)
    #message("src files to change: ${SRC_FILES_TO_CHANGE}")
    set_source_files_properties(${SRC_FILES_TO_CHANGE} TARGET_DIRECTORY ${target_name} PROPERTIES LANGUAGE ${target_lang})
ENDMACRO()

MACRO(ADD_FILE_TO_TARGET infile outfile target_name target_lang)
    configure_file(${infile} ${outfile} COPYONLY)
    target_sources(${target_name} PRIVATE ${outfile})
    set_source_files_properties(${outfile} PROPERTIES LANGUAGE ${target_lang})
ENDMACRO()

FUNCTION(COPY_FILES_AND_ADD_TO_TARGET files target_name target_lang)

    #message("received files list: [${files}]")

    FOREACH(filepath ${files})
        # generate an outfile name
        #message("looking at file: [${filepath}]")
        get_filename_component(CLEAN_FILE_NAME "${filepath}" NAME)
        #message("extracted name: [${CLEAN_FILE_NAME}]")
        set(NEW_FILE_NAME "${target_name}-${CLEAN_FILE_NAME}")
        #message("new name: [${NEW_FILE_NAME}]")

        ADD_FILE_TO_TARGET(${filepath} ${NEW_FILE_NAME} ${target_name} ${target_lang})
    ENDFOREACH()
ENDFUNCTION()


# this is used to find the files that need to get built
# we essentially call `make clean` and extract the object file names
# of the files that need to be built
MACRO(FINDFILESTOBUILD result makefile)

    #message("makefile using ${makefile}")
    GET_FILENAME_COMPONENT(MAKEFILE_DIR "${makefile}" DIRECTORY)

    # lets call the `make clean` because we can extract the filenames of the object files that get deleted
    execute_process(
        COMMAND /bin/bash -c "make -f ${makefile} clean | grep -oP '\\b[a-zA-Z0-9._-]+(?=\\.\\w+)\\b' | sed 's/\\.o$//' | xargs -I{} find . -type f -name '{}.c' -o -name '{}.cpp' -o -name '{}.cu' -o -name '{}.cc' -o -name '{}.cxx' -o -name '{}.c++' | sed 's|^\\./||' | sed -z 's/\\n/ /g'"
        OUTPUT_VARIABLE MATCHED_FILES
        RESULT_VARIABLE RESULT_CODE
        WORKING_DIRECTORY ${MAKEFILE_DIR}
    )

    string(REPLACE " " ";" MATCHED_FILES_LIST "${MATCHED_FILES}")
    set(FULLPATH_FILES "")

    FOREACH(match ${MATCHED_FILES_LIST})
        #message("Matched file: ${match}")
        LIST(APPEND FULLPATH_FILES "${MAKEFILE_DIR}/${match}")
    ENDFOREACH()
  SET(${result} ${FULLPATH_FILES})
ENDMACRO()


FUNCTION(ISZIPFILE filename)
    GET_FILENAME_COMPONENT(FILEEXT "${zipfile}" EXT)

    set(result false)

    IF(FILEEXT MATCHES "\.bz2")
        set(result true)
    ELSEIF(FILEEXT MATCHES "\.tar")
        set(result true)
    ELSEIF(FILEEXT MATCHES "\.zip")
        set(result true)
    ENDIF()

    return(${result})
ENDFUNCTION()


MACRO(UNZIPFILE zipfile)
    
    # get the filetype extension
    GET_FILENAME_COMPONENT(FILEEXT "${zipfile}" EXT)
    GET_FILENAME_COMPONENT(FILEDIR "${zipfile}" DIRECTORY)

    IF(FILEEXT MATCHES "\.bz2")
        set(TOEXEC "bzip2 -dkf ${zipfile}")
    ELSEIF(FILEEXT MATCHES "\.tar")
        set(TOEXEC "tar -xf ${zipfile}")
    ELSEIF(FILEEXT MATCHES "\.zip")
        set(TOEXEC "unzip ${zipfile}")
    ENDIF()

    execute_process(
        COMMAND /bin/bash -c "${TOEXEC}"
        OUTPUT_VARIABLE EXEC_OUTPUT
        RESULT_VARIABLE RESULT_CODE
        WORKING_DIRECTORY ${FILEDIR}
    )

    # if the result code was not 0
    IF (NOT(RESULT_CODE EQUAL 0))
        message("Problem unzipping file ${zipfile}!!") 
    ENDIF()
ENDMACRO()






MACRO(FINDMAKEFILE result curdir)
    # all of the directories have a capital "M" makefile
    # there should only be one for each source 
    IF(${curdir} MATCHES "\-cuda")
        execute_process(
            COMMAND /bin/bash -c "find ./ -name \"Makefile*\" | sort -r"
            OUTPUT_VARIABLE MATCHED_FILES
            WORKING_DIRECTORY ${curdir}
        )
    ELSEIF(${curdir} MATCHES "\-omp")
        execute_process(
            COMMAND /bin/bash -c "find ./ -name \"[Mm]akefile*\""
            OUTPUT_VARIABLE MATCHED_FILES
            WORKING_DIRECTORY ${curdir}
        )
    ENDIF()

    string(REPLACE "./" "" MATCHED_FILES "${MATCHED_FILES}")
    string(REPLACE "\n" " " MATCHED_FILES "${MATCHED_FILES}")
    string(REPLACE " " ";" MATCHED_FILES_LIST "${MATCHED_FILES}")

    # if there are no matches, it's probably a CMAKE build system
    # or lowercase makefile
    LIST(LENGTH MATCHED_FILES_LIST NUM_MATCHED_FILES)
    IF (NUM_MATCHED_FILES EQUAL 0)
        message("skipping ${curdir}!")
        SET(${result} "NONE")
    ELSE()
        # only grab the first element -- it conveniently happens that in the dirs with
        # multiple Makefiles, the first element is the desired makefile

        # if one of the makefiles has .gpu in it, we'll take that one

        LIST(GET MATCHED_FILES_LIST 0 MATCHED)
        SET(${result} "${curdir}/${MATCHED}")
    ENDIF()
ENDMACRO()


MACRO(GETALLSUBDIRS result curdir)
  FILE(GLOB_RECURSE children LIST_DIRECTORIES true "${curdir}/*")
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


MACRO(INC_ADD_ALL_SUBDIRS TARGET_DIR TARGET_NAME)
        GETALLSUBDIRS(INC_SUBDIRS ${TARGET_DIR})

        LIST(LENGTH INC_SUBDIRS NUM_SUBDIRS)

        target_include_directories(${TARGET_NAME} PUBLIC ${TARGET_DIR})
        target_include_directories(${TARGET_NAME} PRIVATE ${TARGET_DIR})

        IF (NUM_SUBDIRS GREATER 0)
            #message("including extra subdirs for ${TARGET_NAME}, ${INC_SUBDIRS}\n")
            FOREACH(SUBDIR ${INC_SUBDIRS})
                target_include_directories(${TARGET_NAME} PUBLIC "${SUBDIR}")
                target_include_directories(${TARGET_NAME} PRIVATE "${SUBDIR}")
            ENDFOREACH()
        ENDIF()
ENDMACRO()





# this is the build include list, by default we just build what's in this list
# unless otherwise instructed
IF(NOT BUILD_ALL)
    message("Only building specific targets!")
    # set the targets you want built here!
    set(TARGET_NAMES permutate)

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
    message("Will build ALL targets!")
    # go into the source directory and gather all the program names
    #message("HECBENCH_SRC_DIR [${HECBENCH_SRC_DIR}]")
    SUBDIRLIST(TARGET_DIRS ${HECBENCH_SRC_DIR})

    SET(TARGET_NAMES "")
ENDIF()


#LIST(LENGTH TARGET_DIRS NUM_RAW_TARGETS)

#set(IGNORE_TARGETS adv-cuda amgmk-cuda aop-cuda aes-cuda asmooth-cuda allreduce-cuda atomicSystemWide-cuda atomicIntrinsics-cuda atomicPerf-cuda)
# some of these are MPI codes we don't want to concern ourselves with yet
# some have zip files or complicated build filestructures that we will avoid for simplicity right now
# Overall, there are 491 cuda codes in HeCBench, this setup is able to build 417 of them (417/491=0.8492 --> 85%)
# allreduce-cuda uses MPI
# ccl-cuda assumes a multi-gpu setup
# langford-cuda, minibude-cuda can't be built even with default script
# fresnel-cuda is doing wonky things with relocatable code -- not linking correctly
# openmp-cuda, qkv-cuda has a malloc visibility bug in clang
# miniWeather-cuda, pingpong-cuda, sparkler-cuda is mpi
# saxpy-ompt-cuda, streamUM-cuda having problems with finding omp.h correctly -- also doesn't build by default
# slit-cuda missing a file it needs to work -- can't default build
# sss-cuda uses gsl
# testSNAP-cuda is missing an include
# lzss-cuda is unable to download required data files -- broken links
# cmp-cuda is unable to download required data files -- files not online need to contact eborin
# bincount-cuda only makes external CUDA library calls (we can't get the source code)
# cm-cuda requires user-input during execution -- need to support
# voxelization-cuda missing required input files -- can't find a link
# dwconv1d-cuda has no makefile -- it's "run" by a python script
# convolutionDeformable requires pytorch be installed
# halo-finder-cuda is mpi-based
# local-ht-cuda has a type definition error that's also fails with regular makefile
# miniDGS-cuda uses MPI
# xlqc-cuda needs to install GNU scientific library to work
# blas-gemm-omp assumes mkl is installed
# cm-omp having a matching function call error during build
# fsm-omp has critical regions in target region -- not allowed
# interval-omp has a linking error I couldn't figure out, undefined reference -- nvlink 255 errorcode XD
# interval-cuda has a ptxas build error I couldn't figure out -- unresolved extern function
# langford-omp having same mysterious issues as langford-cuda -- can't be built
# memcpy-omp abuses the `auto` keyword and we get a not-assignable error
# miniWeather-omp is MPI-based
# saxpy-ompt-hip/sycl are HIP and SYCL based -- we don't support building this for now
# xlqc-omp requires GSL lib to be installed
# cmp-omp and cmp-cuda require a dataset we need to get from ed borin
# mallocFree-omp and mallocFree-cuda doesn't actually do any compute, it has no kernels that get offloaded
# layernorm-cuda using nvrtc intrinsics that are not easily found by clang
# lda-cuda having an LLVM intrinsics error

# SIGNAL 137 Out-of-memory (OOM)
# mpc-cuda lrn-cuda feynman-kac-cuda bscan-cuda 
# stencil3d-cuda gets out-of-memory error during execution (although it runs fine whout `ncu`) -- need to look into this...
# channelShuffle-omp

# Device out of memory
# dense-embedding-omp lrn-omp

# SIGNAL 11 (segfaulting programs)
# face-omp tridiagonal-cuda cc-cuda 

# SIGNAL 255
# linearprobing-omp

# cuStreamSynchronize illegal memory access
# debayer-omp pointwise-omp

# cudaMemGetInfo illegal memory access
# logic-rewrite-cuda -- I also tried fixing the __managed__ variables by adding __device__ to them

# ptxas error
# sobol-cuda interval-cuda

# strange issue with __host__ __device__ variants not genreating correctly
# xsbench-cuda

set(IGNORE_TARGETS allreduce-cuda ccl-cuda diamond-cuda langford-cuda 
                   miniWeather-cuda minibude-cuda pingpong-cuda si-cuda
                   qkv-cuda rowwiseMoments-cuda saxpy-ompt-cuda slit-cuda sparkler-cuda
                   lzss-cuda cmp-cuda cc-cuda sss-cuda streamUM-cuda stsg-cuda testSNAP-cuda
                   bscan-cuda feynman-kac-cuda lrn-cuda cm-cuda voxelization-cuda mpc-cuda
                   stencil3d-cuda tridiagonal-cuda dwconv1d-cuda convolutionDeformable-cuda
                   halo-finder-cuda local-ht-cuda miniDGS-cuda xlqc-cuda mallocFree-cuda interval-cuda
                   layernorm-cuda  lda-cuda logic-rewrite-cuda sobol-cuda xsbench-cuda
                   blas-gemm-omp cm-omp fsm-omp interval-omp langford-omp mallocFree-omp memcpy-omp
                   miniWeather-omp saxpy-ompt-hip saxpy-ompt-sycl xlqc-omp cmp-omp face-omp channelShuffle-omp
                   debayer-omp dense-embedding-omp lrn-omp linearprobing-omp pointwise-omp)

IF (BUILD_CUDA)
    # if we're using IBM-Power9 on the host system
    IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ppc")
        set(X86INTRIN_CODES heat2d-cuda)
        message("DROPPING CODES THAT USE x86 intrinsics! [${X86INTRIN_CODES}]")
        list(APPEND IGNORE_TARGETS ${X86INTRIN_CODES}) 
    ENDIF()
    # if we don't have FP16 support, some codes can't be built
    #IF (NOT CUDA_HAS_FP16)
    #    list(APPEND IGNORE_TARGETS "conversion-cuda") 
    #ENDIF()
ENDIF()

# We manually build slu-omp later
set(EXISTING_CMAKE_TARGETS slu-omp)

IF(BUILD_CUDA)
    # manually unzip files needed for building of some codes
    UNZIPFILE("${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu.tar.bz2")
    UNZIPFILE("${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu.tar")
    UNZIPFILE("${HECBENCH_SRC_DIR}/axhelm-cuda/BlasLapack.tar.gz")
ENDIF()

IF(BUILD_OMP)
    UNZIPFILE("${HECBENCH_SRC_DIR}/axhelm-omp/BlasLapack.tar.gz")
    UNZIPFILE("${HECBENCH_SRC_DIR}/testSNAP-omp/refdata.tar.gz")
ENDIF()

set(TARGETS "")
set(OMP_TARGETS "")
set(CUDA_TARGETS "")

FOREACH(TARGET_DIR ${TARGET_DIRS})
    GET_FILENAME_COMPONENT(TARGET_NAME ${TARGET_DIR} NAME)

    STRING(REPLACE "\-cuda" "" TARGET_BASENAME ${TARGET_NAME})
    STRING(REPLACE "\-omp" "" TARGET_BASENAME ${TARGET_BASENAME})

    # skip undesired builds
    IF(${TARGET_NAME} IN_LIST IGNORE_TARGETS)
        continue()
    ENDIF()
    IF(${TARGET_NAME} IN_LIST EXISTING_CMAKE_TARGETS)
        continue()
    ENDIF()

    # get the subdirectories of this directory
    #set(SUBDIRS "")

    #FILE(GLOB_RECURSE DIR_ELEMS LIST_DIRECTORIES true "${TARGET_DIR}/*")
    #FOREACH(elem ${DIR_ELEMS})
    #    IF (IS_DIRECTORY ${elem})
    #        LIST(APPEND SUBDIRS ${elem})
    #    # if we find a zip file or gz file, let's unzip it
    #    #ELSEIF (ISZIPFILE(${elem}))
    #    #    UNZIPFILE(${elem})
    #    ENDIF()
    #ENDFOREACH()
    #LIST(LENGTH SUBDIRS NUM_SUBDIRS)

    FILE(GLOB SOURCES_CU LIST_DIRECTORIES false "${TARGET_DIR}/*.c?")
    FILE(GLOB SOURCES_CPP LIST_DIRECTORIES false "${TARGET_DIR}/*.cpp")
    FILE(GLOB SOURCES_C LIST_DIRECTORIES false "${TARGET_DIR}/*.c")
    set(SOURCES ${SOURCES_CU} ${SOURCES_CPP} ${SOURCES_C})

    # add any nested dir files too
    FILE(GLOB SOURCES_CU LIST_DIRECTORIES false "${TARGET_DIR}/**/*.c?")
    FILE(GLOB SOURCES_CPP LIST_DIRECTORIES false "${TARGET_DIR}/**/*.cpp")
    FILE(GLOB SOURCES_C LIST_DIRECTORIES false "${TARGET_DIR}/**/*.c")
    LIST(APPEND SOURCES ${SOURCES_CU})
    LIST(APPEND SOURCES ${SOURCES_CPP})
    LIST(APPEND SOURCES ${SOURCES_C})

    LIST(LENGTH SOURCES NUM_SOURCES)

    IF (NUM_SOURCES GREATER 0)
        #message("sources: [${SOURCES}]")

        FINDMAKEFILE(MAKEFILE ${TARGET_DIR})
        #message("makefile [${MAKEFILE}]")
        FINDFILESTOBUILD(BUILDABLE_FILES ${MAKEFILE})

        LIST(LENGTH BUILDABLE_FILES BUILDFILESNUM)

        IF(BUILDFILESNUM EQUAL 0)
            # if we got no files, assume all files to build
            add_executable(${TARGET_NAME} ${SOURCES})
        ELSE()
            add_executable(${TARGET_NAME} ${BUILDABLE_FILES})
        ENDIF()

        INC_ADD_ALL_SUBDIRS(${TARGET_DIR} ${TARGET_NAME})

        #IF(BUILD_OMP)
        IF(${TARGET_NAME} MATCHES "\-omp")
            INC_ADD_ALL_SUBDIRS("${HECBENCH_SRC_DIR}/${TARGET_BASENAME}-cuda" ${TARGET_NAME})
            INC_ADD_ALL_SUBDIRS("${HECBENCH_SRC_DIR}/${TARGET_BASENAME}-sycl" ${TARGET_NAME})
            LIST(APPEND OMP_TARGETS ${TARGET_NAME})
            SET_SOURCES_TO_LANGUAGE(${TARGET_NAME} CXX)
        ENDIF()
        IF(${TARGET_NAME} MATCHES "\-cuda")
            LIST(APPEND CUDA_TARGETS ${TARGET_NAME})
            #SET_SOURCES_TO_LANGUAGE("${TARGET_NAME}" CUDA)
        ENDIF()

        # add the target to the targets list 
        LIST(APPEND TARGETS ${TARGET_NAME})
    ENDIF()

ENDFOREACH()


# set target-specific properties
IF(BUILD_OMP OR BUILD_CUDA)

    FILE(GLOB NICSLU_LIB "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/source/*.c")
    add_library(nicslu STATIC ${NICSLU_LIB})
    #set_target_properties(nicslu PROPERTIES COMPILE_FLAGS "-iquote /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11 -isystem /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11 -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include -nobuiltininc -H")
    #set_target_properties(nicslu PROPERTIES COMPILE_FLAGS "-isystem /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11 -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include -nobuiltininc")
    set_target_properties(nicslu PROPERTIES COMPILE_FLAGS "${CUSTOM_CUDA_FLAGS}")
    set_target_properties(nicslu PROPERTIES CXX_STANDARD "14")
    SET_SOURCES_CUDA_LANGUAGE(nicslu)

    FILE(GLOB NICSLU_UTIL "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/util/nicslu_readfile.c")
    add_library(nicslu-util STATIC ${NICSLU_UTIL})
    target_include_directories(nicslu-util AFTER PRIVATE "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/source")
    #target_include_directories(nicslu-util AFTER PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/include")

    # we need to manually specify the files for meschach lib    
    SET(MESCHACH_TARGETS copy.c err.c matrixio.c memory.c vecop.c 
                         matop.c pxop.c submat.c init.c otherio.c 
                         machine.c matlab.c ivecop.c version.c 
                         meminfo.c memstat.c lufactor.c bkpfacto.c 
                         chfactor.c qrfactor.c solve.c hsehldr.c 
                         givens.c update.c norm.c hessen.c symmeig.c 
                         schur.c svd.c fft.c mfunc.c bdfactor.c 
                         sparse.c sprow.c sparseio.c spchfctr.c 
                         splufctr.c spbkp.c spswap.c iter0.c 
                         itersym.c iternsym.c zmachine.c zcopy.c 
                         zmatio.c zmemory.c zvecop.c zmatop.c 
                         znorm.c zfunc.c zlufctr.c zsolve.c 
                         zmatlab.c zhsehldr.c zqrfctr.c zgivens.c 
                         zhessen.c zschur.c)
    SET(MESCHACH_LIB "") 
    FOREACH(MESCH_TARGET ${MESCHACH_TARGETS})
        LIST(APPEND MESCHACH_LIB "${HECBENCH_SRC_DIR}/leukocyte-cuda/meschach_lib/${MESCH_TARGET}")
    ENDFOREACH()
    add_library(meschach STATIC ${MESCHACH_LIB})
    #add_library(meschach SHARED ${MESCHACH_LIB})
    FOREACH(MESCH_TARGET ${MESCHACH_LIB})
        set_source_files_properties("${MESCH_TARGET}" PROPERTIES LANGUAGE CXX)
        set_source_files_properties("${MESCH_TARGET}" PROPERTIES INCLUDE_DIRECTORIES "${HECBENCH_SRC_DIR}/leukocyte-cuda/meschach_lib")
    ENDFOREACH()
    # for some weird reason CMake is picking these up even though we don't include them...
    set_source_files_properties("${HECBENCH_SRC_DIR}/leukocyte-omp/meschach_lib/iotort.c" PROPERTIES HEADER_FILE_ONLY ON)
    set_source_files_properties("${HECBENCH_SRC_DIR}/leukocyte-cuda/meschach_lib/iotort.c" PROPERTIES HEADER_FILE_ONLY ON)
    #set_target_properties(meschach PROPERTIES C_STANDARD "90")
    set_target_properties(meschach PROPERTIES LANGUAGE CXX)
    target_compile_options(meschach BEFORE PUBLIC "-DHAVE_CONFIG_H" "-DANSI_C")
    target_link_options(meschach BEFORE PRIVATE "-lm")

    FILE(GLOB BL_FORTRAN_SOURCES "${HECBENCH_SRC_DIR}/axhelm-omp/BlasLapack/*.f")
    add_library(BlasLapack STATIC ${BL_FORTRAN_SOURCES})

    # Have to custom build boxfilter-sycl objects because it's used by both boxfilter-omp and boxfilter-cuda
    #set(BF_BASE_SOURCES "${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp;${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp")
    #add_library(boxfilter-base STATIC ${BF_BASE_SOURCES})
    #set_target_properties(boxfilter-base PROPERTIES COMPILE_FLAGS "${CUSTOM_CUDA_FLAGS}")
    #SET_SOURCES_CUDA_LANGUAGE(boxfilter-base)

    # Have to custom build fluidSim-sycl objects because it's used by both fluidSim-omp and fluidSim-cuda
    # need to move this file so that we get a correct build
    #execute_process(
    #    COMMAND /bin/bash -c "cp ${HECBENCH_SRC_DIR}/fluidSim-cuda/main.cpp ./"
    #    RESULT_VARIABLE RESULT_CODE
    #    WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/fluidSim-omp"
    #)

    #set(FS_BASE_SOURCES "${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp")
    #add_library(fluidSim-base STATIC ${FS_BASE_SOURCES})
    #set_source_files_properties("${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp" PROPERTIES LANGUAGE CXX)
    #target_compile_options(fluidSim-base PRIVATE "-DVERIFY")

    #add_library(mcmd-base STATIC "${HECBENCH_SRC_DIR}/mcmd-cuda/main.cpp")
    #set_source_files_properties("${HECBENCH_SRC_DIR}/mcmd-cuda/main.cpp" PROPERTIES LANGUAGE CXX)
    #target_include_directories(mcmd-base BEFORE PUBLIC "${HECBENCH_SRC_DIR}/mcmd-cuda")

    #target_include_directories(fluidSim-base BEFORE PUBLIC "${HECBENCH_SRC_DIR}/fluidSim-omp")
    #target_compile_options(fluidSim-base PRIVATE "-D__STRICT_ANSI__")
    #set_target_properties(fluidSim-base PROPERTIES LANGUAGE CXX)
    #set_target_properties(fluidSim-base PROPERTIES CXX_STANDARD "14")
    #set_target_properties(fluidSim-base PROPERTIES COMPILE_FLAGS "${CUSTOM_OMP_FLAGS}")
    #target_compile_options(fluidSim-base PRIVATE "-includeutils.h")
    #set_target_properties(fluidSim-base PROPERTIES COMPILE_FLAGS "${CUSTOM_CUDA_FLAGS}")
    #SET_SOURCES_CUDA_LANGUAGE(fluidSim-base)


    # had to manually build slu-omp due to bad auto include ordering
    IF (BUILD_OMP)
        set(SLU_OMP_SRCS "${HECBENCH_SRC_DIR}/slu-omp/numeric.cpp")
        add_executable(slu-omp ${SLU_OMP_SRCS})
        set(TOADD "${HECBENCH_SRC_DIR}/slu-cuda/src/Timer.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/symbolic.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/main.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/preprocess.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" slu-omp CXX)
        list(APPEND OMP_TARGETS "slu-omp")
        list(APPEND TARGETS "slu-omp")
        target_include_directories(slu-omp AFTER PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/include")
        target_include_directories(slu-omp AFTER PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/include")
        target_include_directories(slu-omp AFTER PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/util")
        target_link_options(slu-omp BEFORE PRIVATE "-lpthread" )
        add_dependencies(slu-omp nicslu)
        target_link_libraries(slu-omp nicslu)
        add_dependencies(slu-omp nicslu-util)
        target_link_libraries(slu-omp nicslu-util)
        set_target_properties(slu-omp PROPERTIES COMPILE_FLAGS "-fopenmp -fopenmp-offload-mandatory ${CUSTOM_OMP_FLAGS} -fopenmp-targets=nvptx64-nvidia-cuda -DOMP_TARGET -DOPENMP_TARGET -DUSE_OPENMP_TARGET")
        set_target_properties(slu-omp PROPERTIES LINK_FLAGS "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda ${CUSTOM_OMP_LINK_FLAGS}")
        #set(SLU_OMP_SRCS "${HECBENCH_SRC_DIR}/slu-omp/numeric.cpp")
        #set(SLU_OMP_SRCS "${HECBENCH_SRC_DIR}/slu-cuda/src/Timer.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/symbolic.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/main.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/preprocess.cpp")
        #add_executable(slu-omp ${SLU_OMP_SRCS})
        #list(APPEND OMP_TARGETS "slu-omp")
        #list(APPEND TARGETS "slu-omp")
        #target_include_directories(slu-omp AFTER PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/include")
        #target_include_directories(slu-omp AFTER PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/include")
        #target_include_directories(slu-omp AFTER PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/util")
        #target_link_options(slu-omp BEFORE PRIVATE "-lpthread" )
        #add_dependencies(slu-omp nicslu)
        #target_link_libraries(slu-omp nicslu)
        #add_dependencies(slu-omp nicslu-util)
        #target_link_libraries(slu-omp nicslu-util)
    ENDIF()
ENDIF()



# print some stats
LIST(LENGTH TARGET_DIRS NUM_RAW_TARGETS)
LIST(LENGTH TARGETS NUM_TARGETS)
LIST(LENGTH IGNORE_TARGETS NUM_IGNORE_TARGETS)
LIST(LENGTH CUDA_TARGETS NUM_CUDA_TARGETS)
LIST(LENGTH OMP_TARGETS NUM_OMP_TARGETS)
#message("TARGETS: [${TARGETS}]")
message("Building [${NUM_TARGETS}/${NUM_RAW_TARGETS}] targets!")
message("OMP targets: [${NUM_OMP_TARGETS}] \t CUDA targets: [${NUM_CUDA_TARGETS}]")
message("Manually ignored [${NUM_IGNORE_TARGETS}/${NUM_RAW_TARGETS}] targets!")

set_target_properties(${TARGETS} PROPERTIES CXX_STANDARD "14")

# this is to set general flags that applies to all the targets
IF (BUILD_OMP)
    #set_target_properties(${OMP_TARGETS} PROPERTIES COMPILE_FLAGS "-fopenmp -fopenmp-offload-mandatory -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include/openmp_wrappers -isystem /usr/tce/packages/gcc/gcc-11.2.1/rh/usr/include/c++/11 -isystem /usr/tce/packages/clang/clang-18.1.8/release/lib/clang/18/include/cuda_wrappers -nobuiltininc -fopenmp-targets=nvptx64-nvidia-cuda -DOMP_TARGET -DOPENMP_TARGET -DUSE_OPENMP_TARGET")
    set_target_properties(${OMP_TARGETS} PROPERTIES COMPILE_FLAGS "-fopenmp -fopenmp-offload-mandatory ${CUSTOM_OMP_FLAGS} -fopenmp-targets=nvptx64-nvidia-cuda -DOMP_TARGET -DOPENMP_TARGET -DUSE_OPENMP_TARGET")
    set_target_properties(${OMP_TARGETS} PROPERTIES LINK_FLAGS "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda ${CUSTOM_OMP_LINK_FLAGS}")
ENDIF()



IF (BUILD_CUDA)
    set_target_properties(${CUDA_TARGETS} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")

    set_target_properties(${CUDA_TARGETS} PROPERTIES COMPILE_FLAGS "${CUSTOM_CUDA_FLAGS}")

    set_target_properties(${CUDA_TARGETS} PROPERTIES LINK_FLAGS "${CUSTOM_CUDA_LINK_FLAGS}")

ENDIF()


# Below are custom build commands for some of the targets
IF (BUILD_ALL)

    IF(BUILD_OMP)

        target_compile_options(adv-omp BEFORE PRIVATE "-Ddfloat=float" "-Ddlong=int")

        target_include_directories(aes-omp PRIVATE "${HECBENCH_SRC_DIR}/include")

        target_compile_options(axhelm-omp BEFORE PRIVATE "-Ddfloat=float" "-Ddlong=int")
        add_dependencies(axhelm-omp BlasLapack)
        target_link_libraries(axhelm-omp BlasLapack)
        set_source_files_properties("${HECBENCH_SRC_DIR}/axhelm-omp/axhelmReference.cpp" PROPERTIES HEADER_FILE_ONLY ON)

        target_sources(b+tree-omp PRIVATE "${HECBENCH_SRC_DIR}/b+tree-omp/util/num/num.c")
        target_sources(b+tree-omp PRIVATE "${HECBENCH_SRC_DIR}/b+tree-omp/util/timer/timer.c")
        target_link_options(b+tree-omp BEFORE PRIVATE "-lm")

        target_include_directories(bfs-omp PRIVATE "${HECBENCH_SRC_DIR}/bfs-sycl")

        target_include_directories(bonds-omp PRIVATE "${HECBENCH_SRC_DIR}/bonds-sycl")
        #target_link_options(blas-gemm-omp BEFORE PRIVATE "-lcublas")
        #set_target_properties(blas-gemm-omp PROPERTIES CXX_STANDARD "17")

        #target_compile_options(boxfilter-omp BEFORE PRIVATE "-D__STRICT_ANSI__")

        #set(BF_BASE_SOURCES "${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp;${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp")
        #target_sources(boxfilter-omp PRIVATE ${BF_BASE_SOURCES})
        #set_target_properties(boxfilter-omp PROPERTIES LANGUAGE CXX)
        set(TOADD "${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp;${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" boxfilter-omp CXX)
        #add_dependencies(boxfilter-omp boxfilter-base)
        #target_link_libraries(boxfilter-omp boxfilter-base)
        #target_sources(boxfilter-omp PUBLIC "${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp")
        #target_sources(boxfilter-omp PUBLIC "${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp")
        #set_source_files_properties("${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp;${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp" TARGET_DIRECTORY boxfilter-omp PROPERTIES LANGUAGE CUDA COMPILE_FLAGS "${CUSTOM_CUDA_FLAGS}" LINK_FLAGS "${CUSTOM_CUDA_LINK_FLAGS}")

        #target_compile_options(bwt-omp BEFORE PRIVATE "-D__STRICT_ANSI__")
        target_sources(bwt-omp PRIVATE "${HECBENCH_SRC_DIR}/bwt-cuda/main.cpp")
        set_target_properties(bwt-omp PROPERTIES LANGUAGE CXX)

        #target_compile_options(cm-omp BEFORE PRIVATE "-D__STRICT_ANSI__")
        #target_sources(cm-omp PRIVATE "${HECBENCH_SRC_DIR}/cm-cuda/main.cpp")
        #target_sources(cm-omp PRIVATE "${HECBENCH_SRC_DIR}/cm-cuda/io.cpp")

        #target_sources(cmp-omp PRIVATE "${HECBENCH_SRC_DIR}/cmp-cuda/reference.cpp")
        #target_sources(cmp-omp PRIVATE "${HECBENCH_SRC_DIR}/cmp-cuda/su_cdp.cpp")
        #target_sources(cmp-omp PRIVATE "${HECBENCH_SRC_DIR}/cmp-cuda/su_gather.cpp")
        #target_sources(cmp-omp PRIVATE "${HECBENCH_SRC_DIR}/cmp-cuda/su_trace.cpp")
        #target_sources(cmp-omp PRIVATE "${HECBENCH_SRC_DIR}/cmp-cuda/log.cpp")
        #target_sources(cmp-omp PRIVATE "${HECBENCH_SRC_DIR}/cmp-cuda/parser.cpp")

        #target_compile_options(crs-omp BEFORE PUBLIC "-D__STRICT_ANSI__")
        #target_sources(crs-omp PRIVATE "${HECBENCH_SRC_DIR}/crs-sycl/galois.cpp")
        #target_sources(crs-omp PRIVATE "${HECBENCH_SRC_DIR}/crs-sycl/jerasure.cpp")
        #target_sources(crs-omp PRIVATE "${HECBENCH_SRC_DIR}/crs-sycl/GCRSMatrix.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/crs-sycl/galois.cpp;${HECBENCH_SRC_DIR}/crs-sycl/jerasure.cpp;${HECBENCH_SRC_DIR}/crs-sycl/GCRSMatrix.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" crs-omp CXX)
        set_source_files_properties("${HECBENCH_SRC_DIR}/crs-omp/kernels.cpp" PROPERTIES HEADER_FILE_ONLY ON)

        target_compile_options(degrid-omp BEFORE PUBLIC "-DPRECISION=double")
        set_source_files_properties("${HECBENCH_SRC_DIR}/degrid-omp/kernels.cpp" PROPERTIES HEADER_FILE_ONLY ON)

        #FILE(GLOB DIA_SYCL_C_FILES "${HECBENCH_SRC_DIR}/diamond-sycl/src/blast/*.c")
        FILE(GLOB TOADD "${HECBENCH_SRC_DIR}/diamond-sycl/src/blast/*.c")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" diamond-omp CXX)
        #target_sources(diamond-omp PRIVATE ${DIA_SYCL_C_FILES})
        set(DMND_PATH "${HECBENCH_SRC_DIR}/diamond-sycl/")
        set(DIA_SYCL_CPP_FILES 
            ${DMND_PATH}/src/run/main.cpp
            ${DMND_PATH}/src/basic/config.cpp
            ${DMND_PATH}/src/util/tinythread.cpp
            ${DMND_PATH}/src/util/compressed_stream.cpp
            ${DMND_PATH}/src/basic/score_matrix.cpp
            ${DMND_PATH}/src/blast/blast_filter.cpp
            ${DMND_PATH}/src/blast/blast_seg.cpp
            ${DMND_PATH}/src/data/queries.cpp
            ${DMND_PATH}/src/data/reference.cpp
            ${DMND_PATH}/src/data/seed_histogram.cpp
            ${DMND_PATH}/src/output/daa_record.cpp
            ${DMND_PATH}/src/search/search.cpp
            ${DMND_PATH}/src/util/command_line_parser.cpp
            ${DMND_PATH}/src/util/seq_file_format.cpp
            ${DMND_PATH}/src/util/util.cpp 
            ${DMND_PATH}/src/util/Timer.cpp
            ${DMND_PATH}/src/basic/basic.cpp
            ${DMND_PATH}/src/dp/floating_sw.cpp
            ${DMND_PATH}/src/basic/hssp.cpp
            ${DMND_PATH}/src/dp/ungapped_align.cpp
            ${DMND_PATH}/src/run/tools.cpp
            ${DMND_PATH}/src/dp/greedy_align.cpp
            ${DMND_PATH}/src/run/benchmark.cpp
            ${DMND_PATH}/src/search/stage2.cpp
            ${DMND_PATH}/src/output/output_format.cpp
            ${DMND_PATH}/src/output/join_blocks.cpp
            ${DMND_PATH}/src/data/frequent_seeds.cpp
            ${DMND_PATH}/src/align/query_mapper.cpp
            ${DMND_PATH}/src/align/align_target.cpp
            ${DMND_PATH}/src/output/blast_tab_format.cpp
            ${DMND_PATH}/src/dp/padded_banded_sw.cpp
            ${DMND_PATH}/src/dp/needleman_wunsch.cpp
            ${DMND_PATH}/src/output/blast_pairwise_format.cpp
            ${DMND_PATH}/src/extra/roc.cpp
            ${DMND_PATH}/src/dp/comp_based_stats.cpp
            ${DMND_PATH}/src/extra/model_sim.cpp
            ${DMND_PATH}/src/run/double_indexed.cpp
            ${DMND_PATH}/src/search/collision.cpp
            ${DMND_PATH}/src/output/sam_format.cpp
            ${DMND_PATH}/src/align/align.cpp
            ${DMND_PATH}/src/search/setup.cpp
            ${DMND_PATH}/src/extra/opt.cpp
            ${DMND_PATH}/src/dp/diag_scores.cpp
            ${DMND_PATH}/src/data/taxonomy.cpp
            ${DMND_PATH}/src/lib/tantan/tantan.cpp
            ${DMND_PATH}/src/lib/tantan/tantale.cpp
            ${DMND_PATH}/src/dp/swipe.cpp
            ${DMND_PATH}/src/dp/banded_sw.cpp
            ${DMND_PATH}/src/data/sorted_list.cpp
            ${DMND_PATH}/src/data/seed_set.cpp
            ${DMND_PATH}/src/util/binary_file.cpp
            ${DMND_PATH}/src/output/taxon_format.cpp
            ${DMND_PATH}/src/output/view.cpp
            ${DMND_PATH}/src/output/output_sink.cpp
            ${DMND_PATH}/src/output/target_culling.cpp
        )
        target_sources(diamond-omp PRIVATE ${DIA_SYCL_CPP_FILES})
        target_link_options(diamond-omp BEFORE PRIVATE "-lz" "-lpthread")


        set(TOADD "${HECBENCH_SRC_DIR}/extend2-sycl/read_data.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" extend2-omp CXX)
        #target_sources(extend2-omp PRIVATE "${HECBENCH_SRC_DIR}/extend2-sycl/read_data.cpp")

        # need to replace this with a configure_file call to do the copy instead in the build directory
        # leaving it for now since it works
        #execute_process(
        #    COMMAND /bin/bash -c "cp ${HECBENCH_SRC_DIR}/fluidSim-cuda/main.cpp ./"
        #    RESULT_VARIABLE RESULT_CODE
        #    WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/fluidSim-omp"
        #)

        set(TOADD "${HECBENCH_SRC_DIR}/fluidSim-cuda/main.cpp;${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" fluidSim-omp CXX)
        #target_sources(fluidSim-omp PRIVATE "${HECBENCH_SRC_DIR}/fluidSim-omp/main.cpp;${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp")
        #target_sources(fluidSim-omp PRIVATE "${HECBENCH_SRC_DIR}/fluidSim-omp/main.cpp")
        get_target_property(FLUIDSIM_SOURCES fluidSim-omp SOURCES)
        set_source_files_properties(${FLUIDSIM_SOURCES} PROPERTIES LANGUAGE CXX)
        #set_source_files_properties("${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp" TARGET_DIRECTORY fluidSim-omp PROPERTIES LANGUAGE CXX)
        #set_source_files_properties("${HECBENCH_SRC_DIR}/fluidSim-omp/main.cpp" TARGET_DIRECTORY fluidSim-omp PROPERTIES LANGUAGE CXX)
        #target_include_directories(fluidSim-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/fluidSim-omp")
        target_compile_options(fluidSim-omp PUBLIC "-DVERIFY")
        target_compile_options(fluidSim-omp PUBLIC "-includeutils.h")
        #add_dependencies(fluidSim-omp fluidSim-base)
        #target_link_libraries(fluidSim-omp fluidSim-base)

        target_compile_options(frna-omp PRIVATE "-DINT")

        target_include_directories(gc-omp PRIVATE "${HECBENCH_SRC_DIR}/mis-cuda")

        #target_sources(gd-omp PRIVATE "${HECBENCH_SRC_DIR}/gd-sycl/utils.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/gd-sycl/utils.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" gd-omp CXX)
        set_target_properties(gd-omp PROPERTIES CXX_STANDARD "17")

        # need to make this copy for correct building
        #execute_process(
        #    COMMAND /bin/bash -c "cp ${HECBENCH_SRC_DIR}/haversine-cuda/main.cpp ./"
        #    RESULT_VARIABLE RESULT_CODE
        #    WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/haversine-omp"
        #)
        set(TOADD "${HECBENCH_SRC_DIR}/haversine-cuda/main.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" haversine-omp CXX)
        #target_sources(haversine-omp PRIVATE "${HECBENCH_SRC_DIR}/haversine-omp/main.cpp")

        set_target_properties(heartwall-omp PROPERTIES CXX_STANDARD "11")
        target_sources(heartwall-omp PRIVATE "${HECBENCH_SRC_DIR}/heartwall-omp/util/avi/avilib.c")
        target_sources(heartwall-omp PRIVATE "${HECBENCH_SRC_DIR}/heartwall-omp/util/avi/avimod.c")
        target_sources(heartwall-omp PRIVATE "${HECBENCH_SRC_DIR}/heartwall-omp/util/timer/timer.c")
        target_sources(heartwall-omp PRIVATE "${HECBENCH_SRC_DIR}/heartwall-omp/util/file/file.c")
        set_source_files_properties("${HECBENCH_SRC_DIR}/heartwall-omp/main.c" PROPERTIES HEADER_FILE_ONLY ON)

        set(TOADD "${HECBENCH_SRC_DIR}/hogbom-cuda/main.cpp;${HECBENCH_SRC_DIR}/hogbom-cuda/reference.cpp;${HECBENCH_SRC_DIR}/hogbom-cuda/timer.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" hogbom-omp CXX)
        #target_sources(hogbom-omp PRIVATE "${HECBENCH_SRC_DIR}/hogbom-cuda/main.cpp")
        #target_sources(hogbom-omp PRIVATE "${HECBENCH_SRC_DIR}/hogbom-cuda/reference.cpp")
        #target_sources(hogbom-omp PRIVATE "${HECBENCH_SRC_DIR}/hogbom-cuda/timer.cpp")

        set_target_properties(hybridsort-omp PROPERTIES CXX_STANDARD "11")
        target_link_options(hybridsort-omp BEFORE PRIVATE "-lm")

        #target_sources(hypterm-omp PRIVATE "${HECBENCH_SRC_DIR}/hypterm-cuda/main.cpp")
        #target_sources(hypterm-omp PRIVATE "${HECBENCH_SRC_DIR}/hypterm-cuda/reference.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/hypterm-cuda/main.cpp;${HECBENCH_SRC_DIR}/hypterm-cuda/reference.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" hypterm-omp CXX)

        FILE(GLOB KMEANS_C_FILES "${HECBENCH_SRC_DIR}/kmeans-omp/*.c")
        set_source_files_properties(${KMEANS_C_FILES} PROPERTIES LANGUAGE CXX)


        #target_include_directories(langford-omp PRIVATE "${HECBENCH_SRC_DIR}/include")
        #target_compile_options(langford-omp PUBLIC "-includecommon.h")
        #target_compile_options(langford-omp PUBLIC "-Dpos=fpos" "-DPositions=PositionsGPU")

        #target_sources(lebesgue-omp PRIVATE "${HECBENCH_SRC_DIR}/lebesgue-cuda/main.cpp")
        #target_sources(lebesgue-omp PRIVATE "${HECBENCH_SRC_DIR}/lebesgue-cuda/utils.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/lebesgue-cuda/main.cpp;${HECBENCH_SRC_DIR}/lebesgue-cuda/utils.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" lebesgue-omp CXX)


        FILE(GLOB LEUKOCYTE_C_FILES "${HECBENCH_SRC_DIR}/leukocyte-omp/*.c")
        #message("leukocyte-omp c files ${LEUKOCYTE_C_FILES}")
        #set_target_properties(leukocyte-omp PROPERTIES SOURCES ${LEUKOCYTE_C_FILES})
        target_sources(leukocyte-omp PRIVATE ${LEUKOCYTE_C_FILES})
        #get_target_property(LEUK_SOURCES leukocyte-omp SOURCES)
        #message("leukocyte-omp c files ${LEUK_SOURCES}")
        set_target_properties(leukocyte-omp PROPERTIES INCLUDE_DIRECTORIES "${HECBENCH_SRC_DIR}/leukocyte-omp;${HECBENCH_SRC_DIR}/leukocyte-cuda/meschach_lib")
        set_target_properties(leukocyte-omp PROPERTIES CXX_STANDARD "11")
        #set_target_properties(leukocyte-omp PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${LEUKOCYTE_C_FILES} PROPERTIES LANGUAGE CXX)
        #set_source_files_properties("${HECBENCH_SRC_DIR}/leukocyte-omp/find_ellipse.c" PROPERTIES LANGUAGE CXX)
        #set_source_files_properties("${HECBENCH_SRC_DIR}/leukocyte-omp/find_ellipse.c" PROPERTIES LANGUAGE CXX)
        #set_target_properties(meschach PROPERTIES C_STANDARD "90")
        #set_source_files_properties("${HECBENCH_SRC_DIR}/leukocyte-omp/meschach_lib/tutadv.c" PROPERTIES HEADER_FILE_ONLY ON)
        add_dependencies(leukocyte-omp meschach)
        target_link_libraries(leukocyte-omp meschach)
        target_link_options(leukocyte-omp BEFORE PRIVATE "-lm;-v")

        #target_link_options(leukocyte-omp BEFORE PRIVATE "-L${HECBENCH_SRC_DIR}/leukocyte-cuda/me")


        target_sources(lud-omp PRIVATE "${HECBENCH_SRC_DIR}/lud-omp/common/common.c")
        set_source_files_properties("${HECBENCH_SRC_DIR}/lud-omp/common/common.c" PROPERTIES LANGUAGE CXX)
        #set_target_properties(lud-omp PROPERTIES LANGUAGE CXX)
        #set_source_files_properties("${HECBENCH_SRC_DIR}/lud-cuda/lud_kernels.cu" PROPERTIES HEADER_FILE_ONLY ON)

        #target_compile_options(mallocFree-omp BEFORE PUBLIC "-DUM")

        #execute_process(
        #    COMMAND /bin/bash -c "cp ${HECBENCH_SRC_DIR}/mcmd-cuda/main.cpp ./"
        #    RESULT_VARIABLE RESULT_CODE
        #    WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/mcmd-omp"
        #)
        #target_sources(mcmd-omp PRIVATE "${HECBENCH_SRC_DIR}/mcmd-omp/main.cpp")
        #set_source_files_properties("${HECBENCH_SRC_DIR}/mcmd-omp/main.cpp" TARGET_DIRECTORY mcmd-omp PROPERTIES LANGUAGE CXX)
        set(TOADD "${HECBENCH_SRC_DIR}/mcmd-cuda/main.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" mcmd-omp CXX)
        target_compile_options(mcmd-omp PUBLIC "-D_MWAITXINTRIN_H_INCLUDED" "-D_FORCE_INLINES" "-D__STRICT_ANSI__" "-DGPU")
        target_include_directories(mcmd-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/mcmd-cuda")


        set_target_properties(meanshift-omp PROPERTIES CXX_STANDARD "17")

        execute_process(
            COMMAND /bin/bash -c "make common_files && make generate_info"
            RESULT_VARIABLE RESULT_CODE
            WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/miniFE-omp/src"
        )
        target_compile_options(miniFE-omp BEFORE PUBLIC "-DMINIFE_SCALAR=double" "-DMINIFE_LOCAL_ORDINAL=int" "-DMINIFE_GLOBAL_ORDINAL=int" "-DMINIFE_RESTRICT=__restrict__" "-DMINIFE_CSR_MATRIX" "-DMINIFE_INFO=1" "-DMINIFE_KERNELS=0")
        #target_compile_options(miniFE-omp BEFORE PUBLIC "-D__CUDACC__" "-fopenmp")
        #target_sources(lud-cuda PRIVATE "${HECBENCH_SRC_DIR}/miniFE-cuda/src/miniFE_info.hpp")
        #set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-cuda/src/miniFE_info.hpp" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-omp/basic/utest.cpp" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-omp/basic/main.cpp" PROPERTIES HEADER_FILE_ONLY ON)
        FILE(GLOB MINIFE_INC "${HECBENCH_SRC_DIR}/miniFE-omp/basic/*")
        set_source_files_properties(${MINIFE_INC} PROPERTIES HEADER_FILE_ONLY ON)
        # keeps getting positive header file matches in the /basic dir and we don't want that
        target_include_directories(miniFE-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-omp/utils")
        target_include_directories(miniFE-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-omp/src")
        target_include_directories(miniFE-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-omp/fem")
        get_target_property(MINIFE_SRC miniFE-omp SOURCES)
        set_source_files_properties(${MINIFE_SRC} PROPERTIES LANGUAGE CXX)

        set_target_properties(minibude-omp PROPERTIES CXX_STANDARD "17")


        #configure_file("${HECBENCH_SRC_DIR}/mriQ-cuda/file.cc" "c1.cpp" COPYONLY)
        ##target_sources(mriQ-omp PRIVATE "${HECBENCH_SRC_DIR}/mriQ-cuda/file.cc")
        #target_sources(mriQ-omp PRIVATE "c1.cpp")
        #set_source_files_properties("c1.cpp" PROPERTIES LANGUAGE CXX)
        set(TOADD "${HECBENCH_SRC_DIR}/mriQ-cuda/file.cc")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" mriQ-omp CXX)
        #set_source_files_properties("${HECBENCH_SRC_DIR}/mriQ-cuda/file.cc" PROPERTIES LANGUAGE CXX)

        set_source_files_properties("${HECBENCH_SRC_DIR}/myocyte-omp/kernel/master.c" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/myocyte-omp/kernel/embedded_fehlberg_7_8.c" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/myocyte-omp/kernel/solver.c" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/myocyte-omp/kernel/kernel_fin.c" PROPERTIES HEADER_FILE_ONLY ON)
        target_sources(myocyte-omp PRIVATE "${HECBENCH_SRC_DIR}/myocyte-omp/util/file/file.c")
        target_sources(myocyte-omp PRIVATE "${HECBENCH_SRC_DIR}/myocyte-omp/util/num/num.c")
        target_sources(myocyte-omp PRIVATE "${HECBENCH_SRC_DIR}/myocyte-omp/kernel/kernel_wrapper.c")
        target_include_directories(myocyte-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/myocyte-omp/util/file")
        target_include_directories(myocyte-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/myocyte-omp/util/num")
        get_target_property(MYOCYCTE_SRCS myocyte-omp SOURCES)
        set_source_files_properties(${MYOCYCTE_SRCS} PROPERTIES LANGUAGE CXX)


        target_include_directories(permutate-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/include")
        FILE(GLOB TOADD "${HECBENCH_SRC_DIR}/permutate-cuda/*.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" permutate-omp CXX)
        FILE(GLOB TOADD "${HECBENCH_SRC_DIR}/permutate-cuda/bzip/*.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" permutate-omp CXX)
        set_target_properties(permutate-omp PROPERTIES CXX_STANDARD "17" LANGUAGE CXX)

        target_compile_options(prna-omp BEFORE PUBLIC "-DFLOAT" "-DOMP_TARGET")

        #configure_file("${HECBENCH_SRC_DIR}/pso-cuda/kernel_cpu.cpp" "b1.cpp" COPYONLY)
        #configure_file("${HECBENCH_SRC_DIR}/pso-cuda/main.cpp" "b2.cpp" COPYONLY)
        set(TOADD "${HECBENCH_SRC_DIR}/pso-cuda/kernel_cpu.cpp;${HECBENCH_SRC_DIR}/pso-cuda/main.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" pso-omp CXX)
        #target_sources(pso-omp "b1.cpp;b2.cpp")
        #target_sources(pso-omp PRIVATE "${HECBENCH_SRC_DIR}/pso-cuda/kernel_cpu.cpp")
        #target_sources(pso-omp PRIVATE "${HECBENCH_SRC_DIR}/pso-cuda/main.cpp")

        target_link_options(rsbench-omp BEFORE PRIVATE "-lm")

        #target_sources(rsc-omp PRIVATE "${HECBENCH_SRC_DIR}/rsc-cuda/model_fitting.cpp")
        #target_sources(rsc-omp PRIVATE "b3.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/rsc-cuda/model_fitting.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" rsc-omp CXX)
        target_link_options(rsc-omp BEFORE PRIVATE "-lpthread")

        #configure_file("${HECBENCH_SRC_DIR}/simpleSpmv-cuda/utils.cpp" "b4.cpp" COPYONLY)
        #configure_file("${HECBENCH_SRC_DIR}/simpleSpmv-cuda/main.cpp" "b5.cpp" COPYONLY)
        set(TOADD "${HECBENCH_SRC_DIR}/simpleSpmv-cuda/utils.cpp;${HECBENCH_SRC_DIR}/simpleSpmv-cuda/main.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" simpleSpmv-omp CXX)
        #target_sources(simpleSpmv-omp PRIVATE "b4.cpp;b5.cpp")
        #target_sources(simpleSpmv-omp PRIVATE "${HECBENCH_SRC_DIR}/simpleSpmv-cuda/utils.cpp")
        #target_sources(simpleSpmv-omp PRIVATE "${HECBENCH_SRC_DIR}/simpleSpmv-cuda/main.cpp")

        # for slu - we manually build the necessary files
        # too many include directories from auto-adding, so we manually clear it
        #set(SLU_OMP_SRCS "${HECBENCH_SRC_DIR}/slu-omp/numeric.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/Timer.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/symbolic.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/main.cpp;${HECBENCH_SRC_DIR}/slu-cuda/src/preprocess.cpp")
        #add_executable(slu-omp ${SLU_OMP_SRCS})
        #add_dependencies(slu-omp nicslu)
        #target_link_libraries(slu-omp nicslu)
        #add_dependencies(slu-omp nicslu-util)
        #target_link_libraries(slu-omp nicslu-util)
        #target_include_directories(nicslu-util PRIVATE "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/include")
        #target_include_directories(slu-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/include")
        #target_include_directories(slu-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/include")
        #target_include_directories(slu-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/util")
        ##FILE(GLOB NICSLU_DEMO "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/demo/*.c")
        ##set_source_files_properties(${NICSLU_DEMO} PROPERTIES HEADER_FILE_ONLY ON)
        ##target_sources(slu-omp PRIVATE "${HECBENCH_SRC_DIR}/slu-cuda/src/Timer.cpp")
        ##target_sources(slu-omp PRIVATE "${HECBENCH_SRC_DIR}/slu-cuda/src/symbolic.cpp")
        ##target_sources(slu-omp PRIVATE "${HECBENCH_SRC_DIR}/slu-cuda/src/main.cpp")
        ##target_sources(slu-omp PRIVATE "${HECBENCH_SRC_DIR}/slu-cuda/src/preprocess.cpp")
        #target_link_options(slu-omp BEFORE PRIVATE "-lpthread" )

        target_include_directories(sobel-omp PRIVATE "${HECBENCH_SRC_DIR}/include")

        #target_sources(sptrsv-omp PRIVATE "${HECBENCH_SRC_DIR}/sptrsv-sycl/main.cpp")
        # had to manually modify some code for this one -- it was incorrect OMP code
        set(TOADD "${HECBENCH_SRC_DIR}/sptrsv-sycl/main.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" sptrsv-omp CXX)
        target_compile_options(sptrsv-omp PUBLIC "-DVALUE_TYPE=double")

        FILE(GLOB SRAD_UTIL "${HECBENCH_SRC_DIR}/srad-omp/util/timer/*.c")
        target_sources(srad-omp PRIVATE ${SRAD_UTIL})
        FILE(GLOB SRAD_UTIL "${HECBENCH_SRC_DIR}/srad-omp/util/graphics/*.c")
        target_sources(srad-omp PRIVATE ${SRAD_UTIL})
        target_include_directories(srad-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/srad-omp/util/timer")
        target_include_directories(srad-omp BEFORE PUBLIC "${HECBENCH_SRC_DIR}/srad-omp/util/graphics")
        target_link_options(srad-omp PRIVATE "-lm")
        SET_SOURCES_TO_LANGUAGE(srad-omp CXX) 

        target_include_directories(urng-omp PRIVATE "${HECBENCH_SRC_DIR}/include")

        #target_sources(wordcount-omp PRIVATE "${HECBENCH_SRC_DIR}/wordcount-sycl/main.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/wordcount-sycl/main.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" wordcount-omp CXX)

        target_compile_options(wsm5-omp PRIVATE "-DMKX=4")

        #enable_language(CUDA)
        #target_sources(xsbench-omp PRIVATE "${HECBENCH_SRC_DIR}/xsbench-cuda/Main.cu")
        #target_sources(xsbench-omp PRIVATE "${HECBENCH_SRC_DIR}/xsbench-cuda/io.cu")
        #target_sources(xsbench-omp PRIVATE "${HECBENCH_SRC_DIR}/xsbench-cuda/GridInit.cu")
        #target_sources(xsbench-omp PRIVATE "${HECBENCH_SRC_DIR}/xsbench-cuda/Materials.cu")
        #target_sources(xsbench-omp PRIVATE "${HECBENCH_SRC_DIR}/xsbench-cuda/XSutils.cu")
        set(TOADD "${HECBENCH_SRC_DIR}/xsbench-cuda/Main.cu;${HECBENCH_SRC_DIR}/xsbench-cuda/io.cu;${HECBENCH_SRC_DIR}/xsbench-cuda/GridInit.cu;${HECBENCH_SRC_DIR}/xsbench-cuda/Materials.cu;${HECBENCH_SRC_DIR}/xsbench-cuda/XSutils.cu")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" xsbench-omp CXX)
        
        #get_target_property(XSBENCH_SRCS xsbench-omp SOURCES)
        #set_source_files_properties(${XSBENCH_SRCS} PROPERTIES LANGUAGE CXX)
        target_link_options(xsbench-omp PRIVATE "-lm")
        #target_sources(xsbench-omp PRIVATE "${HECBENCH_SRC_DIR}/rsbench-cuda/XSutils.cu")
    ENDIF()










    IF(BUILD_CUDA)
        set_target_properties(adv-cuda PROPERTIES COMPILE_FLAGS "-Ddfloat=float -Ddlong=int")

        target_include_directories(aes-cuda PRIVATE "${HECBENCH_SRC_DIR}/include")

        FILE(GLOB AMGMK_C_FILES "${HECBENCH_SRC_DIR}/amgmk-cuda/*.c")
        set_source_files_properties(${AMGMK_C_FILES} PROPERTIES HEADER_FILE_ONLY ON)

        SET_SOURCES_CUDA_LANGUAGE(ans-cuda)

        set_target_properties(axhelm-cuda PROPERTIES COMPILE_FLAGS "-Ddfloat=float -Ddlong=int")
        set_source_files_properties("${HECBENCH_SRC_DIR}/axhelm-cuda/axhelmReference.cpp" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/axhelm-cuda/axhelmKernel.cpp" PROPERTIES HEADER_FILE_ONLY ON)
        #FILE(GLOB BL_FORTRAN_SOURCES "${HECBENCH_SRC_DIR}/axhelm-cuda/BlasLapack/*.f")
        #enable_language(Fortran)
        #add_library(BlasLapack STATIC ${BL_FORTRAN_SOURCES})
        add_dependencies(axhelm-cuda BlasLapack)
        target_link_libraries(axhelm-cuda BlasLapack)

        # don't build these files
        FILE(GLOB BPTREE_IGNORE "${HECBENCH_SRC_DIR}/b+tree-cuda/kernel/kernel.cu")
        set_source_files_properties(${BPTREE_IGNORE} PROPERTIES HEADER_FILE_ONLY ON)
        FILE(GLOB BPTREE_IGNORE "${HECBENCH_SRC_DIR}/b+tree-cuda/kernel/kernel2.cu")
        set_source_files_properties(${BPTREE_IGNORE} PROPERTIES HEADER_FILE_ONLY ON)
        target_sources(b+tree-cuda PRIVATE "${HECBENCH_SRC_DIR}/b+tree-cuda/util/num/num.c")
        target_sources(b+tree-cuda PRIVATE "${HECBENCH_SRC_DIR}/b+tree-cuda/util/timer/timer.c")

        target_include_directories(bfs-cuda PRIVATE "${HECBENCH_SRC_DIR}/bfs-sycl")

        set_target_properties(bicgstab-cuda PROPERTIES LINK_FLAGS "-lcublas -lcusparse")

        #set_target_properties(bitpacking-cuda PROPERTIES CXX_STANDARD "17")

        set_target_properties(blas-dot-cuda PROPERTIES LINK_FLAGS "-lcublas")

        set_target_properties(blas-fp8gemm-cuda PROPERTIES LINK_FLAGS "-lcublasLt")

        set_target_properties(blas-gemmBatched-cuda PROPERTIES LINK_FLAGS "-lcublas")
        set_target_properties(blas-gemmBatched-cuda PROPERTIES CXX_STANDARD "17")

        set_target_properties(blas-gemm-cuda PROPERTIES LINK_FLAGS "-lcublas")
        set_target_properties(blas-gemm-cuda PROPERTIES CXX_STANDARD "17")

        set_target_properties(blas-gemmEx2-cuda PROPERTIES LINK_FLAGS "-lcublasLt")

        set_target_properties(blas-gemmEx-cuda PROPERTIES LINK_FLAGS "-lcublas")

        target_include_directories(blas-gemmStridedBatched-cuda PRIVATE "${HECBENCH_SRC_DIR}/blas-gemmBatched-cuda")
        set_target_properties(blas-gemmStridedBatched-cuda PROPERTIES LINK_FLAGS "-lcublas")
        set_target_properties(blas-gemmStridedBatched-cuda PROPERTIES CXX_STANDARD "17")

        target_include_directories(blockexchange-cuda PRIVATE "${HECBENCH_SRC_DIR}/blockAccess-cuda")
        target_include_directories(blockexchange-cuda PRIVATE "${HECBENCH_SRC_DIR}/warpexchange-cuda")

        # bm3d-cuda only works if we build it for an older version (sm_60)
        set_target_properties(bm3d-cuda PROPERTIES LINK_FLAGS "-lpthread")
        set_target_properties(bm3d-cuda PROPERTIES CUDA_ARCHITECTURES "60")

        target_compile_options(bmf-cuda BEFORE PRIVATE "-DUSE_GPU")
        target_compile_options(bmf-cuda PRIVATE "-fopenmp")
        target_link_options(bmf-cuda PRIVATE "-fopenmp")

        SET_SOURCES_CUDA_LANGUAGE(boxfilter-cuda)
        target_include_directories(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl")
        #target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp")
        #target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/cmd_arg_reader.cpp")
        #target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp;${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" boxfilter-cuda CXX)
        set(TOADD "${HECBENCH_SRC_DIR}/boxfilter-sycl/cmd_arg_reader.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" boxfilter-cuda CUDA)
        #set_source_files_properties("${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp;${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp" TARGET_DIRECTORY boxfilter-cuda PROPERTIES LANGUAGE CXX)
        #set_source_files_properties("${HECBENCH_SRC_DIR}/boxfilter-sycl/cmd_arg_reader.cpp" TARGET_DIRECTORY boxfilter-cuda PROPERTIES LANGUAGE CUDA CUDA_ARCHITECTURES "70")

        #set_source_files_properties("${HECBENCH_SRC_DIR}/bsw-cuda/driver.cu" PROPERTIES COMPILE_OPTIONS "-fcuda-rdc;--cuda-device-only")
        #set_source_files_properties("${HECBENCH_SRC_DIR}/bsw-cuda/kernel.cu" PROPERTIES COMPILE_OPTIONS "--offload-new-driver")
        #set_source_files_properties("${HECBENCH_SRC_DIR}/bsw-cuda/driver.cu" PROPERTIES COMPILE_OPTIONS "--offload-new-driver")
        #target_compile_options(bsw-cuda PRIVATE "--cuda-device-only" "-fcuda-rdc")
        #target_compile_options(bsw-cuda PUBLIC "-includemath.h" "-relocatable-pch" "-fgpu-rdc" "--cuda-device-only")
        set_source_files_properties("${HECBENCH_SRC_DIR}/bsw-cuda/kernel.cu" PROPERTIES HEADER_FILE_ONLY ON)
        #target_link_options(bsw-cuda PUBLIC "--offload-link")

        # excluded
        #target_include_directories(cc-cuda PRIVATE "${HECBENCH_SRC_DIR}/mis-cuda")
        target_link_libraries(ced-cuda pthread)
        FILE(GLOB CED_IGNORE "${HECBENCH_SRC_DIR}/ced-cuda/main.cpp")
        set_source_files_properties(${CED_IGNORE} PROPERTIES HEADER_FILE_ONLY ON)

        #set_target_properties(d3q19-bgk-cuda PROPERTIES CXX_STANDARD "11")
        target_compile_options(d3q19-bgk-cuda PUBLIC "-Wno-c++11-narrowing")

        target_include_directories(debayer-cuda PRIVATE "${HECBENCH_SRC_DIR}/debayer-sycl")

        set_target_properties(degrid-cuda PROPERTIES COMPILE_FLAGS "-DPRECISION=double")

        #set_target_properties(determinant-cuda PROPERTIES LINK_FLAGS "-lcusolver")
        target_link_options(determinant-cuda PRIVATE "-lcusolver")

        target_link_libraries(dp-cuda "cublas")
        target_compile_options(dp-cuda PUBLIC "-D_PSTL_PAR_BACKEND_SERIAL")
        set_target_properties(dp-cuda PROPERTIES CXX_STANDARD "17")

        target_include_directories(dwconv-cuda PRIVATE "${HECBENCH_SRC_DIR}/tensorAccessor-cuda")

        target_include_directories(extend2-cuda PRIVATE "${HECBENCH_SRC_DIR}/extend2-sycl")
        #target_sources(extend2-cuda PRIVATE "${HECBENCH_SRC_DIR}/extend2-sycl/read_data.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/extend2-sycl/read_data.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" extend2-cuda CUDA)

        set_target_properties(f16sp-cuda PROPERTIES LINK_FLAGS "-lcublas")

        target_compile_options(face-cuda BEFORE PUBLIC "-DGPU")

        target_include_directories(floydwarshall2-cuda PRIVATE "${HECBENCH_SRC_DIR}/mis-cuda")

        target_compile_options(fluidSim-cuda PRIVATE "-DVERIFY")
        SET_SOURCES_CUDA_LANGUAGE(fluidSim-cuda)
        #target_sources(fluidSim-cuda PRIVATE "${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp")
        #set_source_files_properties("${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp" PROPERTIES LANGUAGE CXX)
        set(TOADD "${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" fluidSim-cuda CUDA)
        target_include_directories(fluidSim-cuda PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
        target_compile_options(fluidSim-cuda PUBLIC "-DVERIFY")
        #add_dependencies(fluidSim-cuda fluidSim-base)
        #target_link_libraries(fluidSim-cuda fluidSim-base)

        #set_target_properties(fpdc-cuda PROPERTIES COMPILE_FLAGS "-std=c++14")
        #target_compile_features(fpdc-cuda PRIVATE cxx_std_14)
        #set_target_properties(fpdc-cuda PROPERTIES CXX_STANDARD "14")

        FILE(GLOB FRECHET_CPP_FILES "${HECBENCH_SRC_DIR}/frechet-cuda/*.cpp")
        set_source_files_properties(${FRECHET_CPP_FILES} PROPERTIES HEADER_FILE_ONLY ON)


        # fresnel is having relocatable code issues...
        ##set_target_properties(fresnel-cuda PROPERTIES COMPILE_FLAGS "-use_fast_math")
        #set_target_properties(fresnel-cuda PROPERTIES CUDA_ARCHITECTURES "60")
        #set_target_properties(fresnel-cuda PROPERTIES LINK_DIRECTORIES "/usr/local/cuda-12.6/bin/")
        #target_compile_options(fresnel-cuda BEFORE PRIVATE "-rdc=true")
        #target_compile_options(fresnel-cuda BEFORE PRIVATE "-use_fast_math")
        ##target_compile_options(fresnel-cuda BEFORE PRIVATE "-ccbin=gcc")
        #target_compile_options(fresnel-cuda BEFORE PRIVATE "-dlink")
        ##target_compile_options(fresnel-cuda BEFORE PRIVATE "-ccbin=gcc")
        ##target_link_options(fresnel-cuda PRIVATE "-dlink")
        #target_link_options(fresnel-cuda BEFORE PRIVATE "-fuse-ld=/usr/local/cuda-12.6/bin/nvlink")
        ##target_link_options(fresnel-cuda PRIVATE "-fuse-ld=nvlink")
        ##target_link_options(fresnel-cuda PRIVATE "--offload-link=nvlink")
        ##target_compile_options(fresnel-cuda BEFORE PRIVATE "-fuse-ld=nvlink")
        set_target_properties(fresnel-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_compile_options(frna-cuda BEFORE PRIVATE "-use_fast_math" "-DINT")

        FILE(GLOB GC_IGNORE "${HECBENCH_SRC_DIR}/gamma-correction-cuda/main.cpp")
        set_source_files_properties(${GC_IGNORE} PROPERTIES HEADER_FILE_ONLY ON)

        target_link_options(geam-cuda PRIVATE "-lcublas")

        set_target_properties(gemv-cuda PROPERTIES CXX_STANDARD "17")

        target_include_directories(gc-cuda PRIVATE "${HECBENCH_SRC_DIR}/mis-cuda")

        set_target_properties(gels-cuda PROPERTIES CXX_STANDARD "17")
        target_link_options(gels-cuda PRIVATE "-lcublas")

        target_compile_options(gpp-cuda PRIVATE "-D__NVCC__")

        # need this flag if you're compiling with nvcc
        # add an if statement here later to check if we are using nvcc instead of clang
        #target_compile_options(hausdorff-cuda PRIVATE "--expt-relaxed-constexpr")

        SET_SOURCES_CUDA_LANGUAGE(haversine-cuda)

        SET_SOURCES_CUDA_LANGUAGE(hbc-cuda)

        target_sources(heartwall-cuda PRIVATE "${HECBENCH_SRC_DIR}/heartwall-cuda/util/avi/avilib.c")
        target_sources(heartwall-cuda PRIVATE "${HECBENCH_SRC_DIR}/heartwall-cuda/util/avi/avimod.c")
        target_sources(heartwall-cuda PRIVATE "${HECBENCH_SRC_DIR}/heartwall-cuda/util/timer/timer.c")
        target_sources(heartwall-cuda PRIVATE "${HECBENCH_SRC_DIR}/heartwall-cuda/util/file/file.c")

        FILE(GLOB HS_IGNORE "${HECBENCH_SRC_DIR}/hybridsort-cuda/tmp/*.cpp")
        set_source_files_properties(${HS_IGNORE} PROPERTIES HEADER_FILE_ONLY ON)

        #target_link_options(interval-cuda PRIVATE "-lboost")

        # this technically makes it so that nvcc in invoked to build these files instead of clang++
        #get_target_property(KMC_SOURCES kmc-cuda SOURCES)
        #set_source_files_properties(${KMC_SOURCES} PROPERTIES LANGUAGE CUDA)
        set_source_files_properties("${HECBENCH_SRC_DIR}/kmc-cuda/driver.c" PROPERTIES HEADER_FILE_ONLY ON)
        target_include_directories(kmc-cuda PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
        target_link_options(kmc-cuda PRIVATE  "-L${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib" "-lcublas" "-lcudart")


        # need to remove the MESCHACH_LIB files from the leukocyte source files 
        # but it's okay since they don't get used in the final linking
        #FILE(GLOB MESCHACH_LIB "${HECBENCH_SRC_DIR}/leukocyte-cuda/meschach_lib/*.c")
        #add_library(meschach STATIC ${MESCHACH_LIB})
        #set_target_properties(leukocyte-cuda PROPERTIES CXX_STANDARD "11")
        #set_target_properties(leukocyte-cuda PROPERTIES LANGUAGE CXX)
        FILE(GLOB LEUKOCYTE_C_FILES "${HECBENCH_SRC_DIR}/leukocyte-cuda/*.c")
        #message("leukocyte-omp c files ${LEUKOCYTE_C_FILES}")
        #set_target_properties(leukocyte-omp PROPERTIES SOURCES ${LEUKOCYTE_C_FILES})
        target_sources(leukocyte-cuda PRIVATE ${LEUKOCYTE_C_FILES})
        #get_target_property(LEUK_SOURCES leukocyte-omp SOURCES)
        #message("leukocyte-omp c files ${LEUK_SOURCES}")
        set_target_properties(leukocyte-cuda PROPERTIES INCLUDE_DIRECTORIES "${HECBENCH_SRC_DIR}/leukocyte-cuda;${HECBENCH_SRC_DIR}/leukocyte-cuda/meschach_lib")
        set_target_properties(leukocyte-cuda PROPERTIES CXX_STANDARD "11")
        #set_target_properties(leukocyte-omp PROPERTIES LANGUAGE CXX)
        set_source_files_properties(${LEUKOCYTE_C_FILES} PROPERTIES LANGUAGE CUDA)
        #get_target_property(LEUKO_SRCS leukocyte-cuda SOURCES)
        #set_source_files_properties(${LEUKO_SRCS} PROPERTIES LANGUAGE CUDA)
        add_dependencies(leukocyte-cuda meschach)
        target_link_libraries(leukocyte-cuda meschach)


        #target_compile_definitions(local-ht-cuda PUBLIC "-Dunsigned=uint32_t")
        #set_target_properties(local-ht-cuda PROPERTIES COMPILE_FLAGS "-Uuint32_t -Duint32_t=unsigned")
        #set_target_properties(local-ht-cuda PROPERTIES COMPILE_FLAGS "--include=bits/stdint-uintn.h")
        #set_target_properties(local-ht-cuda PROPERTIES COMPILE_FLAGS "--include=bits/types.h")

        # the maxregcount is needed for nvcc compilation
        #target_compile_options(logan-cuda BEFORE PUBLIC "-DADAPTABLE" "-maxrregcount=32")

        target_compile_options(logan-cuda BEFORE PUBLIC "-DADAPTABLE")
        set_target_properties(logan-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        FILE(GLOB LOGAN_INC "${HECBENCH_SRC_DIR}/logan-cuda/src/*.cuh")
        set_source_files_properties(${LOGAN_INC} PROPERTIES HEADER_FILE_ONLY ON)
        target_compile_options(logan-cuda PRIVATE "-fopenmp")
        target_link_options(logan-cuda PRIVATE "-fopenmp")
        #target_compile_options(logan-cuda PRIVATE "-fopenmp=libomp" "-I${OMP_INCLUDE_DIR}")
        #target_link_options(logan-cuda PRIVATE "${OMP_LINKER_FLAGS}")

        #set_target_properties(logic-rewrite-cuda PROPERTIES CXX_STANDARD "17")
        target_compile_definitions(lr-cuda PUBLIC "-D__NVCC__")

        # manually fixing a mispelling in thier cuda code
        target_compile_definitions(lsqt-cuda PUBLIC "__syncwrap=__syncwarp")

        target_sources(lud-cuda PRIVATE "${HECBENCH_SRC_DIR}/lud-cuda/common/common.cu")
        set_source_files_properties("${HECBENCH_SRC_DIR}/lud-cuda/lud_kernels.cu" PROPERTIES HEADER_FILE_ONLY ON)

        target_link_options(ludb-cuda PRIVATE "-lcublas")

        SET_SOURCES_CUDA_LANGUAGE(mcmd-cuda)

        set_source_files_properties("${HECBENCH_SRC_DIR}/md5hash-cuda/MD5Hash.cpp" PROPERTIES HEADER_FILE_ONLY ON)

        set_target_properties(meanshift-cuda PROPERTIES CXX_STANDARD "17")

        set_target_properties(merkle-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        #target_compile_options(merkle-cuda PRIVATE "--expt-relaxed-constexpr") #"--extended-lambda")

        target_link_options(mf-sgd-cuda PRIVATE "-lpthread" )
        target_include_directories(mf-sgd-cuda PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
        # not allowed to use the __shfl() operation on compute version 7X and higher
        set_target_properties(mf-sgd-cuda PROPERTIES CUDA_ARCHITECTURES "60")
        SET_SOURCES_CUDA_LANGUAGE(mf-sgd-cuda)


        execute_process(
            COMMAND /bin/bash -c "make common_files && make generate_info"
            RESULT_VARIABLE RESULT_CODE
            WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/miniFE-cuda/src"
        )
        target_compile_options(miniFE-cuda BEFORE PUBLIC "-DMINIFE_SCALAR=double" "-DMINIFE_LOCAL_ORDINAL=int" "-DMINIFE_GLOBAL_ORDINAL=int" "-DMINIFE_RESTRICT=__restrict__" "-DMINIFE_CSR_MATRIX" "-DMINIFE_INFO=1" "-DMINIFE_KERNELS=0")
        #target_compile_options(miniFE-omp BEFORE PUBLIC "-D__CUDACC__" "-fopenmp")
        #target_sources(lud-cuda PRIVATE "${HECBENCH_SRC_DIR}/miniFE-cuda/src/miniFE_info.hpp")
        #set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-cuda/src/miniFE_info.hpp" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-cuda/basic/utest.cpp" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-cuda/basic/main.cpp" PROPERTIES HEADER_FILE_ONLY ON)
        FILE(GLOB MINIFE_INC "${HECBENCH_SRC_DIR}/miniFE-cuda/basic/*")
        set_source_files_properties(${MINIFE_INC} PROPERTIES HEADER_FILE_ONLY ON)
        target_include_directories(miniFE-cuda BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-cuda/utils")
        target_include_directories(miniFE-cuda BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-cuda/src")
        target_include_directories(miniFE-cuda BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-cuda/fem")
        #get_target_property(MINIFE_SRC miniFE-cuda SOURCES)
        #set_source_files_properties(${MINIFE_SRC} PROPERTIES LANGUAGE CUDA)
        SET_SOURCES_CUDA_LANGUAGE(miniFE-cuda)
        #set_target_properties(miniFE-cuda PROPERTIES CXX_STANDARD "11")

        set_target_properties(minmax-cuda PROPERTIES CXX_STANDARD "17")
        #target_compile_options(minmax-cuda PRIVATE "-ferror-limit=3")
        #target_include_directories(minmax-cuda PUBLIC "/usr/local/cuda-12.6/targets/x86_64-linux/include")
        #target_compile_options(minmax-cuda PUBLIC "-includecub/cub.cuh" "-D__NVCC__")
        target_compile_options(minmax-cuda PUBLIC "-D__NVCC__")
        #target_compile_options(minmax-cuda PUBLIC "-includecub/thread/thread_reduce.cuh")

        target_include_directories(mmcsf-cuda BEFORE PUBLIC "${Boost_INCLUDE_DIR}")
        target_link_libraries(mmcsf-cuda ${Boost_LIBRARIES})

        target_link_options(mrg32k3a-cuda PRIVATE "-lcurand")

        target_link_options(norm2-cuda PRIVATE "-lcurand" "-lcublas")

        target_compile_options(openmp-cuda PRIVATE "-fopenmp=libomp")
        target_compile_options(openmp-cuda PRIVATE "-I${HECBENCH_SRC_DIR}/openmp-omp" "-fopenmp" )
        target_link_options(openmp-cuda PRIVATE "-fopenmp")
        #target_compile_options(openmp-cuda PRIVATE "-I${HECBENCH_SRC_DIR}/openmp-omp" "-I${OMP_INCLUDE_DIR}" )
        #target_link_options(openmp-cuda PRIVATE "${OMP_LINKER_FLAGS}")

        SET_SOURCES_CUDA_LANGUAGE(pad-cuda)

        target_link_options(pcc-cuda PRIVATE "-lcublas")

        target_include_directories(permute-cuda PUBLIC "${HECBENCH_SRC_DIR}/qkv-cuda")
        #target_link_options(permute-cuda PRIVATE "-lcublas" "-lcublasLt")

        SET_SOURCES_CUDA_LANGUAGE(permutate-cuda)

        target_compile_options(prna-cuda BEFORE PUBLIC "-DFLOAT")

        target_include_directories(qtclustering-cuda PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

        execute_process(
            COMMAND /bin/bash -c "tar -xf data.tar.gz"
            RESULT_VARIABLE RESULT_CODE
            WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/sad-cuda"
        )

        execute_process(
            COMMAND /bin/bash -c "tar -xf image.tar.gz"
            RESULT_VARIABLE RESULT_CODE
            WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/seam-carving-cuda"
        )

        SET_SOURCES_CUDA_LANGUAGE(rsc-cuda)

        SET_SOURCES_CUDA_LANGUAGE(sc-cuda)

        #target_compile_options(segsort-cuda PRIVATE "--expt-relaxed-constexpr")

        # the SSE is for HOST-side, so honestly we don't care about building it in
        #target_compile_options(slu-cuda BEFORE PUBLIC "-DNO_ATOMIC" "-DSSE2" "-msse2" "-fPIE")
        target_compile_options(slu-cuda BEFORE PUBLIC "-DNO_ATOMIC" "-fPIE")
        #FILE(GLOB NICSLU_LIB "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/source/*.c")
        #add_library(nicslu STATIC ${NICSLU_LIB})
        add_dependencies(slu-cuda nicslu)
        target_link_libraries(slu-cuda nicslu)
        #FILE(GLOB NICSLU_UTIL "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/util/nicslu_readfile.c")
        #add_library(nicslu-util STATIC ${NICSLU_UTIL})
        add_dependencies(slu-cuda nicslu-util)
        target_link_libraries(slu-cuda nicslu-util)
        target_include_directories(nicslu-util PRIVATE "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/include")
        target_include_directories(slu-cuda BEFORE PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/include")
        target_include_directories(slu-cuda BEFORE PUBLIC "${HECBENCH_SRC_DIR}/slu-cuda/src")
        target_link_options(slu-cuda PRIVATE "-lm" "-lpthread" "-lrt")
        set_source_files_properties("${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/demo/test.c" PROPERTIES C_STANDARD "90")
        set_source_files_properties("${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/demo/test.c" PROPERTIES COMPILE_FLAGS "-ansi")
        FILE(GLOB NICSLU_DEMO "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/demo/*.c")
        set_source_files_properties(${NICSLU_DEMO} PROPERTIES HEADER_FILE_ONLY ON)

        SET_SOURCES_CUDA_LANGUAGE(slu-cuda)

        #target_compile_options(rowwiseMoments-cuda PRIVATE "-I${CUDA_THRUST_INCLUDE_DIR}")

        #target_compile_options(qkv-cuda PRIVATE "-fopenmp=libomp")
        #target_compile_options(qkv-cuda PRIVATE "-I${OMP_INCLUDE_DIR}")
        #target_link_options(qkv-cuda PRIVATE "-lcublas" "-lcublasLt")

        target_link_options(rayleighBenardConvection-cuda PRIVATE "-lcublas")

        #target_compile_options(rotary-cuda PRIVATE "--expt-relaxed-constexpr" "--extended-lambda")

        #target_compile_options(saxpy-ompt-cuda PRIVATE "-fopenmp")
        #target_include_directories(saxpy-ompt-cuda PRIVATE "-I${OMP_INCLUDE_DIR}")

        target_include_directories(scan3-cuda PRIVATE "${HECBENCH_SRC_DIR}/scan2-cuda")

        target_link_options(sddmm-batch-cuda PRIVATE "-lcusparse")

        # copied si-cuda CMakeLists commands


        #target_link_options(slit-cuda PRIVATE "-lcufft")

        target_link_options(spaxpby-cuda PRIVATE "-lcusparse")

        target_link_options(spd2s-cuda PRIVATE "-lcusparse")

        target_include_directories(spgeam-cuda PRIVATE "${HECBENCH_SRC_DIR}/spgeam-sycl")
        target_link_options(spgeam-cuda PRIVATE "-lcusparse")

        target_link_options(spgemm-cuda PRIVATE "-lcusparse")

        target_link_options(spmm-cuda PRIVATE "-lcusparse")

        target_link_options(spmv-cuda PRIVATE "-lcusparse")

        target_include_directories(spnnz-cuda PRIVATE "${HECBENCH_SRC_DIR}/spd2s-cuda")
        target_link_options(spnnz-cuda PRIVATE "-lcusparse")

        target_include_directories(sobel-cuda PRIVATE "${HECBENCH_SRC_DIR}/include")

        #target_compile_options(softmax-fused-cuda PRIVATE "--expt-relaxed-constexpr" "--use_fast_math")
        #target_compile_options(softmax-fused-cuda PRIVATE "--use_fast_math")

        target_include_directories(sps2d-cuda PRIVATE "${HECBENCH_SRC_DIR}/spd2s-cuda")
        target_link_options(sps2d-cuda PRIVATE "-lcusparse")

        target_link_options(spsm-cuda PRIVATE "-lcusparse")

        target_include_directories(spsort-cuda PRIVATE "${HECBENCH_SRC_DIR}/spsort-sycl")
        target_link_options(spsort-cuda PRIVATE "-lcusparse")
        set_target_properties(spsort-cuda PROPERTIES CXX_STANDARD "17")

        target_include_directories(sptrsv-cuda PRIVATE "${HECBENCH_SRC_DIR}/sptrsv-sycl")
        target_compile_options(sptrsv-cuda PRIVATE "-DWARP_SIZE=32" "-DWARP_PER_BLOCK=8" "-DVALUE_TYPE=double")
        target_sources(sptrsv-cuda PRIVATE "${HECBENCH_SRC_DIR}/sptrsv-sycl/main.cpp")
        target_link_options(sptrsv-cuda PRIVATE "-lcusparse")


        target_link_options(srad-cuda PRIVATE "-lm")
        get_target_property(SRAD_SOURCES srad-cuda SOURCES)
        set_source_files_properties(${SRAD_SOURCES} PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/srad-cuda/main.cu" PROPERTIES HEADER_FILE_ONLY OFF)

        #target_link_options(streamUM-cuda PRIVATE "-lcublas" "-L/usr/lib/llvm-18/lib")
        #target_compile_options(streamUM-cuda PRIVATE "-fopenmp=libomp")
        #target_compile_options(streamUM-cuda PRIVATE "-I${OMP_INCLUDE_DIR}" "-rpath /usr/lib/llvm-18/lib" )

        target_compile_options(ssim-cuda PRIVATE "-D__NVCC__" "-D__CUDACC__")

        target_compile_options(su3-cuda PRIVATE "-DMILC_COMPLEX")


        #FILE(GLOB TPACF_CU_FILES "${HECBENCH_SRC_DIR}/tpacf-cuda/*.cu")
        #target_sources(tpacf-cuda PRIVATE ${TPACF_CU_FILES})
        set_source_files_properties("${HECBENCH_SRC_DIR}/tpacf-cuda/ACF_kernel.cu" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/tpacf-cuda/histogram_kernel.cu" PROPERTIES HEADER_FILE_ONLY ON)
        set_source_files_properties("${HECBENCH_SRC_DIR}/tpacf-cuda/model_io.cu" PROPERTIES HEADER_FILE_ONLY ON)
        SET_SOURCES_CUDA_LANGUAGE(tpacf-cuda)

        target_include_directories(tqs-cuda PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
        SET_SOURCES_CUDA_LANGUAGE(tqs-cuda)

        SET_SOURCES_CUDA_LANGUAGE(triad-cuda)

        #target_compile_options(unfold-cuda PRIVATE "--extended-lambda")

        target_include_directories(urng-cuda PRIVATE "${HECBENCH_SRC_DIR}/include")

        #target_compile_options(vol2col-cuda PRIVATE "--expt-relaxed-constexpr")

        target_include_directories(warpexchange-cuda PRIVATE "${HECBENCH_SRC_DIR}/blockAccess-cuda")

        target_compile_options(warpsort-cuda BEFORE PUBLIC "-DWARP_SIZE=32" "-DHALF_WARP_SIZE=16")
        SET_SOURCES_CUDA_LANGUAGE(warpsort-cuda)

        #target_sources(wordcount-cuda PRIVATE "${HECBENCH_SRC_DIR}/wordcount-sycl/main.cpp")
        set(TOADD "${HECBENCH_SRC_DIR}/wordcount-sycl/main.cpp")
        COPY_FILES_AND_ADD_TO_TARGET("${TOADD}" wordcount-cuda CUDA)

        target_compile_options(wsm5-cuda PRIVATE "-DMKX=4")

        #target_compile_options(xsbench-cuda PUBLIC "-fopenmp=libomp")
        #target_link_options(xsbench-cuda PRIVATE "-fopenmp=libomp")
    ENDIF()

ENDIF() # BUILD_ALL