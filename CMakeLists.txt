
cmake_minimum_required(VERSION 3.28)

project("HeCBench-roofline")


option(OMP_INCLUDE_DIR "OpenMP Include Directory (contains omp.h)" "")


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
    find_package(OpenMP REQUIRED)
ENDIF()

# we will be using clang++ OMP offloading 
IF(BUILD_OMP)
    find_package(OpenMP REQUIRED)
    #find_program(clang REQUIRED)
    #find_program(clang-offload-packager-18 REQUIRED)
ENDIF()

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
            COMMAND /bin/bash -c "find ./ -name \"Makefile*\""
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
            message("including extra subdirs for ${TARGET_NAME}, ${INC_SUBDIRS}\n")
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


LIST(LENGTH TARGET_DIRS NUM_CUDA_CODES)

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
# cc-cuda has some segfault caused by a stack overflow
# bincount-cuda only makes external CUDA library calls (we can't get the source code)
# bscan-cuda gets out-of-memory error during execution
# feynman-kac-cuda gets out-of-memory error during execution
# lrn-cuda gets out-of-memory error during execution
# cm-cuda requires user-input during execution -- need to support
# voxelization-cuda missing required input files -- can't find a link
# mpc-cuda gets out-of-memory error during execution
# stencil3d-cuda gets out-of-memory error during execution (although it runs fine whout `ncu`) -- need to look into this...
# tridiagonal-cuda segfaults during execution
# dwconv1d-cuda has no makefile -- it's "run" by a python script
# convolutionDeformable requires pytorch be installed
# halo-finder-cuda is mpi-based
# local-ht-cuda has a type definition error that's also fails with regular makefile
# miniDGS-cuda uses MPI
# xlqc-cuda needs to install GNU scientific library to work
set(IGNORE_TARGETS allreduce-cuda ccl-cuda diamond-cuda langford-cuda 
                   miniWeather-cuda minibude-cuda pingpong-cuda
                   qkv-cuda rowwiseMoments-cuda saxpy-ompt-cuda slit-cuda sparkler-cuda
                   lzss-cuda cmp-cuda cc-cuda sss-cuda streamUM-cuda stsg-cuda testSNAP-cuda
                   bscan-cuda feynman-kac-cuda lrn-cuda cm-cuda voxelization-cuda mpc-cuda
                   stencil3d-cuda tridiagonal-cuda dwconv1d-cuda convolutionDeformable-cuda
                   halo-finder-cuda local-ht-cuda miniDGS-cuda xlqc-cuda)

# for now we are ignoring this because it requires files
# to be downloaded
set(EXISTING_CMAKE_TARGETS si-cuda)

IF(BUILD_CUDA)
    # manually unzip files needed for building of some codes
    UNZIPFILE("${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu.tar.bz2")
    UNZIPFILE("${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu.tar")
    UNZIPFILE("${HECBENCH_SRC_DIR}/axhelm-cuda/BlasLapack.tar.gz")
ENDIF()

IF(BUILD_OMP)
    UNZIPFILE("${HECBENCH_SRC_DIR}/axhelm-omp/BlasLapack.tar.gz")
ENDIF()

set(TARGETS "")

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

        IF(BUILD_OMP)
            INC_ADD_ALL_SUBDIRS("${HECBENCH_SRC_DIR}/${TARGET_BASENAME}-cuda" ${TARGET_NAME})
        ENDIF()

        # add the target to the targets list 
        LIST(APPEND TARGETS ${TARGET_NAME})
    ENDIF()

ENDFOREACH()


LIST(LENGTH TARGETS NUM_TARGETS)
LIST(LENGTH IGNORE_TARGETS NUM_IGNORE_TARGETS)
message("TARGETS: [${TARGETS}]")
message("Building [${NUM_TARGETS}/${NUM_CUDA_CODES}] targets!")
message("Manually ignored [${NUM_IGNORE_TARGETS}/${NUM_CUDA_CODES}] targets!")
#set_target_properties(${TARGETS} PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(${TARGETS} PROPERTIES CUDA_ARCHITECTURES "86")
set_target_properties(${TARGETS} PROPERTIES CXX_STANDARD "14")


# set target-specific properties
IF(BUILD_OMP OR BUILD_CUDA)

ENDIF()

IF(BUILD_OMP)
    set_target_properties(${TARGETS} PROPERTIES COMPILE_FLAGS "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-offload-mandatory")
    set_target_properties(${TARGETS} PROPERTIES LINK_FLAGS "-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-link -fopenmp-offload-mandatory")

    target_compile_options(accuracy-omp BEFORE PRIVATE "-D__STRICT_ANSI__")

    target_compile_options(adv-omp BEFORE PRIVATE "-Ddfloat=float" "-Ddlong=int")

    target_include_directories(aes-omp PRIVATE "${HECBENCH_SRC_DIR}/include")

    target_compile_options(axhelm-omp BEFORE PRIVATE "-Ddfloat=float" "-Ddlong=int")
    FILE(GLOB BL_FORTRAN_SOURCES "${HECBENCH_SRC_DIR}/axhelm-omp/BlasLapack/*.f")
    enable_language(Fortran)
    add_library(BlasLapack STATIC ${BL_FORTRAN_SOURCES})
    add_dependencies(axhelm-omp BlasLapack)
    target_link_libraries(axhelm-omp BlasLapack)

    target_sources(b+tree-omp PRIVATE "${HECBENCH_SRC_DIR}/b+tree-omp/util/num/num.c")
    target_sources(b+tree-omp PRIVATE "${HECBENCH_SRC_DIR}/b+tree-omp/util/timer/timer.c")
    target_link_options(b+tree-omp BEFORE PRIVATE "-lm")

    target_include_directories(bfs-omp PRIVATE "${HECBENCH_SRC_DIR}/bfs-sycl")

    target_link_options(blas-gemm-omp BEFORE PRIVATE "-lcublas")
    set_target_properties(blas-gemm-omp PROPERTIES CXX_STANDARD "17")

ENDIF()

IF(BUILD_CUDA)
    set_target_properties(adv-cuda PROPERTIES COMPILE_FLAGS "-Ddfloat=float -Ddlong=int")

    target_include_directories(aes-cuda PRIVATE "${HECBENCH_SRC_DIR}/include")

    FILE(GLOB AMGMK_C_FILES "${HECBENCH_SRC_DIR}/amgmk-cuda/*.c")
    set_source_files_properties(${AMGMK_C_FILES} PROPERTIES HEADER_FILE_ONLY ON)

    set_target_properties(axhelm-cuda PROPERTIES COMPILE_FLAGS "-Ddfloat=float -Ddlong=int")
    FILE(GLOB BL_FORTRAN_SOURCES "${HECBENCH_SRC_DIR}/axhelm-cuda/BlasLapack/*.f")
    enable_language(Fortran)
    add_library(BlasLapack STATIC ${BL_FORTRAN_SOURCES})
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
    target_compile_options(bmf-cuda PRIVATE "-fopenmp=libomp")
    target_compile_options(bmf-cuda PRIVATE "-I${OMP_INCLUDE_DIR}" )
    target_link_options(bmf-cuda PRIVATE "${OMP_LINKER_FLAGS}")

    target_include_directories(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl")
    target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp")
    target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/cmd_arg_reader.cpp")
    target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp")

    # excluded
    #target_include_directories(cc-cuda PRIVATE "${HECBENCH_SRC_DIR}/mis-cuda")
    target_link_libraries(ced-cuda pthread)
    FILE(GLOB CED_IGNORE "${HECBENCH_SRC_DIR}/ced-cuda/main.cpp")
    set_source_files_properties(${CED_IGNORE} PROPERTIES HEADER_FILE_ONLY ON)

    target_include_directories(debayer-cuda PRIVATE "${HECBENCH_SRC_DIR}/debayer-sycl")

    set_target_properties(degrid-cuda PROPERTIES COMPILE_FLAGS "-DPRECISION=double")

    #set_target_properties(determinant-cuda PROPERTIES LINK_FLAGS "-lcusolver")
    target_link_options(determinant-cuda PRIVATE "-lcusolver")

    set_target_properties(dp-cuda PROPERTIES LINK_FLAGS "-lcublas")
    set_target_properties(dp-cuda PROPERTIES CXX_STANDARD "17")

    target_include_directories(dwconv-cuda PRIVATE "${HECBENCH_SRC_DIR}/tensorAccessor-cuda")

    target_include_directories(extend2-cuda PRIVATE "${HECBENCH_SRC_DIR}/extend2-sycl")
    target_sources(extend2-cuda PRIVATE "${HECBENCH_SRC_DIR}/extend2-sycl/read_data.cpp")

    set_target_properties(f16sp-cuda PROPERTIES LINK_FLAGS "-lcublas")

    target_include_directories(floydwarshall2-cuda PRIVATE "${HECBENCH_SRC_DIR}/mis-cuda")

    target_include_directories(fluidSim-cuda PRIVATE "${HECBENCH_SRC_DIR}/fluidSim-sycl")
    target_sources(fluidSim-cuda PRIVATE "${HECBENCH_SRC_DIR}/fluidSim-sycl/reference.cpp")
    target_compile_options(fluidSim-cuda PRIVATE "-DVERIFY")
    get_target_property(FLUIDSIM_SOURCES fluidSim-cuda SOURCES)
    set_source_files_properties(${FLUIDSIM_SOURCES} PROPERTIES LANGUAGE CUDA)

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

    target_compile_options(hausdorff-cuda PRIVATE "--expt-relaxed-constexpr")

    get_target_property(HAVERSINE_SOURCES haversine-cuda SOURCES)
    set_source_files_properties(${HAVERSINE_SOURCES} PROPERTIES LANGUAGE CUDA)

    target_sources(heartwall-cuda PRIVATE "${HECBENCH_SRC_DIR}/heartwall-cuda/util/avi/avilib.c")
    target_sources(heartwall-cuda PRIVATE "${HECBENCH_SRC_DIR}/heartwall-cuda/util/avi/avimod.c")
    target_sources(heartwall-cuda PRIVATE "${HECBENCH_SRC_DIR}/heartwall-cuda/util/timer/timer.c")
    target_sources(heartwall-cuda PRIVATE "${HECBENCH_SRC_DIR}/heartwall-cuda/util/file/file.c")

    FILE(GLOB HS_IGNORE "${HECBENCH_SRC_DIR}/hybridsort-cuda/tmp/*.cpp")
    set_source_files_properties(${HS_IGNORE} PROPERTIES HEADER_FILE_ONLY ON)

    # this technically makes it so that nvcc in invoked to build these files instead of clang++
    #get_target_property(KMC_SOURCES kmc-cuda SOURCES)
    #set_source_files_properties(${KMC_SOURCES} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties("${HECBENCH_SRC_DIR}/kmc-cuda/driver.c" PROPERTIES HEADER_FILE_ONLY ON)
    target_link_options(kmc-cuda PRIVATE "-lcublas" "-lcudart")


    # need to remove the MESCHACH_LIB files from the leukocyte source files 
    # but it's okay since they don't get used in the final linking
    FILE(GLOB MESCHACH_LIB "${HECBENCH_SRC_DIR}/leukocyte-cuda/meschach_lib/*.c")
    add_library(meschach STATIC ${MESCHACH_LIB})
    set_target_properties(meschach PROPERTIES C_STANDARD "90")
    target_compile_options(meschach BEFORE PUBLIC "-DHAVE_CONFIG_H" "-DANSI_C")
    set_source_files_properties("${HECBENCH_SRC_DIR}/leukocyte-cuda/meschach_lib/tutadv.c" PROPERTIES HEADER_FILE_ONLY ON)
    add_dependencies(leukocyte-cuda meschach)
    target_link_libraries(leukocyte-cuda meschach)


    #target_compile_definitions(local-ht-cuda PUBLIC "-Dunsigned=uint32_t")
    #set_target_properties(local-ht-cuda PROPERTIES COMPILE_FLAGS "-Uuint32_t -Duint32_t=unsigned")
    #set_target_properties(local-ht-cuda PROPERTIES COMPILE_FLAGS "--include=bits/stdint-uintn.h")
    #set_target_properties(local-ht-cuda PROPERTIES COMPILE_FLAGS "--include=bits/types.h")

    target_compile_options(logan-cuda BEFORE PUBLIC "-DADAPTABLE" "-maxrregcount=32")
    set_target_properties(logan-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    FILE(GLOB LOGAN_INC "${HECBENCH_SRC_DIR}/logan-cuda/src/*.cuh")
    set_source_files_properties(${LOGAN_INC} PROPERTIES HEADER_FILE_ONLY ON)
    target_compile_options(logan-cuda PRIVATE "-fopenmp=libomp" "-I${OMP_INCLUDE_DIR}")
    target_link_options(logan-cuda PRIVATE "${OMP_LINKER_FLAGS}")

    # manually fixing a mispelling in thier cuda code
    target_compile_definitions(lsqt-cuda PUBLIC "__syncwrap=__syncwarp")

    target_sources(lud-cuda PRIVATE "${HECBENCH_SRC_DIR}/lud-cuda/common/common.cu")
    set_source_files_properties("${HECBENCH_SRC_DIR}/lud-cuda/lud_kernels.cu" PROPERTIES HEADER_FILE_ONLY ON)

    target_link_options(ludb-cuda PRIVATE "-lcublas")

    set_source_files_properties("${HECBENCH_SRC_DIR}/md5hash-cuda/MD5Hash.cpp" PROPERTIES HEADER_FILE_ONLY ON)

    set_target_properties(meanshift-cuda PROPERTIES CXX_STANDARD "17")

    set_target_properties(merkle-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(merkle-cuda PRIVATE "--expt-relaxed-constexpr") #"--extended-lambda")

    target_link_options(mf-sgd-cuda PRIVATE "-lpthread" )
    # not allowed to use the __shfl() operation on compute version 7X and higher
    set_target_properties(mf-sgd-cuda PROPERTIES CUDA_ARCHITECTURES "60")


    execute_process(
        COMMAND /bin/bash -c "make common_files && make generate_info"
        RESULT_VARIABLE RESULT_CODE
        WORKING_DIRECTORY "${HECBENCH_SRC_DIR}/miniFE-cuda/src"
    )
    target_compile_options(miniFE-cuda BEFORE PUBLIC "-DMINIFE_SCALAR=double" "-DMINIFE_LOCAL_ORDINAL=int" "-DMINIFE_GLOBAL_ORDINAL=int" "-DMINIFE_RESTRICT=__restrict__" "-DMINIFE_CSR_MATRIX" "-DMINIFE_INFO=1" "-DMINIFE_KERNELS=0")
    #target_sources(lud-cuda PRIVATE "${HECBENCH_SRC_DIR}/miniFE-cuda/src/miniFE_info.hpp")
    #set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-cuda/src/miniFE_info.hpp" PROPERTIES HEADER_FILE_ONLY ON)
    set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-cuda/basic/utest.cpp" PROPERTIES HEADER_FILE_ONLY ON)
    set_source_files_properties("${HECBENCH_SRC_DIR}/miniFE-cuda/basic/main.cpp" PROPERTIES HEADER_FILE_ONLY ON)
    FILE(GLOB MINIFE_INC "${HECBENCH_SRC_DIR}/miniFE-cuda/basic/*")
    set_source_files_properties(${MINIFE_INC} PROPERTIES HEADER_FILE_ONLY ON)
    target_include_directories(miniFE-cuda BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-cuda/utils")
    target_include_directories(miniFE-cuda BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-cuda/src")
    target_include_directories(miniFE-cuda BEFORE PUBLIC "${HECBENCH_SRC_DIR}/miniFE-cuda/fem")
    get_target_property(MINIFE_SRC miniFE-cuda SOURCES)
    set_source_files_properties(${MINIFE_SRC} PROPERTIES LANGUAGE CUDA)
    #set_target_properties(miniFE-cuda PROPERTIES CXX_STANDARD "11")

    target_link_options(mrg32k3a-cuda PRIVATE "-lcurand")

    target_link_options(norm2-cuda PRIVATE "-lcurand" "-lcublas")

    target_compile_options(openmp-cuda PRIVATE "-fopenmp=libomp")
    target_compile_options(openmp-cuda PRIVATE "-I${HECBENCH_SRC_DIR}/openmp-omp" "-I${OMP_INCLUDE_DIR}" )
    target_link_options(openmp-cuda PRIVATE "${OMP_LINKER_FLAGS}")

    target_link_options(pcc-cuda PRIVATE "-lcublas")

    target_include_directories(permute-cuda PRIVATE "${HECBENCH_SRC_DIR}/qkv-cuda")

    target_compile_options(prna-cuda BEFORE PUBLIC "-DFLOAT")

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

    target_compile_options(segsort-cuda PRIVATE "--expt-relaxed-constexpr")

    target_compile_options(slu-cuda BEFORE PUBLIC "-DNO_ATOMIC" "-DSSE2" "-msse2" "-fPIE")
    FILE(GLOB NICSLU_LIB "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/source/*.c")
    add_library(nicslu STATIC ${NICSLU_LIB})
    add_dependencies(slu-cuda nicslu)
    target_link_libraries(slu-cuda nicslu)
    FILE(GLOB NICSLU_UTIL "${HECBENCH_SRC_DIR}/slu-cuda/src/nicslu/util/nicslu_readfile.c")
    add_library(nicslu-util STATIC ${NICSLU_UTIL})
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

    #target_compile_options(rowwiseMoments-cuda PRIVATE "-I${CUDA_THRUST_INCLUDE_DIR}")

    #target_compile_options(qkv-cuda PRIVATE "-fopenmp=libomp")
    #target_compile_options(qkv-cuda PRIVATE "-I${OMP_INCLUDE_DIR}")
    #target_link_options(qkv-cuda PRIVATE "-lcublas" "-lcublasLt")

    target_link_options(rayleighBenardConvection-cuda PRIVATE "-lcublas")

    target_compile_options(rotary-cuda PRIVATE "--expt-relaxed-constexpr" "--extended-lambda")

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

    target_compile_options(softmax-fused-cuda PRIVATE "--expt-relaxed-constexpr" "--use_fast_math")

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


    target_compile_options(su3-cuda PRIVATE "-DMILC_COMPLEX")


    #FILE(GLOB TPACF_CU_FILES "${HECBENCH_SRC_DIR}/tpacf-cuda/*.cu")
    #target_sources(tpacf-cuda PRIVATE ${TPACF_CU_FILES})
    #set_source_files_properties("${HECBENCH_SRC_DIR}/tpacf-cuda/ACF_kernel.cu" PROPERTIES HEADER_FILE_ONLY ON)
    #set_source_files_properties("${HECBENCH_SRC_DIR}/tpacf-cuda/histogram_kernel.cu" PROPERTIES HEADER_FILE_ONLY ON)
    #set_source_files_properties("${HECBENCH_SRC_DIR}/tpacf-cuda/model_io.cu" PROPERTIES HEADER_FILE_ONLY ON)
    get_target_property(TPACF_SOURCES tpacf-cuda SOURCES)
    set_source_files_properties(${TPACF_SOURCES} PROPERTIES LANGUAGE CUDA)



    target_compile_options(unfold-cuda PRIVATE "--extended-lambda")

    target_include_directories(urng-cuda PRIVATE "${HECBENCH_SRC_DIR}/include")

    target_compile_options(vol2col-cuda PRIVATE "--expt-relaxed-constexpr")

    target_include_directories(warpexchange-cuda PRIVATE "${HECBENCH_SRC_DIR}/blockAccess-cuda")

    target_compile_options(warpsort-cuda BEFORE PUBLIC "-DWARP_SIZE=32" "-DHALF_WARP_SIZE=16")

    target_sources(wordcount-cuda PRIVATE "${HECBENCH_SRC_DIR}/wordcount-sycl/main.cpp")

    target_compile_options(wsm5-cuda PRIVATE "-DMKX=4")
ENDIF()