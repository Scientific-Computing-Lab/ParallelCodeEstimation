
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
MACRO(FINDFILESTOBUILD result curdir)
    # lets call the `make clean` because we can extract the filenames of the object files that get deleted
    execute_process(
        COMMAND /bin/bash -c "make clean | grep -oP '\\b[a-zA-Z0-9._-]+(?=\\.\\w+)\\b' | sed 's/\\.o$//' | xargs -I{} find . -type f -name '{}.c' -o -name '{}.cpp' -o -name '{}.cu' -o -name '{}.cc' -o -name '{}.cxx' -o -name '{}.c++' | sed 's|^\\./||' | sed -z 's/\\n/ /g'"
        OUTPUT_VARIABLE MATCHED_FILES
        WORKING_DIRECTORY ${curdir}
    )

    string(REPLACE " " ";" MATCHED_FILES_LIST "${MATCHED_FILES}")
    set(FULLPATH_FILES "")

    FOREACH(match ${MATCHED_FILES_LIST})
        #message("Matched file: ${match}")
        LIST(APPEND FULLPATH_FILES "${curdir}/${match}")
    ENDFOREACH()
  SET(${result} ${FULLPATH_FILES})
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


#set(IGNORE_TARGETS adv-cuda amgmk-cuda aop-cuda aes-cuda asmooth-cuda allreduce-cuda atomicSystemWide-cuda atomicIntrinsics-cuda atomicPerf-cuda)
# some of these are MPI codes we don't want to concern ourselves with yet
# some have zip files or complicated build filestructures that we will avoid for simplicity right now
# Overall, there are 491 cuda codes in HeCBench, this setup is able to build 417 of them (417/491=0.8492 --> 85%)
# langford-cuda, minibude-cuda can't be built even with default script
# fresnel-cuda is doing wonky things with relocatable code -- not linking correctly
# openmp-cuda, qkv-cuda has a malloc visibility bug in clang
# miniWeather-cuda, pingpong-cuda, sparkler-cuda is mpi
# saxpy-ompt-cude, streamUM-cuda having problems with finding omp.h correctly -- also doesn't build by default
# slit-cude missing a file it needs to work -- can't default build
# sss-cuda uses gsl
# testSNAP-cuda is missing an include
set(IGNORE_TARGETS allreduce-cuda ccl-cuda diamond-cuda fresnel-cuda langford-cuda 
                   merkle-cuda miniWeather-cuda minibude-cuda pingpong-cuda
                   qkv-cuda rowwiseMoments-cuda saxpy-ompt-cuda slit-cuda sparkler-cuda
                   sss-cuda streamUM-cuda stsg-cuda testSNAP-cuda)
set(TARGETS "")

FOREACH(TARGET_DIR ${TARGET_DIRS})
    GET_FILENAME_COMPONENT(TARGET_NAME ${TARGET_DIR} NAME)

    # skip undesired builds
    IF(${TARGET_NAME} IN_LIST IGNORE_TARGETS)
        continue()
    ENDIF()

    #message("${TARGET_NAME}")
    #message("${TARGET_DIR}")

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

    #FILE(GLOB SOURCES_H LIST_DIRECTORIES false "${TARGET_DIR}/*.h")
    #FILE(GLOB SOURCES_HPP LIST_DIRECTORIES false "${TARGET_DIR}/*.hpp")
    FILE(GLOB SOURCES_CU LIST_DIRECTORIES false "${TARGET_DIR}/*.cu")
    FILE(GLOB SOURCES_CPP LIST_DIRECTORIES false "${TARGET_DIR}/*.cpp")
    FILE(GLOB SOURCES_C LIST_DIRECTORIES false "${TARGET_DIR}/*.c")
    set(SOURCES ${SOURCES_CU} ${SOURCES_CPP} ${SOURCES_C})

    # if SOURCES didn't pick anything up, the program probably is large and complex
    # so we're going to omit those files for now. If there are other subdirectories,
    # we will also skip the program for now.
    LIST(LENGTH SOURCES NUM_SOURCES)
    IF ((NUM_SOURCES GREATER 0) AND (NUM_SUBDIRS EQUAL 0))
        #message("sources: [${SOURCES}]")

        FINDFILESTOBUILD(BUILDABLE_FILES ${TARGET_DIR})
        LIST(LENGTH BUILDABLE_FILES BUILDFILESNUM)

        IF(BUILDFILESNUM EQUAL 0)
            # if we got no files, assume all files to build
            add_executable(${TARGET_NAME} ${SOURCES})
        ELSE()
            add_executable(${TARGET_NAME} ${BUILDABLE_FILES})
        ENDIF()

        #add_executable(${TARGET_NAME} ${SOURCES})

        # add its own directory to be included during build
        target_include_directories(${TARGET_NAME} PRIVATE "${TARGET_DIR}")

        # add the target to the targets list 
        LIST(APPEND TARGETS ${TARGET_NAME})
    ENDIF()

ENDFOREACH()

LIST(LENGTH TARGETS NUM_TARGETS)
message("TARGETS: [${TARGETS}]")
message("Building [${NUM_TARGETS}] targets!")
#set_target_properties(${TARGETS} PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(${TARGETS} PROPERTIES CUDA_ARCHITECTURES "86")
set_target_properties(${TARGETS} PROPERTIES CXX_STANDARD "14")


# set target-specific properties
set_target_properties(adv-cuda PROPERTIES COMPILE_FLAGS "-Ddfloat=float -Ddlong=int")

target_include_directories(aes-cuda PRIVATE "${HECBENCH_SRC_DIR}/include")

FILE(GLOB AMGMK_C_FILES "${HECBENCH_SRC_DIR}/amgmk-cuda/*.c")
set_source_files_properties(${AMGMK_C_FILES} PROPERTIES HEADER_FILE_ONLY ON)

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

target_include_directories(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl")
target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/shrUtils.cpp")
target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/cmd_arg_reader.cpp")
target_sources(boxfilter-cuda PRIVATE "${HECBENCH_SRC_DIR}/boxfilter-sycl/reference.cpp")

target_include_directories(cc-cuda PRIVATE "${HECBENCH_SRC_DIR}/mis-cuda")

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

target_compile_options(frna-cuda BEFORE PRIVATE "-use_fast_math" "-DINT")

target_link_options(geam-cuda PRIVATE "-lcublas")

set_target_properties(gemv-cuda PROPERTIES CXX_STANDARD "17")

target_include_directories(gc-cuda PRIVATE "${HECBENCH_SRC_DIR}/mis-cuda")

set_target_properties(gels-cuda PROPERTIES CXX_STANDARD "17")
target_link_options(gels-cuda PRIVATE "-lcublas")

target_compile_options(hausdorff-cuda PRIVATE "--expt-relaxed-constexpr")

get_target_property(HAVERSINE_SOURCES haversine-cuda SOURCES)
set_source_files_properties(${HAVERSINE_SOURCES} PROPERTIES LANGUAGE CUDA)


# this technically makes it so that nvcc in invoked to build these files instead of clang++
#get_target_property(KMC_SOURCES kmc-cuda SOURCES)
#set_source_files_properties(${KMC_SOURCES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties("${HECBENCH_SRC_DIR}/kmc-cuda/driver.c" PROPERTIES HEADER_FILE_ONLY ON)
target_link_options(kmc-cuda PRIVATE "-lcublas" "-lcudart")

target_link_options(ludb-cuda PRIVATE "-lcublas")

set_source_files_properties("${HECBENCH_SRC_DIR}/md5hash-cuda/MD5Hash.cpp" PROPERTIES HEADER_FILE_ONLY ON)

target_link_options(mrg32k3a-cuda PRIVATE "-lcurand")

target_link_options(norm2-cuda PRIVATE "-lcurand" "-lcublas")

target_compile_options(openmp-cuda PRIVATE "-fopenmp=libomp")
#target_compile_options(openmp-cuda PRIVATE "-I${HECBENCH_SRC_DIR}/openmp-omp" "-I/usr/lib/gcc/x86_64-linux-gnu/13/include" )
target_compile_options(openmp-cuda PRIVATE "-I${HECBENCH_SRC_DIR}/openmp-omp" "-I${OMP_INCLUDE_DIR}" )
target_link_options(openmp-cuda PRIVATE "${OMP_LINKER_FLAGS}")

target_link_options(pcc-cuda PRIVATE "-lcublas")

target_include_directories(permute-cuda PRIVATE "${HECBENCH_SRC_DIR}/qkv-cuda")


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

target_sources(wordcount-cuda PRIVATE "${HECBENCH_SRC_DIR}/wordcount-sycl/main.cpp")

target_compile_options(wsm5-cuda PRIVATE "-DMKX=4")

# merkle also suffering from relocatable device code issues
#set_target_properties(merkle-cuda PROPERTIES CXX_STANDARD "17")
#target_compile_options(merkle-cuda BEFORE PRIVATE "--expt-relaxed-constexpr")
#target_compile_options(merkle-cuda BEFORE PRIVATE "--relocatable-pch")
##target_compile_options(merkle-cuda BEFORE PRIVATE "-rdc=true")
#target_compile_options(merkle-cuda BEFORE PRIVATE "-fcuda-rdc")
#get_target_property(MERKLE_SOURCES merkle-cuda SOURCES)
##message("merkle sources ${MERKLE_SOURCES}")
##set_source_files_properties(${MERKLE_SOURCES} PROPERTIES LANGUAGE CUDA)

#set_target_properties(axhelm-cuda PROPERTIES LINK_FLAGS "BlasLapack/libBlasLapack.a -lgfortran")
#file(GLOB_RECURSE CFILES "${HECBENCH_SRC_DIR}/amgmk-cuda/*.c")
#set_source_files_properties(${CFILES} PROPERTIES LANGUAGE CUDA)

# these flags/args get applied to all targets
#add_compile_options("-Wall -O3 -std=c++14 -x cu --offload-arch=${OFFLOAD_ARCH} --arch=${OFFLOAD_ARCH}")

