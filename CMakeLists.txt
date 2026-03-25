cmake_minimum_required(VERSION 3.24)
project(kan_credit_cuda LANGUAGES CXX CUDA)

option(KAN_USE_FAST_MATH "Enable CUDA fast math in optimized builds." ON)
option(KAN_ENABLE_CUDA_DEBUG "Enable device debug info (-G) for Debug builds." ON)
option(KAN_ENABLE_CUDA_LINEINFO "Emit CUDA line information for profilers." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 CACHE STRING "CUDA SM architectures to build for.")
endif()

find_package(CUDAToolkit REQUIRED)

set(KAN_CREDIT_SOURCES
    src/train.cpp
    src/csv_reader.cpp
    src/preprocess.cpp
    src/kan_model.cpp
    src/kan_cuda_kernels.cu
)

add_executable(kan_credit_train ${KAN_CREDIT_SOURCES})

target_include_directories(kan_credit_train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kan_credit_train PRIVATE CUDA::cudart)

set_target_properties(kan_credit_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${KAN_CREDIT_SOURCES})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/train-hmda-data.csv
    ${CMAKE_CURRENT_BINARY_DIR}/train-hmda-data.csv
    COPYONLY
)

if(MSVC)
    target_compile_options(kan_credit_train PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /EHsc /bigobj>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od /Zi>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2 /Zi>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MinSizeRel>>:/O1>
    )
else()
    target_compile_options(kan_credit_train PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-O2 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MinSizeRel>>:-Os>
    )
endif()

target_compile_options(kan_credit_train PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-O2>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:MinSizeRel>>:-O2>
)

if(KAN_ENABLE_CUDA_DEBUG)
    target_compile_options(kan_credit_train PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    )
endif()

if(KAN_ENABLE_CUDA_LINEINFO)
    target_compile_options(kan_credit_train PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:MinSizeRel>>:-lineinfo>
    )
endif()

if(KAN_USE_FAST_MATH)
    target_compile_options(kan_credit_train PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:--use_fast_math>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:MinSizeRel>>:--use_fast_math>
    )
endif()

if(MSVC)
    target_compile_options(kan_credit_train PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/Zi>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/Od>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-Xcompiler=/Zi>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-Xcompiler=/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:MinSizeRel>>:-Xcompiler=/O1>
    )
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT kan_credit_train)
    set_property(TARGET kan_credit_train PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    set_property(
        TARGET kan_credit_train PROPERTY
        VS_DEBUGGER_COMMAND_ARGUMENTS
        "train-hmda-data.csv 16 8 3 20 512 0.01"
    )
endif()
