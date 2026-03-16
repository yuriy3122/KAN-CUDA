cmake_minimum_required(VERSION 3.24)
project(kan_credit_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(kan_credit_train
    src/train.cpp
    src/csv_reader.cpp
    src/preprocess.cpp
    src/kan_model.cpp
    src/kan_cuda_kernels.cu
)

target_compile_options(kan_credit_train PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
)

set_target_properties(kan_credit_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
