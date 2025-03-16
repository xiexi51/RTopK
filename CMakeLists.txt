cmake_minimum_required(VERSION 3.5)
project (rtopk C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(${PROJECT_NAME} rtopk_kernel.cu test.cu)

target_link_libraries(${PROJECT_NAME}  curand stdc++fs)