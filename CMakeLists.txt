cmake_minimum_required(VERSION 3.20)

project(cNeRF VERSION 1.0
              DESCRIPTION "Neural radiance fields with libtorch"
              LANGUAGES CXX)

# Configure CUDA
find_package(CUDA)
find_library(LIBNVTOOLSEXT
    NAMES nvToolsExt
    HINTS
        ENV CUDA_TOOLKIT_ROOT_DIR
        ENV CONDA_PREFIX
    PATH_SUFFIXES lib64 lib
)

# Configure libtorch and OpenCV
list(APPEND CMAKE_PREFIX_PATH "libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cNeRF src/main.cpp src/model.cpp src/renderer.cpp src/utils.cpp)
target_include_directories(cNeRF PUBLIC include)
target_link_libraries(cNeRF ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
