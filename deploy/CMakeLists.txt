cmake_minimum_required(VERSION 3.10)
project(yolox_visdrone_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

add_executable(yolox_visdrone_demo src/main.cpp)
target_include_directories(yolox_visdrone_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolox_visdrone_demo
  ${OpenCV_LIBS}
  vart-runner
  vart-dpu-controller
  xir
  unilog
  glog
  pthread
)

# Eski GCC'lerde std::filesystem ayri libstdc++fs ister.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++17")
check_cxx_source_compiles("
  #include <filesystem>
  int main() { return std::filesystem::temp_directory_path().empty() ? 1 : 0; }
" YOLOX_HAS_STD_FILESYSTEM)
if(NOT YOLOX_HAS_STD_FILESYSTEM)
  target_link_libraries(yolox_visdrone_demo stdc++fs)
endif()
