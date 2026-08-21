cmake_minimum_required(VERSION 3.20)
project(aiinfra_raii_lab LANGUAGES CXX)

add_executable(aiinfra_raii_lab main.cpp)
target_compile_features(aiinfra_raii_lab PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(aiinfra_raii_lab PRIVATE /W4)
else()
  target_compile_options(aiinfra_raii_lab PRIVATE -Wall -Wextra -Wpedantic)
endif()
