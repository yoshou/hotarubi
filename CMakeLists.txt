cmake_minimum_required(VERSION 3.25)
project(hotarubi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -ffp-contract=off)

option(HOTARUBI_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(HOTARUBI_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

option(HOTARUBI_COVERAGE "Build with Clang source-based coverage" OFF)
if(HOTARUBI_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "HOTARUBI_COVERAGE requires Clang")
  endif()
  add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
  add_link_options(-fprofile-instr-generate)
endif()

if(CMAKE_CROSSCOMPILING)
  add_subdirectory(tools/runtime)
  return()
endif()

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)
find_package(verilator REQUIRED)

function(hotarubi_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE GTest::gtest_main)
  gtest_discover_tests(${name})
  set_property(GLOBAL APPEND PROPERTY HOTARUBI_TEST_TARGETS ${name})
endfunction()

function(hotarubi_add_coverage_report)
  get_property(test_targets GLOBAL PROPERTY HOTARUBI_TEST_TARGETS)
  if(NOT test_targets)
    add_custom_target(coverage-report COMMAND ${CMAKE_COMMAND} -E echo "No test targets")
    return()
  endif()
  find_program(LLVM_PROFDATA NAMES llvm-profdata-21 REQUIRED)
  find_program(LLVM_COV NAMES llvm-cov-21 REQUIRED)
  set(objects "")
  foreach(target IN LISTS test_targets)
    list(APPEND objects -object $<TARGET_FILE:${target}>)
  endforeach()
  add_custom_target(coverage-report
    COMMAND sh -c "${LLVM_PROFDATA} merge -sparse ${CMAKE_BINARY_DIR}/profiles/*.profraw -o ${CMAKE_BINARY_DIR}/coverage.profdata"
    COMMAND ${LLVM_COV} report -instr-profile=${CMAKE_BINARY_DIR}/coverage.profdata -ignore-filename-regex=/test/ ${objects}
    VERBATIM)
endfunction()

if(HOTARUBI_COVERAGE)
  cmake_language(DEFER CALL hotarubi_add_coverage_report)
endif()

add_subdirectory(rtl)
add_subdirectory(tb)
add_subdirectory(fpga)
add_subdirectory(tools/runtime)
