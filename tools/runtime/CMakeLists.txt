add_library(hotarubi_runtime STATIC
  driver.cpp
  exact.cpp
  formats.cpp
  json.cpp
  require.cpp
  runtime.cpp
  safetensors.cpp
  sha256.cpp)
target_include_directories(hotarubi_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(hotarubi_ps STATIC frame_files.cpp smolvla_ps.cpp sparsedrive_ps.cpp)
target_link_libraries(hotarubi_ps PUBLIC hotarubi_runtime)

add_executable(hotarubi_run run.cpp)
target_link_libraries(hotarubi_run PRIVATE hotarubi_ps hotarubi_iss)

if(NOT CMAKE_CROSSCOMPILING)
  foreach(test exact_test formats_test json_test safetensors_test sha256_test)
    hotarubi_add_test(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE hotarubi_runtime)
  endforeach()
  foreach(test smolvla_ps_test sparsedrive_ps_test)
    hotarubi_add_test(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE hotarubi_ps)
  endforeach()
  foreach(test driver_test runtime_test)
    hotarubi_add_test(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE hotarubi_runtime hotarubi_iss tb_npu_bench)
  endforeach()
endif()
