add_library(hotarubi_runtime STATIC driver.cpp runtime.cpp)
target_include_directories(hotarubi_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(NOT CMAKE_CROSSCOMPILING)
  foreach(test driver_test runtime_test)
    hotarubi_add_test(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE hotarubi_runtime tb_npu_bench)
  endforeach()
endif()
