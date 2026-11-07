add_library(hotarubi_trace_lib STATIC
  analysis.cpp
  input.cpp
  predict.cpp
  report.cpp)
target_include_directories(hotarubi_trace_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hotarubi_trace_lib PUBLIC hotarubi_iss hotarubi_runtime)

add_executable(hotarubi_trace main.cpp)
target_link_libraries(hotarubi_trace PRIVATE hotarubi_trace_lib)

hotarubi_add_test(trace_test trace_test.cpp)
target_link_libraries(trace_test PRIVATE hotarubi_trace_lib hotarubi_compiler)
