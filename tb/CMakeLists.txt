function(hotarubi_add_rtl_model name top)
  add_library(${name} STATIC)
  verilate(${name}
    SOURCES ${HOTARUBI_RTL_SOURCES}
    TOP_MODULE ${top}
    PREFIX V${top}
    VERILATOR_ARGS --x-assign 0 --x-initial 0)
  set_target_properties(${name} PROPERTIES CXX_CLANG_TIDY "")
  target_compile_options(${name} PRIVATE -w -O2)
  get_target_property(includes ${name} INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(${name} PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${includes}")
endfunction()

hotarubi_add_rtl_model(rtl_fp32_add fp32_add)
hotarubi_add_rtl_model(rtl_fp32_mul fp32_mul)
hotarubi_add_rtl_model(rtl_npu_top npu_top)
hotarubi_add_rtl_model(rtl_vec_transpose vec_transpose)
hotarubi_add_rtl_model(rtl_vec_engine vec_engine)

add_library(tb_fp32_ref STATIC fp32_ref.cpp)
target_include_directories(tb_fp32_ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(tb_vec_ref STATIC vec_ref.cpp)
target_link_libraries(tb_vec_ref PUBLIC tb_fp32_ref)

add_library(tb_npu_bench STATIC npu_bench.cpp npu_data.cpp gemm_ref.cpp)
target_link_libraries(tb_npu_bench PUBLIC rtl_npu_top tb_fp32_ref)

hotarubi_add_test(fp32_unit_test fp32_unit_test.cpp)
target_link_libraries(fp32_unit_test PRIVATE rtl_fp32_add rtl_fp32_mul tb_fp32_ref)

foreach(test gemm_test dma_test split8_test gacc_test sequencer_test)
  hotarubi_add_test(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE tb_npu_bench)
endforeach()

foreach(test vec_test vec_program_test)
  hotarubi_add_test(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE tb_npu_bench tb_vec_ref)
endforeach()

hotarubi_add_test(vec_transpose_test vec_transpose_test.cpp)
target_link_libraries(vec_transpose_test PRIVATE rtl_vec_transpose)

add_library(tb_vec_engine_bench STATIC vec_engine_bench.cpp)
target_link_libraries(tb_vec_engine_bench PUBLIC rtl_vec_engine hotarubi_iss)

hotarubi_add_test(vec_engine_bench_test vec_engine_bench_test.cpp)
target_link_libraries(vec_engine_bench_test PRIVATE tb_vec_engine_bench iss_reference)

add_executable(hotarubi_vec_bench vec_engine_bench_main.cpp)
target_link_libraries(hotarubi_vec_bench PRIVATE tb_vec_engine_bench)

add_library(tb_kernel_bench STATIC kernel_bench.cpp)
target_link_libraries(tb_kernel_bench PUBLIC tb_npu_bench hotarubi_trace_lib)

add_executable(hotarubi_kernel_bench kernel_bench_main.cpp)
target_link_libraries(hotarubi_kernel_bench PRIVATE tb_kernel_bench)

hotarubi_add_test(kernel_bench_test kernel_bench_test.cpp)
target_link_libraries(kernel_bench_test PRIVATE tb_kernel_bench hotarubi_compiler)
