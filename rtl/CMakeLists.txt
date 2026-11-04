set(rtl_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/vec_pkg.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/vec_rom_pkg.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/sfifo.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/skew.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/fp32_add.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/fp32_mul.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/fp32_ufmul.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/bdu2_dsp.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_engine.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/vec_lane.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/vec_reduce.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/vec_scalar.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/vec_engine.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/dma_burst.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/dma_rowfmt.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/dma_t32.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/dma_gacc.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/dma_engine.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/npu_trace.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/npu_seq.sv
  ${CMAKE_CURRENT_SOURCE_DIR}/npu_top.sv)
set(HOTARUBI_RTL_SOURCES ${rtl_sources} PARENT_SCOPE)

add_test(NAME rtl_lint
  COMMAND ${VERILATOR_BIN} --lint-only -Wall --top-module npu_top ${rtl_sources})
