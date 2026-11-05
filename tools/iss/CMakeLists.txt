set(rom_package ${PROJECT_SOURCE_DIR}/rtl/vec_rom_pkg.sv)
set(rom_source ${CMAKE_CURRENT_BINARY_DIR}/rom.cpp)
add_custom_command(
  OUTPUT ${rom_source}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${rom_package} -DOUTPUT=${rom_source}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/rom.cmake
  DEPENDS ${rom_package} ${CMAKE_CURRENT_SOURCE_DIR}/rom.cmake
  VERBATIM)

add_library(hotarubi_iss STATIC
  ddr.cpp
  dma_engine.cpp
  gemm_array.cpp
  iss_bus.cpp
  lane.cpp
  machine.cpp
  memories.cpp
  scalar.cpp
  vec_engine.cpp
  ${rom_source})
target_include_directories(hotarubi_iss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hotarubi_iss PUBLIC hotarubi_runtime)

add_library(iss_reference STATIC encoding.cpp reference.cpp)
target_include_directories(iss_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(iss_reference PUBLIC hotarubi_runtime)

foreach(test dma_engine_test fp32_test gemm_array_test machine_test vec_engine_test
    vec_pass_test)
  hotarubi_add_test(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE hotarubi_iss iss_reference)
endforeach()
