add_library(hotarubi_compiler STATIC
  checkpoint.cpp
  ddr_image.cpp
  exact.cpp
  isa.cpp
  output.cpp
  program.cpp
  safetensors.cpp
  smolvla.cpp
  smolvla_common.cpp
  smolvla_expert.cpp
  smolvla_lm.cpp
  smolvla_vision.cpp
  sparsedrive.cpp
  sparsedrive_common.cpp
  sparsedrive_encoder.cpp
  sparsedrive_head.cpp
  sparsedrive_kernels.cpp
  sparsedrive_layers.cpp
  sparsedrive_motion.cpp
  tensors.cpp
  vprog.cpp
  weights.cpp)
target_include_directories(hotarubi_compiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hotarubi_compiler PUBLIC hotarubi_runtime)

add_executable(hotarubi_compile main.cpp)
target_link_libraries(hotarubi_compile PRIVATE hotarubi_compiler)

foreach(test checkpoint_test ddr_image_test exact_test isa_test output_test program_test
    safetensors_test smolvla_parts_test sparsedrive_layers_test sparsedrive_parts_test
    tensors_test vprog_test weights_test)
  hotarubi_add_test(${test} ${test}.cpp)
  target_link_libraries(${test} PRIVATE hotarubi_compiler)
endforeach()
