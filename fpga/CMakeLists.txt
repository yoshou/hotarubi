find_program(VIVADO_EXECUTABLE vivado)
find_program(BOOTGEN_EXECUTABLE bootgen)
find_program(DTC_EXECUTABLE dtc)
if(NOT VIVADO_EXECUTABLE OR NOT BOOTGEN_EXECUTABLE OR NOT DTC_EXECUTABLE)
  message(STATUS "Vivado, bootgen or dtc not found; FPGA targets are not available")
  return()
endif()

set(HOTARUBI_BOARD_REPO "" CACHE PATH "Vivado board repository that contains the KV260 SOM board files")
set(HOTARUBI_IMPL_STRATEGY "" CACHE STRING "Vivado implementation strategy; empty keeps Performance_NetDelay_high")

set(fpga_dir ${CMAKE_CURRENT_BINARY_DIR})
set(build_tcl ${CMAKE_CURRENT_SOURCE_DIR}/build.tcl)
set(vivado_command ${VIVADO_EXECUTABLE} -mode batch -nojournal -source ${build_tcl})

add_custom_command(
  OUTPUT ${fpga_dir}/synth.stamp
  COMMAND ${vivado_command} -log ${fpga_dir}/synth.log
    -tclargs synth ${fpga_dir} "${HOTARUBI_BOARD_REPO}"
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_top_v.v ${HOTARUBI_RTL_SOURCES}
  COMMAND ${CMAKE_COMMAND} -E touch ${fpga_dir}/synth.stamp
  DEPENDS ${build_tcl} ${CMAKE_CURRENT_SOURCE_DIR}/npu_top_v.v ${HOTARUBI_RTL_SOURCES}
  WORKING_DIRECTORY ${fpga_dir}
  USES_TERMINAL
  VERBATIM)
add_custom_target(fpga-synth DEPENDS ${fpga_dir}/synth.stamp)

add_custom_command(
  OUTPUT ${fpga_dir}/impl.stamp
  COMMAND ${vivado_command} -log ${fpga_dir}/impl.log -tclargs impl ${fpga_dir} "${HOTARUBI_IMPL_STRATEGY}"
  COMMAND ${CMAKE_COMMAND} -E touch ${fpga_dir}/impl.stamp
  DEPENDS ${fpga_dir}/synth.stamp
  WORKING_DIRECTORY ${fpga_dir}
  USES_TERMINAL
  VERBATIM)
add_custom_target(fpga-impl DEPENDS ${fpga_dir}/impl.stamp)

file(WRITE ${fpga_dir}/npu.bif "all:\n{\n  [destination_device = pl] ${fpga_dir}/npu.bit\n}\n")

add_custom_command(
  OUTPUT ${fpga_dir}/npu.bit.bin ${fpga_dir}/npu.dtbo ${fpga_dir}/shell.json
  COMMAND ${vivado_command} -log ${fpga_dir}/bitstream.log -tclargs bitstream ${fpga_dir}
  COMMAND ${BOOTGEN_EXECUTABLE} -arch zynqmp -image ${fpga_dir}/npu.bif -process_bitstream bin -w on
  COMMAND ${DTC_EXECUTABLE} -@ -I dts -O dtb -o ${fpga_dir}/npu.dtbo ${CMAKE_CURRENT_SOURCE_DIR}/npu.dtso
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shell.json ${fpga_dir}/shell.json
  DEPENDS ${fpga_dir}/impl.stamp ${CMAKE_CURRENT_SOURCE_DIR}/npu.dtso ${CMAKE_CURRENT_SOURCE_DIR}/shell.json
  WORKING_DIRECTORY ${fpga_dir}
  USES_TERMINAL
  VERBATIM)
add_custom_target(fpga-bitstream DEPENDS ${fpga_dir}/npu.bit.bin ${fpga_dir}/npu.dtbo ${fpga_dir}/shell.json)
