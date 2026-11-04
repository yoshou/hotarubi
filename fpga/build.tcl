set step [lindex $argv 0]
set out_dir [file normalize [lindex $argv 1]]
set project_name hotarubi_kv260
set project_dir [file join $out_dir $project_name]
set jobs [exec nproc]

proc check_run {run} {
  set status [get_property STATUS [get_runs $run]]
  if {[get_property PROGRESS [get_runs $run]] ne "100%" || [string match "*ERROR*" $status]} {
    error "$run did not complete: $status"
  }
}

proc connect_axi {master slave object} {
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} \
    Clk_slave {Auto} \
    Clk_xbar {Auto} \
    Master $master \
    Slave $slave \
    ddr_seg {Auto} \
    intc_ip {New AXI SmartConnect} \
    master_apm {0}] [get_bd_intf_pins $object]
}

proc create_npu_bd {} {
  create_bd_design npu_bd

  set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
  apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset 1} $ps
  set_property -dict {
    CONFIG.PSU__USE__M_AXI_GP0 1
    CONFIG.PSU__MAXIGP0__DATA_WIDTH 128
    CONFIG.PSU__USE__M_AXI_GP1 0
    CONFIG.PSU__USE__M_AXI_GP2 0
    CONFIG.PSU__USE__S_AXI_GP2 1
    CONFIG.PSU__SAXIGP2__DATA_WIDTH 128
    CONFIG.PSU__USE__S_AXI_GP3 1
    CONFIG.PSU__SAXIGP3__DATA_WIDTH 128
    CONFIG.PSU__USE__IRQ0 1
    CONFIG.PSU__FPGA_PL0_ENABLE 1
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ 250
    CONFIG.PSU__FPGA_PL1_ENABLE 0
    CONFIG.PSU__FPGA_PL2_ENABLE 0
    CONFIG.PSU__FPGA_PL3_ENABLE 0
  } $ps

  create_bd_cell -type module -reference npu_top_v npu
  connect_axi /ps/M_AXI_HPM0_FPD /npu/s_axil npu/s_axil
  connect_axi /npu/m0_axi /ps/S_AXI_HP0_FPD ps/S_AXI_HP0_FPD
  connect_axi /npu/m1_axi /ps/S_AXI_HP1_FPD ps/S_AXI_HP1_FPD
  connect_bd_net [get_bd_pins npu/irq] [get_bd_pins ps/pl_ps_irq0]

  assign_bd_address
  set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data] -filter {NAME =~ *npu*}]
  if {[llength $seg] != 1} {
    error "expected one NPU address segment, found: $seg"
  }
  set_property offset 0xA0000000 $seg
  set_property range 4K $seg

  validate_bd_design
  save_bd_design
}

switch -- $step {
  synth {
    set board_repo [lindex $argv 2]
    set sources [lrange $argv 3 end]
    if {$board_repo ne ""} {
      set_param board.repoPaths [list [file normalize $board_repo]]
    }
    create_project $project_name $project_dir -part xck26-sfvc784-2LV-c -force
    set board_part [lindex [lsort -dictionary [get_board_parts -quiet xilinx.com:kv260_som:part0:*]] end]
    if {$board_part eq ""} {
      error "KV260 SOM board files not found; set HOTARUBI_BOARD_REPO to a Vivado board repository"
    }
    set_property board_part $board_part [current_project]
    add_files -norecurse $sources
    update_compile_order -fileset sources_1
    create_npu_bd
    make_wrapper -files [get_files npu_bd.bd] -top -import
    set_property top npu_bd_wrapper [current_fileset]
    update_compile_order -fileset sources_1
    set_property strategy Performance_NetDelay_high [get_runs impl_1]
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
    launch_runs synth_1 -jobs $jobs
    wait_on_run synth_1
    check_run synth_1
  }
  impl {
    open_project [file join $project_dir $project_name.xpr]
    launch_runs impl_1 -jobs $jobs
    wait_on_run impl_1
    check_run impl_1
    open_run impl_1
    report_utilization -file [file join $out_dir utilization.rpt]
    report_timing_summary -max_paths 10 -file [file join $out_dir timing_summary.rpt]
    set wns [get_property STATS.WNS [get_runs impl_1]]
    set whs [get_property STATS.WHS [get_runs impl_1]]
    puts "WNS $wns ns, WHS $whs ns"
    if {$wns < 0 || $whs < 0} {
      error "timing not met: WNS $wns ns, WHS $whs ns"
    }
  }
  bitstream {
    open_project [file join $project_dir $project_name.xpr]
    launch_runs impl_1 -to_step write_bitstream -jobs $jobs
    wait_on_run impl_1
    check_run impl_1
    file copy -force [file join [get_property DIRECTORY [get_runs impl_1]] npu_bd_wrapper.bit] \
      [file join $out_dir npu.bit]
  }
  default {
    error "unknown step: $step"
  }
}
