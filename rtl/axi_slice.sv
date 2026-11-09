module axi_slice (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic [31:0]  ar_addr_i,
  input  logic [7:0]   ar_len_i,
  input  logic [2:0]   ar_size_i,
  input  logic [1:0]   ar_burst_i,
  input  logic         ar_valid_i,
  output logic         ar_ready_o,
  output logic [127:0] r_data_o,
  output logic [1:0]   r_resp_o,
  output logic         r_last_o,
  output logic         r_valid_o,
  input  logic [31:0]  aw_addr_i,
  input  logic [7:0]   aw_len_i,
  input  logic [2:0]   aw_size_i,
  input  logic [1:0]   aw_burst_i,
  input  logic         aw_valid_i,
  output logic         aw_ready_o,
  input  logic [127:0] w_data_i,
  input  logic [15:0]  w_strb_i,
  input  logic         w_last_i,
  input  logic         w_valid_i,
  output logic         w_ready_o,
  output logic [1:0]   b_resp_o,
  output logic         b_valid_o,
  output logic [31:0]  m_axi_araddr_o,
  output logic [7:0]   m_axi_arlen_o,
  output logic [2:0]   m_axi_arsize_o,
  output logic [1:0]   m_axi_arburst_o,
  output logic         m_axi_arvalid_o,
  input  logic         m_axi_arready_i,
  input  logic [127:0] m_axi_rdata_i,
  input  logic [1:0]   m_axi_rresp_i,
  input  logic         m_axi_rlast_i,
  input  logic         m_axi_rvalid_i,
  output logic         m_axi_rready_o,
  output logic [31:0]  m_axi_awaddr_o,
  output logic [7:0]   m_axi_awlen_o,
  output logic [2:0]   m_axi_awsize_o,
  output logic [1:0]   m_axi_awburst_o,
  output logic         m_axi_awvalid_o,
  input  logic         m_axi_awready_i,
  output logic [127:0] m_axi_wdata_o,
  output logic [15:0]  m_axi_wstrb_o,
  output logic         m_axi_wlast_o,
  output logic         m_axi_wvalid_o,
  input  logic         m_axi_wready_i,
  input  logic [1:0]   m_axi_bresp_i,
  input  logic         m_axi_bvalid_i,
  output logic         m_axi_bready_o
);

  reg_slice #(
    .Width(45)
  ) u_ar (
    .clk_i,
    .rst_ni,
    .in_valid_i (ar_valid_i),
    .in_ready_o (ar_ready_o),
    .in_data_i  ({ar_addr_i, ar_len_i, ar_size_i, ar_burst_i}),
    .out_valid_o(m_axi_arvalid_o),
    .out_ready_i(m_axi_arready_i),
    .out_data_o ({m_axi_araddr_o, m_axi_arlen_o, m_axi_arsize_o, m_axi_arburst_o})
  );

  reg_slice #(
    .Width(45)
  ) u_aw (
    .clk_i,
    .rst_ni,
    .in_valid_i (aw_valid_i),
    .in_ready_o (aw_ready_o),
    .in_data_i  ({aw_addr_i, aw_len_i, aw_size_i, aw_burst_i}),
    .out_valid_o(m_axi_awvalid_o),
    .out_ready_i(m_axi_awready_i),
    .out_data_o ({m_axi_awaddr_o, m_axi_awlen_o, m_axi_awsize_o, m_axi_awburst_o})
  );

  reg_slice #(
    .Width(145)
  ) u_w (
    .clk_i,
    .rst_ni,
    .in_valid_i (w_valid_i),
    .in_ready_o (w_ready_o),
    .in_data_i  ({w_data_i, w_strb_i, w_last_i}),
    .out_valid_o(m_axi_wvalid_o),
    .out_ready_i(m_axi_wready_i),
    .out_data_o ({m_axi_wdata_o, m_axi_wstrb_o, m_axi_wlast_o})
  );

  assign m_axi_rready_o = 1'b1;
  assign m_axi_bready_o = 1'b1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      r_valid_o <= 1'b0;
      b_valid_o <= 1'b0;
    end else begin
      r_valid_o <= m_axi_rvalid_i;
      b_valid_o <= m_axi_bvalid_i;
    end
  end

  always_ff @(posedge clk_i) begin
    r_data_o <= m_axi_rdata_i;
    r_resp_o <= m_axi_rresp_i;
    r_last_o <= m_axi_rlast_i;
    b_resp_o <= m_axi_bresp_i;
  end

endmodule
