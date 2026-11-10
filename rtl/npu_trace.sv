module npu_trace
  import isa_pkg::*;
#(
  parameter int Depth = 64
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         clear_i,
  input  logic [31:0]  base_i,
  input  logic         ev_push_i,
  input  logic [TraceRecordBits-1:0] ev_data_i,
  output logic         ev_afull_o,
  output logic [31:0]  count_o,
  output logic         idle_o,
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

  logic                   empty, pop;
  logic [TraceRecordBits-1:0] dout;
  logic [$clog2(Depth):0] fifo_count;

  sfifo #(
    .Width      (TraceRecordBits),
    .Depth      (Depth),
    .AfullMargin(8)
  ) u_f (
    .clk_i,
    .rst_ni,
    .clear_i,
    .push_i (ev_push_i),
    .din_i  (ev_data_i),
    .pop_i  (pop),
    .dout_o (dout),
    .empty_o(empty),
    .afull_o(ev_afull_o),
    .count_o(fifo_count)
  );

  logic        aw_p, w_p, b_p;
  logic [31:0] wptr;

  assign pop    = !b_p && !empty && !clear_i;
  assign idle_o = empty && !b_p;

  assign m_axi_awaddr_o  = wptr;
  assign m_axi_awlen_o   = 8'd0;
  assign m_axi_awsize_o  = 3'b100;
  assign m_axi_awburst_o = 2'b01;
  assign m_axi_awvalid_o = aw_p;
  assign m_axi_wstrb_o   = 16'hFFFF;
  assign m_axi_wlast_o   = 1'b1;
  assign m_axi_wvalid_o  = w_p;
  assign m_axi_bready_o  = 1'b1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      aw_p          <= 1'b0;
      w_p           <= 1'b0;
      b_p           <= 1'b0;
      wptr          <= '0;
      count_o       <= '0;
      m_axi_wdata_o <= '0;
    end else begin
      if (clear_i) begin
        wptr    <= base_i;
        count_o <= '0;
      end
      if (pop) begin
        aw_p          <= 1'b1;
        w_p           <= 1'b1;
        b_p           <= 1'b1;
        m_axi_wdata_o <= dout;
      end
      if (aw_p && m_axi_awready_i) begin
        aw_p <= 1'b0;
      end
      if (w_p && m_axi_wready_i) begin
        w_p <= 1'b0;
      end
      if (b_p && m_axi_bvalid_i && !aw_p && !w_p) begin
        b_p     <= 1'b0;
        count_o <= count_o + 1'b1;
        wptr    <= wptr + 32'd16;
      end
    end
  end

  logic unused_trace;

  assign unused_trace = ^{fifo_count, m_axi_bresp_i};

endmodule
