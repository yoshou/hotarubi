module npu_top
  import isa_pkg::*;
  import vec_pkg::*;
  import vec_rom_pkg::*;
(
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic [11:0]  s_axil_awaddr_i,
  input  logic         s_axil_awvalid_i,
  output logic         s_axil_awready_o,
  input  logic [31:0]  s_axil_wdata_i,
  input  logic [3:0]   s_axil_wstrb_i,
  input  logic         s_axil_wvalid_i,
  output logic         s_axil_wready_o,
  output logic [1:0]   s_axil_bresp_o,
  output logic         s_axil_bvalid_o,
  input  logic         s_axil_bready_i,
  input  logic [11:0]  s_axil_araddr_i,
  input  logic         s_axil_arvalid_i,
  output logic         s_axil_arready_o,
  output logic [31:0]  s_axil_rdata_o,
  output logic [1:0]   s_axil_rresp_o,
  output logic         s_axil_rvalid_o,
  input  logic         s_axil_rready_i,
  output logic         irq_o,
  output logic [31:0]  m0_axi_araddr_o,
  output logic [7:0]   m0_axi_arlen_o,
  output logic [2:0]   m0_axi_arsize_o,
  output logic [1:0]   m0_axi_arburst_o,
  output logic         m0_axi_arvalid_o,
  input  logic         m0_axi_arready_i,
  input  logic [127:0] m0_axi_rdata_i,
  input  logic [1:0]   m0_axi_rresp_i,
  input  logic         m0_axi_rlast_i,
  input  logic         m0_axi_rvalid_i,
  output logic         m0_axi_rready_o,
  output logic [31:0]  m0_axi_awaddr_o,
  output logic [7:0]   m0_axi_awlen_o,
  output logic [2:0]   m0_axi_awsize_o,
  output logic [1:0]   m0_axi_awburst_o,
  output logic         m0_axi_awvalid_o,
  input  logic         m0_axi_awready_i,
  output logic [127:0] m0_axi_wdata_o,
  output logic [15:0]  m0_axi_wstrb_o,
  output logic         m0_axi_wlast_o,
  output logic         m0_axi_wvalid_o,
  input  logic         m0_axi_wready_i,
  input  logic [1:0]   m0_axi_bresp_i,
  input  logic         m0_axi_bvalid_i,
  output logic         m0_axi_bready_o,
  output logic [31:0]  m1_axi_araddr_o,
  output logic [7:0]   m1_axi_arlen_o,
  output logic [2:0]   m1_axi_arsize_o,
  output logic [1:0]   m1_axi_arburst_o,
  output logic         m1_axi_arvalid_o,
  input  logic         m1_axi_arready_i,
  input  logic [127:0] m1_axi_rdata_i,
  input  logic [1:0]   m1_axi_rresp_i,
  input  logic         m1_axi_rlast_i,
  input  logic         m1_axi_rvalid_i,
  output logic         m1_axi_rready_o,
  output logic [31:0]  m1_axi_awaddr_o,
  output logic [7:0]   m1_axi_awlen_o,
  output logic [2:0]   m1_axi_awsize_o,
  output logic [1:0]   m1_axi_awburst_o,
  output logic         m1_axi_awvalid_o,
  input  logic         m1_axi_awready_i,
  output logic [127:0] m1_axi_wdata_o,
  output logic [15:0]  m1_axi_wstrb_o,
  output logic         m1_axi_wlast_o,
  output logic         m1_axi_wvalid_o,
  input  logic         m1_axi_wready_i,
  input  logic [1:0]   m1_axi_bresp_i,
  input  logic         m1_axi_bvalid_i,
  output logic         m1_axi_bready_o
);

  logic        gemm_cmd_valid, gemm_busy, vec_cmd_valid, vec_busy;
  logic        dma_cmd_valid, dma_busy;
  logic [19:0] gemm_a_base, gemm_w_base, gemm_c_base;
  logic [9:0]  gemm_a_stride, gemm_kb, gemm_m, gemm_w_stride;
  logic [7:0]  gemm_n8;
  logic [1:0]  gemm_wmode;
  logic        gemm_acc;
  logic [7:0]  vec_entry;
  logic        vec_q;
  logic [11:0] vec_src, vec_src2;
  logic [12:0] vec_dst;
  logic [8:0]  vec_rows_q;
  logic [7:0]  vec_n8;
  logic [8:0]  vec_p_base;
  logic [10:0] vec_aux;
  logic [31:0] vec_imm;
  logic        dma_st, dma_acc, dma_flush;
  logic [27:0] dma_ddr_addr;
  logic [2:0]  dma_mem, dma_kind;
  logic [12:0] dma_oc_addr;
  logic [11:0] dma_rows, dma_row_words;
  logic [19:0] dma_ddr_stride;
  logic [5:0]  dma_tok;
  logic        dma_axi_err;
  logic        gemm_stat_fed, vec_stat_issue, vec_stat_drain, vec_stat_scalar, vec_stat_pass;
  logic        dma_stat_rd_beat, dma_stat_wr_beat, dma_stat_denied;
  logic [1:0]  dma_stat_null;

  logic [31:0]  d_axi_araddr, d_axi_awaddr, q_axi_araddr, q_axi_awaddr;
  logic [7:0]   d_axi_arlen, d_axi_awlen, q_axi_arlen, q_axi_awlen;
  logic [2:0]   d_axi_arsize, d_axi_awsize, q_axi_arsize, q_axi_awsize;
  logic [1:0]   d_axi_arburst, d_axi_awburst, q_axi_arburst, q_axi_awburst;
  logic [1:0]   d_axi_rresp, d_axi_bresp, q_axi_rresp, q_axi_bresp;
  logic [127:0] d_axi_rdata, d_axi_wdata, q_axi_rdata, q_axi_wdata;
  logic [15:0]  d_axi_wstrb, q_axi_wstrb;
  logic         d_axi_arvalid, d_axi_arready, d_axi_rlast, d_axi_rvalid, d_axi_rready;
  logic         d_axi_awvalid, d_axi_awready, d_axi_wlast, d_axi_wvalid, d_axi_wready;
  logic         d_axi_bvalid, d_axi_bready;
  logic         q_axi_arvalid, q_axi_arready, q_axi_rlast, q_axi_rvalid, q_axi_rready;
  logic         q_axi_awvalid, q_axi_awready, q_axi_wlast, q_axi_wvalid, q_axi_wready;
  logic         q_axi_bvalid, q_axi_bready;

  axi_slice u_m0 (
    .clk_i,
    .rst_ni,
    .ar_addr_i      (d_axi_araddr),
    .ar_len_i       (d_axi_arlen),
    .ar_size_i      (d_axi_arsize),
    .ar_burst_i     (d_axi_arburst),
    .ar_valid_i     (d_axi_arvalid),
    .ar_ready_o     (d_axi_arready),
    .r_data_o       (d_axi_rdata),
    .r_resp_o       (d_axi_rresp),
    .r_last_o       (d_axi_rlast),
    .r_valid_o      (d_axi_rvalid),
    .aw_addr_i      (d_axi_awaddr),
    .aw_len_i       (d_axi_awlen),
    .aw_size_i      (d_axi_awsize),
    .aw_burst_i     (d_axi_awburst),
    .aw_valid_i     (d_axi_awvalid),
    .aw_ready_o     (d_axi_awready),
    .w_data_i       (d_axi_wdata),
    .w_strb_i       (d_axi_wstrb),
    .w_last_i       (d_axi_wlast),
    .w_valid_i      (d_axi_wvalid),
    .w_ready_o      (d_axi_wready),
    .b_resp_o       (d_axi_bresp),
    .b_valid_o      (d_axi_bvalid),
    .m_axi_araddr_o (m0_axi_araddr_o),
    .m_axi_arlen_o  (m0_axi_arlen_o),
    .m_axi_arsize_o (m0_axi_arsize_o),
    .m_axi_arburst_o(m0_axi_arburst_o),
    .m_axi_arvalid_o(m0_axi_arvalid_o),
    .m_axi_arready_i(m0_axi_arready_i),
    .m_axi_rdata_i  (m0_axi_rdata_i),
    .m_axi_rresp_i  (m0_axi_rresp_i),
    .m_axi_rlast_i  (m0_axi_rlast_i),
    .m_axi_rvalid_i (m0_axi_rvalid_i),
    .m_axi_rready_o (m0_axi_rready_o),
    .m_axi_awaddr_o (m0_axi_awaddr_o),
    .m_axi_awlen_o  (m0_axi_awlen_o),
    .m_axi_awsize_o (m0_axi_awsize_o),
    .m_axi_awburst_o(m0_axi_awburst_o),
    .m_axi_awvalid_o(m0_axi_awvalid_o),
    .m_axi_awready_i(m0_axi_awready_i),
    .m_axi_wdata_o  (m0_axi_wdata_o),
    .m_axi_wstrb_o  (m0_axi_wstrb_o),
    .m_axi_wlast_o  (m0_axi_wlast_o),
    .m_axi_wvalid_o (m0_axi_wvalid_o),
    .m_axi_wready_i (m0_axi_wready_i),
    .m_axi_bresp_i  (m0_axi_bresp_i),
    .m_axi_bvalid_i (m0_axi_bvalid_i),
    .m_axi_bready_o (m0_axi_bready_o)
  );

  axi_slice u_m1 (
    .clk_i,
    .rst_ni,
    .ar_addr_i      (q_axi_araddr),
    .ar_len_i       (q_axi_arlen),
    .ar_size_i      (q_axi_arsize),
    .ar_burst_i     (q_axi_arburst),
    .ar_valid_i     (q_axi_arvalid),
    .ar_ready_o     (q_axi_arready),
    .r_data_o       (q_axi_rdata),
    .r_resp_o       (q_axi_rresp),
    .r_last_o       (q_axi_rlast),
    .r_valid_o      (q_axi_rvalid),
    .aw_addr_i      (q_axi_awaddr),
    .aw_len_i       (q_axi_awlen),
    .aw_size_i      (q_axi_awsize),
    .aw_burst_i     (q_axi_awburst),
    .aw_valid_i     (q_axi_awvalid),
    .aw_ready_o     (q_axi_awready),
    .w_data_i       (q_axi_wdata),
    .w_strb_i       (q_axi_wstrb),
    .w_last_i       (q_axi_wlast),
    .w_valid_i      (q_axi_wvalid),
    .w_ready_o      (q_axi_wready),
    .b_resp_o       (q_axi_bresp),
    .b_valid_o      (q_axi_bvalid),
    .m_axi_araddr_o (m1_axi_araddr_o),
    .m_axi_arlen_o  (m1_axi_arlen_o),
    .m_axi_arsize_o (m1_axi_arsize_o),
    .m_axi_arburst_o(m1_axi_arburst_o),
    .m_axi_arvalid_o(m1_axi_arvalid_o),
    .m_axi_arready_i(m1_axi_arready_i),
    .m_axi_rdata_i  (m1_axi_rdata_i),
    .m_axi_rresp_i  (m1_axi_rresp_i),
    .m_axi_rlast_i  (m1_axi_rlast_i),
    .m_axi_rvalid_i (m1_axi_rvalid_i),
    .m_axi_rready_o (m1_axi_rready_o),
    .m_axi_awaddr_o (m1_axi_awaddr_o),
    .m_axi_awlen_o  (m1_axi_awlen_o),
    .m_axi_awsize_o (m1_axi_awsize_o),
    .m_axi_awburst_o(m1_axi_awburst_o),
    .m_axi_awvalid_o(m1_axi_awvalid_o),
    .m_axi_awready_i(m1_axi_awready_i),
    .m_axi_wdata_o  (m1_axi_wdata_o),
    .m_axi_wstrb_o  (m1_axi_wstrb_o),
    .m_axi_wlast_o  (m1_axi_wlast_o),
    .m_axi_wvalid_o (m1_axi_wvalid_o),
    .m_axi_wready_i (m1_axi_wready_i),
    .m_axi_bresp_i  (m1_axi_bresp_i),
    .m_axi_bvalid_i (m1_axi_bvalid_i),
    .m_axi_bready_o (m1_axi_bready_o)
  );

  assign dma_axi_err = (d_axi_rvalid && d_axi_rresp[1]) || (d_axi_bvalid && d_axi_bresp[1]);

  npu_seq u_seq (
    .clk_i,
    .rst_ni,
    .s_axil_awaddr_i (s_axil_awaddr_i[7:0]),
    .s_axil_awvalid_i,
    .s_axil_awready_o,
    .s_axil_wdata_i,
    .s_axil_wstrb_i,
    .s_axil_wvalid_i,
    .s_axil_wready_o,
    .s_axil_bresp_o,
    .s_axil_bvalid_o,
    .s_axil_bready_i,
    .s_axil_araddr_i (s_axil_araddr_i[7:0]),
    .s_axil_arvalid_i,
    .s_axil_arready_o,
    .s_axil_rdata_o,
    .s_axil_rresp_o,
    .s_axil_rvalid_o,
    .s_axil_rready_i,
    .irq_o,
    .m_axi_araddr_o  (q_axi_araddr),
    .m_axi_arlen_o   (q_axi_arlen),
    .m_axi_arsize_o  (q_axi_arsize),
    .m_axi_arburst_o (q_axi_arburst),
    .m_axi_arvalid_o (q_axi_arvalid),
    .m_axi_arready_i (q_axi_arready),
    .m_axi_rdata_i   (q_axi_rdata),
    .m_axi_rresp_i   (q_axi_rresp),
    .m_axi_rlast_i   (q_axi_rlast),
    .m_axi_rvalid_i  (q_axi_rvalid),
    .m_axi_rready_o  (q_axi_rready),
    .m_axi_awaddr_o  (q_axi_awaddr),
    .m_axi_awlen_o   (q_axi_awlen),
    .m_axi_awsize_o  (q_axi_awsize),
    .m_axi_awburst_o (q_axi_awburst),
    .m_axi_awvalid_o (q_axi_awvalid),
    .m_axi_awready_i (q_axi_awready),
    .m_axi_wdata_o   (q_axi_wdata),
    .m_axi_wstrb_o   (q_axi_wstrb),
    .m_axi_wlast_o   (q_axi_wlast),
    .m_axi_wvalid_o  (q_axi_wvalid),
    .m_axi_wready_i  (q_axi_wready),
    .m_axi_bresp_i   (q_axi_bresp),
    .m_axi_bvalid_i  (q_axi_bvalid),
    .m_axi_bready_o  (q_axi_bready),
    .dma_axi_err_i   (dma_axi_err),
    .gemm_cmd_valid_o(gemm_cmd_valid),
    .gemm_busy_i     (gemm_busy),
    .gemm_a_base_o   (gemm_a_base),
    .gemm_a_stride_o (gemm_a_stride),
    .gemm_kb_o       (gemm_kb),
    .gemm_m_o        (gemm_m),
    .gemm_w_base_o   (gemm_w_base),
    .gemm_w_stride_o (gemm_w_stride),
    .gemm_n8_o       (gemm_n8),
    .gemm_c_base_o   (gemm_c_base),
    .gemm_wmode_o    (gemm_wmode),
    .gemm_acc_o      (gemm_acc),
    .vec_cmd_valid_o (vec_cmd_valid),
    .vec_busy_i      (vec_busy),
    .vec_entry_o     (vec_entry),
    .vec_q_o         (vec_q),
    .vec_src_o       (vec_src),
    .vec_src2_o      (vec_src2),
    .vec_dst_o       (vec_dst),
    .vec_rows_q_o    (vec_rows_q),
    .vec_n8_o        (vec_n8),
    .vec_p_base_o    (vec_p_base),
    .vec_aux_o       (vec_aux),
    .vec_imm_o       (vec_imm),
    .dma_cmd_valid_o (dma_cmd_valid),
    .dma_busy_i      (dma_busy),
    .dma_st_o        (dma_st),
    .dma_ddr_addr_o  (dma_ddr_addr),
    .dma_mem_o       (dma_mem),
    .dma_oc_addr_o   (dma_oc_addr),
    .dma_rows_o      (dma_rows),
    .dma_row_words_o (dma_row_words),
    .dma_ddr_stride_o(dma_ddr_stride),
    .dma_kind_o      (dma_kind),
    .dma_tok_o       (dma_tok),
    .dma_acc_o       (dma_acc),
    .dma_flush_o     (dma_flush),
    .gemm_stat_fed_i (gemm_stat_fed),
    .vec_stat_issue_i(vec_stat_issue),
    .vec_stat_drain_i(vec_stat_drain),
    .vec_stat_scalar_i(vec_stat_scalar),
    .vec_stat_pass_i (vec_stat_pass),
    .dma_stat_rd_beat_i(dma_stat_rd_beat),
    .dma_stat_wr_beat_i(dma_stat_wr_beat),
    .dma_stat_denied_i(dma_stat_denied),
    .dma_stat_null_i (dma_stat_null)
  );

  logic [19:0]   g_abuf_addr, g_wbuf_addr, g_fbuf_raddr, g_fbuf_waddr;
  logic [1055:0] abuf_qa, abuf_qb2;
  logic [1151:0] wbuf_q;
  logic [1023:0] g_fbuf_wdata;
  (* keep = "true" *) logic [1023:0] fbuf_q2;
  logic          g_fbuf_rd_req, g_fbuf_we;

  gemm_engine #(
    .Mp       (2),
    .Nt       (8),
    .Nacc     (8),
    .AddrWidth(20),
    .FbLat    (3)
  ) u_gemm (
    .clk_i,
    .rst_ni        (rst_ni),
    .cmd_valid_i   (gemm_cmd_valid),
    .cmd_a_base_i  (gemm_a_base),
    .cmd_a_stride_i(gemm_a_stride),
    .cmd_kb_i      (gemm_kb),
    .cmd_m_i       (gemm_m),
    .cmd_w_base_i  (gemm_w_base),
    .cmd_w_stride_i(gemm_w_stride),
    .cmd_n8_i      (gemm_n8),
    .cmd_c_base_i  (gemm_c_base),
    .cmd_wmode_i   (gemm_wmode),
    .cmd_acc_i     (gemm_acc),
    .busy_o        (gemm_busy),
    .abuf_addr_o   (g_abuf_addr),
    .abuf_rdata_i  (abuf_qa[1023:0]),
    .ascl_rdata_i  (abuf_qa[1055:1024]),
    .wbuf_addr_o   (g_wbuf_addr),
    .wbuf_rdata_i  (wbuf_q),
    .fbuf_rd_req_o (g_fbuf_rd_req),
    .fbuf_raddr_o  (g_fbuf_raddr),
    .fbuf_rdata_i  (fbuf_q2),
    .fbuf_we_o     (g_fbuf_we),
    .fbuf_waddr_o  (g_fbuf_waddr),
    .fbuf_wdata_o  (g_fbuf_wdata),
    .stat_fed_o    (gemm_stat_fed)
  );

  logic          v_fb_rd_req, v_fb_rd_grant, v_fb_wr_req, v_fb_wr_grant, v_ab_wr_req;
  logic [11:0]   v_fb_raddr, v_fb_waddr;
  logic [1023:0] v_fb_wdata, v_fb_tdata, pbuf_q;
  logic          v_fb_wtr;
  logic [8:0]    v_pb_addr;
  logic [9:0]    tab_addr [Lanes];
  logic [63:0]   tab_data [Lanes];
  logic [12:0]   v_ab_waddr;
  logic [1055:0] v_ab_wdata;
  logic [9:0]    cb_addr;
  logic [127:0]  cb_rdata;

  vec_engine u_vec (
    .clk_i,
    .rst_ni         (rst_ni),
    .cmd_valid_i    (vec_cmd_valid),
    .cmd_entry_i    (vec_entry),
    .cmd_q_i        (vec_q),
    .cmd_src_i      (vec_src),
    .cmd_src2_i     (vec_src2),
    .cmd_dst_i      (vec_dst),
    .cmd_rows_q_i   (vec_rows_q),
    .cmd_n8_i       (vec_n8),
    .cmd_p_base_i   (vec_p_base),
    .cmd_aux_i      (vec_aux),
    .cmd_imm_i      (vec_imm),
    .busy_o         (vec_busy),
    .fbuf_rd_req_o  (v_fb_rd_req),
    .fbuf_rd_grant_i(v_fb_rd_grant),
    .fbuf_raddr_o   (v_fb_raddr),
    .fbuf_rdata_i   (fbuf_q2),
    .fbuf_wr_req_o  (v_fb_wr_req),
    .fbuf_wr_grant_i(v_fb_wr_grant),
    .fbuf_waddr_o   (v_fb_waddr),
    .fbuf_wdata_o   (v_fb_wdata),
    .fbuf_wtr_o     (v_fb_wtr),
    .fbuf_tdata_o   (v_fb_tdata),
    .pbuf_a_addr_o  (v_pb_addr),
    .pbuf_a_rdata_i (pbuf_q),
    .tab_addr_o     (tab_addr),
    .tab_data_i     (tab_data),
    .abuf_wr_req_o  (v_ab_wr_req),
    .abuf_waddr_o   (v_ab_waddr),
    .abuf_wdata_o   (v_ab_wdata),
    .cbuf_addr_o    (cb_addr),
    .cbuf_rdata_i   (cb_rdata),
    .stat_issue_o   (vec_stat_issue),
    .stat_drain_o   (vec_stat_drain),
    .stat_scalar_o  (vec_stat_scalar),
    .stat_pass_o    (vec_stat_pass)
  );

  logic          d_wr_req, d_wr_grant, d_rd_req, d_rd_grant;
  logic [2:0]    d_wr_mem, d_rd_mem;
  logic [12:0]   d_wr_addr, d_rd_addr;
  logic [1151:0] d_wr_data;
  logic [1055:0] d_rd_data;

  dma_engine u_dma (
    .clk_i,
    .rst_ni          (rst_ni),
    .cmd_valid_i     (dma_cmd_valid),
    .cmd_st_i        (dma_st),
    .cmd_ddr_addr_i  (dma_ddr_addr),
    .cmd_mem_i       (dma_mem),
    .cmd_oc_addr_i   (dma_oc_addr),
    .cmd_rows_i      (dma_rows),
    .cmd_row_words_i (dma_row_words),
    .cmd_ddr_stride_i(dma_ddr_stride),
    .cmd_kind_i      (dma_kind),
    .cmd_tok_i       (dma_tok),
    .cmd_acc_i       (dma_acc),
    .cmd_flush_i     (dma_flush),
    .busy_o          (dma_busy),
    .m_axi_araddr_o  (d_axi_araddr),
    .m_axi_arlen_o   (d_axi_arlen),
    .m_axi_arsize_o  (d_axi_arsize),
    .m_axi_arburst_o (d_axi_arburst),
    .m_axi_arvalid_o (d_axi_arvalid),
    .m_axi_arready_i (d_axi_arready),
    .m_axi_rdata_i   (d_axi_rdata),
    .m_axi_rresp_i   (d_axi_rresp),
    .m_axi_rlast_i   (d_axi_rlast),
    .m_axi_rvalid_i  (d_axi_rvalid),
    .m_axi_rready_o  (d_axi_rready),
    .m_axi_awaddr_o  (d_axi_awaddr),
    .m_axi_awlen_o   (d_axi_awlen),
    .m_axi_awsize_o  (d_axi_awsize),
    .m_axi_awburst_o (d_axi_awburst),
    .m_axi_awvalid_o (d_axi_awvalid),
    .m_axi_awready_i (d_axi_awready),
    .m_axi_wdata_o   (d_axi_wdata),
    .m_axi_wstrb_o   (d_axi_wstrb),
    .m_axi_wlast_o   (d_axi_wlast),
    .m_axi_wvalid_o  (d_axi_wvalid),
    .m_axi_wready_i  (d_axi_wready),
    .m_axi_bresp_i   (d_axi_bresp),
    .m_axi_bvalid_i  (d_axi_bvalid),
    .m_axi_bready_o  (d_axi_bready),
    .wr_req_o        (d_wr_req),
    .wr_grant_i      (d_wr_grant),
    .wr_mem_o        (d_wr_mem),
    .wr_addr_o       (d_wr_addr),
    .wr_data_o       (d_wr_data),
    .rd_req_o        (d_rd_req),
    .rd_grant_i      (d_rd_grant),
    .rd_mem_o        (d_rd_mem),
    .rd_addr_o       (d_rd_addr),
    .rd_data_i       (d_rd_data),
    .stat_rd_beat_o  (dma_stat_rd_beat),
    .stat_wr_beat_o  (dma_stat_wr_beat),
    .stat_denied_o   (dma_stat_denied),
    .stat_null_o     (dma_stat_null)
  );

  logic d_wr_abuf, d_wr_wbuf, d_wr_fbuf, d_wr_pbuf, d_wr_tbuf, d_wr_cbuf;

  assign d_wr_abuf = d_wr_req && (d_wr_mem == MemAbuf);
  assign d_wr_wbuf = d_wr_req && (d_wr_mem == MemWbuf);
  assign d_wr_fbuf = d_wr_req && (d_wr_mem == MemFbuf);
  assign d_wr_pbuf = d_wr_req && (d_wr_mem == MemPbuf);
  assign d_wr_tbuf = d_wr_req && (d_wr_mem == MemTbuf);
  assign d_wr_cbuf = d_wr_req && (d_wr_mem == MemCbuf);

  assign v_fb_wr_grant = v_fb_wr_req && !g_fbuf_we;
  assign v_fb_rd_grant = v_fb_rd_req && !g_fbuf_rd_req;

  always_comb begin
    case (d_wr_mem)
      MemAbuf:    d_wr_grant = d_wr_req && !v_ab_wr_req;
      MemFbuf:    d_wr_grant = d_wr_req && !g_fbuf_we && !v_fb_wr_req;
      default:    d_wr_grant = d_wr_req;
    endcase
  end

  assign d_rd_grant = d_rd_req && ((d_rd_mem == MemAbuf) ? !v_ab_wr_req :
                                   (d_rd_mem == MemFbuf) ? (!g_fbuf_rd_req && !v_fb_rd_req) : 1'b1);
  assign d_rd_data  = (d_rd_mem == MemAbuf) ? abuf_qb2 : {32'd0, fbuf_q2};

  (* ram_style = "ultra" *) logic [1055:0] abuf [8192];
  logic [1055:0] abuf_qa1, abuf_qa2, abuf_qb1, ab_wd;
  logic [12:0]   ab_addr;
  logic          ab_we;

  always_ff @(posedge clk_i) begin
    ab_we   <= v_ab_wr_req | d_wr_abuf;
    ab_addr <= v_ab_wr_req ? v_ab_waddr : d_wr_abuf ? d_wr_addr : d_rd_addr;
    ab_wd   <= v_ab_wr_req ? v_ab_wdata : d_wr_data[1055:0];
  end

  always_ff @(posedge clk_i) begin
    abuf_qa1 <= abuf[g_abuf_addr[12:0]];
    abuf_qa2 <= abuf_qa1;
  end

  assign abuf_qa = abuf_qa2;

  always_ff @(posedge clk_i) begin
    if (ab_we) begin
      abuf[ab_addr] <= ab_wd;
    end else begin
      abuf_qb1 <= abuf[ab_addr];
    end
    abuf_qb2 <= abuf_qb1;
  end

  (* ram_style = "block" *) logic [1151:0] wbuf [2048];
  logic [1151:0] wbuf_q1;

  always_ff @(posedge clk_i) begin
    wbuf_q1 <= wbuf[g_wbuf_addr[10:0]];
    wbuf_q  <= wbuf_q1;
    if (d_wr_wbuf) begin
      wbuf[d_wr_addr[10:0]] <= d_wr_data;
    end
  end

  (* ram_style = "ultra" *) logic [1023:0] fbuf [4096];
  logic [1023:0] fbuf_q1, fb_wd, fb_wd_r;
  logic [11:0]   fb_wa, fb_wa_r, fb_ra, g_ra_r, d_ra_r;
  logic          fb_we, fb_we_r, sel_g_r, sel_v_r;

  assign fb_we = g_fbuf_we | v_fb_wr_req | d_wr_fbuf;
  assign fb_wa = g_fbuf_we ? g_fbuf_waddr[11:0] : v_fb_wr_req ? v_fb_waddr : d_wr_addr[11:0];
  logic [1:0] fb_src;

  assign fb_src = g_fbuf_we ? 2'd0 : v_fb_wr_req ? (v_fb_wtr ? 2'd2 : 2'd1) : 2'd3;

  always_comb begin
    case (fb_src)
      2'd0:    fb_wd = g_fbuf_wdata;
      2'd1:    fb_wd = v_fb_wdata;
      2'd2:    fb_wd = v_fb_tdata;
      default: fb_wd = d_wr_data[1023:0];
    endcase
  end

  always_ff @(posedge clk_i) begin
    fb_we_r <= fb_we;
    fb_wa_r <= fb_wa;
    fb_wd_r <= fb_wd;
  end

  always_ff @(posedge clk_i) begin
    sel_g_r <= g_fbuf_rd_req;
    sel_v_r <= v_fb_rd_req && v_fb_rd_grant;
    g_ra_r  <= g_fbuf_raddr[11:0];
    d_ra_r  <= d_rd_addr[11:0];
  end

  assign fb_ra = sel_g_r ? g_ra_r : sel_v_r ? v_fb_raddr : d_ra_r;

  always_ff @(posedge clk_i) begin
    if (fb_we_r) begin
      fbuf[fb_wa_r] <= fb_wd_r;
    end
    fbuf_q1 <= fbuf[fb_ra];
    fbuf_q2 <= fbuf_q1;
  end

  (* ram_style = "block" *) logic [1023:0] pbuf [512];
  logic [1023:0] pbuf_q1, pb_wd;
  logic [8:0]    pb_wa;
  logic          pb_we;

  always_ff @(posedge clk_i) begin
    pb_we <= d_wr_pbuf;
    pb_wa <= d_wr_addr[8:0];
    pb_wd <= d_wr_data[1023:0];
  end

  always_ff @(posedge clk_i) begin
    if (pb_we) begin
      pbuf[pb_wa] <= pb_wd;
    end
    pbuf_q1 <= pbuf[v_pb_addr];
    pbuf_q  <= pbuf_q1;
  end

  logic        tb_we_r;
  logic [9:0]  tb_wa_r;
  logic [63:0] tb_wd_r;

  always_ff @(posedge clk_i) begin
    tb_we_r <= d_wr_tbuf;
    tb_wa_r <= d_wr_addr[9:0];
    tb_wd_r <= d_wr_data[63:0];
  end

  for (genvar i = 0; i < Lanes / 2; i++) begin : g_tb
    (* ram_style = "block" *) logic [63:0] tb [1024];
    logic [63:0] ra_r, rb_r;
    logic [9:0]  addr_b;

    assign addr_b = tb_we_r ? tb_wa_r : tab_addr[2*i+1];

    always_ff @(posedge clk_i) begin
      ra_r <= tb[tab_addr[2*i]];
      if (tb_we_r) begin
        tb[addr_b] <= tb_wd_r;
      end
      rb_r <= tb[addr_b];
    end

    assign tab_data[2*i]   = ra_r;
    assign tab_data[2*i+1] = rb_r;
  end

  (* ram_style = "block" *) logic [127:0] cbuf [1024];

  initial begin
    for (int i = 0; i < 1024; i++) begin
      cbuf[i] = (i < VecRomWords) ? Rom[i] : 128'd0;
    end
  end

  logic [127:0] cb_q1;
  logic         cb_we_r;
  logic [9:0]   cb_wa_r;
  logic [127:0] cb_wd_r;

  always_ff @(posedge clk_i) begin
    cb_we_r <= d_wr_cbuf;
    cb_wa_r <= d_wr_addr[9:0];
    cb_wd_r <= d_wr_data[127:0];
  end

  always_ff @(posedge clk_i) begin
    if (cb_we_r) begin
      cbuf[cb_wa_r] <= cb_wd_r;
    end
    cb_q1    <= cbuf[cb_addr];
    cb_rdata <= cb_q1;
  end

  logic unused_top;

  assign unused_top = ^{d_axi_rready, d_axi_bready, q_axi_rready, q_axi_bready,
                        s_axil_awaddr_i[11:8], s_axil_araddr_i[11:8], g_abuf_addr[19:13],
                        g_wbuf_addr[19:11], g_fbuf_raddr[19:12], g_fbuf_waddr[19:12],
                        d_wr_data[1151:1056]};

endmodule
