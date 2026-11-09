module npu_seq #(
  parameter int IbufDepth = 64
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic [7:0]   s_axil_awaddr_i,
  input  logic         s_axil_awvalid_i,
  output logic         s_axil_awready_o,
  input  logic [31:0]  s_axil_wdata_i,
  input  logic [3:0]   s_axil_wstrb_i,
  input  logic         s_axil_wvalid_i,
  output logic         s_axil_wready_o,
  output logic [1:0]   s_axil_bresp_o,
  output logic         s_axil_bvalid_o,
  input  logic         s_axil_bready_i,
  input  logic [7:0]   s_axil_araddr_i,
  input  logic         s_axil_arvalid_i,
  output logic         s_axil_arready_o,
  output logic [31:0]  s_axil_rdata_o,
  output logic [1:0]   s_axil_rresp_o,
  output logic         s_axil_rvalid_o,
  input  logic         s_axil_rready_i,
  output logic         irq_o,
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
  output logic         m_axi_bready_o,
  input  logic         dma_axi_err_i,
  output logic         eng_rst_no,
  output logic         gemm_cmd_valid_o,
  input  logic         gemm_cmd_ready_i,
  input  logic         gemm_busy_i,
  output logic [19:0]  gemm_a_base_o,
  output logic [9:0]   gemm_a_stride_o,
  output logic [9:0]   gemm_kb_o,
  output logic [9:0]   gemm_m_o,
  output logic [19:0]  gemm_w_base_o,
  output logic [9:0]   gemm_w_stride_o,
  output logic [7:0]   gemm_n8_o,
  output logic [19:0]  gemm_c_base_o,
  output logic [1:0]   gemm_wmode_o,
  output logic         gemm_acc_o,
  output logic         vec_cmd_valid_o,
  input  logic         vec_cmd_ready_i,
  input  logic         vec_busy_i,
  output logic [4:0]   vec_op_o,
  output logic         vec_q_o,
  output logic [11:0]  vec_src_o,
  output logic [11:0]  vec_src2_o,
  output logic [12:0]  vec_dst_o,
  output logic [8:0]   vec_rows_q_o,
  output logic [7:0]   vec_n8_o,
  output logic [9:0]   vec_p_base_o,
  output logic [10:0]  vec_aux_o,
  output logic [31:0]  vec_imm_o,
  output logic         dma_cmd_valid_o,
  input  logic         dma_cmd_ready_i,
  input  logic         dma_busy_i,
  output logic         dma_st_o,
  output logic [31:0]  dma_ddr_addr_o,
  output logic [2:0]   dma_mem_o,
  output logic [19:0]  dma_oc_addr_o,
  output logic [11:0]  dma_rows_o,
  output logic [11:0]  dma_row_words_o,
  output logic [23:0]  dma_ddr_stride_o,
  output logic [1:0]   dma_mode_o,
  output logic [1:0]   dma_flags_o,
  output logic [6:0]   dma_tok_o,
  input  logic         gemm_stat_fed_i,
  input  logic         vec_stat_issue_i,
  input  logic         vec_stat_drain_i,
  input  logic         vec_stat_scalar_i,
  input  logic         vec_stat_pass_i,
  input  logic         dma_stat_rd_beat_i,
  input  logic         dma_stat_wr_beat_i,
  input  logic         dma_stat_denied_i,
  input  logic         dma_stat_null_i
);

  localparam logic [7:0] OpDmaLd    = 8'd1;
  localparam logic [7:0] OpDmaSt    = 8'd2;
  localparam logic [7:0] OpMove     = 8'd3;
  localparam logic [7:0] OpGemm     = 8'd4;
  localparam logic [7:0] OpVec      = 8'd5;
  localparam logic [7:0] OpSignal   = 8'd6;
  localparam logic [7:0] OpHalt     = 8'd7;
  localparam logic [7:0] OpWaitHost = 8'd8;

  logic        aw_v, w_v, start_req, soft_req;
  logic [7:0]  aw_a;
  logic [31:0] w_d;
  logic [31:0] instr_base, instr_count, irq_enable, irq_status;
  logic [3:0]  irq_set, irq_clr;
  logic        running, halted, error, ibuf_empty;
  logic [7:0]  sig_id;
  logic [15:0] sig_count;
  logic [31:0] sig_value, n_issued, cyc, cyc_gemm, cyc_vec, cyc_dma, cyc_stall, cyc_starve;
  logic [31:0] tr_base, tr_count, host_go;
  logic        tr_en, tr_stats, tr_idle, waiting_host, start_now;
  logic        wr_fire;

  assign s_axil_awready_o = !aw_v;
  assign s_axil_wready_o  = !w_v;
  assign s_axil_bresp_o   = 2'b00;
  assign wr_fire          = aw_v && w_v && !s_axil_bvalid_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      aw_v            <= 1'b0;
      w_v             <= 1'b0;
      s_axil_bvalid_o <= 1'b0;
      start_req       <= 1'b0;
      soft_req        <= 1'b0;
      instr_base      <= '0;
      instr_count     <= '0;
      irq_enable      <= '0;
      irq_clr         <= '0;
      tr_base         <= '0;
      tr_en           <= 1'b0;
      tr_stats        <= 1'b0;
      host_go         <= '0;
    end else begin
      start_req <= 1'b0;
      soft_req  <= 1'b0;
      irq_clr   <= '0;
      if (start_now) begin
        host_go <= '0;
      end
      if (s_axil_awvalid_i && s_axil_awready_o) begin
        aw_v <= 1'b1;
        aw_a <= s_axil_awaddr_i;
      end
      if (s_axil_wvalid_i && s_axil_wready_o) begin
        w_v <= 1'b1;
        w_d <= s_axil_wdata_i;
      end
      if (wr_fire) begin
        aw_v            <= 1'b0;
        w_v             <= 1'b0;
        s_axil_bvalid_o <= 1'b1;
        case (aw_a[7:2])
          6'h00: begin
            start_req <= w_d[0];
            soft_req  <= w_d[1];
          end
          6'h02:   instr_base <= w_d;
          6'h03:   instr_count <= w_d;
          6'h04:   irq_clr <= w_d[3:0];
          6'h05:   irq_enable <= w_d;
          6'h0F:   tr_base <= w_d;
          6'h10: begin
            tr_en    <= w_d[0];
            tr_stats <= w_d[2];
          end
          6'h12:   host_go <= w_d;
          default: begin
          end
        endcase
      end
      if (s_axil_bvalid_o && s_axil_bready_i) begin
        s_axil_bvalid_o <= 1'b0;
      end
    end
  end

  assign s_axil_arready_o = !s_axil_rvalid_o;
  assign s_axil_rresp_o   = 2'b00;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s_axil_rvalid_o <= 1'b0;
    end else begin
      if (s_axil_arvalid_i && s_axil_arready_o) begin
        s_axil_rvalid_o <= 1'b1;
        case (s_axil_araddr_i[7:2])
          6'h01: s_axil_rdata_o <= {24'd0, ibuf_empty, dma_busy_i, vec_busy_i, gemm_busy_i,
                                    waiting_host, error, halted, running};
          6'h02:   s_axil_rdata_o <= instr_base;
          6'h03:   s_axil_rdata_o <= instr_count;
          6'h04:   s_axil_rdata_o <= irq_status;
          6'h05:   s_axil_rdata_o <= irq_enable;
          6'h06:   s_axil_rdata_o <= {sig_count, 8'd0, sig_id};
          6'h07:   s_axil_rdata_o <= sig_value;
          6'h08:   s_axil_rdata_o <= n_issued;
          6'h09:   s_axil_rdata_o <= cyc;
          6'h0A:   s_axil_rdata_o <= cyc_gemm;
          6'h0B:   s_axil_rdata_o <= cyc_vec;
          6'h0C:   s_axil_rdata_o <= cyc_dma;
          6'h0D:   s_axil_rdata_o <= cyc_stall;
          6'h0E:   s_axil_rdata_o <= cyc_starve;
          6'h0F:   s_axil_rdata_o <= tr_base;
          6'h10:   s_axil_rdata_o <= {29'd0, tr_stats, tr_idle, tr_en};
          6'h11:   s_axil_rdata_o <= tr_count;
          6'h12:   s_axil_rdata_o <= host_go;
          default: s_axil_rdata_o <= 32'd0;
        endcase
      end else if (s_axil_rready_i) begin
        s_axil_rvalid_o <= 1'b0;
      end
    end
  end

  logic [2:0] soft_cnt;
  logic       srst;

  assign srst = (soft_cnt != 3'd0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      soft_cnt <= '0;
    end else if (soft_req) begin
      soft_cnt <= 3'd7;
    end else if (srst) begin
      soft_cnt <= soft_cnt - 1'b1;
    end
  end

  assign eng_rst_no = rst_ni;

  logic                       pend_start, fetch_arv;
  logic [31:0]                fetch_pc, fetch_left;
  logic [7:0]                 inflight;
  logic [4:0]                 ar_beats;
  logic                       ib_push, ib_pop, ib_clear, ib_fifo_empty, ib_afull, ins_v;
  logic [127:0]               ins_r;
  logic [127:0]               ib_dout;
  logic [$clog2(IbufDepth):0] ib_count;

  assign ar_beats        = (fetch_left >= 32'd16) ? 5'd16 : fetch_left[4:0];
  assign m_axi_araddr_o  = fetch_pc;
  assign m_axi_arlen_o   = 8'(ar_beats) - 8'd1;
  assign m_axi_arsize_o  = 3'b100;
  assign m_axi_arburst_o = 2'b01;
  assign m_axi_arvalid_o = fetch_arv;
  assign m_axi_rready_o  = 1'b1;
  assign start_now       = pend_start && (inflight == 8'd0) && !fetch_arv;
  assign ib_clear        = start_now || srst;
  assign ib_push         = m_axi_rvalid_i && !ib_clear;

  sfifo #(
    .Width      (128),
    .Depth      (IbufDepth),
    .AfullMargin(1)
  ) u_ib (
    .clk_i,
    .rst_ni,
    .clear_i(ib_clear),
    .push_i (ib_push),
    .din_i  (m_axi_rdata_i),
    .pop_i  (ib_pop),
    .dout_o (ib_dout),
    .empty_o(ib_fifo_empty),
    .afull_o(ib_afull),
    .count_o(ib_count)
  );

  logic fetch_credit;

  assign fetch_credit = (8'(ib_count) + inflight <= 8'(IbufDepth - 16));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      pend_start <= 1'b0;
      fetch_arv  <= 1'b0;
      fetch_pc   <= '0;
      fetch_left <= '0;
      inflight   <= '0;
    end else begin
      inflight <= inflight + ((fetch_arv && m_axi_arready_i) ? 8'(ar_beats) : 8'd0) -
                  8'(m_axi_rvalid_i);
      if (srst) begin
        pend_start <= 1'b0;
        fetch_left <= '0;
        if (m_axi_arready_i) begin
          fetch_arv <= 1'b0;
        end
      end else begin
        if (start_req) begin
          pend_start <= 1'b1;
        end
        if (start_now) begin
          pend_start <= 1'b0;
          fetch_pc   <= instr_base;
          fetch_left <= instr_count;
        end
        if (fetch_arv && m_axi_arready_i) begin
          fetch_arv  <= 1'b0;
          fetch_pc   <= fetch_pc + 32'({ar_beats, 4'b0});
          fetch_left <= fetch_left - 32'(ar_beats);
        end else if (!fetch_arv && running && fetch_left != 32'd0 && fetch_credit) begin
          fetch_arv <= 1'b1;
        end
      end
    end
  end

  logic [127:0] ins;
  logic [7:0]   opc;
  logic [3:0]   wt;
  logic         out_gemm, out_vec, out_dma, d_gemm, d_vec, d_dma;
  logic         wait_ok, eng_ok, issue, host_ok;

  assign ins        = ins_r;
  assign ibuf_empty = ib_fifo_empty && !ins_v;
  assign opc        = ins[127:120];
  assign wt         = ins[119:116];
  assign wait_ok    = !(wt[0] && out_dma) && !(wt[2] && out_gemm) && !(wt[3] && out_vec);

  always_comb begin
    case (opc)
      OpDmaLd, OpDmaSt: eng_ok = dma_cmd_ready_i && !out_dma;
      OpGemm:           eng_ok = gemm_cmd_ready_i && !out_gemm;
      OpVec:            eng_ok = vec_cmd_ready_i && !out_vec;
      OpWaitHost:       eng_ok = host_ok;
      default:          eng_ok = 1'b1;
    endcase
  end

  assign waiting_host = running && ins_v && (opc == OpWaitHost) && !host_ok;

  logic tr_ok, ins_load;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      host_ok <= 1'b0;
    end else begin
      host_ok <= !ins_load && !ib_clear && (host_go >= ins[31:0]);
    end
  end

  assign issue    = running && ins_v && wait_ok && eng_ok && !srst && tr_ok;
  assign ins_load = !ib_fifo_empty && (!ins_v || issue);
  assign ib_pop   = ins_load;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ins_v <= 1'b0;
    end else if (ib_clear) begin
      ins_v <= 1'b0;
    end else if (ins_load) begin
      ins_v <= 1'b1;
      ins_r <= ib_dout;
    end else if (issue) begin
      ins_v <= 1'b0;
    end
  end

  assign gemm_cmd_valid_o = issue && (opc == OpGemm);
  assign vec_cmd_valid_o  = issue && (opc == OpVec);
  assign dma_cmd_valid_o  = issue && (opc == OpDmaLd || opc == OpDmaSt);

  assign gemm_a_base_o   = ins[111:92];
  assign gemm_a_stride_o = ins[91:82];
  assign gemm_kb_o       = ins[81:72];
  assign gemm_m_o        = ins[71:62];
  assign gemm_w_base_o   = ins[61:42];
  assign gemm_w_stride_o = ins[41:32];
  assign gemm_n8_o       = ins[31:24];
  assign gemm_c_base_o   = ins[23:4];
  assign gemm_wmode_o    = ins[3:2];
  assign gemm_acc_o      = ins[1];

  assign vec_op_o     = ins[111:107];
  assign vec_src_o    = ins[106:95];
  assign vec_src2_o   = ins[94:83];
  assign vec_dst_o    = ins[82:70];
  assign vec_rows_q_o = ins[69:61];
  assign vec_n8_o     = ins[60:53];
  assign vec_p_base_o = ins[52:43];
  assign vec_aux_o    = ins[42:32];
  assign vec_imm_o    = ins[31:0];
  assign vec_q_o      = ins[112];

  assign dma_st_o         = (opc == OpDmaSt);
  assign dma_ddr_addr_o   = ins[111:80];
  assign dma_mem_o        = ins[79:77];
  assign dma_oc_addr_o    = ins[76:57];
  assign dma_rows_o       = ins[56:45];
  assign dma_row_words_o  = ins[44:33];
  assign dma_ddr_stride_o = ins[32:9];
  assign dma_mode_o       = ins[8:7];
  assign dma_flags_o      = ins[113:112];
  assign dma_tok_o        = ins[6:0];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || srst) begin
      out_gemm <= 1'b0;
      out_vec  <= 1'b0;
      out_dma  <= 1'b0;
      d_gemm   <= 1'b0;
      d_vec    <= 1'b0;
      d_dma    <= 1'b0;
    end else begin
      d_gemm <= gemm_cmd_valid_o;
      d_vec  <= vec_cmd_valid_o;
      d_dma  <= dma_cmd_valid_o;
      if (gemm_cmd_valid_o) begin
        out_gemm <= 1'b1;
      end else if (out_gemm && !d_gemm && !gemm_busy_i) begin
        out_gemm <= 1'b0;
      end
      if (vec_cmd_valid_o) begin
        out_vec <= 1'b1;
      end else if (out_vec && !d_vec && !vec_busy_i) begin
        out_vec <= 1'b0;
      end
      if (dma_cmd_valid_o) begin
        out_dma <= 1'b1;
      end else if (out_dma && !d_dma && !dma_busy_i) begin
        out_dma <= 1'b0;
      end
    end
  end

  localparam int SrcDma  = 0;
  localparam int SrcGemm = 1;
  localparam int SrcVec  = 2;
  localparam int SrcCtrl = 3;

  logic [31:0]  iss_cyc [3];
  logic [31:0]  iss_idx [3];
  logic [15:0]  iss_tag [3];
  logic         slot_v  [4];
  logic [127:0] slot_d  [4];
  logic [127:0] slot_s  [3];
  logic [23:0]  st_a    [3];
  logic [23:0]  st_b    [3];
  logic [23:0]  st_c    [3];
  logic [7:0]   st_d    [3];
  logic         st_in_a [3];
  logic         st_in_b [3];
  logic         st_in_c [3];
  logic         st_in_d [3];
  logic         st_clr  [3];
  logic         tr_pend_v;
  logic [127:0] tr_pend_d;
  logic [1:0]   src_sel;
  logic         tr_afull, tr_push;
  logic [127:0] tr_data;
  logic [15:0]  cur_tag;

  assign st_in_a[SrcDma]  = dma_stat_rd_beat_i;
  assign st_in_b[SrcDma]  = dma_stat_wr_beat_i;
  assign st_in_c[SrcDma]  = dma_stat_denied_i;
  assign st_in_d[SrcDma]  = dma_stat_null_i;
  assign st_in_a[SrcGemm] = gemm_stat_fed_i;
  assign st_in_b[SrcGemm] = 1'b0;
  assign st_in_c[SrcGemm] = 1'b0;
  assign st_in_d[SrcGemm] = 1'b0;
  assign st_in_a[SrcVec]  = vec_stat_issue_i;
  assign st_in_b[SrcVec]  = vec_stat_drain_i;
  assign st_in_c[SrcVec]  = vec_stat_scalar_i;
  assign st_in_d[SrcVec]  = vec_stat_pass_i;
  assign st_clr[SrcDma]   = dma_cmd_valid_o;
  assign st_clr[SrcGemm]  = gemm_cmd_valid_o;
  assign st_clr[SrcVec]   = vec_cmd_valid_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < 3; i++) begin
        st_a[i] <= '0;
        st_b[i] <= '0;
        st_c[i] <= '0;
        st_d[i] <= '0;
      end
    end else begin
      for (int i = 0; i < 3; i++) begin
        if (st_clr[i]) begin
          st_a[i] <= '0;
          st_b[i] <= '0;
          st_c[i] <= '0;
          st_d[i] <= '0;
        end else begin
          if (st_in_a[i] && st_a[i] != 24'hFFFFFF) begin
            st_a[i] <= st_a[i] + 24'd1;
          end
          if (st_in_b[i] && st_b[i] != 24'hFFFFFF) begin
            st_b[i] <= st_b[i] + 24'd1;
          end
          if (st_in_c[i] && st_c[i] != 24'hFFFFFF) begin
            st_c[i] <= st_c[i] + 24'd1;
          end
          if (st_in_d[i] && st_d[i] != 8'hFF) begin
            st_d[i] <= st_d[i] + 8'd1;
          end
        end
      end
    end
  end

  function automatic logic [127:0] stat_record(input logic [1:0] src, input logic [31:0] idx,
                                               input logic [23:0] a, input logic [23:0] b,
                                               input logic [23:0] c, input logic [7:0] d);
    return {d, 4'd1, 4'd0, c, b, a, 4'd0, 2'b00, src, idx};
  endfunction

  assign cur_tag = {ins[115:112], wt, opc};

  always_comb begin
    case (opc)
      OpDmaLd, OpDmaSt: src_sel = 2'(SrcDma);
      OpGemm:           src_sel = 2'(SrcGemm);
      OpVec:            src_sel = 2'(SrcVec);
      default:          src_sel = 2'(SrcCtrl);
    endcase
  end

  assign tr_ok = !tr_en || (!slot_v[src_sel] && !tr_afull);

  logic done_gemm, done_vec, done_dma, done_ctrl;

  assign done_gemm = out_gemm && !d_gemm && !gemm_busy_i && !gemm_cmd_valid_o;
  assign done_vec  = out_vec && !d_vec && !vec_busy_i && !vec_cmd_valid_o;
  assign done_dma  = out_dma && !d_dma && !dma_busy_i && !dma_cmd_valid_o;
  assign done_ctrl = issue && (src_sel == 2'(SrcCtrl));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < 4; i++) begin
        slot_v[i] <= 1'b0;
      end
      tr_push   <= 1'b0;
      tr_pend_v <= 1'b0;
    end else begin
      if (gemm_cmd_valid_o) begin
        iss_cyc[SrcGemm] <= cyc;
        iss_idx[SrcGemm] <= n_issued;
        iss_tag[SrcGemm] <= cur_tag;
      end
      if (vec_cmd_valid_o) begin
        iss_cyc[SrcVec] <= cyc;
        iss_idx[SrcVec] <= n_issued;
        iss_tag[SrcVec] <= cur_tag;
      end
      if (dma_cmd_valid_o) begin
        iss_cyc[SrcDma] <= cyc;
        iss_idx[SrcDma] <= n_issued;
        iss_tag[SrcDma] <= cur_tag;
      end
      if (tr_en && done_gemm) begin
        slot_v[SrcGemm] <= 1'b1;
        slot_d[SrcGemm] <= {12'd0, iss_tag[SrcGemm], 4'(SrcGemm), cyc, iss_cyc[SrcGemm],
                            iss_idx[SrcGemm]};
        slot_s[SrcGemm] <= stat_record(2'(SrcGemm), iss_idx[SrcGemm], st_a[SrcGemm],
                                       st_b[SrcGemm], st_c[SrcGemm], st_d[SrcGemm]);
      end
      if (tr_en && done_vec) begin
        slot_v[SrcVec] <= 1'b1;
        slot_d[SrcVec] <= {12'd0, iss_tag[SrcVec], 4'(SrcVec), cyc, iss_cyc[SrcVec],
                           iss_idx[SrcVec]};
        slot_s[SrcVec] <= stat_record(2'(SrcVec), iss_idx[SrcVec], st_a[SrcVec], st_b[SrcVec],
                                      st_c[SrcVec], st_d[SrcVec]);
      end
      if (tr_en && done_dma) begin
        slot_v[SrcDma] <= 1'b1;
        slot_d[SrcDma] <= {12'd0, iss_tag[SrcDma], 4'(SrcDma), cyc, iss_cyc[SrcDma],
                           iss_idx[SrcDma]};
        slot_s[SrcDma] <= stat_record(2'(SrcDma), iss_idx[SrcDma], st_a[SrcDma], st_b[SrcDma],
                                      st_c[SrcDma], st_d[SrcDma]);
      end
      if (tr_en && done_ctrl) begin
        slot_v[SrcCtrl] <= 1'b1;
        slot_d[SrcCtrl] <= {12'd0, cur_tag, 4'(SrcCtrl), cyc, cyc, n_issued};
      end
      tr_push <= 1'b0;
      if (tr_pend_v) begin
        tr_pend_v <= 1'b0;
        tr_push   <= 1'b1;
        tr_data   <= tr_pend_d;
      end else if (slot_v[SrcDma] && !(tr_en && done_dma)) begin
        slot_v[SrcDma] <= 1'b0;
        tr_push        <= 1'b1;
        tr_data        <= slot_d[SrcDma];
        tr_pend_v      <= tr_stats;
        tr_pend_d      <= slot_s[SrcDma];
      end else if (slot_v[SrcGemm] && !(tr_en && done_gemm)) begin
        slot_v[SrcGemm] <= 1'b0;
        tr_push         <= 1'b1;
        tr_data         <= slot_d[SrcGemm];
        tr_pend_v       <= tr_stats;
        tr_pend_d       <= slot_s[SrcGemm];
      end else if (slot_v[SrcVec] && !(tr_en && done_vec)) begin
        slot_v[SrcVec] <= 1'b0;
        tr_push        <= 1'b1;
        tr_data        <= slot_d[SrcVec];
        tr_pend_v      <= tr_stats;
        tr_pend_d      <= slot_s[SrcVec];
      end else if (slot_v[SrcCtrl] && !(tr_en && done_ctrl)) begin
        slot_v[SrcCtrl] <= 1'b0;
        tr_push         <= 1'b1;
        tr_data         <= slot_d[SrcCtrl];
      end
      if (srst) begin
        for (int i = 0; i < 4; i++) begin
          slot_v[i] <= 1'b0;
        end
        tr_pend_v <= 1'b0;
      end
    end
  end

  npu_trace u_trace (
    .clk_i,
    .rst_ni,
    .clear_i        (start_now || srst),
    .base_i         (tr_base),
    .ev_push_i      (tr_push),
    .ev_data_i      (tr_data),
    .ev_afull_o     (tr_afull),
    .count_o        (tr_count),
    .idle_o         (tr_idle),
    .m_axi_awaddr_o,
    .m_axi_awlen_o,
    .m_axi_awsize_o,
    .m_axi_awburst_o,
    .m_axi_awvalid_o,
    .m_axi_awready_i,
    .m_axi_wdata_o,
    .m_axi_wstrb_o,
    .m_axi_wlast_o,
    .m_axi_wvalid_o,
    .m_axi_wready_i,
    .m_axi_bresp_i,
    .m_axi_bvalid_i,
    .m_axi_bready_o
  );

  always_comb begin
    irq_set = '0;
    if (issue && opc == OpSignal) begin
      irq_set[0] = 1'b1;
    end
    if (issue && opc == OpHalt) begin
      irq_set[1] = 1'b1;
    end
    if (issue && (opc == OpMove || opc > OpWaitHost)) begin
      irq_set[2] = 1'b1;
    end
    if (dma_axi_err_i || (m_axi_rvalid_i && m_axi_rresp_i[1]) ||
        (m_axi_bvalid_i && m_axi_bresp_i[1])) begin
      irq_set[3] = 1'b1;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      running    <= 1'b0;
      halted     <= 1'b0;
      error      <= 1'b0;
      irq_status <= '0;
      sig_id     <= '0;
      sig_count  <= '0;
      sig_value  <= '0;
      n_issued   <= '0;
      cyc        <= '0;
      cyc_gemm   <= '0;
      cyc_vec    <= '0;
      cyc_dma    <= '0;
      cyc_stall  <= '0;
      cyc_starve <= '0;
    end else begin
      irq_status <= (irq_status & ~32'(irq_clr)) | 32'(irq_set);
      if (srst) begin
        running <= 1'b0;
        halted  <= 1'b0;
        error   <= 1'b0;
      end else if (start_now) begin
        running    <= 1'b1;
        halted     <= 1'b0;
        error      <= 1'b0;
        n_issued   <= '0;
        cyc        <= '0;
        cyc_gemm   <= '0;
        cyc_vec    <= '0;
        cyc_dma    <= '0;
        cyc_stall  <= '0;
        cyc_starve <= '0;
      end else if (running) begin
        cyc <= cyc + 1'b1;
        if (gemm_busy_i) begin
          cyc_gemm <= cyc_gemm + 1'b1;
        end
        if (vec_busy_i) begin
          cyc_vec <= cyc_vec + 1'b1;
        end
        if (dma_busy_i) begin
          cyc_dma <= cyc_dma + 1'b1;
        end
        if (ins_v && !issue) begin
          cyc_stall <= cyc_stall + 1'b1;
        end
        if (!ins_v) begin
          cyc_starve <= cyc_starve + 1'b1;
        end
        if (issue) begin
          n_issued <= n_issued + 1'b1;
          if (opc == OpSignal) begin
            sig_id    <= ins[111:104];
            sig_value <= ins[31:0];
            sig_count <= sig_count + 1'b1;
          end
          if (opc == OpHalt) begin
            running <= 1'b0;
            halted  <= 1'b1;
          end
          if (irq_set[2]) begin
            running <= 1'b0;
            halted  <= 1'b1;
            error   <= 1'b1;
          end
        end
      end
    end
  end

  assign irq_o = |(irq_status & 32'(irq_enable[3:0]));

  logic unused_seq;

  assign unused_seq = ^{s_axil_wstrb_i, s_axil_araddr_i[1:0], aw_a[1:0], m_axi_rresp_i[0],
                        m_axi_rlast_i, m_axi_bresp_i[0], ib_afull, irq_enable[31:4]};

endmodule
