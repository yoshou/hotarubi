module npu_seq
  import isa_pkg::*;
#(
  parameter int IbufDepth = 64
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic [RegAddrBits-1:0] s_axil_awaddr_i,
  input  logic         s_axil_awvalid_i,
  output logic         s_axil_awready_o,
  input  logic [31:0]  s_axil_wdata_i,
  input  logic [3:0]   s_axil_wstrb_i,
  input  logic         s_axil_wvalid_i,
  output logic         s_axil_wready_o,
  output logic [1:0]   s_axil_bresp_o,
  output logic         s_axil_bvalid_o,
  input  logic         s_axil_bready_i,
  input  logic [RegAddrBits-1:0] s_axil_araddr_i,
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
  output logic         gemm_cmd_valid_o,
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
  input  logic         vec_busy_i,
  output logic [7:0]   vec_entry_o,
  output logic         vec_q_o,
  output logic [11:0]  vec_src_o,
  output logic [11:0]  vec_src2_o,
  output logic [12:0]  vec_dst_o,
  output logic [8:0]   vec_rows_q_o,
  output logic [7:0]   vec_n8_o,
  output logic [8:0]   vec_p_base_o,
  output logic [10:0]  vec_aux_o,
  output logic [31:0]  vec_imm_o,
  output logic         dma_cmd_valid_o,
  input  logic         dma_busy_i,
  output logic         dma_st_o,
  output logic [27:0]  dma_ddr_addr_o,
  output logic [2:0]   dma_mem_o,
  output logic [12:0]  dma_oc_addr_o,
  output logic [11:0]  dma_rows_o,
  output logic [11:0]  dma_row_words_o,
  output logic [19:0]  dma_ddr_stride_o,
  output logic [2:0]   dma_kind_o,
  output logic [5:0]   dma_tok_o,
  output logic         dma_acc_o,
  output logic         dma_flush_o,
  input  logic         gemm_stat_fed_i,
  input  logic         vec_stat_issue_i,
  input  logic         vec_stat_drain_i,
  input  logic         vec_stat_scalar_i,
  input  logic         vec_stat_pass_i,
  input  logic         dma_stat_rd_beat_i,
  input  logic         dma_stat_wr_beat_i,
  input  logic         dma_stat_denied_i,
  input  logic [1:0]   dma_stat_null_i
);

  localparam int TraceSourceBits = TraceSourceMsb - TraceSourceLsb + 1;
  localparam int OpcodeBits      = OpcodeMsb - OpcodeLsb + 1;
  localparam int WaitBits        = WaitMsb - WaitLsb + 1;
  localparam int StatABits       = TraceStatAMsb - TraceStatALsb + 1;
  localparam int StatBBits       = TraceStatBMsb - TraceStatBLsb + 1;
  localparam int StatCBits       = TraceStatCMsb - TraceStatCLsb + 1;
  localparam int StatDBits       = TraceStatDMsb - TraceStatDLsb + 1;

  function automatic logic legal_op(input logic [7:0] op);
    return op inside {OpNop, OpDmaLoad, OpDmaStore, OpGemm, OpVec, OpSignal, OpHalt, OpWaitHost};
  endfunction

  logic        aw_v, w_v, start_req, soft_req;
  logic [RegAddrBits-1:0] aw_a;
  logic [31:0] w_d;
  logic [31:0] instr_base, instr_count, irq_enable, irq_status;
  logic [IrqBits-1:0] irq_set, irq_clr;
  logic        running, halted, error, ibuf_empty;
  logic [7:0]  sig_id;
  logic [15:0] sig_count;
  logic [31:0] sig_value, n_issued, cyc, cyc_stall, cyc_starve;
  logic [31:0] cyc_busy [Engines];
  logic        eng_busy [Engines];
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
        case (aw_a[RegAddrBits-1:RegByteBits])
          RegCtrl: begin
            start_req <= w_d[CtrlStart];
            soft_req  <= w_d[CtrlSoftReset];
          end
          RegInstrBase:  instr_base <= w_d;
          RegInstrCount: instr_count <= w_d;
          RegIrqStatus:  irq_clr <= w_d[IrqBits-1:0];
          RegIrqEnable:  irq_enable <= w_d;
          RegTraceBase:  tr_base <= w_d;
          RegTraceCtrl: begin
            tr_en    <= w_d[TraceCtrlEnable];
            tr_stats <= w_d[TraceCtrlStatistics];
          end
          RegHostGo:     host_go <= w_d;
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

  logic [31:0] status_word, signal_word, trace_ctrl_word;

  always_comb begin
    status_word                    = '0;
    status_word[StatusRunning]     = running;
    status_word[StatusHalted]      = halted;
    status_word[StatusError]       = error;
    status_word[StatusWaitingHost] = waiting_host;
    for (int e = 0; e < Engines; e++) begin
      status_word[StatusBusyLsb+e] = eng_busy[e];
    end
    status_word[StatusIbufEmpty] = ibuf_empty;
    signal_word                                      = '0;
    signal_word[SignalRegCountMsb:SignalRegCountLsb] = sig_count;
    signal_word[SignalRegIdMsb:SignalRegIdLsb]       = sig_id;
    trace_ctrl_word                      = '0;
    trace_ctrl_word[TraceCtrlEnable]     = tr_en;
    trace_ctrl_word[TraceCtrlIdle]       = tr_idle;
    trace_ctrl_word[TraceCtrlStatistics] = tr_stats;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      s_axil_rvalid_o <= 1'b0;
    end else begin
      if (s_axil_arvalid_i && s_axil_arready_o) begin
        s_axil_rvalid_o <= 1'b1;
        case (s_axil_araddr_i[RegAddrBits-1:RegByteBits])
          RegStatus:       s_axil_rdata_o <= status_word;
          RegInstrBase:    s_axil_rdata_o <= instr_base;
          RegInstrCount:   s_axil_rdata_o <= instr_count;
          RegIrqStatus:    s_axil_rdata_o <= irq_status;
          RegIrqEnable:    s_axil_rdata_o <= irq_enable;
          RegSignal:       s_axil_rdata_o <= signal_word;
          RegSignalValue:  s_axil_rdata_o <= sig_value;
          RegIssued:       s_axil_rdata_o <= n_issued;
          RegCycles:       s_axil_rdata_o <= cyc;
          RegDmaCycles:    s_axil_rdata_o <= cyc_busy[EngDma];
          RegGemmCycles:   s_axil_rdata_o <= cyc_busy[EngGemm];
          RegVecCycles:    s_axil_rdata_o <= cyc_busy[EngVec];
          RegStallCycles:  s_axil_rdata_o <= cyc_stall;
          RegStarveCycles: s_axil_rdata_o <= cyc_starve;
          RegTraceBase:    s_axil_rdata_o <= tr_base;
          RegTraceCtrl:    s_axil_rdata_o <= trace_ctrl_word;
          RegTraceCount:   s_axil_rdata_o <= tr_count;
          RegHostGo:       s_axil_rdata_o <= host_go;
          default:         s_axil_rdata_o <= 32'd0;
        endcase
      end else if (s_axil_rready_i) begin
        s_axil_rvalid_o <= 1'b0;
      end
    end
  end

  logic srst;

  assign srst = soft_req;

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
  assign ib_push         = m_axi_rvalid_i && running && !ib_clear;

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
      end else if (start_now) begin
        pend_start <= 1'b0;
        fetch_pc   <= instr_base;
        fetch_left <= instr_count;
      end else if (start_req) begin
        pend_start <= 1'b1;
      end
      if (fetch_arv && m_axi_arready_i) begin
        fetch_arv  <= 1'b0;
        fetch_pc   <= fetch_pc + 32'({ar_beats, 4'b0});
        fetch_left <= fetch_left - 32'(ar_beats);
      end else if (!fetch_arv && running && !srst && fetch_left != 32'd0 && fetch_credit) begin
        fetch_arv <= 1'b1;
      end
    end
  end

  logic [127:0]       ins;
  logic [7:0]         opc;
  logic [3:0]         wt;
  logic [Engines-1:0] out_eng, d_eng, done_eng, eng_cmd, eng_sel;
  logic               wait_ok, eng_ok, issue, host_ok;

  assign ins        = ins_r;
  assign ibuf_empty = ib_fifo_empty && !ins_v;
  assign opc        = ins[OpcodeMsb:OpcodeLsb];
  assign wt         = ins[WaitMsb:WaitLsb];
  assign eng_busy   = '{dma_busy_i, gemm_busy_i, vec_busy_i};

  always_comb begin
    eng_sel          = '0;
    eng_sel[EngDma]  = (opc == OpDmaLoad) || (opc == OpDmaStore);
    eng_sel[EngGemm] = (opc == OpGemm);
    eng_sel[EngVec]  = (opc == OpVec);
    wait_ok          = 1'b1;
    eng_ok           = (opc != OpWaitHost) || host_ok;
    for (int e = 0; e < Engines; e++) begin
      if (wt[e] && out_eng[e]) begin
        wait_ok = 1'b0;
      end
      if (eng_sel[e] && (eng_busy[e] || out_eng[e])) begin
        eng_ok = 1'b0;
      end
    end
  end

  assign waiting_host = running && ins_v && (opc == OpWaitHost) && !host_ok;

  logic tr_ok, ins_load;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      host_ok <= 1'b0;
    end else begin
      host_ok <= !ins_load && !ib_clear && (host_go >= ins[WaitHostValueMsb:WaitHostValueLsb]);
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

  assign eng_cmd          = issue ? eng_sel : '0;
  assign dma_cmd_valid_o  = eng_cmd[EngDma];
  assign gemm_cmd_valid_o = eng_cmd[EngGemm];
  assign vec_cmd_valid_o  = eng_cmd[EngVec];

  assign gemm_a_base_o   = ins[GemmABaseMsb:GemmABaseLsb];
  assign gemm_a_stride_o = ins[GemmAStrideMsb:GemmAStrideLsb];
  assign gemm_kb_o       = ins[GemmKbMsb:GemmKbLsb];
  assign gemm_m_o        = ins[GemmMMsb:GemmMLsb];
  assign gemm_w_base_o   = ins[GemmWBaseMsb:GemmWBaseLsb];
  assign gemm_w_stride_o = ins[GemmWStrideMsb:GemmWStrideLsb];
  assign gemm_n8_o       = ins[GemmN8Msb:GemmN8Lsb];
  assign gemm_c_base_o   = ins[GemmCBaseMsb:GemmCBaseLsb];
  assign gemm_wmode_o    = ins[GemmWmodeMsb:GemmWmodeLsb];
  assign gemm_acc_o      = ins[GemmAccLsb];

  assign vec_entry_o  = ins[VecEntryMsb:VecEntryLsb];
  assign vec_q_o      = ins[VecQLsb];
  assign vec_src_o    = ins[VecSrcMsb:VecSrcLsb];
  assign vec_src2_o   = ins[VecSrc2Msb:VecSrc2Lsb];
  assign vec_dst_o    = ins[VecDstMsb:VecDstLsb];
  assign vec_rows_q_o = ins[VecRowsQMsb:VecRowsQLsb];
  assign vec_n8_o     = ins[VecN8Msb:VecN8Lsb];
  assign vec_p_base_o = ins[VecPBaseMsb:VecPBaseLsb];
  assign vec_aux_o    = ins[VecAuxMsb:VecAuxLsb];
  assign vec_imm_o    = ins[VecImmMsb:VecImmLsb];

  assign dma_st_o         = (opc == OpDmaStore);
  assign dma_ddr_addr_o   = ins[DmaDdrAddrMsb:DmaDdrAddrLsb];
  assign dma_mem_o        = ins[DmaMemMsb:DmaMemLsb];
  assign dma_oc_addr_o    = ins[DmaAddrMsb:DmaAddrLsb];
  assign dma_rows_o       = ins[DmaRowsMsb:DmaRowsLsb];
  assign dma_row_words_o  = ins[DmaRowWordsMsb:DmaRowWordsLsb];
  assign dma_ddr_stride_o = ins[DmaDdrStrideMsb:DmaDdrStrideLsb];
  assign dma_kind_o       = ins[DmaKindMsb:DmaKindLsb];
  assign dma_tok_o        = ins[DmaTokMsb:DmaTokLsb];
  assign dma_acc_o        = ins[DmaAccLsb];
  assign dma_flush_o      = ins[DmaFlushLsb];

  always_comb begin
    for (int e = 0; e < Engines; e++) begin
      done_eng[e] = out_eng[e] && !d_eng[e] && !eng_busy[e];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || srst) begin
      out_eng <= '0;
      d_eng   <= '0;
    end else begin
      d_eng <= eng_cmd;
      for (int e = 0; e < Engines; e++) begin
        if (eng_cmd[e]) begin
          out_eng[e] <= 1'b1;
        end else if (done_eng[e]) begin
          out_eng[e] <= 1'b0;
        end
      end
    end
  end

  logic [31:0]  iss_cyc [Engines];
  logic [31:0]  iss_idx [Engines];
  logic [WaitBits+OpcodeBits-1:0] iss_tag [Engines];
  logic         slot_v  [Engines + 1];
  logic [31:0]  slot_c  [Engines];
  logic [TraceRecordBits-1:0] slot_s [Engines];
  logic [TraceRecordBits-1:0] ctrl_d;
  logic [StatABits-1:0] st_a [Engines];
  logic [StatBBits-1:0] st_b [Engines];
  logic [StatCBits-1:0] st_c [Engines];
  logic [StatDBits-1:0] st_d [Engines];
  logic         st_in_a [Engines];
  logic         st_in_b [Engines];
  logic         st_in_c [Engines];
  logic [1:0]   st_in_d [Engines];
  logic         tr_pend_v;
  logic [TraceRecordBits-1:0] tr_pend_d;
  logic [1:0]   src_sel;
  logic         tr_afull, tr_push;
  logic [TraceRecordBits-1:0] tr_data;
  logic [WaitBits+OpcodeBits-1:0] cur_tag;

  assign st_in_a = '{dma_stat_rd_beat_i, gemm_stat_fed_i, vec_stat_issue_i};
  assign st_in_b = '{dma_stat_wr_beat_i, 1'b0, vec_stat_drain_i};
  assign st_in_c = '{dma_stat_denied_i, 1'b0, vec_stat_scalar_i};
  assign st_in_d = '{dma_stat_null_i, 2'd0, {1'b0, vec_stat_pass_i}};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < Engines; i++) begin
        st_a[i] <= '0;
        st_b[i] <= '0;
        st_c[i] <= '0;
        st_d[i] <= '0;
      end
    end else begin
      for (int i = 0; i < Engines; i++) begin
        if (eng_cmd[i]) begin
          st_a[i] <= '0;
          st_b[i] <= '0;
          st_c[i] <= '0;
          st_d[i] <= '0;
        end else begin
          if (st_in_a[i] && st_a[i] != {StatABits{1'b1}}) begin
            st_a[i] <= st_a[i] + StatABits'(1);
          end
          if (st_in_b[i] && st_b[i] != {StatBBits{1'b1}}) begin
            st_b[i] <= st_b[i] + StatBBits'(1);
          end
          if (st_in_c[i] && st_c[i] != {StatCBits{1'b1}}) begin
            st_c[i] <= st_c[i] + StatCBits'(1);
          end
          if (st_in_d[i] != 2'd0) begin
            st_d[i] <= (st_d[i] > {StatDBits{1'b1}} - StatDBits'(st_in_d[i])) ?
                       {StatDBits{1'b1}} : st_d[i] + StatDBits'(st_in_d[i]);
          end
        end
      end
    end
  end

  function automatic logic [TraceRecordBits-1:0] done_record(
      input logic [WaitBits+OpcodeBits-1:0] tag, input logic [1:0] src, input logic [31:0] done,
      input logic [31:0] iss, input logic [31:0] idx);
    logic [TraceRecordBits-1:0] r;
    r                                        = '0;
    r[TraceIndexMsb:TraceIndexLsb]           = idx;
    r[TraceIssueMsb:TraceIssueLsb]           = iss;
    r[TraceCompletionMsb:TraceCompletionLsb] = done;
    r[TraceSourceMsb:TraceSourceLsb]         = TraceSourceBits'(src);
    r[TraceOpcodeMsb:TraceOpcodeLsb]         = tag[OpcodeBits-1:0];
    r[TraceWaitMsb:TraceWaitLsb]             = tag[OpcodeBits+:WaitBits];
    r[TraceTypeMsb:TraceTypeLsb]             = TraceTypeCompletion;
    return r;
  endfunction

  function automatic logic [TraceRecordBits-1:0] stat_record(
      input logic [1:0] src, input logic [31:0] idx, input logic [StatABits-1:0] a,
      input logic [StatBBits-1:0] b, input logic [StatCBits-1:0] c, input logic [StatDBits-1:0] d);
    logic [TraceRecordBits-1:0] r;
    r                                        = '0;
    r[TraceIndexMsb:TraceIndexLsb]           = idx;
    r[TraceStatSourceMsb:TraceStatSourceLsb] = TraceSourceBits'(src);
    r[TraceStatAMsb:TraceStatALsb]           = a;
    r[TraceStatBMsb:TraceStatBLsb]           = b;
    r[TraceStatCMsb:TraceStatCLsb]           = c;
    r[TraceStatDMsb:TraceStatDLsb]           = d;
    r[TraceTypeMsb:TraceTypeLsb]             = TraceTypeStatistics;
    return r;
  endfunction

  assign cur_tag = {wt, opc};

  always_comb begin
    src_sel = 2'(TraceSourceSequencer);
    for (int e = Engines - 1; e >= 0; e--) begin
      if (eng_sel[e]) begin
        src_sel = 2'(e);
      end
    end
  end

  assign tr_ok = !tr_en || (!slot_v[src_sel] && !tr_afull);

  logic       done_ctrl, pop_v, pop_ctrl;
  logic [1:0] pop_src;

  assign done_ctrl = issue && (src_sel == 2'(TraceSourceSequencer));

  always_comb begin
    pop_v   = 1'b0;
    pop_src = 2'(TraceSourceSequencer);
    for (int i = Engines; i >= 0; i--) begin
      if (slot_v[i]) begin
        pop_v   = 1'b1;
        pop_src = 2'(i);
      end
    end
  end

  assign pop_ctrl = (pop_src == 2'(TraceSourceSequencer));

  always_ff @(posedge clk_i) begin
    for (int e = 0; e < Engines; e++) begin
      if (eng_cmd[e]) begin
        iss_cyc[e] <= cyc;
        iss_idx[e] <= n_issued;
        iss_tag[e] <= cur_tag;
      end
      if (tr_en && done_eng[e]) begin
        slot_c[e] <= cyc;
        slot_s[e] <= stat_record(2'(e), iss_idx[e], st_a[e], st_b[e], st_c[e], st_d[e]);
      end
    end
    if (tr_en && done_ctrl) begin
      ctrl_d <= done_record(cur_tag, 2'(TraceSourceSequencer), cyc, cyc, n_issued);
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i <= Engines; i++) begin
        slot_v[i] <= 1'b0;
      end
      tr_push   <= 1'b0;
      tr_pend_v <= 1'b0;
    end else begin
      for (int e = 0; e < Engines; e++) begin
        if (tr_en && done_eng[e]) begin
          slot_v[e] <= 1'b1;
        end
      end
      if (tr_en && done_ctrl) begin
        slot_v[TraceSourceSequencer] <= 1'b1;
      end
      tr_push <= 1'b0;
      if (tr_pend_v) begin
        tr_pend_v <= 1'b0;
        tr_push   <= 1'b1;
        tr_data   <= tr_pend_d;
      end else if (pop_v) begin
        slot_v[pop_src] <= 1'b0;
        tr_push         <= 1'b1;
        tr_data         <= pop_ctrl ? ctrl_d :
                           done_record(iss_tag[pop_src], pop_src, slot_c[pop_src],
                                       iss_cyc[pop_src], iss_idx[pop_src]);
        tr_pend_v       <= !pop_ctrl && tr_stats;
        tr_pend_d       <= slot_s[pop_src];
      end
      if (srst) begin
        for (int i = 0; i <= Engines; i++) begin
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
      irq_set[IrqSignal] = 1'b1;
    end
    if (issue && opc == OpHalt) begin
      irq_set[IrqHalt] = 1'b1;
    end
    if (issue && !legal_op(opc)) begin
      irq_set[IrqIllegal] = 1'b1;
    end
    if (dma_axi_err_i || (m_axi_rvalid_i && m_axi_rresp_i[1]) ||
        (m_axi_bvalid_i && m_axi_bresp_i[1])) begin
      irq_set[IrqAxiError] = 1'b1;
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
      cyc_stall  <= '0;
      cyc_starve <= '0;
      for (int e = 0; e < Engines; e++) begin
        cyc_busy[e] <= '0;
      end
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
        cyc_stall  <= '0;
        cyc_starve <= '0;
        for (int e = 0; e < Engines; e++) begin
          cyc_busy[e] <= '0;
        end
      end else if (running) begin
        cyc <= cyc + 1'b1;
        for (int e = 0; e < Engines; e++) begin
          if (eng_busy[e]) begin
            cyc_busy[e] <= cyc_busy[e] + 1'b1;
          end
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
            sig_id    <= ins[SignalIdMsb:SignalIdLsb];
            sig_value <= ins[SignalValueMsb:SignalValueLsb];
            sig_count <= sig_count + 1'b1;
          end
          if (opc == OpHalt) begin
            running <= 1'b0;
            halted  <= 1'b1;
          end
          if (irq_set[IrqIllegal]) begin
            running <= 1'b0;
            halted  <= 1'b1;
            error   <= 1'b1;
          end
        end
      end
    end
  end

  assign irq_o = |(irq_status & 32'(irq_enable[IrqBits-1:0]));

  logic unused_seq;

  assign unused_seq = ^{s_axil_wstrb_i, s_axil_araddr_i[RegByteBits-1:0], aw_a[RegByteBits-1:0],
                        m_axi_rresp_i[0], m_axi_rlast_i, m_axi_bresp_i[0], ib_afull,
                        irq_enable[31:IrqBits]};

endmodule
