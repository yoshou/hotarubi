module dma_engine (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          cmd_valid_i,
  output logic          cmd_ready_o,
  input  logic          cmd_st_i,
  input  logic [31:0]   cmd_ddr_addr_i,
  input  logic [2:0]    cmd_mem_i,
  input  logic [19:0]   cmd_oc_addr_i,
  input  logic [11:0]   cmd_rows_i,
  input  logic [11:0]   cmd_row_words_i,
  input  logic [23:0]   cmd_ddr_stride_i,
  input  logic [1:0]    cmd_mode_i,
  input  logic [1:0]    cmd_flags_i,
  input  logic [6:0]    cmd_tok_i,
  output logic          busy_o,
  output logic [31:0]   m_axi_araddr_o,
  output logic [7:0]    m_axi_arlen_o,
  output logic [2:0]    m_axi_arsize_o,
  output logic [1:0]    m_axi_arburst_o,
  output logic          m_axi_arvalid_o,
  input  logic          m_axi_arready_i,
  input  logic [127:0]  m_axi_rdata_i,
  input  logic [1:0]    m_axi_rresp_i,
  input  logic          m_axi_rlast_i,
  input  logic          m_axi_rvalid_i,
  output logic          m_axi_rready_o,
  output logic [31:0]   m_axi_awaddr_o,
  output logic [7:0]    m_axi_awlen_o,
  output logic [2:0]    m_axi_awsize_o,
  output logic [1:0]    m_axi_awburst_o,
  output logic          m_axi_awvalid_o,
  input  logic          m_axi_awready_i,
  output logic [127:0]  m_axi_wdata_o,
  output logic [15:0]   m_axi_wstrb_o,
  output logic          m_axi_wlast_o,
  output logic          m_axi_wvalid_o,
  input  logic          m_axi_wready_i,
  input  logic [1:0]    m_axi_bresp_i,
  input  logic          m_axi_bvalid_i,
  output logic          m_axi_bready_o,
  output logic          wr_req_o,
  input  logic          wr_grant_i,
  output logic [2:0]    wr_mem_o,
  output logic [19:0]   wr_addr_o,
  output logic [1151:0] wr_data_o,
  output logic          rd_req_o,
  input  logic          rd_grant_i,
  output logic [2:0]    rd_mem_o,
  output logic [19:0]   rd_addr_o,
  input  logic [1055:0] rd_data_i,
  output logic          stat_rd_beat_o,
  output logic          stat_wr_beat_o,
  output logic          stat_denied_o,
  output logic [1:0]    stat_null_o
);

  localparam int RfDepth = 128;
  localparam int WbDepth = 64;
  localparam int IbDepth = 64;
  localparam int UfDepth = 64;
  localparam int WqDepth = 64;

  localparam logic [1:0] ModeRaw    = 2'd0;
  localparam logic [1:0] ModeSplit8 = 2'd1;
  localparam logic [1:0] ModeT32    = 2'd2;
  localparam logic [1:0] ModeGacc   = 2'd3;

  typedef enum logic [1:0] {
    SIdle,
    SSetup1,
    SSetup2,
    SRun
  } state_e;

  state_e      st;
  logic        c_st, c_idx, c_row, c_abuf, c_gacc;
  logic [1:0]  c_gflags;
  logic [31:0] c_ddr;
  logic [2:0]  c_mem;
  logic [19:0] c_oc;
  logic [11:0] c_rows, c_w;
  logic [23:0] c_stride;
  logic [1:0]  c_mode;
  logic [5:0]  c_tok;
  logic [3:0]  bpw;
  logic [7:0]  ubeats;
  logic [6:0]  nwq;
  logic [11:0] n_out, n_w;
  logic [3:0]  n_in;
  logic [26:0] stride_out;
  logic [15:0] row_beats;
  (* use_dsp = "yes" *) logic [23:0] rows_w;
  logic [23:0] oc_words;
  logic [19:0] n_half, kbt, kbt2, kbt3, kbt4;
  logic [13:0] list_beats;

  assign cmd_ready_o = (st == SIdle);
  assign busy_o      = (st != SIdle);

  logic [1:0] ga_null;

  assign stat_rd_beat_o = m_axi_rvalid_i;
  assign stat_wr_beat_o = m_axi_wvalid_o && m_axi_wready_i;
  assign stat_denied_o  = (rd_req_o && !rd_grant_i) || (wr_req_o && !wr_grant_i);
  assign stat_null_o    = ga_null + {1'b0, gi_step && gi_null};

  always_comb begin
    case (c_mem)
      3'd0, 3'd1: bpw = 4'd9;
      3'd4, 3'd5: bpw = 4'd1;
      default:    bpw = 4'd8;
    endcase
  end

  logic [1:0] nsc;
  logic [7:0] tbeats;

  assign nsc    = c_abuf ? ((c_tok > 6'd16) ? 2'd2 : 2'd1) : 2'd0;
  assign tbeats = (c_abuf ? 8'({c_tok, 1'b0}) : 8'({c_tok, 3'b000})) + 8'(nsc);

  logic          ld_done, st_done;
  logic          bout_fire;
  logic          ga_ent_valid, ga_ent_ready, ga_req_valid, ga_req_ready, ga_row_ready, ga_w_ready;
  logic          ga_out_valid, ga_out_ready, ga_idle, ga_start;
  logic [31:0]   ga_req_addr;
  logic [15:0]   ga_req_beats;
  logic [1:0]    ga_req_tag;
  logic [1023:0] ga_out_word;
  logic [12:0]   ga_out_addr;
  logic          wq_push, wq_pop, wq_empty;
  logic [127:0]  wq_dout;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st <= SIdle;
    end else begin
      case (st)
        SIdle: begin
          if (cmd_valid_i) begin
            st       <= SSetup1;
            c_st     <= cmd_st_i;
            c_ddr    <= cmd_ddr_addr_i;
            c_mem    <= cmd_mem_i;
            c_oc     <= cmd_oc_addr_i;
            c_rows   <= cmd_rows_i;
            c_w      <= cmd_row_words_i;
            c_stride <= cmd_ddr_stride_i;
            c_mode   <= cmd_st_i ? ModeRaw : cmd_mode_i;
            c_idx    <= cmd_flags_i[0] && cmd_flags_i[1] && !cmd_st_i && (cmd_mode_i == ModeRaw);
            c_row    <= cmd_flags_i[1] && (cmd_st_i || cmd_mode_i == ModeRaw);
            c_abuf   <= (cmd_mem_i == 3'd0);
            c_gacc   <= !cmd_st_i && (cmd_mode_i == ModeGacc);
            c_gflags <= cmd_flags_i;
            c_tok    <= (cmd_tok_i == 7'd0) ? 6'd1 : (cmd_tok_i > 7'd32) ? 6'd32 : cmd_tok_i[5:0];
          end
        end
        SSetup1: begin
          st     <= SSetup2;
          rows_w <= c_rows * c_w;
          ubeats <= c_row ? tbeats : 8'(bpw);
          nwq    <= c_abuf ? 7'(c_tok) : 7'({c_tok, 2'b00});
          kbt    <= 20'(c_rows[11:3]);
          case (c_mode)
            ModeSplit8: begin
              n_out      <= c_rows >> 1;
              n_w        <= c_w;
              n_in       <= 4'd2;
              stride_out <= {2'b0, c_stride, 1'b0};
            end
            ModeT32: begin
              n_out      <= c_rows >> 3;
              n_w        <= c_w;
              n_in       <= 4'd8;
              stride_out <= {c_stride, 3'b0};
            end
            default: begin
              n_out      <= c_rows;
              n_w        <= 12'd1;
              n_in       <= 4'd1;
              stride_out <= {3'b0, c_stride};
            end
          endcase
        end
        SSetup2: begin
          st         <= SRun;
          n_half     <= 20'(rows_w >> 1);
          kbt2       <= kbt << 1;
          kbt3       <= kbt + (kbt << 1);
          kbt4       <= kbt << 2;
          row_beats  <= 16'(c_w) * 16'(ubeats);
          oc_words   <= c_row ? 24'((rows_w >> 2) * 24'(nwq)) : rows_w;
          list_beats <= 14'((rows_w + 24'd3) >> 2);
        end
        SRun: begin
          if (c_st ? st_done : ld_done) begin
            st <= SIdle;
          end
        end
      endcase
    end
  end

  logic run, setup_pulse;

  assign run         = (st == SRun);
  assign setup_pulse = (st == SSetup2);

  logic        gen_done, g_valid, g_ready;
  logic [11:0] g_out, g_w;
  logic [3:0]  g_in;
  logic [31:0] g_pa, g_wa, g_ia;
  logic [15:0] g_beats;

  assign g_valid = run && !gen_done && !c_idx && !c_gacc;
  assign g_beats = (c_mode == ModeRaw) ? row_beats : 16'd9;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      gen_done <= 1'b1;
      g_out    <= '0;
      g_w      <= '0;
      g_in     <= '0;
      g_pa     <= '0;
      g_wa     <= '0;
      g_ia     <= '0;
    end else if (setup_pulse) begin
      gen_done <= (c_rows == 12'd0) || (c_w == 12'd0) || c_gacc;
      g_out    <= '0;
      g_w      <= '0;
      g_in     <= '0;
      g_pa     <= c_ddr;
      g_wa     <= c_ddr;
      g_ia     <= c_ddr;
    end else if (g_valid && g_ready) begin
      if (g_in == n_in - 1'b1) begin
        g_in <= '0;
        if (g_w == n_w - 1'b1) begin
          g_w  <= '0;
          g_pa <= g_pa + 32'(stride_out);
          g_wa <= g_pa + 32'(stride_out);
          g_ia <= g_pa + 32'(stride_out);
          if (g_out == n_out - 1'b1) begin
            gen_done <= 1'b1;
          end else begin
            g_out <= g_out + 1'b1;
          end
        end else begin
          g_w  <= g_w + 1'b1;
          g_wa <= g_wa + 32'd144;
          g_ia <= g_wa + 32'd144;
        end
      end else begin
        g_in <= g_in + 1'b1;
        g_ia <= g_ia + 32'(c_stride);
      end
    end
  end

  logic                     lf_valid, lf_ready, lf_done;
  logic [31:0]              lf_pc;
  logic [13:0]              lf_left;
  logic [6:0]               lf_inflight;
  logic [4:0]               lf_beats;
  logic [8:0]               lf_to4k;
  logic                     ib_push, ib_pop, ib_empty, ibr_load, ib_afull;
  logic [127:0]             ib_dout;
  logic [$clog2(IbDepth):0] ib_count;

  assign lf_to4k = 9'd256 - {1'b0, lf_pc[11:4]};

  always_comb begin
    lf_beats = (lf_left >= 14'd16) ? 5'd16 : lf_left[4:0];
    if (9'(lf_beats) > lf_to4k) begin
      lf_beats = lf_to4k[4:0];
    end
  end

  assign lf_valid = run && (c_idx || c_gacc) && !lf_done &&
                    (7'(ib_count) + lf_inflight <= 7'(IbDepth - 16));

  sfifo #(
    .Width      (128),
    .Depth      (IbDepth),
    .AfullMargin(1)
  ) u_ib (
    .clk_i,
    .rst_ni,
    .clear_i(setup_pulse),
    .push_i (ib_push),
    .din_i  (m_axi_rdata_i),
    .pop_i  (ib_pop),
    .dout_o (ib_dout),
    .empty_o(ib_empty),
    .afull_o(ib_afull),
    .count_o(ib_count)
  );

  logic        gi_valid, gi_ready, gi_done, gi_null;
  logic [1:0]  gi_sub;
  logic [23:0] gi_cnt;
  logic [31:0] gi_entry;
  logic        uf_push, uf_pop, uf_empty, uf_afull;
  logic        uf_dout;
  logic        gi_can, gi_step;

  assign gi_entry = ib_dout[gi_sub * 32 +: 32];
  assign gi_null  = (gi_entry == 32'hFFFFFFFF);
  assign gi_can   = run && c_idx && !gi_done && !ib_empty && !uf_afull;
  assign gi_valid = gi_can && !gi_null;
  assign gi_step  = gi_can && (gi_null || gi_ready);
  assign uf_push  = gi_step;
  assign ib_pop   = c_gacc ? ibr_load : (gi_step && (gi_sub == 2'd3 || gi_cnt == rows_w - 1'b1));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      gi_done     <= 1'b1;
      gi_sub      <= '0;
      gi_cnt      <= '0;
      lf_done     <= 1'b1;
      lf_pc       <= '0;
      lf_left     <= '0;
      lf_inflight <= '0;
    end else if (setup_pulse) begin
      gi_done     <= !c_idx || (rows_w == 24'd0);
      gi_sub      <= '0;
      gi_cnt      <= '0;
      lf_done     <= !(c_idx || c_gacc) || (!c_gacc && rows_w == 24'd0);
      lf_pc       <= c_ddr;
      lf_inflight <= '0;
      lf_left     <= c_gacc ? 14'({c_rows, 3'b000} + 15'd1) : 14'((rows_w + 24'd3) >> 2);
    end else begin
      lf_inflight <= lf_inflight + ((lf_valid && lf_ready) ? 7'(lf_beats) : 7'd0) - 7'(ib_push);
      if (lf_valid && lf_ready) begin
        lf_pc   <= lf_pc + 32'({lf_beats, 4'b0});
        lf_left <= lf_left - 14'(lf_beats);
        if (lf_left == 14'(lf_beats)) begin
          lf_done <= 1'b1;
        end
      end
      if (gi_step) begin
        gi_sub <= gi_sub + 1'b1;
        if (gi_cnt == rows_w - 1'b1) begin
          gi_done <= 1'b1;
        end else begin
          gi_cnt <= gi_cnt + 1'b1;
        end
      end
    end
  end

  logic [$clog2(UfDepth):0] uf_count;

  sfifo #(
    .Width      (1),
    .Depth      (UfDepth),
    .AfullMargin(2)
  ) u_uf (
    .clk_i,
    .rst_ni,
    .clear_i(setup_pulse),
    .push_i (uf_push),
    .din_i  (gi_null),
    .pop_i  (uf_pop),
    .dout_o (uf_dout),
    .empty_o(uf_empty),
    .afull_o(uf_afull),
    .count_o(uf_count)
  );

  logic        b_valid, b_ready, b_idle, req_valid, req_ready;
  logic [1:0]  b_tag;
  logic [31:0] b_addr, req_addr;
  logic [15:0] req_beats;
  logic [3:0]  b_len;
  logic [1:0]  req_tag;

  always_comb begin
    if (c_gacc) begin
      req_valid = ga_req_valid;
      req_addr  = ga_req_addr;
      req_beats = ga_req_beats;
      req_tag   = ga_req_tag;
    end else if (c_idx) begin
      req_valid = gi_valid;
      req_addr  = {gi_entry[27:0], 4'b0};
      req_beats = 16'(ubeats);
      req_tag   = 2'd0;
    end else begin
      req_valid = g_valid;
      req_addr  = g_ia;
      req_beats = g_beats;
      req_tag   = 2'd0;
    end
  end

  assign gi_ready     = req_ready && c_idx && !c_gacc;
  assign g_ready      = req_ready && !c_idx && !c_gacc;
  assign ga_req_ready = req_ready && c_gacc;

  dma_burst u_burst (
    .clk_i,
    .rst_ni,
    .req_valid_i(req_valid),
    .req_ready_o(req_ready),
    .req_addr_i (req_addr),
    .req_beats_i(req_beats),
    .req_tag_i  (req_tag),
    .b_valid_o  (b_valid),
    .b_ready_i  (b_ready),
    .b_addr_o   (b_addr),
    .b_len_o    (b_len),
    .b_tag_o    (b_tag),
    .idle_o     (b_idle)
  );

  logic gen_all_done;

  assign gen_all_done = c_gacc ? (lf_done && ga_idle) : c_idx ? (gi_done && lf_done) : gen_done;

  logic [7:0]               reserved;
  logic                     credit_ok, rf_push, rf_pop, rf_empty, rf_afull;
  logic [127:0]             rf_dout;
  logic [$clog2(RfDepth):0] rf_count;
  logic                     tf_push, tf_pop, tf_empty, tf_afull;
  logic [1:0]               tf_dout;

  assign credit_ok = (reserved <= 8'(RfDepth - 16)) && !tf_afull;

  logic ar_lock, ar_lock_list;
  logic ar_list, ar_data;

  assign ar_list = ar_lock ? ar_lock_list : (lf_valid && credit_ok);
  assign ar_data = ar_lock ? !ar_lock_list : (b_valid && !c_st && credit_ok && !lf_valid);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ar_lock      <= 1'b0;
      ar_lock_list <= 1'b0;
    end else begin
      ar_lock      <= m_axi_arvalid_o && !m_axi_arready_i;
      ar_lock_list <= ar_list;
    end
  end

  assign m_axi_araddr_o  = ar_list ? lf_pc : b_addr;
  assign m_axi_arlen_o   = ar_list ? 8'(lf_beats) - 8'd1 : {4'b0, b_len};
  assign m_axi_arsize_o  = 3'b100;
  assign m_axi_arburst_o = 2'b01;
  assign m_axi_arvalid_o = ar_list || ar_data;
  assign m_axi_rready_o  = 1'b1;
  assign lf_ready        = ar_list && m_axi_arready_i;
  assign tf_push         = m_axi_arvalid_o && m_axi_arready_i;

  logic [5:0] tf_count;

  sfifo #(
    .Width      (2),
    .Depth      (32),
    .AfullMargin(1)
  ) u_tf (
    .clk_i,
    .rst_ni,
    .clear_i(setup_pulse),
    .push_i (tf_push),
    .din_i  (ar_list ? 2'd1 : b_tag),
    .pop_i  (tf_pop),
    .dout_o (tf_dout),
    .empty_o(tf_empty),
    .afull_o(tf_afull),
    .count_o(tf_count)
  );

  logic r_is_list, r_is_w;

  assign r_is_list = !tf_empty && (tf_dout == 2'd1);
  assign r_is_w    = !tf_empty && (tf_dout == 2'd2);
  assign rf_push   = m_axi_rvalid_i && !r_is_list && !r_is_w;
  assign ib_push   = m_axi_rvalid_i && r_is_list;
  assign wq_push   = m_axi_rvalid_i && r_is_w;
  assign tf_pop    = m_axi_rvalid_i && m_axi_rlast_i;

  logic                     wq_afull;
  logic [$clog2(WqDepth):0] wq_count;

  sfifo #(
    .Width      (128),
    .Depth      (WqDepth),
    .AfullMargin(1)
  ) u_wq (
    .clk_i,
    .rst_ni,
    .clear_i(setup_pulse),
    .push_i (wq_push),
    .din_i  (m_axi_rdata_i),
    .pop_i  (wq_pop),
    .dout_o (wq_dout),
    .empty_o(wq_empty),
    .afull_o(wq_afull),
    .count_o(wq_count)
  );

  sfifo #(
    .Width      (128),
    .Depth      (RfDepth),
    .AfullMargin(4)
  ) u_rf (
    .clk_i,
    .rst_ni,
    .clear_i(1'b0),
    .push_i (rf_push),
    .din_i  (m_axi_rdata_i),
    .pop_i  (rf_pop),
    .dout_o (rf_dout),
    .empty_o(rf_empty),
    .afull_o(rf_afull),
    .count_o(rf_count)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      reserved <= '0;
    end else begin
      reserved <= reserved + ((m_axi_arvalid_o && m_axi_arready_i) ?
                              (ar_list ? 8'(lf_beats) : 8'(b_len) + 8'd1) : 8'd0) -
                  8'(rf_pop) - 8'(ib_push) - 8'(wq_push);
    end
  end

  logic         unit_known, null_act;
  logic [7:0]   zcnt;
  logic [23:0]  u_left;
  logic         src_v, src_pop, cons_ok;
  logic [127:0] src_d, zbeat;
  logic         need_uf;

  assign need_uf = run && c_idx && !unit_known;
  assign uf_pop  = need_uf && !uf_empty;

  function automatic logic [127:0] zb(input logic [7:0] z);
    logic [127:0] r;
    logic [7:0]   d;
    r = '0;
    d = z - 8'({c_tok, 1'b0});
    if (c_abuf && z >= 8'({c_tok, 1'b0})) begin
      for (int i = 0; i < 16; i++) begin
        if ({d, 4'(i)} < 12'(c_tok)) begin
          r[8 * i +: 8] = 8'h7F;
        end
      end
    end
    return r;
  endfunction

  logic       z_step;
  logic [7:0] zcnt_n;

  assign z_step = src_pop && c_idx;
  assign zcnt_n = (setup_pulse || (z_step && (zcnt == ubeats - 1'b1))) ? 8'd0 :
                  z_step ? zcnt + 1'b1 : zcnt;

  always_ff @(posedge clk_i) begin
    zbeat <= zb(zcnt_n);
  end

  assign src_v   = c_gacc ? 1'b0 :
                   c_idx ? (unit_known && (null_act || !rf_empty)) :
                   !rf_empty;
  assign src_d   = null_act ? zbeat : rf_dout;
  assign src_pop = src_v && cons_ok && run && !c_st;
  assign rf_pop  = c_gacc ? (!rf_empty && ga_row_ready && run) : (src_pop && !null_act);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      unit_known <= 1'b1;
      null_act   <= 1'b0;
      zcnt       <= '0;
      u_left     <= '0;
    end else if (setup_pulse) begin
      unit_known <= !c_idx;
      null_act   <= 1'b0;
      zcnt       <= '0;
      u_left     <= c_idx ? rows_w : 24'd0;
    end else begin
      if (uf_pop) begin
        unit_known <= 1'b1;
        null_act   <= uf_dout;
      end
      zcnt <= zcnt_n;
      if (z_step && (zcnt == ubeats - 1'b1)) begin
        unit_known <= 1'b0;
        null_act   <= 1'b0;
        u_left     <= u_left - 1'b1;
      end
    end
  end

  logic          t32_in_ready, t32_out_valid, t32_out_ready, t32_lo, t32_idle;
  logic [1151:0] t32_out;
  logic [1:0]    t32_cg;
  logic          rf_in_ready, rf_out_valid, rf_out_ready, rf_idle, rf_win_ready, rf_bout_valid;
  logic          rf_bout_ready;
  logic [1151:0] rf_out_word;
  logic [127:0]  rf_bout;
  logic          wf_push, wf_pop, wf_empty, wf_afull;
  logic [1174:0] wf_din, wf_dout;
  logic          pend_hi, pend_lo;

  assign cons_ok = (c_mode == ModeT32) ? t32_in_ready :
                   c_row ? rf_in_ready : (!wf_afull && !pend_hi && !pend_lo);

  logic [3:0]    bcnt;
  logic          wv, c_rr;
  logic [1151:0] asm_r;
  logic [19:0]   dst_cnt, idx;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      bcnt    <= '0;
      wv      <= 1'b0;
      c_rr    <= 1'b0;
      dst_cnt <= '0;
      idx     <= '0;
    end else if (setup_pulse) begin
      bcnt    <= '0;
      wv      <= 1'b0;
      c_rr    <= 1'b0;
      dst_cnt <= c_oc;
      idx     <= '0;
    end else begin
      wv <= 1'b0;
      if (src_pop && c_mode != ModeT32 && !c_row) begin
        if (bcnt == bpw - 1'b1) begin
          bcnt <= '0;
          wv   <= 1'b1;
        end else begin
          bcnt <= bcnt + 1'b1;
        end
      end
      if ((wv && c_mode == ModeRaw && !c_row) || (wf_push && c_row)) begin
        dst_cnt <= dst_cnt + 1'b1;
      end
      if (wv && c_mode == ModeSplit8) begin
        c_rr <= ~c_rr;
        if (c_rr) begin
          idx <= idx + 1'b1;
        end
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (!setup_pulse && src_pop && c_mode != ModeT32 && !c_row) begin
      asm_r[bcnt * 128 +: 128] <= src_d;
    end
  end

  logic [1087:0] s_hi, s_lo;
  logic [511:0]  nib_hi, nib_lo;

  always_comb begin
    for (int k = 0; k < 128; k++) begin
      nib_hi[k*4 +: 4] = asm_r[k*8 + 4 +: 4];
      nib_lo[k*4 +: 4] = asm_r[k*8 +: 4];
    end
  end

  always_ff @(posedge clk_i) begin
    if (wv && c_mode == ModeSplit8) begin
      if (!c_rr) begin
        s_hi[511:0]       <= nib_hi;
        s_lo[511:0]       <= nib_lo;
        s_hi[1055:1024]   <= asm_r[1055:1024];
        s_lo[1055:1024]   <= asm_r[1055:1024];
      end else begin
        s_hi[1023:512]    <= nib_hi;
        s_lo[1023:512]    <= nib_lo;
        s_hi[1087:1056]   <= asm_r[1055:1024];
        s_lo[1087:1056]   <= asm_r[1055:1024];
      end
    end
  end

  logic [19:0] idx_hi;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      pend_hi <= 1'b0;
      pend_lo <= 1'b0;
    end else if (setup_pulse) begin
      pend_hi <= 1'b0;
      pend_lo <= 1'b0;
    end else begin
      if (wv && c_mode == ModeSplit8 && c_rr) begin
        pend_hi <= 1'b1;
        idx_hi  <= idx;
      end else if (pend_hi) begin
        pend_hi <= 1'b0;
        pend_lo <= 1'b1;
      end else if (pend_lo) begin
        pend_lo <= 1'b0;
      end
    end
  end

  dma_t32 u_t32 (
    .clk_i,
    .rst_ni,
    .clear_i    (setup_pulse),
    .in_valid_i (src_pop && c_mode == ModeT32),
    .in_ready_o (t32_in_ready),
    .in_data_i  (src_d),
    .out_valid_o(t32_out_valid),
    .out_ready_i(t32_out_ready),
    .out_data_o (t32_out),
    .out_cg_o   (t32_cg),
    .out_lo_o   (t32_lo),
    .idle_o     (t32_idle)
  );

  assign t32_out_ready = !wf_afull && run;

  logic          t_v;
  logic [1151:0] t_d;
  logic [19:0]   t_addr, blk_base, rb;
  logic [11:0]   t_w;
  logic [2:0]    t_cnt;
  logic [19:0]   cg_off;

  always_comb begin
    case (t32_cg)
      2'd0:    cg_off = '0;
      2'd1:    cg_off = kbt;
      2'd2:    cg_off = kbt2;
      default: cg_off = kbt3;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      t_v   <= 1'b0;
      t_cnt <= '0;
      t_w   <= '0;
    end else if (setup_pulse) begin
      t_v      <= 1'b0;
      t_cnt    <= '0;
      t_w      <= '0;
      blk_base <= c_oc;
      rb       <= c_oc;
    end else begin
      t_v <= t32_out_valid && t32_out_ready;
      if (t32_out_valid && t32_out_ready) begin
        t_addr <= blk_base + cg_off + (t32_lo ? n_half : 20'd0);
        t_cnt  <= t_cnt + 1'b1;
        if (t_cnt == 3'd7) begin
          if (t_w == c_w - 1'b1) begin
            t_w      <= '0;
            rb       <= rb + 1'b1;
            blk_base <= rb + 1'b1;
          end else begin
            t_w      <= t_w + 1'b1;
            blk_base <= blk_base + kbt4;
          end
        end
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (!setup_pulse && t32_out_valid && t32_out_ready) begin
      t_d <= t32_out;
    end
  end

  logic [1151:0] rf_win_word;
  logic          rf_win_valid;

  dma_rowfmt u_rowfmt (
    .clk_i,
    .rst_ni,
    .clear_i     (setup_pulse),
    .cfg_unpack_i(c_st),
    .cfg_abuf_i  (c_abuf),
    .cfg_tok_i   (c_tok),
    .in_valid_i  (src_v && run && !c_st && c_row),
    .in_ready_o  (rf_in_ready),
    .in_data_i   (src_d),
    .out_valid_o (rf_out_valid),
    .out_ready_i (rf_out_ready),
    .out_word_o  (rf_out_word),
    .win_valid_i (rf_win_valid),
    .win_ready_o (rf_win_ready),
    .win_word_i  (rf_win_word),
    .bout_valid_o(rf_bout_valid),
    .bout_ready_i(rf_bout_ready),
    .bout_data_o (rf_bout),
    .idle_o      (rf_idle)
  );

  assign rf_out_ready = !wf_afull && run && !c_st && c_row;

  logic         ibr_v;
  logic [127:0] ibr_d;

  assign ibr_load = c_gacc && run && !ib_empty && (!ibr_v || ga_ent_ready);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ibr_v <= 1'b0;
    end else if (setup_pulse) begin
      ibr_v <= 1'b0;
    end else if (ibr_load) begin
      ibr_v <= 1'b1;
    end else if (ibr_v && ga_ent_ready) begin
      ibr_v <= 1'b0;
    end
  end

  always_ff @(posedge clk_i) begin
    if (!setup_pulse && ibr_load) begin
      ibr_d <= ib_dout;
    end
  end

  assign ga_ent_valid = ibr_v;
  assign wq_pop       = c_gacc && run && !wq_empty && ga_w_ready;

  dma_gacc u_gacc (
    .clk_i,
    .rst_ni,
    .clear_i    (setup_pulse),
    .start_i    (ga_start),
    .cfg_slots_i(c_rows),
    .cfg_kb_i   (c_w[3:0]),
    .cfg_q_i    (c_stride[15:0]),
    .cfg_acc_i  (c_gflags[0]),
    .cfg_flush_i(c_gflags[1]),
    .cfg_dst_i  (c_oc[12:0]),
    .ent_valid_i(ga_ent_valid),
    .ent_ready_o(ga_ent_ready),
    .ent_data_i (ibr_d),
    .req_valid_o(ga_req_valid),
    .req_ready_i(ga_req_ready),
    .req_addr_o (ga_req_addr),
    .req_beats_o(ga_req_beats),
    .req_tag_o  (ga_req_tag),
    .row_valid_i(c_gacc && run && !rf_empty),
    .row_ready_o(ga_row_ready),
    .row_data_i (rf_dout),
    .w_valid_i  (c_gacc && run && !wq_empty),
    .w_ready_o  (ga_w_ready),
    .w_data_i   (wq_dout),
    .out_valid_o(ga_out_valid),
    .out_ready_i(ga_out_ready),
    .out_word_o (ga_out_word),
    .out_addr_o (ga_out_addr),
    .idle_o     (ga_idle),
    .null_o     (ga_null)
  );

  assign ga_out_ready = !wf_afull && run && c_gacc;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ga_start <= 1'b0;
    end else begin
      ga_start <= setup_pulse && c_gacc;
    end
  end

  always_comb begin
    wf_push = 1'b0;
    wf_din  = '0;
    if (c_gacc) begin
      wf_push = ga_out_valid && ga_out_ready;
      wf_din  = {3'd2, 7'd0, ga_out_addr, 128'd0, ga_out_word};
    end else if (c_row) begin
      wf_push = rf_out_valid && rf_out_ready;
      wf_din  = {c_mem, dst_cnt, rf_out_word};
    end else if (c_mode == ModeT32) begin
      wf_push = t_v;
      wf_din  = {3'd1, t_addr, t_d};
    end else if (c_mode == ModeSplit8) begin
      if (pend_hi) begin
        wf_push = 1'b1;
        wf_din  = {3'd1, c_oc + idx_hi, 64'd0, s_hi};
      end else if (pend_lo) begin
        wf_push = 1'b1;
        wf_din  = {3'd1, c_oc + n_half + idx_hi, 64'd0, s_lo};
      end
    end else begin
      wf_push = wv;
      wf_din  = {c_mem, dst_cnt, asm_r};
    end
  end

  logic [3:0] wf_count;

  sfifo #(
    .Width      (1175),
    .Depth      (8),
    .AfullMargin(3)
  ) u_wf (
    .clk_i,
    .rst_ni,
    .clear_i(1'b0),
    .push_i (wf_push),
    .din_i  (wf_din),
    .pop_i  (wf_pop),
    .dout_o (wf_dout),
    .empty_o(wf_empty),
    .afull_o(wf_afull),
    .count_o(wf_count)
  );

  logic          wo_v;
  logic [1174:0] wo_d;

  assign wf_pop = !wf_empty && (!wo_v || wr_grant_i);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wo_v <= 1'b0;
    end else if (!wo_v || wr_grant_i) begin
      wo_v <= !wf_empty;
    end
  end

  always_ff @(posedge clk_i) begin
    if (!wo_v || wr_grant_i) begin
      wo_d <= wf_dout;
    end
  end

  assign wr_req_o  = wo_v;
  assign wr_mem_o  = wo_d[1174:1172];
  assign wr_addr_o = wo_d[1171:1152];
  assign wr_data_o = wo_d[1151:0];

  assign ld_done = gen_all_done && b_idle && (reserved == 8'd0) && rf_empty && tf_empty && !wv &&
                   !pend_hi && !pend_lo && !t_v && t32_idle && (!c_row || rf_idle) && wf_empty &&
                   !wo_v && (bcnt == 4'd0) && !null_act &&
                   (!c_idx || (uf_empty && u_left == 24'd0)) &&
                   (!c_gacc || (ga_idle && !ga_start && ib_empty && !ibr_v && wq_empty));

  logic [23:0]              rd_cnt;
  logic                     rd_all, inflight, hold_v, ser_v;
  logic [2:0]               rdv;
  logic [1055:0]            hold;
  logic [1151:0]            ser;
  logic [3:0]               scnt;
  logic                     wb_push, wb_pop, wb_empty, wb_afull;
  logic [127:0]             wb_din, wb_dout;
  logic [$clog2(WbDepth):0] wb_count;
  logic                     rd_can, ser_take;

  assign rd_can      = run && c_st && !rd_all && !hold_v && !inflight;
  assign rd_req_o    = rd_can;
  assign rd_mem_o    = c_mem;
  assign ser_take    = !c_row && hold_v && (!ser_v || (wb_push && scnt == bpw - 1'b1));

  logic [19:0] rd_addr_r;

  assign rd_addr_o = rd_addr_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rd_cnt   <= '0;
      rd_all   <= 1'b1;
      inflight <= 1'b0;
      hold_v   <= 1'b0;
      ser_v    <= 1'b0;
      rdv      <= '0;
      scnt     <= '0;
    end else if (setup_pulse) begin
      rd_cnt    <= '0;
      rd_all    <= (rows_w == 24'd0);
      inflight  <= 1'b0;
      hold_v    <= 1'b0;
      ser_v     <= 1'b0;
      rdv       <= '0;
      scnt      <= '0;
      rd_addr_r <= c_oc;
    end else begin
      rdv <= {rdv[1:0], rd_req_o && rd_grant_i};
      if (rd_req_o && rd_grant_i) begin
        inflight  <= 1'b1;
        rd_cnt    <= rd_cnt + 1'b1;
        rd_addr_r <= rd_addr_r + 1'b1;
        if (rd_cnt == oc_words - 1'b1) begin
          rd_all <= 1'b1;
        end
      end
      if (rdv[2]) begin
        hold_v   <= 1'b1;
        inflight <= 1'b0;
      end
      if (!c_row) begin
        if (ser_take) begin
          ser_v  <= 1'b1;
          scnt   <= '0;
          hold_v <= 1'b0;
        end else if (wb_push) begin
          if (scnt == bpw - 1'b1) begin
            ser_v <= 1'b0;
            scnt  <= '0;
          end else begin
            scnt <= scnt + 1'b1;
          end
        end
      end else if (hold_v && rf_win_ready) begin
        hold_v <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (!setup_pulse && rdv[2]) begin
      hold <= rd_data_i;
    end
    if (!setup_pulse && ser_take) begin
      ser <= {96'd0, hold};
    end
  end

  assign rf_win_valid  = c_st && c_row && hold_v;
  assign rf_win_word   = {96'd0, hold};
  assign bout_fire     = rf_bout_valid && rf_bout_ready;
  assign rf_bout_ready = c_st && c_row && !wb_afull;

  logic [127:0] ser_beat;

  assign ser_beat = ser[scnt * 128 +: 128];
  assign wb_push  = c_row ? bout_fire : (ser_v && !wb_afull);
  assign wb_din   = c_row ? rf_bout : ser_beat;

  sfifo #(
    .Width      (128),
    .Depth      (WbDepth),
    .AfullMargin(2)
  ) u_wb (
    .clk_i,
    .rst_ni,
    .clear_i(1'b0),
    .push_i (wb_push),
    .din_i  (wb_din),
    .pop_i  (wb_pop),
    .dout_o (wb_dout),
    .empty_o(wb_empty),
    .afull_o(wb_afull),
    .count_o(wb_count)
  );

  logic       w_active, wl_full, aw_fire;
  logic [3:0] w_len, w_cnt;
  logic [7:0] aw_cnt, b_cnt;
  logic [3:0] wl_mem [4];
  logic [2:0] wl_wp, wl_rp;
  logic [6:0] w_avail;

  assign w_active        = wl_wp != wl_rp;
  assign wl_full         = (wl_wp[1:0] == wl_rp[1:0]) && (wl_wp[2] != wl_rp[2]);
  assign w_len           = wl_mem[wl_rp[1:0]];
  assign aw_fire         = m_axi_awvalid_o && m_axi_awready_i;
  assign m_axi_awaddr_o  = b_addr;
  assign m_axi_awlen_o   = {4'b0, b_len};
  assign m_axi_awsize_o  = 3'b100;
  assign m_axi_awburst_o = 2'b01;
  assign m_axi_awvalid_o = b_valid && c_st && !wl_full &&
                           (w_avail > {3'b0, b_len});
  assign b_ready         = c_st ? (m_axi_awvalid_o && m_axi_awready_i) : (ar_data && m_axi_arready_i);
  assign m_axi_wvalid_o  = w_active;
  assign m_axi_wdata_o   = wb_dout;
  assign m_axi_wstrb_o   = '1;
  assign m_axi_wlast_o   = w_active && (w_cnt == w_len);
  assign wb_pop          = m_axi_wvalid_o && m_axi_wready_i;
  assign m_axi_bready_o  = 1'b1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wl_wp   <= '0;
      wl_rp   <= '0;
      w_cnt   <= '0;
      w_avail <= '0;
      aw_cnt  <= '0;
      b_cnt   <= '0;
    end else if (setup_pulse) begin
      wl_wp   <= '0;
      wl_rp   <= '0;
      w_cnt   <= '0;
      w_avail <= '0;
      aw_cnt  <= '0;
      b_cnt   <= '0;
    end else begin
      if (aw_fire) begin
        wl_wp  <= wl_wp + 3'd1;
        aw_cnt <= aw_cnt + 1'b1;
      end
      w_avail <= w_avail + 7'(wb_push) - (aw_fire ? 7'(b_len) + 7'd1 : 7'd0);
      if (wb_pop) begin
        if (m_axi_wlast_o) begin
          w_cnt <= '0;
          wl_rp <= wl_rp + 3'd1;
        end else begin
          w_cnt <= w_cnt + 1'b1;
        end
      end
      if (m_axi_bvalid_i) begin
        b_cnt <= b_cnt + 1'b1;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (aw_fire) begin
      wl_mem[wl_wp[1:0]] <= b_len;
    end
  end

  assign st_done = gen_all_done && b_idle && !w_active && rd_all && !inflight && !hold_v &&
                   !ser_v && wb_empty && (aw_cnt == b_cnt) && (rdv == 3'b000) &&
                   (!c_row || rf_idle) &&
                   (reserved == 8'd0) && tf_empty;

  logic unused_dma;

  assign unused_dma = ^{m_axi_rresp_i, m_axi_bresp_i, ib_afull, uf_count, rf_afull,
                        rf_count, tf_count, wq_afull, wq_count, wf_count, list_beats, wb_count};

endmodule
