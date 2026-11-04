module dma_gacc (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          clear_i,
  input  logic          start_i,
  input  logic [11:0]   cfg_slots_i,
  input  logic [3:0]    cfg_kb_i,
  input  logic [15:0]   cfg_q_i,
  input  logic          cfg_acc_i,
  input  logic          cfg_flush_i,
  input  logic [12:0]   cfg_dst_i,
  input  logic          ent_valid_i,
  output logic          ent_ready_o,
  input  logic [127:0]  ent_data_i,
  output logic          req_valid_o,
  input  logic          req_ready_i,
  output logic [31:0]   req_addr_o,
  output logic [15:0]   req_beats_o,
  output logic [1:0]    req_tag_o,
  input  logic          row_valid_i,
  output logic          row_ready_o,
  input  logic [127:0]  row_data_i,
  input  logic          w_valid_i,
  output logic          w_ready_o,
  input  logic [127:0]  w_data_i,
  output logic          out_valid_o,
  input  logic          out_ready_i,
  output logic [1023:0] out_word_o,
  output logic [12:0]   out_addr_o,
  output logic          idle_o
);

  localparam int Nl     = 8;
  localparam int LMul   = 6;
  localparam int LAdd   = 6;
  localparam int LWb    = 2 + LMul + LAdd;
  localparam int MinGap = LWb + 3;

  logic [11:0] slots;
  logic [3:0]  kb;
  logic [15:0] quad;
  logic        acc_keep, do_flush;
  logic [12:0] dst;
  logic [31:0] w_base;
  logic [15:0] w_gstride, w_qwords;
  logic [31:0] w_qoff, w_grp, wreq_addr;
  logic [4:0]  nd;
  logic [5:0]  row_beats;

  assign nd        = {kb, 1'b0};
  assign row_beats = 6'(nd) + 6'd1;

  typedef enum logic [3:0] {
    GIdle,
    GZero,
    GHdr,
    GWmul,
    GWadd,
    GWreq,
    GWwait,
    GEnt,
    GDrain,
    GFlush
  } st_e;

  st_e          st;
  logic [11:0]  s_cur;
  logic [2:0]   s_in_grp;
  logic [1:0]   e_t, e_k;
  logic         e_half;
  logic [3:0]   wreq_kb, wrcv_kb;
  logic [2:0]   wrcv_b;
  logic         wr_we;
  logic [5:0]   wr_wa, wr_ra;
  logic [31:0]  wr_rd [4];
  logic [6:0]   zcnt;
  logic [3:0]   drain_cnt;
  logic         cm_busy, cm_pend;
  logic [2:0]   cm_i;
  logic         cm_v [LMul + 1];
  logic [2:0]   cm_i_d [LMul + 1];
  logic [31:0]  cm_bw, cm_w, cm_y;
  logic [1:0]   cm_t;
  logic [2:0]   cm_col;
  logic [255:0] cm_c;
  logic [3:0]   cm_cnt;
  logic         pf_push, pf_pop, pf_empty, pf_afull;
  logic [257:0] pf_din, pf_dout;
  (* ram_style = "distributed" *) logic [127:0] rbuf [32];
  logic         rb_fill, rb_mac;
  logic [4:0]   rb_b;
  logic [1:0]   rfull;
  logic [127:0] rscale [2];
  logic [257:0] rinfo [2];
  logic         mac_busy, mac_row_end, mac_start, pipe_empty;
  logic [4:0]   mac_b;
  logic [7:0]   gap [4];
  logic         p_v [LWb + 1];
  logic [1:0]   p_t [LWb + 1];
  logic [4:0]   p_b [LWb + 1];
  logic [6:0]   rd_idx_d [LMul + 1];
  logic [63:0]  md_r;
  logic [31:0]  mc_r;
  logic [7:0]   me_r;
  logic [31:0]  xf [Nl], prod [Nl], acc_q [Nl], sum [Nl], fl_q [Nl];
  logic [31:0]  cs, cs_d;
  logic [9:0]   cs_e;
  logic         acc_we;
  logic [6:0]   acc_wa;
  logic [31:0]  acc_wd [Nl];
  logic         fl_rd_v, fl_word_v;
  logic [5:0]   fl_g, fl_g_d;
  logic [1:0]   fl_t, fl_t_d;
  logic [1023:0] fl_word;

  typedef enum logic [1:0] {
    FRead,
    FWait,
    FEnd
  } fl_e;

  fl_e fl_st;

  logic [63:0] ent_cur;
  logic        ent_null, ent_ok, ent_take;

  assign ent_cur     = e_half ? ent_data_i[127:64] : ent_data_i[63:0];
  assign ent_null    = (ent_cur[31:0] == 32'hFFFFFFFF);
  assign ent_ok      = (st == GEnt) && ent_valid_i && !cm_busy && !cm_pend && !pf_afull &&
                       (ent_null || req_ready_i);
  assign ent_take    = ent_ok;
  assign ent_ready_o = (st == GHdr) ? ent_valid_i : (ent_take && e_half);

  always_comb begin
    req_valid_o = 1'b0;
    req_addr_o  = '0;
    req_beats_o = '0;
    req_tag_o   = 2'd0;
    w_ready_o   = 1'b0;
    if (st == GWreq) begin
      req_valid_o = 1'b1;
      req_tag_o   = 2'd2;
      req_beats_o = 16'd8;
      req_addr_o  = wreq_addr;
    end else if (st == GEnt) begin
      req_valid_o = ent_valid_i && !ent_null && !cm_busy && !cm_pend && !pf_afull;
      req_addr_o  = {ent_cur[27:0], 4'b0};
      req_beats_o = 16'(row_beats);
      req_tag_o   = 2'd0;
    end else if (st == GWwait) begin
      w_ready_o = 1'b1;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      st        <= GIdle;
      s_cur     <= '0;
      s_in_grp  <= '0;
      e_t       <= '0;
      e_k       <= '0;
      e_half    <= 1'b0;
      wreq_kb   <= '0;
      wrcv_kb   <= '0;
      wrcv_b    <= '0;
      zcnt      <= '0;
      drain_cnt <= '0;
    end else begin
      case (st)
        GIdle: begin
          if (start_i) begin
            slots     <= cfg_slots_i;
            kb        <= cfg_kb_i;
            quad      <= cfg_q_i;
            acc_keep  <= cfg_acc_i;
            do_flush  <= cfg_flush_i;
            dst       <= cfg_dst_i;
            s_cur     <= '0;
            s_in_grp  <= '0;
            e_t       <= '0;
            e_k       <= '0;
            e_half    <= 1'b0;
            zcnt      <= '0;
            drain_cnt <= '0;
            st        <= cfg_acc_i ? GHdr : GZero;
          end
        end
        GZero: begin
          zcnt <= zcnt + 1'b1;
          if (zcnt == 7'd127) begin
            st <= GHdr;
          end
        end
        GHdr: begin
          if (ent_valid_i) begin
            w_base    <= {ent_data_i[27:0], 4'b0};
            w_gstride <= ent_data_i[47:32];
            w_qwords  <= ent_data_i[63:48];
            if (slots == 12'd0) begin
              st <= GDrain;
            end else begin
              st      <= GWmul;
              wreq_kb <= '0;
              wrcv_kb <= '0;
              wrcv_b  <= '0;
            end
          end
        end
        GWmul: st <= GWadd;
        GWadd: begin
          w_grp     <= w_base + (w_qoff << 7);
          wreq_addr <= w_base + (w_qoff << 7);
          st        <= GWreq;
        end
        GWreq: begin
          if (req_ready_i) begin
            wreq_kb   <= wreq_kb + 1'b1;
            wreq_addr <= wreq_addr + 32'({w_gstride, 7'b0});
            if (wreq_kb == kb - 1'b1) begin
              st <= GWwait;
            end
          end
        end
        GWwait: begin
          if (w_valid_i) begin
            wrcv_b <= wrcv_b + 1'b1;
            if (wrcv_b == 3'd7) begin
              if (wrcv_kb == kb - 1'b1) begin
                wrcv_kb  <= '0;
                st       <= GEnt;
                s_in_grp <= '0;
              end else begin
                wrcv_kb <= wrcv_kb + 1'b1;
              end
            end
          end
        end
        GEnt: begin
          if (ent_take) begin
            e_half <= ~e_half;
            if (e_k == 2'd3) begin
              e_k <= '0;
              if (e_t == 2'd3) begin
                e_t      <= '0;
                s_cur    <= s_cur + 1'b1;
                s_in_grp <= s_in_grp + 1'b1;
                if (s_cur == slots - 1'b1) begin
                  st <= GDrain;
                end else if (s_in_grp == 3'd7) begin
                  st        <= GWreq;
                  wreq_kb   <= '0;
                  w_grp     <= w_grp + 32'd128;
                  wreq_addr <= w_grp + 32'd128;
                end
              end else begin
                e_t <= e_t + 1'b1;
              end
            end else begin
              e_k <= e_k + 1'b1;
            end
          end
        end
        GDrain: begin
          if (pf_empty && rfull == 2'b00 && !mac_busy && pipe_empty && !cm_busy && !cm_pend) begin
            drain_cnt <= drain_cnt + 1'b1;
            if (drain_cnt == 4'd3) begin
              drain_cnt <= '0;
              st        <= do_flush ? GFlush : GIdle;
            end
          end else begin
            drain_cnt <= '0;
          end
        end
        GFlush: begin
          if (fl_st == FEnd) begin
            st <= GIdle;
          end
        end
        default: st <= GIdle;
      endcase
    end
  end

  always_ff @(posedge clk_i) begin
    w_qoff <= 32'(quad) * 32'(w_qwords);
  end

  assign wr_we = (st == GWwait) && w_valid_i;
  assign wr_wa = {wrcv_kb[2:0], wrcv_b};
  assign wr_ra = {cm_i, cm_t, cm_col[2]};

  for (genvar gb = 0; gb < 4; gb++) begin : wbank
    (* ram_style = "distributed" *) logic [31:0] mem [64];

    always_ff @(posedge clk_i) begin
      if (wr_we) begin
        mem[wr_wa] <= w_data_i[32 * gb +: 32];
      end
    end

    assign wr_rd[gb] = mem[wr_ra];
  end

  logic cm_y_valid;

  assign cm_w = wr_rd[cm_col[1:0]];

  fp32_mul u_cmul (
    .clk_i,
    .rst_ni,
    .in_valid_i (cm_busy),
    .a_i        (cm_w),
    .b_i        (cm_bw),
    .out_valid_o(cm_y_valid),
    .y_o        (cm_y)
  );

  assign pf_push = cm_pend && (cm_cnt == 4'd8);
  assign pf_din  = {cm_t, cm_c};

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      cm_busy <= 1'b0;
      cm_pend <= 1'b0;
      cm_i    <= '0;
      cm_cnt  <= '0;
      for (int i = 0; i <= LMul; i++) begin
        cm_v[i] <= 1'b0;
      end
    end else begin
      cm_v[0]   <= cm_busy;
      cm_i_d[0] <= cm_i;
      for (int i = 1; i <= LMul; i++) begin
        cm_v[i]   <= cm_v[i-1];
        cm_i_d[i] <= cm_i_d[i-1];
      end
      if (ent_take && !ent_null) begin
        cm_busy <= 1'b1;
        cm_pend <= 1'b1;
        cm_i    <= '0;
        cm_cnt  <= '0;
        cm_bw   <= ent_cur[63:32];
        cm_t    <= e_t;
        cm_col  <= s_cur[2:0];
      end else if (cm_busy) begin
        cm_i <= cm_i + 1'b1;
        if (cm_i == 3'd7) begin
          cm_busy <= 1'b0;
        end
      end
      if (cm_v[LMul - 1]) begin
        cm_c[cm_i_d[LMul - 1] * 32 +: 32] <= cm_y;
        cm_cnt                            <= cm_cnt + 1'b1;
      end
      if (pf_push) begin
        cm_pend <= 1'b0;
      end
    end
  end

  logic [4:0] pf_count;

  sfifo #(
    .Width      (258),
    .Depth      (16),
    .AfullMargin(3)
  ) u_pf (
    .clk_i,
    .rst_ni,
    .clear_i,
    .push_i (pf_push),
    .din_i  (pf_din),
    .pop_i  (pf_pop),
    .dout_o (pf_dout),
    .empty_o(pf_empty),
    .afull_o(pf_afull),
    .count_o(pf_count)
  );

  logic row_fire, row_last;

  assign row_ready_o = !rfull[rb_fill] && !pf_empty;
  assign row_fire    = row_valid_i && row_ready_o;
  assign row_last    = row_fire && (rb_b == 5'(row_beats) - 1'b1);
  assign pf_pop      = row_last;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      rb_fill <= 1'b0;
      rb_b    <= '0;
      rfull   <= '0;
    end else begin
      if (row_fire) begin
        if (rb_b >= nd) begin
          rscale[rb_fill] <= row_data_i;
        end
        if (row_last) begin
          rb_b           <= '0;
          rfull[rb_fill] <= 1'b1;
          rb_fill        <= ~rb_fill;
          rinfo[rb_fill] <= pf_dout;
        end else begin
          rb_b <= rb_b + 1'b1;
        end
      end
      if (mac_row_end) begin
        rfull[rb_mac] <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (row_fire && rb_b < nd) begin
      rbuf[{rb_fill, rb_b[3:0]}] <= row_data_i;
    end
  end

  logic [1:0] mac_t;

  assign mac_t       = rinfo[rb_mac][257:256];
  assign mac_start   = !mac_busy && rfull[rb_mac] && (gap[mac_t] == 8'd0) && (st != GFlush) &&
                       (st != GZero);
  assign mac_row_end = mac_busy && (mac_b == 5'(6'({nd, 1'b0}) - 6'd1));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      rb_mac   <= 1'b0;
      mac_busy <= 1'b0;
      mac_b    <= '0;
      for (int t = 0; t < 4; t++) begin
        gap[t] <= '0;
      end
    end else begin
      for (int t = 0; t < 4; t++) begin
        if (gap[t] != 8'd0) begin
          gap[t] <= gap[t] - 1'b1;
        end
      end
      if (mac_start) begin
        mac_busy   <= 1'b1;
        mac_b      <= '0;
        gap[mac_t] <= 8'(MinGap);
      end else if (mac_busy) begin
        mac_b <= mac_b + 1'b1;
        if (mac_row_end) begin
          mac_busy <= 1'b0;
          rb_mac   <= ~rb_mac;
        end
      end
    end
  end

  logic [3:0]  mac_kb;
  logic [7:0]  mac_e;
  logic [31:0] mac_c;

  assign mac_kb = 4'(mac_b[4:2]);
  assign mac_e  = rscale[rb_mac][mac_kb * 8 +: 8];
  assign mac_c  = rinfo[rb_mac][mac_kb * 32 +: 32];

  always_comb begin
    cs_e = {2'b0, mc_r[30:23]} + {2'b0, me_r} - 10'd133;
    if (mc_r[30:23] == 8'd0 || $signed(cs_e) <= 10'sd0) begin
      cs = 32'd0;
    end else if (cs_e >= 10'd255) begin
      cs = {mc_r[31], 8'hFE, 23'h7FFFFF};
    end else begin
      cs = {mc_r[31], cs_e[7:0], mc_r[22:0]};
    end
  end

  logic [127:0] mac_data;
  logic [6:0]   rd_idx;

  assign mac_data = rbuf[{rb_mac, mac_b[4:1]}];
  assign rd_idx   = {mac_t, mac_b};

  function automatic logic [31:0] i8_to_f32(input logic [7:0] q);
    logic [7:0]  m;
    logic [31:0] r;
    int          sh;
    if (q == 8'd0) begin
      return 32'd0;
    end
    m  = q[7] ? (~q + 8'd1) : q;
    sh = 0;
    for (int i = 7; i >= 0; i--) begin
      if (m[i]) begin
        sh = i;
        break;
      end
    end
    r       = {q[7], 8'(127 + sh), 23'd0};
    r[22:0] = 23'((32'(m) << (23 - sh)) & 32'h7FFFFF);
    return r;
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      for (int i = 0; i <= LWb; i++) begin
        p_v[i] <= 1'b0;
      end
    end else begin
      p_v[0] <= mac_busy;
      p_t[0] <= mac_t;
      p_b[0] <= mac_b;
      for (int i = 1; i <= LWb; i++) begin
        p_v[i] <= p_v[i-1];
        p_t[i] <= p_t[i-1];
        p_b[i] <= p_b[i-1];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    md_r <= mac_b[0] ? mac_data[127:64] : mac_data[63:0];
    mc_r <= mac_c;
    me_r <= mac_e;
    for (int l = 0; l < Nl; l++) begin
      xf[l] <= i8_to_f32(md_r[8 * l +: 8]);
    end
    cs_d        <= cs;
    rd_idx_d[0] <= rd_idx;
    for (int i = 1; i <= LMul; i++) begin
      rd_idx_d[i] <= rd_idx_d[i-1];
    end
  end

  logic [Nl-1:0] prod_valid, sum_valid;

  for (genvar gl = 0; gl < Nl; gl++) begin : lane
    fp32_mul u_m (
      .clk_i,
      .rst_ni,
      .in_valid_i (1'b1),
      .a_i        (xf[gl]),
      .b_i        (cs_d),
      .out_valid_o(prod_valid[gl]),
      .y_o        (prod[gl])
    );
    fp32_add u_a (
      .clk_i,
      .rst_ni,
      .in_valid_i (1'b1),
      .a_i        (acc_q[gl]),
      .b_i        (prod[gl]),
      .out_valid_o(sum_valid[gl]),
      .y_o        (sum[gl])
    );
  end

  logic       wb_v;
  logic [6:0] wb_idx;

  assign wb_v   = p_v[LWb - 1];
  assign wb_idx = {p_t[LWb - 1], p_b[LWb - 1]};
  assign acc_we = (st == GZero) || wb_v;
  assign acc_wa = (st == GZero) ? zcnt : wb_idx;

  always_comb begin
    for (int l = 0; l < Nl; l++) begin
      acc_wd[l] = (st == GZero) ? 32'd0 : sum[l];
    end
  end

  logic [6:0] fl_idx;

  assign fl_idx = {fl_t, fl_g[4:0]};

  for (genvar gl = 0; gl < Nl; gl++) begin : accl
    (* ram_style = "distributed" *) logic [31:0] accm [128];

    always_ff @(posedge clk_i) begin
      if (acc_we) begin
        accm[acc_wa] <= acc_wd[gl];
      end
    end

    always_ff @(posedge clk_i) begin
      acc_q[gl] <= accm[rd_idx_d[LMul]];
      fl_q[gl]  <= accm[fl_idx];
    end
  end

  always_comb begin
    pipe_empty = 1'b1;
    for (int i = 0; i <= LWb; i++) begin
      if (p_v[i]) begin
        pipe_empty = 1'b0;
      end
    end
  end

  assign out_word_o  = fl_word;
  assign out_valid_o = fl_word_v;
  assign out_addr_o  = dst + 13'(fl_g_d);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      fl_st     <= FRead;
      fl_g      <= '0;
      fl_t      <= '0;
      fl_rd_v   <= 1'b0;
      fl_word_v <= 1'b0;
    end else begin
      fl_rd_v <= 1'b0;
      if (st != GFlush) begin
        fl_st     <= FRead;
        fl_g      <= '0;
        fl_t      <= '0;
        fl_word_v <= 1'b0;
      end else begin
        case (fl_st)
          FRead: begin
            fl_rd_v <= 1'b1;
            fl_t_d  <= fl_t;
            fl_g_d  <= fl_g;
            if (fl_t == 2'd3) begin
              fl_st <= FWait;
            end else begin
              fl_t <= fl_t + 1'b1;
            end
          end
          FWait: begin
            if (fl_word_v && out_ready_i) begin
              fl_word_v <= 1'b0;
              fl_t      <= '0;
              if (fl_g == 6'({kb, 2'b00}) - 1'b1) begin
                fl_st <= FEnd;
              end else begin
                fl_g  <= fl_g + 1'b1;
                fl_st <= FRead;
              end
            end
          end
          default: begin
          end
        endcase
      end
      if (fl_rd_v) begin
        for (int c = 0; c < 8; c++) begin
          fl_word[(fl_t_d * 8 + c) * 32 +: 32] <= fl_q[c];
        end
        if (fl_t_d == 2'd3) begin
          fl_word_v <= 1'b1;
        end
      end
    end
  end

  assign idle_o = (st == GIdle);

  logic unused_gacc;

  assign unused_gacc = ^{cm_y_valid, pf_count, acc_keep, prod_valid, sum_valid};

endmodule
