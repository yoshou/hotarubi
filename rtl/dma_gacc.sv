module dma_gacc (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          clear_i,
  input  logic          start_i,
  input  logic [11:0]   cfg_slots_i,
  input  logic [3:0]    cfg_kb_i,
  input  logic [19:0]   cfg_stride_i,
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
  output logic          idle_o,
  output logic [1:0]    null_o
);

  localparam int Nl     = 16;
  localparam int LMul   = 6;
  localparam int LAdd   = 6;
  localparam int LWb    = 2 + LMul + LAdd;
  localparam int MinGap = LWb + 3;

  logic [11:0] slots;
  logic [3:0]  kb;
  logic        do_flush;
  logic [12:0] dst;
  logic [19:0] w_stride;
  logic [31:0] w_grp0, pq_addr;
  logic [4:0]  nd;
  logic [5:0]  row_beats;

  assign nd        = {kb, 1'b0};
  assign row_beats = 6'(nd) + 6'd1;

  typedef enum logic [3:0] {
    GIdle,
    GZero,
    GHdr,
    GEnt,
    GDrain,
    GFlush
  } st_e;

  st_e          st;
  logic         zeroing, flushing;
  logic [11:0]  s_cur;
  logic [1:0]   e_t, e_k;
  logic         e_half;
  logic [3:0]   pq_kb, rx_kb;
  logic [2:0]   rx_b;
  logic [8:0]   ngrp, cur_g, f_g, g_fsm, f_g_new;
  logic         pq_on, rx_on, rx_half, nx_ready, wh_cur, wh_cm, cur_v;
  logic         ent_req, pq_fire, w_fire, w_here, need, sw, od, pfx, f_start, dbl;
  logic         wr_we;
  logic [6:0]   wr_wa, wr_ra;
  logic [31:0]  wr_rd [4];
  logic [5:0]   zcnt;
  logic [3:0]   drain_cnt;
  logic         cm_busy, cm_pend;
  logic [2:0]   cm_i;
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
  logic [3:0]   mac_b;
  logic [7:0]   gap [4];
  logic         p_v [LWb + 1];
  logic [1:0]   p_t [LWb + 1];
  logic [3:0]   p_b [LWb + 1];
  logic [127:0] md_r;
  logic [31:0]  mc_r;
  logic [7:0]   me_r;
  logic [31:0]  xf [Nl], prod [Nl], acc_q [Nl], sum [Nl], fl_q [Nl];
  logic [31:0]  cs, cs_d;
  logic [9:0]   cs_e;
  logic         acc_we;
  logic [5:0]   acc_wa;
  logic [31:0]  acc_wd [Nl];
  logic         fl_rd_v, fl_rd_v2, fl_word_v;
  logic [5:0]   fl_g, fl_g_d;
  logic [1:0]   fl_t, fl_t_d, fl_t_d2;
  logic [1023:0] fl_word;

  typedef enum logic [1:0] {
    FRead,
    FWait,
    FEnd
  } fl_e;

  fl_e fl_st;

  logic [63:0] ent_cur;
  logic        ent_null, ent_ok, ent_take;

  assign g_fsm       = s_cur[11:3];
  assign w_here      = cur_v && (cur_g == g_fsm);
  assign ent_cur     = e_half ? ent_data_i[127:64] : ent_data_i[63:0];
  assign ent_null    = (ent_cur[31:0] == 32'hFFFFFFFF);
  assign dbl         = (st == GEnt) && ent_valid_i && !e_half && ent_null &&
                       (ent_data_i[95:64] == 32'hFFFFFFFF);
  assign ent_ok      = (st == GEnt) && ent_valid_i && !dbl &&
                       (ent_null || (w_here && !cm_busy && !cm_pend && !pf_afull && req_ready_i));
  assign ent_take    = ent_ok;
  assign ent_ready_o = (st == GHdr) ? ent_valid_i : ((ent_take && e_half) || dbl);

  assign ent_req = (st == GEnt) && ent_valid_i && !dbl && !ent_null && w_here && !cm_busy &&
                   !cm_pend && !pf_afull;
  assign pq_fire = pq_on && !ent_req && req_ready_i;
  assign need    = (st == GEnt) && ent_valid_i && !dbl && !ent_null && !w_here;
  assign sw      = need && nx_ready && (f_g == g_fsm);
  assign od      = need && !pq_on && !rx_on && !(nx_ready && (f_g == g_fsm));
  assign pfx     = sw && (g_fsm + 9'd1 < ngrp);
  assign f_start = od || pfx;
  assign f_g_new = od ? g_fsm : (g_fsm + 9'd1);

  always_comb begin
    req_valid_o = 1'b0;
    req_addr_o  = '0;
    req_beats_o = '0;
    req_tag_o   = 2'd0;
    if (ent_req) begin
      req_valid_o = 1'b1;
      req_addr_o  = {ent_cur[27:0], 4'b0};
      req_beats_o = 16'(row_beats);
      req_tag_o   = 2'd0;
    end else if (pq_on) begin
      req_valid_o = 1'b1;
      req_tag_o   = 2'd2;
      req_beats_o = 16'd8;
      req_addr_o  = pq_addr;
    end
  end

  assign w_ready_o = rx_on && !(cm_busy && (wh_cm == rx_half));
  assign w_fire    = w_valid_i && w_ready_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      pq_on    <= 1'b0;
      rx_on    <= 1'b0;
      nx_ready <= 1'b0;
      cur_v    <= 1'b0;
    end else begin
      if (sw) begin
        cur_v    <= 1'b1;
        nx_ready <= 1'b0;
      end
      if (f_start) begin
        pq_on    <= 1'b1;
        rx_on    <= 1'b1;
        nx_ready <= 1'b0;
      end else begin
        if (pq_fire && pq_kb == kb - 1'b1) begin
          pq_on <= 1'b0;
        end
        if (w_fire && rx_b == 3'd7 && rx_kb == kb - 1'b1) begin
          rx_on    <= 1'b0;
          nx_ready <= 1'b1;
        end
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (st == GHdr && ent_valid_i) begin
      w_grp0 <= {ent_data_i[27:0], 4'b0};
      ngrp   <= 9'((slots + 12'd7) >> 3);
    end
    if (sw) begin
      wh_cur <= rx_half;
      cur_g  <= f_g;
    end
    if (f_start) begin
      f_g     <= f_g_new;
      pq_addr <= w_grp0 + {16'd0, f_g_new, 7'd0};
      pq_kb   <= '0;
      rx_kb   <= '0;
      rx_b    <= '0;
      rx_half <= od ? ~wh_cur : wh_cur;
    end else begin
      if (pq_fire) begin
        pq_kb   <= pq_kb + 1'b1;
        pq_addr <= pq_addr + 32'({w_stride, 4'b0});
      end
      if (w_fire) begin
        rx_b <= rx_b + 1'b1;
        if (rx_b == 3'd7) begin
          rx_kb <= rx_kb + 1'b1;
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      st        <= GIdle;
      s_cur     <= '0;
      e_t       <= '0;
      e_k       <= '0;
      e_half    <= 1'b0;
      zcnt      <= '0;
      drain_cnt <= '0;
      zeroing   <= 1'b0;
      flushing  <= 1'b0;
    end else begin
      case (st)
        GIdle: begin
          if (start_i) begin
            slots     <= cfg_slots_i;
            kb        <= cfg_kb_i;
            w_stride  <= cfg_stride_i;
            do_flush  <= cfg_flush_i;
            dst       <= cfg_dst_i;
            s_cur     <= '0;
            e_t       <= '0;
            e_k       <= '0;
            e_half    <= 1'b0;
            zcnt      <= '0;
            drain_cnt <= '0;
            st        <= cfg_acc_i ? GHdr : GZero;
            zeroing   <= !cfg_acc_i;
          end
        end
        GZero: begin
          zcnt <= zcnt + 1'b1;
          if (zcnt == 6'd63) begin
            st      <= GHdr;
            zeroing <= 1'b0;
          end
        end
        GHdr: begin
          if (ent_valid_i) begin
            st <= GEnt;
          end
        end
        GEnt: begin
          if (dbl || ent_take) begin
            if (!dbl) begin
              e_half <= ~e_half;
            end
            if (e_k == 2'd3 || (dbl && e_k == 2'd2)) begin
              e_k <= '0;
              if (e_t == 2'd3) begin
                e_t   <= '0;
                s_cur <= s_cur + 1'b1;
                if (s_cur == slots - 1'b1) begin
                  st <= GDrain;
                end
              end else begin
                e_t <= e_t + 1'b1;
              end
            end else begin
              e_k <= dbl ? e_k + 2'd2 : e_k + 1'b1;
            end
          end
        end
        GDrain: begin
          if (pf_empty && rfull == 2'b00 && !mac_busy && pipe_empty && !cm_busy && !cm_pend &&
              !pq_on && !rx_on) begin
            drain_cnt <= drain_cnt + 1'b1;
            if (drain_cnt == 4'd3) begin
              drain_cnt <= '0;
              st        <= do_flush ? GFlush : GIdle;
              flushing  <= do_flush;
            end
          end else begin
            drain_cnt <= '0;
          end
        end
        GFlush: begin
          if (fl_st == FEnd) begin
            st       <= GIdle;
            flushing <= 1'b0;
          end
        end
        default: begin
          st       <= GIdle;
          zeroing  <= 1'b0;
          flushing <= 1'b0;
        end
      endcase
    end
  end

  assign wr_we = w_fire;
  assign wr_wa = {rx_half, rx_kb[2:0], rx_b};
  assign wr_ra = {wh_cm, cm_i, cm_t, cm_col[2]};

  for (genvar gb = 0; gb < 4; gb++) begin : wbank
    (* ram_style = "distributed" *) logic [31:0] mem [128];

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
    end else begin
      if (ent_take && !ent_null) begin
        cm_busy <= 1'b1;
        cm_pend <= 1'b1;
        cm_i    <= '0;
        cm_cnt  <= '0;
      end else if (cm_busy) begin
        cm_i <= cm_i + 1'b1;
        if (cm_i == 3'd7) begin
          cm_busy <= 1'b0;
        end
      end
      if (cm_y_valid) begin
        cm_cnt <= cm_cnt + 1'b1;
      end
      if (pf_push) begin
        cm_pend <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (!clear_i && ent_take && !ent_null) begin
      cm_bw  <= ent_cur[63:32];
      cm_t   <= e_t;
      cm_col <= s_cur[2:0];
      wh_cm  <= wh_cur;
    end
    if (!clear_i && cm_y_valid) begin
      cm_c[cm_cnt[2:0] * 32 +: 32] <= cm_y;
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
        if (row_last) begin
          rb_b           <= '0;
          rfull[rb_fill] <= 1'b1;
          rb_fill        <= ~rb_fill;
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
    if (!clear_i && row_fire && rb_b >= nd) begin
      rscale[rb_fill] <= row_data_i;
    end
    if (!clear_i && row_last) begin
      rinfo[rb_fill] <= pf_dout;
    end
  end

  logic [1:0] mac_t;

  assign mac_t       = rinfo[rb_mac][257:256];
  assign mac_start   = !mac_busy && rfull[rb_mac] && (gap[mac_t] == 8'd0) && !flushing &&
                       !zeroing;
  assign mac_row_end = mac_busy && (mac_b == 4'(nd - 5'd1));

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

  assign mac_kb = 4'(mac_b[3:1]);
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

  assign mac_data = rbuf[{rb_mac, mac_b}];

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
      for (int i = 1; i <= LWb; i++) begin
        p_v[i] <= p_v[i-1];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    p_t[0] <= mac_t;
    p_b[0] <= mac_b;
    for (int i = 1; i <= LWb; i++) begin
      p_t[i] <= p_t[i-1];
      p_b[i] <= p_b[i-1];
    end
  end

  always_ff @(posedge clk_i) begin
    md_r <= mac_data;
    mc_r <= mac_c;
    me_r <= mac_e;
    for (int l = 0; l < Nl; l++) begin
      xf[l] <= i8_to_f32(md_r[8 * l +: 8]);
    end
    cs_d <= cs;
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
  logic [5:0] wb_idx;

  assign wb_v   = p_v[LWb - 1];
  assign wb_idx = {p_t[LWb - 1], p_b[LWb - 1]};
  assign acc_we = zeroing || wb_v;
  assign acc_wa = zeroing ? zcnt : wb_idx;

  always_comb begin
    for (int l = 0; l < Nl; l++) begin
      acc_wd[l] = zeroing ? 32'd0 : sum[l];
    end
  end

  logic [5:0] fl_idx;

  assign fl_idx = {fl_t, fl_g[4:1]};

  (* ram_style = "block" *) logic [Nl*32-1:0] accm [64];
  logic [Nl*32-1:0] acc_wd_w, acc_rd_m, acc_rd;
  logic [5:0]       acc_ra;

  assign acc_ra = flushing ? fl_idx : {p_t[LMul - 1], p_b[LMul - 1]};

  always_comb begin
    for (int l = 0; l < Nl; l++) begin
      acc_wd_w[l*32 +: 32] = acc_wd[l];
      acc_q[l]             = acc_rd[l*32 +: 32];
      fl_q[l]              = acc_rd[l*32 +: 32];
    end
  end

  always_ff @(posedge clk_i) begin
    if (acc_we) begin
      accm[acc_wa] <= acc_wd_w;
    end
    acc_rd_m <= accm[acc_ra];
    acc_rd   <= acc_rd_m;
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
      fl_rd_v2  <= 1'b0;
      fl_word_v <= 1'b0;
    end else begin
      fl_rd_v  <= 1'b0;
      fl_rd_v2 <= fl_rd_v;
      if (!flushing) begin
        fl_st     <= FRead;
        fl_g      <= '0;
        fl_t      <= '0;
        fl_word_v <= 1'b0;
      end else begin
        case (fl_st)
          FRead: begin
            fl_rd_v <= 1'b1;
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
      if (fl_rd_v2 && fl_t_d2 == 2'd3) begin
        fl_word_v <= 1'b1;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    fl_t_d2 <= fl_t_d;
    if (!clear_i && flushing && fl_st == FRead) begin
      fl_t_d <= fl_t;
      fl_g_d <= fl_g;
    end
    if (!clear_i && fl_rd_v2) begin
      for (int c = 0; c < 8; c++) begin
        fl_word[(fl_t_d2 * 8 + c) * 32 +: 32] <= fl_q[{fl_g_d[0], 3'(c)}];
      end
    end
  end

  assign idle_o = (st == GIdle);
  assign null_o = dbl ? 2'd2 : {1'b0, ent_take && ent_null};

  logic unused_gacc;

  assign unused_gacc = ^{pf_count, prod_valid, sum_valid};

endmodule
