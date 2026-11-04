module dma_rowfmt (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          clear_i,
  input  logic          cfg_unpack_i,
  input  logic          cfg_abuf_i,
  input  logic [5:0]    cfg_tok_i,
  input  logic          in_valid_i,
  output logic          in_ready_o,
  input  logic [127:0]  in_data_i,
  output logic          out_valid_o,
  input  logic          out_ready_i,
  output logic [1151:0] out_word_o,
  input  logic          win_valid_i,
  output logic          win_ready_o,
  input  logic [1151:0] win_word_i,
  output logic          bout_valid_o,
  input  logic          bout_ready_i,
  output logic [127:0]  bout_data_o,
  output logic          idle_o
);

  logic [6:0] nw;
  logic [1:0] nsc;
  logic [7:0] nd, tbeats;

  assign nw     = cfg_abuf_i ? 7'(cfg_tok_i) : 7'({cfg_tok_i, 2'b00});
  assign nsc    = cfg_abuf_i ? ((cfg_tok_i > 6'd16) ? 2'd2 : 2'd1) : 2'd0;
  assign nd     = 8'({nw, 1'b0});
  assign tbeats = nd + 8'(nsc);

  logic          p_q;
  logic [1:0]    p_j;
  logic [7:0]    p_b;
  logic [1:0]    full;
  logic          o_q;
  logic [6:0]    o_w;
  logic [1:0]    o_j;
  logic          o_busy, o_rd_v, o_rd_last, o_word_v, out_v, o_pend;
  logic [1:0]    o_rd_j;
  logic [1023:0] o_acc;
  logic [1151:0] out_r;
  logic          u_q, u_busy;
  logic [6:0]    u_w;
  logic [1:0]    u_j;
  logic [1151:0] u_word;
  logic [1:0]    ufull;
  logic          v_q, v_busy, v_rd_v, v_last_d, v_is_data_d;
  logic [1:0]    v_j, v_j_d;
  logic [7:0]    v_b, v_b_d;
  logic [2:0]    credit;
  logic          bf_push, bf_pop, bf_empty, bf_afull;
  logic [127:0]  bf_din, bf_dout;
  logic          sc_we [16];
  logic [3:0]    sc_wa, sc_ra;
  logic [7:0]    sc_wd [16], sc_rd [16];
  logic [7:0]    o_sc [4];
  logic          wl_en, wh_en;
  logic [7:0]    w_addr, r_addr;
  logic [127:0]  wl_data, wh_data, rl_data, rh_data;

  (* ram_style = "block" *) logic [127:0] buf_lo [256];
  (* ram_style = "block" *) logic [127:0] buf_hi [256];

  always_ff @(posedge clk_i) begin
    if (wl_en) begin
      buf_lo[w_addr] <= wl_data;
    end
    if (wh_en) begin
      buf_hi[w_addr] <= wh_data;
    end
    rl_data <= buf_lo[r_addr];
    rh_data <= buf_hi[r_addr];
  end

  for (genvar gs = 0; gs < 16; gs++) begin : scb
    (* ram_style = "distributed" *) logic [7:0] mem [16];

    always_ff @(posedge clk_i) begin
      if (sc_we[gs]) begin
        mem[sc_wa] <= sc_wd[gs];
      end
    end

    assign sc_rd[gs] = mem[sc_ra];
  end

  function automatic logic [7:0] baddr(input logic q, input logic [1:0] j, input logic [4:0] w);
    return {q, j, w};
  endfunction

  logic in_fire, in_is_data, in_tok_end;

  assign in_ready_o = !cfg_unpack_i && !full[p_q];
  assign in_fire    = in_valid_i && in_ready_o;
  assign in_is_data = (p_b < nd);
  assign in_tok_end = in_fire && (p_b == tbeats - 1'b1);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      p_q <= 1'b0;
      p_j <= '0;
      p_b <= '0;
    end else if (in_fire) begin
      if (p_b == tbeats - 1'b1) begin
        p_b <= '0;
        if (p_j == 2'd3) begin
          p_j <= '0;
          p_q <= ~p_q;
        end else begin
          p_j <= p_j + 1'b1;
        end
      end else begin
        p_b <= p_b + 1'b1;
      end
    end
  end

  logic o_start, o_issue;

  assign o_start     = !cfg_unpack_i && !o_busy && !o_pend && full[o_q];
  assign o_issue     = o_busy && !(o_word_v && out_v && !out_ready_i);
  assign out_valid_o = out_v;
  assign out_word_o  = out_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      full      <= '0;
      o_q       <= 1'b0;
      o_w       <= '0;
      o_j       <= '0;
      o_busy    <= 1'b0;
      o_rd_v    <= 1'b0;
      o_rd_last <= 1'b0;
      o_word_v  <= 1'b0;
      out_v     <= 1'b0;
      o_pend    <= 1'b0;
    end else begin
      if (in_tok_end && p_j == 2'd3 && !cfg_unpack_i) begin
        full[p_q] <= 1'b1;
      end
      o_rd_v    <= 1'b0;
      o_rd_last <= 1'b0;
      if (o_start) begin
        o_busy <= 1'b1;
        o_pend <= 1'b1;
        o_j    <= '0;
      end else if (o_issue) begin
        o_rd_v    <= 1'b1;
        o_rd_j    <= o_j;
        o_rd_last <= (o_j == 2'd3);
        o_j       <= o_j + 1'b1;
        if (o_j == 2'd3) begin
          o_busy <= 1'b0;
        end
      end
      if (o_rd_v) begin
        o_acc[o_rd_j * 256 +: 256] <= {rh_data, rl_data};
        o_sc[o_rd_j]               <= sc_rd[o_w[3:0]];
        if (o_rd_last) begin
          o_word_v <= 1'b1;
        end
      end
      if (o_word_v && (!out_v || out_ready_i)) begin
        o_word_v       <= 1'b0;
        o_pend         <= 1'b0;
        out_v          <= 1'b1;
        out_r          <= '0;
        out_r[1023:0]  <= o_acc;
        if (cfg_abuf_i) begin
          for (int j = 0; j < 4; j++) begin
            out_r[1024 + 8 * j +: 8] <= o_sc[j];
          end
        end
        if (o_w == nw - 1'b1) begin
          o_w       <= '0;
          full[o_q] <= 1'b0;
          o_q       <= ~o_q;
        end else begin
          o_w <= o_w + 1'b1;
        end
      end else if (out_v && out_ready_i) begin
        out_v <= 1'b0;
      end
    end
  end

  logic win_fire;

  assign win_ready_o = cfg_unpack_i && !u_busy && !ufull[u_q];
  assign win_fire    = win_valid_i && win_ready_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      u_q    <= 1'b0;
      u_w    <= '0;
      u_j    <= '0;
      u_busy <= 1'b0;
      ufull  <= '0;
    end else begin
      if (win_fire) begin
        u_word <= win_word_i;
        u_busy <= 1'b1;
        u_j    <= '0;
      end else if (u_busy) begin
        u_j <= u_j + 1'b1;
        if (u_j == 2'd3) begin
          u_busy <= 1'b0;
          if (u_w == nw - 1'b1) begin
            u_w        <= '0;
            ufull[u_q] <= 1'b1;
            u_q        <= ~u_q;
          end else begin
            u_w <= u_w + 1'b1;
          end
        end
      end
      if (v_rd_v && v_last_d) begin
        ufull[v_q] <= 1'b0;
      end
    end
  end

  logic [2:0] bf_count;

  sfifo #(
    .Width      (128),
    .Depth      (4),
    .AfullMargin(1)
  ) u_bf (
    .clk_i,
    .rst_ni,
    .clear_i,
    .push_i (bf_push),
    .din_i  (bf_din),
    .pop_i  (bf_pop),
    .dout_o (bf_dout),
    .empty_o(bf_empty),
    .afull_o(bf_afull),
    .count_o(bf_count)
  );

  logic v_is_data, v_issue;

  assign v_is_data    = (v_b < nd);
  assign v_issue      = cfg_unpack_i && ufull[v_q] && (credit != 3'd0) && !(v_rd_v && v_last_d);
  assign bout_valid_o = !bf_empty;
  assign bout_data_o  = bf_dout;
  assign bf_pop       = bout_valid_o && bout_ready_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      v_q      <= 1'b0;
      v_j      <= '0;
      v_b      <= '0;
      v_rd_v   <= 1'b0;
      v_busy   <= 1'b0;
      v_last_d <= 1'b0;
      credit   <= 3'd4;
    end else begin
      credit   <= credit - 3'(v_issue) + 3'(bf_pop);
      v_rd_v   <= v_issue;
      v_last_d <= 1'b0;
      if (v_issue) begin
        v_is_data_d <= v_is_data;
        v_b_d       <= v_b;
        v_j_d       <= v_j;
        v_busy      <= 1'b1;
        if (v_b == tbeats - 1'b1) begin
          v_b <= '0;
          if (v_j == 2'd3) begin
            v_j      <= '0;
            v_last_d <= 1'b1;
          end else begin
            v_j <= v_j + 1'b1;
          end
        end else begin
          v_b <= v_b + 1'b1;
        end
      end
      if (v_rd_v && v_last_d) begin
        v_busy <= 1'b0;
        v_q    <= ~v_q;
      end
    end
  end

  always_comb begin
    bf_push = v_rd_v;
    bf_din  = '0;
    if (v_is_data_d) begin
      bf_din = v_b_d[0] ? rh_data : rl_data;
    end else begin
      for (int i = 0; i < 16; i++) begin
        if (int'(32'(v_b_d) - 32'(nd)) * 16 + i < int'(cfg_tok_i)) begin
          bf_din[8 * i +: 8] = sc_rd[i];
        end
      end
    end
  end

  always_comb begin
    if (!cfg_unpack_i) begin
      sc_wa = {p_q, p_j, p_b[0]};
      sc_ra = {o_q, o_rd_j, o_w[4]};
      for (int i = 0; i < 16; i++) begin
        sc_we[i] = in_fire && !in_is_data && cfg_abuf_i;
        sc_wd[i] = in_data_i[8 * i +: 8];
      end
    end else begin
      sc_wa = {u_q, u_j, u_w[4]};
      sc_ra = {v_q, v_j_d, v_b_d[0]};
      for (int i = 0; i < 16; i++) begin
        sc_we[i] = u_busy && cfg_abuf_i && (u_w[3:0] == 4'(i));
        sc_wd[i] = u_word[1024 + 8 * u_j +: 8];
      end
    end
  end

  always_comb begin
    if (!cfg_unpack_i) begin
      wl_en   = in_fire && in_is_data && !p_b[0];
      wh_en   = in_fire && in_is_data && p_b[0];
      w_addr  = baddr(p_q, p_j, p_b[5:1]);
      wl_data = in_data_i;
      wh_data = in_data_i;
      r_addr  = baddr(o_q, o_j, o_w[4:0]);
    end else begin
      wl_en   = u_busy;
      wh_en   = u_busy;
      w_addr  = baddr(u_q, u_j, u_w[4:0]);
      wl_data = u_word[u_j * 256 +: 128];
      wh_data = u_word[u_j * 256 + 128 +: 128];
      r_addr  = baddr(v_q, v_j, v_b[5:1]);
    end
  end

  assign idle_o = (full == 2'b00) && (ufull == 2'b00) && !o_busy && !o_pend && !out_v &&
                  !u_busy && !v_busy && bf_empty && (p_b == 8'd0) && (p_j == 2'd0) &&
                  (u_w == 7'd0);

  logic unused_rowfmt;

  assign unused_rowfmt = ^{bf_afull, bf_count};

endmodule
