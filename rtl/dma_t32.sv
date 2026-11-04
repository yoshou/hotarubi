module dma_t32 (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          clear_i,
  input  logic          in_valid_i,
  output logic          in_ready_o,
  input  logic [127:0]  in_data_i,
  output logic          out_valid_o,
  input  logic          out_ready_i,
  output logic [1151:0] out_data_o,
  output logic [1:0]    out_cg_o,
  output logic          out_lo_o,
  output logic          idle_o
);

  logic        fb;
  logic [2:0]  f_w;
  logic [3:0]  f_b;
  logic [1:0]  full;
  (* ram_style = "distributed" *) logic [127:0] buf_mem [128];
  logic [7:0]  e_r [2][32];
  logic [31:0] nz [2];
  logic        in_fire, nzw_v;
  logic [5:0]  nzw_idx;
  logic [127:0] nzw_d;

  assign in_ready_o = !full[fb];

  always_ff @(posedge clk_i) begin
    nzw_v   <= in_fire && (f_b != 4'd8);
    nzw_idx <= {fb, f_w, f_b[2:1]};
    nzw_d   <= in_data_i;
  end

  assign in_fire = in_valid_i && in_ready_o;

  always_ff @(posedge clk_i) begin
    if (in_fire && f_b != 4'd8) begin
      buf_mem[{fb, f_w, f_b[2:0]}] <= in_data_i;
    end
  end

  typedef enum logic [2:0] {
    DIdle,
    DMax1,
    DMax2,
    DRun,
    DFlush,
    DEmit
  } dstate_e;

  dstate_e    ds;
  logic       db, d_h;
  logic [4:0] d_t;
  logic [1:0] d_cnt;
  logic [7:0] emax;
  logic [7:0] m8 [8];
  logic       any_nz;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni || clear_i) begin
      fb    <= 1'b0;
      f_w   <= '0;
      f_b   <= '0;
      full  <= '0;
      nz[0] <= '0;
      nz[1] <= '0;
      ds    <= DIdle;
      db    <= 1'b0;
      d_h   <= 1'b0;
      d_t   <= '0;
      d_cnt <= '0;
    end else begin
      if (in_fire) begin
        if (f_b == 4'd8) begin
          for (int j = 0; j < 4; j++) begin
            e_r[fb][{f_w, j[1:0]}] <= in_data_i[j*8 +: 8];
          end
          f_b <= '0;
          if (f_w == 3'd7) begin
            f_w      <= '0;
            full[fb] <= 1'b1;
            fb       <= ~fb;
          end else begin
            f_w <= f_w + 1'b1;
          end
        end else begin
          f_b <= f_b + 1'b1;
        end
      end
      if (nzw_v && (|nzw_d)) begin
        nz[nzw_idx[5]][nzw_idx[4:0]] <= 1'b1;
      end
      case (ds)
        DIdle: begin
          if (full[db]) begin
            ds <= DMax1;
          end
        end
        DMax1: ds <= DMax2;
        DMax2: begin
          ds  <= DRun;
          d_t <= '0;
        end
        DRun: begin
          d_t <= d_t + 1'b1;
          if (d_t == 5'd31) begin
            ds    <= DFlush;
            d_cnt <= '0;
          end
        end
        DFlush: begin
          d_cnt <= d_cnt + 1'b1;
          if (d_cnt == 2'd2) begin
            ds    <= DEmit;
            d_cnt <= '0;
          end
        end
        DEmit: begin
          if (out_ready_i) begin
            d_cnt <= d_cnt + 1'b1;
            if (d_cnt == 2'd3) begin
              if (!d_h) begin
                d_h <= 1'b1;
                ds  <= DRun;
                d_t <= '0;
              end else begin
                d_h      <= 1'b0;
                ds       <= DIdle;
                full[db] <= 1'b0;
                nz[db]   <= '0;
                db       <= ~db;
              end
            end
          end
        end
        default: ds <= DIdle;
      endcase
    end
  end

  function automatic logic [7:0] max2(input logic [7:0] a, input logic [7:0] b);
    return (a > b) ? a : b;
  endfunction

  always_ff @(posedge clk_i) begin
    if (ds == DMax1) begin
      for (int k = 0; k < 8; k++) begin
        m8[k] <= max2(max2(nz[db][4*k]   ? e_r[db][4*k]   : 8'd0,
                           nz[db][4*k+1] ? e_r[db][4*k+1] : 8'd0),
                      max2(nz[db][4*k+2] ? e_r[db][4*k+2] : 8'd0,
                           nz[db][4*k+3] ? e_r[db][4*k+3] : 8'd0));
      end
      any_nz <= |nz[db];
    end
    if (ds == DMax2) begin
      emax <= any_nz ? max2(max2(max2(m8[0], m8[1]), max2(m8[2], m8[3])),
                            max2(max2(m8[4], m8[5]), max2(m8[6], m8[7]))) : 8'd127;
    end
  end

  logic         a_v, b_v, c1_v, c2_v, a_zero, b_zero, c1_zero;
  logic [6:0]   a_addr;
  logic [4:0]   a_t, b_t, c1_t;
  logic [3:0]   a_s, b_s;
  logic [8:0]   sdiff;
  logic [127:0] b_d;
  logic [15:0]  c1_v16 [16];

  assign sdiff = {1'b0, emax} - {1'b0, e_r[db][d_t]};

  always_ff @(posedge clk_i) begin
    a_v     <= (ds == DRun);
    a_addr  <= {db, d_t, d_h};
    a_t     <= d_t;
    a_zero  <= sdiff[8] | (sdiff[7:0] >= 8'd9) | !nz[db][d_t];
    a_s     <= sdiff[3:0];
    b_v     <= a_v;
    b_t     <= a_t;
    b_s     <= a_s;
    b_zero  <= a_zero;
    b_d     <= buf_mem[a_addr];
    c1_v    <= b_v;
    c1_t    <= b_t;
    c1_zero <= b_zero;
    for (int k = 0; k < 16; k++) begin
      c1_v16[k] <= $signed({b_d[k*8 +: 8], 8'b0}) >>> b_s;
    end
    c2_v <= c1_v;
  end

  logic [1023:0] acc_hi [2], acc_lo [2];

  always_ff @(posedge clk_i) begin
    if (c1_v) begin
      for (int k = 0; k < 16; k++) begin
        logic [7:0] r;
        logic       rnd;
        rnd = c1_v16[k][7] & ((|c1_v16[k][6:0]) | c1_v16[k][8]);
        r   = c1_zero ? 8'd0 : (c1_v16[k][15:8] + 8'(rnd));
        acc_hi[k[3]][(k[2:0] * 128) + (c1_t * 4) +: 4] <= r[7:4];
        acc_lo[k[3]][(k[2:0] * 128) + (c1_t * 4) +: 4] <= r[3:0];
      end
    end
  end

  logic [1023:0] out_sel;

  always_comb begin
    case (d_cnt)
      2'd0:    out_sel = acc_hi[0];
      2'd1:    out_sel = acc_hi[1];
      2'd2:    out_sel = acc_lo[0];
      default: out_sel = acc_lo[1];
    endcase
  end

  assign out_valid_o = (ds == DEmit);
  assign out_data_o  = {64'd0, {8{emax}}, out_sel};
  assign out_cg_o    = {d_h, d_cnt[0]};
  assign out_lo_o    = d_cnt[1];
  assign idle_o      = (ds == DIdle) && (full == 2'b00) && (f_w == 3'd0) && (f_b == 4'd0);

  logic unused_c2_v;

  assign unused_c2_v = c2_v;

endmodule
