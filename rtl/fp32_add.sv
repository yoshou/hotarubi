module fp32_add (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        in_valid_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  output logic        out_valid_o,
  output logic [31:0] y_o
);

  logic        sa, sb;
  logic [7:0]  ea, eb;
  logic [23:0] ma, mb;

  assign sa = a_i[31];
  assign sb = b_i[31];
  assign ea = a_i[30:23];
  assign eb = b_i[30:23];
  assign ma = (ea == 8'd0) ? 24'd0 : {1'b1, a_i[22:0]};
  assign mb = (eb == 8'd0) ? 24'd0 : {1'b1, b_i[22:0]};

  logic a_ge_b;

  assign a_ge_b = ({ea, ma} >= {eb, mb});

  logic        v1, sub1, sbig1;
  logic [7:0]  ebig1;
  logic [8:0]  d1;
  logic [23:0] mbig1, msmall1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v1 <= 1'b0;
    end else begin
      v1 <= in_valid_i;
      if (a_ge_b) begin
        sbig1   <= sa;
        sub1    <= sa ^ sb;
        ebig1   <= ea;
        mbig1   <= ma;
        msmall1 <= mb;
        d1      <= {1'b0, ea} - {1'b0, eb};
      end else begin
        sbig1   <= sb;
        sub1    <= sa ^ sb;
        ebig1   <= eb;
        mbig1   <= mb;
        msmall1 <= ma;
        d1      <= {1'b0, eb} - {1'b0, ea};
      end
    end
  end

  logic [26:0] msmall_ext, msmall_sh;
  logic        sticky;

  assign msmall_ext = {msmall1, 3'b000};

  always_comb begin
    if (msmall1 == 24'd0) begin
      msmall_sh = 27'd0;
      sticky    = 1'b0;
    end else if (d1 > 9'd26) begin
      msmall_sh = 27'd0;
      sticky    = 1'b1;
    end else begin
      msmall_sh = msmall_ext >> d1[4:0];
      sticky    = |(msmall_ext & ((27'd1 << d1[4:0]) - 27'd1));
    end
  end

  logic        v2, sub2, sbig2;
  logic [7:0]  ebig2;
  logic [26:0] mbig2, msmall2;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v2 <= 1'b0;
    end else begin
      v2      <= v1;
      sub2    <= sub1;
      sbig2   <= sbig1;
      ebig2   <= ebig1;
      mbig2   <= {mbig1, 3'b000};
      msmall2 <= msmall_sh | {26'd0, sticky};
    end
  end

  logic        v3, sbig3;
  logic [7:0]  ebig3;
  logic [27:0] sum3;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v3 <= 1'b0;
    end else begin
      v3    <= v2;
      sbig3 <= sbig2;
      ebig3 <= ebig2;
      sum3  <= sub2 ? ({1'b0, mbig2} - {1'b0, msmall2}) : ({1'b0, mbig2} + {1'b0, msmall2});
    end
  end

  logic [4:0] lzc;

  always_comb begin
    lzc = 5'd0;
    for (int i = 27; i >= 0; i--) begin
      if (sum3[i]) begin
        lzc = 5'(27 - i);
        break;
      end
    end
  end

  logic        v4, sbig4, zero4;
  logic [7:0]  ebig4;
  logic [27:0] sum4;
  logic [4:0]  lzc4;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v4 <= 1'b0;
    end else begin
      v4    <= v3;
      sbig4 <= sbig3;
      ebig4 <= ebig3;
      zero4 <= (sum3 == 28'd0);
      sum4  <= sum3;
      lzc4  <= lzc;
    end
  end

  logic        v5, sbig5, zero5;
  logic [27:0] norm5;
  logic [9:0]  e_adj5;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v5 <= 1'b0;
    end else begin
      v5     <= v4;
      sbig5  <= sbig4;
      zero5  <= zero4;
      norm5  <= sum4 << lzc4;
      e_adj5 <= {2'b00, ebig4} + 10'd1 - {5'd0, lzc4};
    end
  end

  logic [23:0] mant_r;
  logic        round_up;
  logic        unused_norm5;

  assign unused_norm5 = norm5[27];

  assign round_up = norm5[3] & ((|norm5[2:0]) | norm5[4]);
  assign mant_r   = {1'b0, norm5[26:4]} + {23'd0, round_up};

  logic [9:0]  e_fin;
  logic [22:0] m_fin;

  always_comb begin
    if (mant_r[23]) begin
      e_fin = e_adj5 + 10'd1;
      m_fin = 23'd0;
    end else begin
      e_fin = e_adj5;
      m_fin = mant_r[22:0];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      out_valid_o <= 1'b0;
    end else begin
      out_valid_o <= v5;
      if (zero5) begin
        y_o <= 32'd0;
      end else if ($signed(e_fin) <= 10'sd0) begin
        y_o <= 32'd0;
      end else if (e_fin >= 10'd255) begin
        y_o <= {sbig5, 8'hFF, 23'd0};
      end else begin
        y_o <= {sbig5, e_fin[7:0], m_fin};
      end
    end
  end

endmodule
