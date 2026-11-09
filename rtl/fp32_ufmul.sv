module fp32_ufmul (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        in_valid_i,
  input  logic [31:0] s_i,
  input  logic [15:0] frac_i,
  output logic        out_valid_o,
  output logic [31:0] y_o
);

  logic       v1, v2, v3, sg1, z1;
  logic [7:0] e1;
  (* use_dsp = "yes" *) logic [39:0] p1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v1          <= 1'b0;
      v2          <= 1'b0;
      v3          <= 1'b0;
      out_valid_o <= 1'b0;
    end else begin
      v1          <= in_valid_i;
      v2          <= v1;
      v3          <= v2;
      out_valid_o <= v3;
    end
  end

  always_ff @(posedge clk_i) begin
    sg1 <= s_i[31];
    e1  <= s_i[30:23];
    z1  <= (s_i[30:23] == 8'd0) | (frac_i == 16'd0);
    p1  <= {1'b1, s_i[22:0]} * frac_i;
  end

  logic        sg2, z2;
  logic [7:0]  e2;
  logic [39:0] p2;
  logic [5:0]  lzc2;
  logic [5:0]  lzc;

  always_comb begin
    lzc = 6'd40;
    for (int i = 39; i >= 0; i--) begin
      if (p1[i]) begin
        lzc = 6'(39 - i);
        break;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    sg2  <= sg1;
    z2   <= z1;
    e2   <= e1;
    p2   <= p1;
    lzc2 <= lzc;
  end

  logic        sg3, z3;
  logic [39:0] pn3;
  logic [9:0]  e3;

  always_ff @(posedge clk_i) begin
    sg3 <= sg2;
    z3  <= z2;
    pn3 <= p2 << lzc2;
    e3  <= {2'b00, e2} - {4'd0, lzc2};
  end

  logic        round_up;
  logic [24:0] mant_r;
  logic [9:0]  e_fin;
  logic [22:0] m_fin;
  logic        unused_mant_r;

  assign round_up      = pn3[15] & ((|pn3[14:0]) | pn3[16]);
  assign mant_r        = {1'b0, pn3[39:16]} + {24'd0, round_up};
  assign unused_mant_r = mant_r[23];

  always_comb begin
    if (mant_r[24]) begin
      e_fin = e3 + 10'd1;
      m_fin = 23'd0;
    end else begin
      e_fin = e3;
      m_fin = mant_r[22:0];
    end
  end

  always_ff @(posedge clk_i) begin
    if (z3) begin
      y_o <= 32'd0;
    end else if ($signed(e_fin) <= 10'sd0) begin
      y_o <= 32'd0;
    end else if (e_fin >= 10'd255) begin
      y_o <= {sg3, 8'hFF, 23'd0};
    end else begin
      y_o <= {sg3, e_fin[7:0], m_fin};
    end
  end

endmodule
