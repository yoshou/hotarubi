module fp32_mul (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        in_valid_i,
  input  logic [31:0] a_i,
  input  logic [31:0] b_i,
  output logic        out_valid_o,
  output logic [31:0] y_o
);

  logic        v1, s1, z1;
  logic [23:0] ma1, mb1;
  logic [8:0]  esum1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v1 <= 1'b0;
    end else begin
      v1    <= in_valid_i;
      s1    <= a_i[31] ^ b_i[31];
      z1    <= (a_i[30:23] == 8'd0) | (b_i[30:23] == 8'd0);
      ma1   <= {1'b1, a_i[22:0]};
      mb1   <= {1'b1, b_i[22:0]};
      esum1 <= {1'b0, a_i[30:23]} + {1'b0, b_i[30:23]};
    end
  end

  (* use_dsp = "yes" *) logic [47:0] p2, p3, p4;
  logic       v2, s2, z2, v3, s3, z3, v4a, s4a, z4a;
  logic [8:0] esum2, esum3, esum4a;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v2  <= 1'b0;
      v3  <= 1'b0;
      v4a <= 1'b0;
    end else begin
      v2     <= v1;
      s2     <= s1;
      z2     <= z1;
      esum2  <= esum1;
      p2     <= ma1 * mb1;
      v3     <= v2;
      s3     <= s2;
      z3     <= z2;
      esum3  <= esum2;
      p3     <= p2;
      v4a    <= v3;
      s4a    <= s3;
      z4a    <= z3;
      esum4a <= esum3;
      p4     <= p3;
    end
  end

  logic        v4, s4, z4, sticky4;
  logic [24:0] m4;
  logic [9:0]  e4;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v4 <= 1'b0;
    end else begin
      v4 <= v4a;
      s4 <= s4a;
      z4 <= z4a;
      if (p4[47]) begin
        m4      <= p4[47:23];
        sticky4 <= |p4[22:0];
        e4      <= {1'b0, esum4a} - 10'd126;
      end else begin
        m4      <= p4[46:22];
        sticky4 <= |p4[21:0];
        e4      <= {1'b0, esum4a} - 10'd127;
      end
    end
  end

  logic        round_up;
  logic [24:0] mant_r;
  logic [9:0]  e_fin;
  logic [22:0] m_fin;
  logic        unused_mant_r;

  assign round_up      = m4[0] & (sticky4 | m4[1]);
  assign mant_r        = {1'b0, m4[24:1]} + {24'd0, round_up};
  assign unused_mant_r = mant_r[23];

  always_comb begin
    if (mant_r[24]) begin
      e_fin = e4 + 10'd1;
      m_fin = 23'd0;
    end else begin
      e_fin = e4;
      m_fin = mant_r[22:0];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      out_valid_o <= 1'b0;
    end else begin
      out_valid_o <= v4;
      if (z4) begin
        y_o <= 32'd0;
      end else if ($signed(e_fin) <= 10'sd0) begin
        y_o <= 32'd0;
      end else if (e_fin >= 10'd255) begin
        y_o <= {s4, 8'hFF, 23'd0};
      end else begin
        y_o <= {s4, e_fin[7:0], m_fin};
      end
    end
  end

endmodule
