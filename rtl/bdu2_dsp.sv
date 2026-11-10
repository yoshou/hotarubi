module bdu2_dsp
  import isa_pkg::*;
#(
  parameter int Nacc     = 8,
  parameter int SelWidth = 3,
  parameter int TagWidth = 16,
  parameter int InitLead = 2
) (
  input  logic                clk_i,
  input  logic                rst_ni,
  input  logic                in_valid_i,
  input  logic                first_i,
  input  logic                last_i,
  input  logic [SelWidth-1:0] acc_sel_i,
  input  logic [TagWidth-1:0] tag_i,
  input  logic [1:0]          wmode_i,
  input  logic [7:0]          ea0_i,
  input  logic [7:0]          ea1_i,
  input  logic [7:0]          ew_i,
  input  logic [15:0]         a_sk_i [32],
  input  logic [3:0]          w_sk_i [32],
  output logic                init_req_o,
  output logic [TagWidth-1:0] init_tag_o,
  input  logic [31:0]         init0_i,
  input  logic [31:0]         init1_i,
  output logic                out_valid_o,
  output logic [TagWidth-1:0] out_tag_o,
  output logic [31:0]         out_acc0_o,
  output logic [31:0]         out_acc1_o
);

  localparam int Ndsp = 32;
  localparam int Lat  = 4 + Ndsp - 1;

  function automatic logic signed [4:0] decode_w(input logic [3:0] c, input logic [1:0] mode);
    logic [3:0] mag;
    case (mode)
      WeightModeFp4: begin
        case (c[2:0])
          3'd0:    mag = 4'd0;
          3'd1:    mag = 4'd1;
          3'd2:    mag = 4'd2;
          3'd3:    mag = 4'd3;
          3'd4:    mag = 4'd4;
          3'd5:    mag = 4'd6;
          3'd6:    mag = 4'd8;
          default: mag = 4'd12;
        endcase
        decode_w = c[3] ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
      end
      WeightModeW8Hi: decode_w = $signed({c[3], c});
      default:        decode_w = $signed({1'b0, c});
    endcase
  endfunction

  function automatic logic signed [9:0] sbias(input logic [1:0] mode);
    case (mode)
      WeightModeFp4:  sbias = -10'sd7;
      WeightModeW8Hi: sbias = -10'sd8;
      default:        sbias = -10'sd12;
    endcase
  endfunction

  logic [1:0] wm [Ndsp];

  always_ff @(posedge clk_i) begin
    wm[0] <= wmode_i;
    for (int k = 1; k < Ndsp; k++) begin
      wm[k] <= wm[k-1];
    end
  end

  logic signed [26:0] a_r [Ndsp], d_r [Ndsp], ad_r [Ndsp];
  logic signed [17:0] b_r [Ndsp], b2_r [Ndsp];
  logic signed [44:0] m_r [Ndsp];
  logic signed [47:0] p_r [Ndsp];

  always_ff @(posedge clk_i) begin
    for (int k = 0; k < Ndsp; k++) begin
      a_r[k]  <= 27'($signed(a_sk_i[k][15:8])) <<< 18;
      d_r[k]  <= 27'($signed(a_sk_i[k][7:0]));
      b_r[k]  <= 18'(decode_w(w_sk_i[k], (k == 0) ? wmode_i : wm[k-1]));
      ad_r[k] <= a_r[k] + d_r[k];
      b2_r[k] <= b_r[k];
      m_r[k]  <= ad_r[k] * b2_r[k];
    end
    p_r[0] <= 48'(m_r[0]);
    for (int k = 1; k < Ndsp; k++) begin
      p_r[k] <= 48'(m_r[k]) + p_r[k-1];
    end
  end

  logic                vq [Lat], fq [Lat], lq [Lat];
  logic [SelWidth-1:0] sq [Lat];
  logic [TagWidth-1:0] tq [Lat];
  logic signed [9:0]   e0q [Lat], e1q [Lat];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < Lat; i++) begin
        vq[i] <= 1'b0;
      end
    end else begin
      vq[0] <= in_valid_i;
      for (int i = 1; i < Lat; i++) begin
        vq[i] <= vq[i-1];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    fq[0]  <= first_i;
    lq[0]  <= last_i;
    sq[0]  <= acc_sel_i;
    tq[0]  <= tag_i;
    e0q[0] <= $signed({2'b00, ea0_i}) + $signed({2'b00, ew_i}) - 10'sd254 + sbias(wmode_i);
    e1q[0] <= $signed({2'b00, ea1_i}) + $signed({2'b00, ew_i}) - 10'sd254 + sbias(wmode_i);
    for (int i = 1; i < Lat; i++) begin
      fq[i]  <= fq[i-1];
      lq[i]  <= lq[i-1];
      sq[i]  <= sq[i-1];
      tq[i]  <= tq[i-1];
      e0q[i] <= e0q[i-1];
      e1q[i] <= e1q[i-1];
    end
  end

  assign init_req_o = vq[Lat-1-InitLead] & fq[Lat-1-InitLead];
  assign init_tag_o = tq[Lat-1-InitLead];

  logic [31:0] init0_a, init1_a, init0_b, init1_b, init0_f, init1_f;

  always_ff @(posedge clk_i) begin
    init0_a <= init0_i;
    init1_a <= init1_i;
    init0_b <= init0_a;
    init1_b <= init1_a;
    init0_f <= init0_b;
    init1_f <= init1_b;
  end

  logic signed [47:0] pf;
  logic signed [17:0] s0;
  logic signed [18:0] s1;

  assign pf = p_r[Ndsp-1];
  assign s0 = pf[17:0];
  assign s1 = 19'((pf - 48'(s0)) >>> 18);

  logic                va, vb, vf, fa, la;
  logic [SelWidth-1:0] sa;
  logic [TagWidth-1:0] ta;
  logic                sg0, sg1;
  logic [16:0]         mg0, mg1;
  logic signed [9:0]   ex0a, ex1a;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      va <= 1'b0;
      vb <= 1'b0;
      vf <= 1'b0;
    end else begin
      va <= vq[Lat-1];
      vb <= va;
      vf <= vb;
    end
  end

  always_ff @(posedge clk_i) begin
    fa   <= fq[Lat-1];
    la   <= lq[Lat-1];
    sa   <= sq[Lat-1];
    ta   <= tq[Lat-1];
    sg0  <= s0[17];
    mg0  <= s0[17] ? 17'(-s0) : 17'(s0);
    sg1  <= s1[18];
    mg1  <= s1[18] ? 17'(-s1) : 17'(s1);
    ex0a <= e0q[Lat-1];
    ex1a <= e1q[Lat-1];
  end

  function automatic logic [4:0] msb_of(input logic [16:0] mag);
    msb_of = 5'd0;
    for (int i = 16; i >= 0; i--) begin
      if (mag[i]) begin
        msb_of = 5'(i);
        break;
      end
    end
  endfunction

  function automatic logic [31:0] pack_f(input logic sgn, input logic [16:0] mag,
                                         input logic [4:0] msb, input logic signed [9:0] ex,
                                         input logic zero);
    logic [22:0] mant;
    mant = 23'(24'(mag) << (5'd23 - msb));
    if (zero || ex <= 10'sd0) begin
      pack_f = 32'd0;
    end else if (ex >= 10'sd255) begin
      pack_f = {sgn, 8'hFF, 23'd0};
    end else begin
      pack_f = {sgn, ex[7:0], mant[22:0]};
    end
  endfunction

  logic                fb, lb, z0b, z1b, sg0b, sg1b;
  logic [SelWidth-1:0] sb;
  logic [TagWidth-1:0] tb;
  logic [16:0]         mg0b, mg1b;
  logic [4:0]          ms0b, ms1b;
  logic signed [9:0]   ex0b, ex1b;

  always_ff @(posedge clk_i) begin
    fb   <= fa;
    lb   <= la;
    sb   <= sa;
    tb   <= ta;
    sg0b <= sg0;
    sg1b <= sg1;
    mg0b <= mg0;
    mg1b <= mg1;
    z0b  <= (mg0 == 17'd0);
    z1b  <= (mg1 == 17'd0);
    ms0b <= msb_of(mg0);
    ms1b <= msb_of(mg1);
    ex0b <= ex0a + $signed({5'd0, msb_of(mg0)}) + 10'sd127;
    ex1b <= ex1a + $signed({5'd0, msb_of(mg1)}) + 10'sd127;
  end

  logic                ff, lf;
  logic [SelWidth-1:0] sf;
  logic [TagWidth-1:0] tf;
  logic [31:0]         f0, f1;

  always_ff @(posedge clk_i) begin
    ff <= fb;
    lf <= lb;
    sf <= sb;
    tf <= tb;
    f0 <= pack_f(sg0b, mg0b, ms0b, ex0b, z0b);
    f1 <= pack_f(sg1b, mg1b, ms1b, ex1b, z1b);
  end

  logic [31:0]         acc0 [Nacc], acc1 [Nacc];
  logic [31:0]         acc0_rd, acc1_rd, acc0_wd, acc1_wd;
  logic                acc0_we, acc1_we;
  logic [SelWidth-1:0] acc_wa;
  logic                add_valid0, add_valid1;
  logic [31:0]         y0, y1;

  always_ff @(posedge clk_i) begin
    if (acc0_we) begin
      acc0[acc_wa] <= acc0_wd;
    end
    if (acc1_we) begin
      acc1[acc_wa] <= acc1_wd;
    end
    acc0_rd <= (acc0_we && acc_wa == sb) ? acc0_wd : acc0[sb];
    acc1_rd <= (acc1_we && acc_wa == sb) ? acc1_wd : acc1[sb];
  end

  fp32_add u_add0 (
    .clk_i,
    .rst_ni,
    .in_valid_i (vf),
    .a_i        (ff ? init0_f : acc0_rd),
    .b_i        (f0),
    .out_valid_o(add_valid0),
    .y_o        (y0)
  );

  fp32_add u_add1 (
    .clk_i,
    .rst_ni,
    .in_valid_i (vf),
    .a_i        (ff ? init1_f : acc1_rd),
    .b_i        (f1),
    .out_valid_o(add_valid1),
    .y_o        (y1)
  );

  logic [SelWidth-1:0] s5, s6, s7, s8, s9, s10;
  logic [TagWidth-1:0] t5, t6, t7, t8, t9, t10;
  logic                l5, l6, l7, l8, l9, l10;

  always_ff @(posedge clk_i) begin
    s5  <= sf;
    s6  <= s5;
    s7  <= s6;
    s8  <= s7;
    s9  <= s8;
    s10 <= s9;
    t5  <= tf;
    t6  <= t5;
    t7  <= t6;
    t8  <= t7;
    t9  <= t8;
    t10 <= t9;
    l5  <= lf;
    l6  <= l5;
    l7  <= l6;
    l8  <= l7;
    l9  <= l8;
    l10 <= l9;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      out_valid_o <= 1'b0;
    end else begin
      out_valid_o <= add_valid0 & l10;
    end
  end

  always_ff @(posedge clk_i) begin
    acc0_we    <= add_valid0;
    acc1_we    <= add_valid1;
    acc_wa     <= s10;
    acc0_wd    <= y0;
    acc1_wd    <= y1;
    out_tag_o  <= t10;
    out_acc0_o <= y0;
    out_acc1_o <= y1;
  end

endmodule
