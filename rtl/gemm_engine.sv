module gemm_engine #(
  parameter int Mp        = 2,
  parameter int Nt        = 8,
  parameter int Nacc      = 8,
  parameter int AddrWidth = 20,
  parameter int FbLat     = 2
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  input  logic                 cmd_valid_i,
  output logic                 cmd_ready_o,
  input  logic [AddrWidth-1:0] cmd_a_base_i,
  input  logic [9:0]           cmd_a_stride_i,
  input  logic [9:0]           cmd_kb_i,
  input  logic [9:0]           cmd_m_i,
  input  logic [AddrWidth-1:0] cmd_w_base_i,
  input  logic [9:0]           cmd_w_stride_i,
  input  logic [7:0]           cmd_n8_i,
  input  logic [AddrWidth-1:0] cmd_c_base_i,
  input  logic [1:0]           cmd_wmode_i,
  input  logic                 cmd_acc_i,
  output logic                 busy_o,
  output logic [AddrWidth-1:0] abuf_addr_o,
  input  logic [1023:0]        abuf_rdata_i,
  input  logic [31:0]          ascl_rdata_i,
  output logic [AddrWidth-1:0] wbuf_addr_o,
  input  logic [1151:0]        wbuf_rdata_i,
  output logic                 fbuf_rd_req_o,
  output logic [AddrWidth-1:0] fbuf_raddr_o,
  input  logic [1023:0]        fbuf_rdata_i,
  output logic                 fbuf_we_o,
  output logic [AddrWidth-1:0] fbuf_waddr_o,
  output logic [1023:0]        fbuf_wdata_o
);

  logic [AddrWidth-1:0] a_base, w_base, c_base;
  logic [9:0]           kb_count, m_rows, a_stride, w_stride;
  logic [7:0]           n8;
  logic [1:0]           wmode;
  logic                 acc;
  logic                 running;
  logic [7:0]           g;
  logic [4:0]           qg;
  logic [9:0]           kb;
  logic [2:0]           i;
  logic [15:0]          issued, retired;
  logic                 issue_done;
  logic                 issue_v;
  logic [AddrWidth-1:0] a_i, a_qg, w_g, c_i, c_qg;

  assign issue_v     = running & !issue_done;
  assign cmd_ready_o = !running;
  assign busy_o      = running;

  logic last_i, last_kb, last_qg, last_g;

  assign last_i  = (i == 3'd7);
  assign last_kb = (32'(kb) == 32'(kb_count) - 32'd1);
  assign last_qg = (32'(qg) == 32'(10'(m_rows >> 5)) - 32'd1);
  assign last_g  = (32'(g) == 32'(n8) - 32'd1);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      running    <= 1'b0;
      issued     <= '0;
      retired    <= '0;
      issue_done <= 1'b0;
    end else begin
      if (!running) begin
        if (cmd_valid_i) begin
          a_base     <= cmd_a_base_i;
          w_base     <= cmd_w_base_i;
          c_base     <= cmd_c_base_i;
          kb_count   <= cmd_kb_i;
          m_rows     <= cmd_m_i;
          n8         <= cmd_n8_i;
          wmode      <= cmd_wmode_i;
          acc        <= cmd_acc_i;
          a_stride   <= cmd_a_stride_i;
          w_stride   <= cmd_w_stride_i;
          g          <= '0;
          qg         <= '0;
          kb         <= '0;
          i          <= '0;
          issued     <= '0;
          retired    <= '0;
          issue_done <= 1'b0;
          a_i        <= '0;
          a_qg       <= '0;
          w_g        <= '0;
          c_i        <= '0;
          c_qg       <= '0;
          running    <= 1'b1;
        end
      end else if (!issue_done) begin
        if (last_kb) begin
          issued <= issued + 16'd1;
        end
        i   <= i + 3'd1;
        a_i <= a_i + AddrWidth'(a_stride);
        c_i <= c_i + AddrWidth'(n8);
        if (last_i) begin
          a_i <= '0;
          c_i <= '0;
          kb  <= kb + 10'd1;
          if (last_kb) begin
            kb   <= '0;
            qg   <= qg + 5'd1;
            a_qg <= a_qg + (AddrWidth'(a_stride) << 3);
            c_qg <= c_qg + (AddrWidth'(n8) << 3);
            if (last_qg) begin
              qg   <= '0;
              a_qg <= '0;
              c_qg <= '0;
              g    <= g + 8'd1;
              w_g  <= w_g + AddrWidth'(w_stride);
              if (last_g) begin
                issue_done <= 1'b1;
              end
            end
          end
        end
      end else if (retired == issued) begin
        running <= 1'b0;
      end
      if (fbuf_we_o) begin
        retired <= retired + 16'd1;
      end
    end
  end

  always_ff @(posedge clk_i) begin
    abuf_addr_o <= a_base + a_qg + a_i + AddrWidth'(kb);
    wbuf_addr_o <= w_base + w_g + AddrWidth'(kb);
  end

  logic [AddrWidth-1:0] c_addr;

  assign c_addr = c_base + c_qg + c_i + AddrWidth'(g);

  localparam int CmdDelay = 3;

  logic                 vd [CmdDelay], fd [CmdDelay], ld [CmdDelay];
  logic [2:0]           sd [CmdDelay];
  logic [AddrWidth-1:0] td [CmdDelay];
  logic [1:0]           md [CmdDelay];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int k = 0; k < CmdDelay; k++) begin
        vd[k] <= 1'b0;
      end
    end else begin
      vd[0] <= issue_v;
      fd[0] <= (kb == 10'd0);
      ld[0] <= last_kb;
      sd[0] <= i;
      td[0] <= c_addr;
      md[0] <= wmode;
      for (int k = 1; k < CmdDelay; k++) begin
        vd[k] <= vd[k-1];
        fd[k] <= fd[k-1];
        ld[k] <= ld[k-1];
        sd[k] <= sd[k-1];
        td[k] <= td[k-1];
        md[k] <= md[k-1];
      end
    end
  end

  logic                 v1, first1, last1;
  logic [2:0]           sel1;
  logic [AddrWidth-1:0] tag1;
  logic [1:0]           wmode1;

  assign v1     = vd[CmdDelay-1];
  assign first1 = fd[CmdDelay-1];
  assign last1  = ld[CmdDelay-1];
  assign sel1   = sd[CmdDelay-1];
  assign tag1   = td[CmdDelay-1];
  assign wmode1 = md[CmdDelay-1];

  logic [15:0] a_sk [Mp][32];
  logic [3:0]  w_sk [Nt][32];

  for (genvar p = 0; p < Mp; p++) begin : g_skew_a
    logic [511:0] pair_in;

    always_comb begin
      for (int k = 0; k < 32; k++) begin
        pair_in[k*16 +: 16] = {abuf_rdata_i[(2*p+1)*256 + k*8 +: 8],
                               abuf_rdata_i[(2*p)*256 + k*8 +: 8]};
      end
    end

    skew #(
      .Width(16)
    ) u (
      .clk_i,
      .in_i (pair_in),
      .out_o(a_sk[p])
    );
  end

  for (genvar c = 0; c < Nt; c++) begin : g_skew_w
    skew #(
      .Width(4)
    ) u (
      .clk_i,
      .in_i (wbuf_rdata_i[c*128 +: 128]),
      .out_o(w_sk[c])
    );
  end

  logic                 o_valid   [Mp][Nt];
  logic [2:0]           o_sel     [Mp][Nt];
  logic [AddrWidth-1:0] o_tag     [Mp][Nt];
  logic [31:0]          o_acc0    [Mp][Nt], o_acc1 [Mp][Nt];
  logic                 ireq      [Mp][Nt];
  logic [AddrWidth-1:0] itag      [Mp][Nt];
  logic [31:0]          init_word [32];

  for (genvar p = 0; p < Mp; p++) begin : g_row
    for (genvar c = 0; c < Nt; c++) begin : g_col
      bdu2_dsp #(
        .Nacc    (Nacc),
        .SelWidth(3),
        .TagWidth(AddrWidth),
        .InitLead(FbLat + 1)
      ) u_bdu (
        .clk_i,
        .rst_ni,
        .in_valid_i (v1),
        .first_i    (first1),
        .last_i     (last1),
        .acc_sel_i  (sel1),
        .tag_i      (tag1),
        .wmode_i    (wmode1),
        .ea0_i      (ascl_rdata_i[(2*p)*8 +: 8]),
        .ea1_i      (ascl_rdata_i[(2*p+1)*8 +: 8]),
        .ew_i       (wbuf_rdata_i[1024 + c*8 +: 8]),
        .a_sk_i     (a_sk[p]),
        .w_sk_i     (w_sk[c]),
        .init_req_o (ireq[p][c]),
        .init_tag_o (itag[p][c]),
        .init0_i    (init_word[(2*p)*8 + c]),
        .init1_i    (init_word[(2*p+1)*8 + c]),
        .out_valid_o(o_valid[p][c]),
        .out_sel_o  (o_sel[p][c]),
        .out_tag_o  (o_tag[p][c]),
        .out_acc0_o (o_acc0[p][c]),
        .out_acc1_o (o_acc1[p][c])
      );
    end
  end

  assign fbuf_raddr_o  = itag[0][0];
  assign fbuf_rd_req_o = ireq[0][0] & acc;

  always_ff @(posedge clk_i) begin
    for (int j = 0; j < 32; j++) begin
      init_word[j] <= acc ? fbuf_rdata_i[j*32 +: 32] : 32'd0;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      fbuf_we_o <= 1'b0;
    end else begin
      fbuf_we_o    <= o_valid[0][0];
      fbuf_waddr_o <= o_tag[0][0];
      for (int p = 0; p < Mp; p++) begin
        for (int c = 0; c < Nt; c++) begin
          fbuf_wdata_o[((2*p)*8 + c)*32 +: 32]   <= o_acc0[p][c];
          fbuf_wdata_o[((2*p+1)*8 + c)*32 +: 32] <= o_acc1[p][c];
        end
      end
    end
  end

  logic unused_sel, unused_gemm;

  always_comb begin
    unused_sel = 1'b0;
    for (int p = 0; p < Mp; p++) begin
      for (int c = 0; c < Nt; c++) begin
        unused_sel ^= ^o_sel[p][c];
      end
    end
  end

  assign unused_gemm = ^{unused_sel, wbuf_rdata_i[1151:1088], m_rows[4:0]};

endmodule
