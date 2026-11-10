module vec_transpose #(
  parameter int ReadLatency = 3
) (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          start_i,
  input  logic [11:0]   src_i,
  input  logic [11:0]   dst_i,
  input  logic [8:0]    tiles_i,
  input  logic [11:0]   src_tile_stride_i,
  input  logic [11:0]   src_word_stride_i,
  input  logic [11:0]   dst_tile_stride_i,
  input  logic [11:0]   dst_word_stride_i,
  input  logic [1:0]    rotation_i,
  output logic          busy_o,
  output logic          rd_req_o,
  input  logic          rd_grant_i,
  output logic [11:0]   rd_addr_o,
  input  logic [1023:0] rd_data_i,
  output logic          wr_req_o,
  input  logic          wr_grant_i,
  output logic [11:0]   wr_addr_o,
  output logic [1023:0] wr_data_o
);

  logic [127:0] tile_read [8];
  logic [1:0] reserved, full;
  logic [11:0] tile_dst [2];
  logic [11:0] src_base, dst_base, read_addr;
  logic [11:0] src_tile_stride, src_word_stride, dst_tile_stride, dst_word_stride;
  logic [8:0] tiles, read_tile;
  logic       read_bank, write_bank, all_read;
  logic [1:0] rotation;
  logic [2:0] read_row, write_row;
  logic running, read_fire, write_fire, load_output, output_valid;
  logic [1023:0] tile_data;
  logic output_last, output_bank;

  typedef struct packed {
    logic valid;
    logic bank;
    logic [2:0] row;
  } tag_t;

  tag_t tags [ReadLatency];

  assign busy_o = running;
  assign rd_req_o = running && !all_read &&
                    ((read_row != 3'd0) || !reserved[read_bank]);
  assign rd_addr_o = read_addr;
  assign wr_req_o = output_valid;
  assign load_output = running && full[write_bank] &&
                       (!output_valid || wr_grant_i);
  assign read_fire = rd_req_o && rd_grant_i;
  assign write_fire = wr_req_o && wr_grant_i;

  for (genvar b = 0; b < 8; b++) begin : g_tile
    (* ram_style = "distributed" *) logic [127:0] mem [16];
    logic [127:0] write_data;
    logic [2:0] column_index, read_row_index;
    assign column_index = 3'(b) - tags[ReadLatency-1].row;
    assign read_row_index = 3'(b) - write_row;
    for (genvar t = 0; t < 4; t++) begin : g_matrix
      assign write_data[t*32 +: 32] = rd_data_i[(t*8+32'(column_index))*32 +: 32];
    end
    always_ff @(posedge clk_i) begin
      if (tags[ReadLatency-1].valid) begin
        mem[{tags[ReadLatency-1].bank, tags[ReadLatency-1].row}] <= write_data;
      end
    end
    assign tile_read[b] = mem[{write_bank, read_row_index}];
  end

  always_comb begin
    for (int t = 0; t < 4; t++) begin
      for (int f = 0; f < 8; f++) begin
        logic [2:0] column;
        case (rotation)
          2'd1:    column = {1'(f), 2'(f >> 1)};
          2'd2:    column = {2'(f), 1'(f >> 2)};
          default: column = 3'(f);
        endcase
        tile_data[(t*8+f)*32 +: 32] =
            tile_read[3'(write_row + column)][t*32 +: 32];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (load_output) begin
      wr_data_o <= tile_data;
      wr_addr_o <= (write_row == 3'd0) ? tile_dst[write_bank] :
                   wr_addr_o + dst_word_stride;
      output_last <= (write_row == 3'd7);
      output_bank <= write_bank;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      running <= 1'b0;
      output_valid <= 1'b0;
      reserved <= '0;
      full <= '0;
      all_read <= 1'b1;
      read_bank <= 1'b0;
      write_bank <= 1'b0;
      read_row <= '0;
      write_row <= '0;
      for (int i = 0; i < ReadLatency; i++) begin
        tags[i] <= '0;
      end
    end else begin
      tags[0] <= '{valid: read_fire, bank: read_bank, row: read_row};
      for (int i = 1; i < ReadLatency; i++) begin
        tags[i] <= tags[i-1];
      end
      if (start_i && !running) begin
        src_base <= src_i;
        dst_base <= dst_i;
        read_addr <= src_i;
        tiles <= tiles_i;
        src_tile_stride <= src_tile_stride_i;
        src_word_stride <= src_word_stride_i;
        dst_tile_stride <= dst_tile_stride_i;
        dst_word_stride <= dst_word_stride_i;
        rotation <= rotation_i;
        read_tile <= '0;
        read_row <= '0;
        write_row <= '0;
        read_bank <= 1'b0;
        write_bank <= 1'b0;
        reserved <= '0;
        full <= '0;
        all_read <= (tiles_i == 9'd0);
        running <= (tiles_i != 9'd0);
        output_valid <= 1'b0;
      end else if (running) begin
        if (read_fire) begin
          if (read_row == 3'd0) begin
            reserved[read_bank] <= 1'b1;
            tile_dst[read_bank] <= dst_base;
          end
          read_row <= read_row + 3'd1;
          if (read_row == 3'd7) begin
            read_bank <= !read_bank;
            read_tile <= read_tile + 9'd1;
            src_base <= src_base + src_tile_stride;
            read_addr <= src_base + src_tile_stride;
            dst_base <= dst_base + dst_tile_stride;
            if (read_tile == tiles - 9'd1) begin
              all_read <= 1'b1;
            end
          end else begin
            read_addr <= read_addr + src_word_stride;
          end
        end
        if (tags[ReadLatency-1].valid && tags[ReadLatency-1].row == 3'd7) begin
          full[tags[ReadLatency-1].bank] <= 1'b1;
        end
        if (load_output) begin
          output_valid <= 1'b1;
          write_row <= write_row + 3'd1;
          if (write_row == 3'd7) begin
            write_bank <= !write_bank;
          end
        end else if (write_fire) begin
          output_valid <= 1'b0;
        end
        if (write_fire && output_last) begin
          reserved[output_bank] <= 1'b0;
          full[output_bank] <= 1'b0;
        end
        if (all_read && reserved == 2'b00 && !output_valid) begin
          running <= 1'b0;
        end
      end
    end
  end

endmodule
