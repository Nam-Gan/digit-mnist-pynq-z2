`default_nettype none
module mnist_core (
    input wire clk,
    input wire nrst,
    input wire wr_en,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,
    input wire start,
    output reg done,
    output reg [3:0] digit
);
  reg [2:0] state;

  // layer 1 signals
  reg start_l1;
  wire [7:0] img_data;
  wire l1_ready;
  wire [9:0] img_addr;
  wire [64*8-1:0] l1_flat;

  // layer 2 signals
  reg start_l2;
  reg [7:0] l1_data;
  wire l2_ready;
  wire [5:0] l1_addr;
  wire [32*8-1:0] l2_flat;

  // layer 3 signals
  reg start_l3;
  reg [7:0] l2_data;
  wire l3_ready;
  wire [4:0] l2_addr;
  wire [10*32-1:0] l3_flat;

  // argmax signals
  reg argmax_en;
  wire [3:0] max_digit;
  wire argmax_valid;

  // instantiations

  img_ram u_img (
      .clk(clk),
      .wr_en(wr_en),
      .wr_addr(wr_addr),
      .wr_data(wr_data),
      .rd_addr(img_addr),
      .rd_data(img_data)
  );

  argmax #(
      .LENGTH(10),
      .DEPTH (32)
  ) u_argmax (
      .nrst(nrst),
      .clk(clk),
      .en(argmax_en),
      .arr(l3_flat),
      .max(max_digit),
      .valid(argmax_valid)
  );

  dense_layer #(
      .in_depth(8),
      .prev_neurons(784),
      .activation_depth(8),
      .neurons(64),
      .SHIFT(12),
      .W_FILE("mem/l1_w.mem"),
      .LAST(0),
      .B_FILE("mem/l1_b.mem")
  ) layer1 (
      .clk(clk),
      .nrst(nrst),
      .start(start_l1),
      .in_data(img_data),
      .ready_q(l1_ready),
      .in_addr(img_addr),
      .activation_flat(l1_flat)
  );
  dense_layer #(
      .in_depth(8),
      .prev_neurons(64),
      .activation_depth(8),
      .neurons(32),
      .SHIFT(8),
      .W_FILE("mem/l2_w.mem"),
      .LAST(0),
      .B_FILE("mem/l2_b.mem")
  ) layer2 (
      .clk(clk),
      .nrst(nrst),
      .start(start_l2),
      .in_data(l1_data),
      .ready_q(l2_ready),
      .in_addr(l1_addr),
      .activation_flat(l2_flat)
  );
  dense_layer #(
      .in_depth(8),
      .prev_neurons(32),
      .activation_depth(8),
      .neurons(10),
      .LAST(1),
      .W_FILE("mem/l3_w.mem"),
      .B_FILE("mem/l3_b.mem")
  ) layer3 (
      .clk(clk),
      .nrst(nrst),
      .start(start_l3),
      .in_data(l2_data),
      .in_addr(l2_addr),
      .activation_flat(l3_flat),
      .ready_q(l3_ready)
  );

  localparam [2:0] IDLE = 3'd0, L1_ACTIVATE = 3'd1, L2_ACTIVATE = 3'd2, L3_ACTIVATE = 3'd3, DONE = 3'd4;
  always @(posedge clk) begin
    l1_data <= l1_flat[l1_addr*8+:8];
    l2_data <= l2_flat[l2_addr*8+:8];
  end
  always @(posedge clk) begin
    if (!nrst) begin
      state <= IDLE;
      start_l1 <= 1'b0;
      start_l2 <= 1'b0;
      start_l3 <= 1'b0;
      argmax_en <= 1'b0;
      done <= 1'b0;
    end else begin
      case (state)
        IDLE: begin
          done <= 1'b0;
          argmax_en <= 1'b0;
          if (start) begin
            state <= L1_ACTIVATE;
            start_l1 <= 1'b1;
          end else state <= IDLE;
        end
        L1_ACTIVATE: begin
          start_l1 <= 1'b0;
          if (!l1_ready) begin
            state <= L1_ACTIVATE;
          end else begin
            state <= L2_ACTIVATE;
            start_l2 <= 1'b1;
          end
        end
        L2_ACTIVATE: begin
          start_l2 <= 1'b0;
          if (!l2_ready) begin
            state <= L2_ACTIVATE;
          end else begin
            state <= L3_ACTIVATE;
            start_l3 <= 1'b1;
          end
        end
        L3_ACTIVATE: begin
          start_l3 <= 1'b0;
          if (!l3_ready) begin
            state <= L3_ACTIVATE;
          end else begin
            state <= DONE;
            argmax_en <= 1'b1;
          end
        end
        DONE: begin
          argmax_en <= 1'b0;
          if (!argmax_valid) begin
            done  <= 1'b0;
            state <= DONE;
          end else begin
            done  <= 1'b1;
            digit <= max_digit;
            state <= IDLE;
          end
        end
      endcase
    end
  end
endmodule
