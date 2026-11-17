module mnist_core (
    input wire clk,
    input wire nrst,
    input wire wr_en,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,
    input wire start,
    output wire done,
    output wire [3:0] digit
);
  reg [7:0] img_buf[0:783];
  reg [1:0] state;
  reg [1:0] n_state;

  // layer 1 signals
  reg start_l1;
  reg [7:0] img_data;
  wire l1_ready;
  wire [9:0] img_addr;
  reg [64*8-1:0] l1_accs;

  // layer 2 signals
  reg start_l2;
  reg [7:0] l1_data;
  wire l2_ready;
  wire l2_addr;
  reg [32*8-1:0] l2_accs;

  // layer 3 signals
  reg start_l3;
  reg [7:0] l2_data;
  wire l3_ready;
  wire l2_addr;
  reg [10*32-1:0] l3_accs;

  // instantiations

  dense_layer #(
      .in_depth(8),
      .prev_neurons(784),
      .activation_depth(8),
      .neurons(64),
      .SHIFT(12),
      .W_FILE("mem/l1_w.mem"),
      .LAST(0),
      .B_FILE("mem/l1_b.mem")) layer1 (
      .clk(clk),
      .nrst(nrst),
      .start(start_l1),
      .in_data(img_data),
      .ready_q(l1_ready),
      .in_addr(img_addr),
      .activation_flat(l1_accs));
  dense_layer #(
      .in_depth(8),
      .prev_neurons(64),
      .activation_depth(8),
      .neurons(32),
      .SHIFT(8),
      .W_FILE("mem/l2_w.mem"),
      .LAST(0),
      .B_FILE("mem/l2_b.mem")) layer2 (
      .clk(clk),
      .nrst(nrst),
      .start(start_l2),
      .in_data(l1_data),
      .ready_q(l2_ready),
      .in_addr(l1_addr),
      .activation_flat(l2_accs));
  dense_layer #(
      .in_depth(8),
      .prev_neurons(32),
      .activation_depth(8),
      .neurons(10),
      .LAST(1),
      .W_FILE("mem/l3_w.mem"),
      .B_FILE("mem/l3_b.mem")) layer3 (
      .clk(clk),
      .nrst(nrst),
      .start(start_l3),
      .in_data(l2_data),
      .in_addr(l2_addr),
      .activation_flat(l3_accs));

  localparam [1:0] IDLE = 2'b00, IMAGE_GET = 2'b01, BUSY = 2'd3;
  always @(posedge clk) begin
      if (!nrst) begin
          n_state <= IDLE;
          state <= IDLE;
      end
      else
      begin
          state <= n_state;
      end
  end
  always @(*) begin
      case(state)
          IDLE:
              if (start) n_state <= 
endmodule
