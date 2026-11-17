module dense_layer #(
    parameter in_depth         = 8,
    parameter prev_neurons     = 784,
    parameter activation_depth = 8,
    parameter neurons          = 64,
    parameter SHIFT            = 8,
    parameter W_FILE           = "mem/l1_w.mem",
    parameter LAST             = 0,
    parameter OW               = LAST ? 32 : activation_depth,
    parameter B_FILE           = "mem/l1_b.mem"
) (
    input wire clk,
    input wire nrst,
    input wire start,
    input wire [in_depth-1:0] in_data,
    output reg ready_q,
    output reg [$clog2(prev_neurons)-1:0] in_addr,
    output reg [neurons*OW-1:0] activation_flat
);

  reg clr, inc, store;
  reg load, en, en_q, ready;
  wire [OW-1:0] activation[neurons-1:0];
  wire [neurons*8-1:0] w_row;
  reg [1:0] n_state;
  reg [1:0] state;
  wire signed [31:0] acc[neurons-1:0];
  reg signed [31:0] bias[0:neurons-1];

  initial $readmemh(B_FILE, bias);

  genvar i;
  generate
    for (i = 0; i < neurons; i = i + 1) begin : neuron_gen
      neuron #(
          .in_depth(in_depth),
          .weight_depth(8),
          .bias_depth(32)
      ) neuron_i (
          .clk(clk),
          .load(load),
          .en(en_q),
          .bias(bias[i]),
          .w(w_row[8*(i+1)-1:8*i]),
          .x(in_data),
          .acc(acc[i])
      );
      if (LAST) begin : g_raw
        assign activation[i] = acc[i];
      end else begin : g_req
        requant #(
            .SHIFT(SHIFT),
            .activation_depth(activation_depth)
        ) requant_i (
            .acc(acc[i]),
            .activation(activation[i])
        );
      end
    end
  endgenerate

  weight_rom #(
      .prev_neurons(prev_neurons),
      .neurons(neurons),
      .FILE(W_FILE)
  ) w_l1 (
      .clk (clk),
      .addr(in_addr),
      .row (w_row)
  );

  localparam [1:0] IDLE = 2'b00, BUSY = 2'b01, DRAIN = 2'd2, STORE = 2'd3;
  always @(posedge clk) begin
    if (!nrst) begin
      state   <= IDLE;
      en_q    <= 1'b0;
      ready_q <= 1'b0;
      in_addr <= 0;
    end else begin
      state   <= n_state;
      en_q    <= en;
      ready_q <= ready;
      if (clr) in_addr <= 0;
      else if (inc) in_addr <= in_addr + 1'b1;
      if (store) for (k = 0; k < neurons; k = k + 1) activation_flat[k*OW+:OW] <= activation[k];
    end
  end

  always @(*) begin
    n_state = state;
    case (state)
      IDLE: if (start) n_state = BUSY;
      BUSY: if (in_addr == prev_neurons - 1) n_state = DRAIN;
      DRAIN: n_state = STORE;
      STORE: n_state = IDLE;
      default: n_state = IDLE;
    endcase
  end

  integer k;
  always @(*) begin
    load = 1'b0;
    en = 1'b0;
    clr = 1'b0;
    inc = 1'b0;
    store = 1'b0;
    ready = 1'b0;
    case (state)
      IDLE: begin
        load = 1'b1;
        clr  = 1'b1;
      end
      BUSY: begin
        en  = 1'b1;
        inc = 1'b1;
      end
      STORE: begin
        store = 1'b1;
        ready = 1'b1;
      end
      default: ;
    endcase
  end
endmodule
