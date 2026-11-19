module argmax #(
    parameter  LENGTH = 10,
    parameter  DEPTH  = 32,
    localparam SIZE   = LENGTH * DEPTH
) (
    input wire nrst,
    input wire clk,
    input wire en,
    input wire [SIZE-1:0] arr,
    output reg [$clog2(LENGTH)-1:0] max,
    output reg valid
);
  integer i;
  reg [1:0] state;
  reg [DEPTH-1:0] max_val;
  reg [$clog2(LENGTH)-1:0] max_idx;
  localparam [1:0] IDLE = 2'd0, DONE = 2'd1;
  always @(posedge clk) begin
    if (!nrst) begin
      state <= IDLE;
      valid <= 1'b0;
      max   <= 0;
    end else begin
      case (state)
        IDLE: begin
          valid <= 1'b0;
          if (en) begin
            max_val = arr[0+:DEPTH];
            max_idx = 0;
            for (i = 0; i < LENGTH; i = i + 1) begin
              if ($signed(arr[i*DEPTH+:DEPTH]) > $signed(max_val)) begin
                max_val = arr[i*DEPTH+:DEPTH];
                max_idx = i;
              end
              max <= max_idx;
            end
            state <= DONE;
          end else state <= IDLE;
        end
        DONE: begin
          valid <= 1'b1;
          state <= IDLE;
        end
      endcase
    end
  end
endmodule
