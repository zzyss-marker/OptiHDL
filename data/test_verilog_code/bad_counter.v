module bad_counter(
    input wire clk,
    input wire reset,
    input wire enable,
    output wire [7:0] count_out,
    output wire overflow
);

// 糟糕的设计选择1: 使用过多的中间寄存器
reg [7:0] count_reg;
reg [7:0] count_temp1;
reg [7:0] count_temp2;
reg [7:0] count_temp3;
reg [7:0] count_temp4;
reg enable_reg1, enable_reg2, enable_reg3;
reg reset_reg1, reset_reg2;

// 糟糕的设计选择2: 创建并使用分频时钟（会被综合但效率低）
reg clk_div2;
reg [7:0] secondary_counter;
always @(posedge clk) begin
    if (reset) begin
        clk_div2 <= 1'b0;
        secondary_counter <= 8'b0;
    end else begin
        clk_div2 <= ~clk_div2;
        if (clk_div2) 
            secondary_counter <= secondary_counter + 1'b1;
    end
end

// 糟糕的设计选择3: 过度的流水线，增加不必要的延迟
always @(posedge clk) begin
    if (reset) begin
        enable_reg1 <= 1'b0;
        enable_reg2 <= 1'b0;
        enable_reg3 <= 1'b0;
        reset_reg1 <= 1'b0;
        reset_reg2 <= 1'b0;
    end else begin
        enable_reg1 <= enable;
        enable_reg2 <= enable_reg1;
        enable_reg3 <= enable_reg2;
        reset_reg1 <= 1'b0;  // 固定为0，避免被优化
        reset_reg2 <= reset_reg1;
    end
end

// 糟糕的设计选择4: 手工实现加法器，增加逻辑深度
wire [7:0] next_count;
wire [8:0] carry_chain;  // 包含进位链

assign carry_chain[0] = 1'b1;  // 加1的进位输入

// 全加器实现每一位
assign next_count[0] = count_reg[0] ^ carry_chain[0];
assign carry_chain[1] = count_reg[0] & carry_chain[0];

assign next_count[1] = count_reg[1] ^ carry_chain[1];
assign carry_chain[2] = count_reg[1] & carry_chain[1];

assign next_count[2] = count_reg[2] ^ carry_chain[2];
assign carry_chain[3] = count_reg[2] & carry_chain[2];

assign next_count[3] = count_reg[3] ^ carry_chain[3];
assign carry_chain[4] = count_reg[3] & carry_chain[3];

assign next_count[4] = count_reg[4] ^ carry_chain[4];
assign carry_chain[5] = count_reg[4] & carry_chain[4];

assign next_count[5] = count_reg[5] ^ carry_chain[5];
assign carry_chain[6] = count_reg[5] & carry_chain[5];

assign next_count[6] = count_reg[6] ^ carry_chain[6];
assign carry_chain[7] = count_reg[6] & carry_chain[6];

assign next_count[7] = count_reg[7] ^ carry_chain[7];
assign carry_chain[8] = count_reg[7] & carry_chain[7];

// 糟糕的设计选择5: 不必要的多级寄存器流水线
always @(posedge clk) begin
    if (reset) begin
        count_temp1 <= 8'b0;
        count_temp2 <= 8'b0;
        count_temp3 <= 8'b0;
        count_temp4 <= 8'b0;
    end else if (enable_reg3) begin
        count_temp1 <= next_count;
        count_temp2 <= count_temp1;
        count_temp3 <= count_temp2;
        count_temp4 <= count_temp3;
    end
end

// 主计数器寄存器
always @(posedge clk) begin
    if (reset) begin
        count_reg <= 8'b0;
    end else if (enable_reg3) begin
        count_reg <= count_temp4;
    end
end

// 糟糕的设计选择6: 复杂的溢出检测逻辑
wire [7:0] overflow_detect_stage1, overflow_detect_stage2;
wire overflow_temp1, overflow_temp2, overflow_temp3;

// 用复杂的组合逻辑检测0xFF
assign overflow_detect_stage1 = count_reg | 8'h00;  // 无意义操作但会被综合
assign overflow_detect_stage2 = overflow_detect_stage1 & 8'hFF;

assign overflow_temp1 = &overflow_detect_stage2[3:0];  // 检测低4位全1
assign overflow_temp2 = &overflow_detect_stage2[7:4];  // 检测高4位全1
assign overflow_temp3 = overflow_temp1 & overflow_temp2 & enable_reg3;

// 糟糕的设计选择7: 不必要的输出寄存器链
reg [7:0] output_reg1, output_reg2, output_reg3;
reg overflow_reg1, overflow_reg2, overflow_reg3;

always @(posedge clk) begin
    if (reset) begin
        output_reg1 <= 8'b0;
        output_reg2 <= 8'b0;
        output_reg3 <= 8'b0;
        overflow_reg1 <= 1'b0;
        overflow_reg2 <= 1'b0;
        overflow_reg3 <= 1'b0;
    end else begin
        output_reg1 <= count_reg;
        output_reg2 <= output_reg1;
        output_reg3 <= output_reg2;
        overflow_reg1 <= overflow_temp3;
        overflow_reg2 <= overflow_reg1;
        overflow_reg3 <= overflow_reg2;
    end
end

// 糟糕的设计选择8: 冗余但会被综合的逻辑
reg [15:0] wide_register;
reg [7:0] redundant_counter;
wire [7:0] xor_pattern;

// 这些逻辑会影响输出，所以不会被完全优化掉
assign xor_pattern = secondary_counter ^ count_reg;

always @(posedge clk) begin
    if (reset) begin
        wide_register <= 16'b0;
        redundant_counter <= 8'b0;
    end else begin
        wide_register <= {secondary_counter, count_reg};
        redundant_counter <= xor_pattern;
    end
end

// 糟糕的设计选择9: 混合输出逻辑
wire [7:0] mixed_output;
assign mixed_output = output_reg3 ^ (redundant_counter & 8'h01);  // 轻微影响输出

assign count_out = mixed_output;
assign overflow = overflow_reg3 | wide_register[15];  // 让wide_register影响输出

endmodule
