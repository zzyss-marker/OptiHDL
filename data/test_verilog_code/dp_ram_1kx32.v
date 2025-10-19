//==============================================================================
// File Name    : dp_ram_1kx32.v
// Description  : True Dual Port RAM with Write-First Mode and Byte Enable
// Date         : 2024-03-06
// Version      : v3.5.2
// Target       : Xilinx Artix-7 FPGA, 200MHz+ timing closure
// Synthesis    : < 180 LUTs estimated
//==============================================================================
// Features:
// - True dual port with independent A/B interfaces
// - 1024x32 bit capacity using 4x256x32 sub-blocks
// - Write-first mode for read/write collision handling
// - Byte-wise write enable (4 bytes per 32-bit word)
// - Single clock domain, synchronous reset
// - Configurable pipeline stages for timing optimization
//==============================================================================

module dp_ram_1kx32 #(
    // Memory Configuration Parameters
    parameter DEPTH         = 1024,           // Total memory depth
    parameter WIDTH         = 32,             // Data width in bits
    parameter ADDR_WIDTH    = 10,             // Address width (log2(DEPTH))
    parameter SUB_DEPTH     = 256,            // Sub-block depth
    parameter SUB_ADDR_WIDTH = 8,             // Sub-block address width
    parameter NUM_BLOCKS    = 4,              // Number of sub-blocks
    
    // Performance Tuning Parameters
    parameter PIPELINE_STAGES = 0,            // 0=combinational, 1=registered output
    parameter WRITE_FIRST   = 1,              // 1=write-first, 0=read-first
    parameter CONFLICT_CHECK = 1,             // Enable address conflict detection
    
    // Synthesis Optimization Parameters
    parameter RAM_STYLE     = "block",        // "block", "distributed", "auto"
    parameter INIT_FILE     = "",             // Memory initialization file
    parameter RESET_VALUE   = 32'h00000000    // Reset value for output registers
)(
    // Clock and Reset Interface
    input  wire                 clk,          // System clock
    input  wire                 rst_n,        // Active-low synchronous reset
    
    // Port A Interface
    input  wire [ADDR_WIDTH-1:0] addr_a,      // Port A address [9:0]
    input  wire                 ena,          // Port A enable
    input  wire                 wea,          // Port A write enable
    input  wire [3:0]           bea,          // Port A byte enable [3:0]
    input  wire [WIDTH-1:0]     dina,         // Port A write data [31:0]
    output wire [WIDTH-1:0]     douta,        // Port A read data [31:0]
    
    // Port B Interface  
    input  wire [ADDR_WIDTH-1:0] addr_b,      // Port B address [9:0]
    input  wire                 enb,          // Port B enable
    input  wire                 web,          // Port B write enable
    input  wire [3:0]           beb,          // Port B byte enable [3:0]
    input  wire [WIDTH-1:0]     dinb,         // Port B write data [31:0]
    output wire [WIDTH-1:0]     doutb,        // Port B read data [31:0]
    
    // Optional Status Signals
    output wire                 conflict_a,   // Port A address conflict indicator
    output wire                 conflict_b    // Port B address conflict indicator
);

//==============================================================================
// Internal Signal Declarations
//==============================================================================

// Address decoding signals
wire [1:0] block_sel_a;                      // Block selection for port A
wire [1:0] block_sel_b;                      // Block selection for port B
wire [SUB_ADDR_WIDTH-1:0] sub_addr_a;        // Sub-block address for port A
wire [SUB_ADDR_WIDTH-1:0] sub_addr_b;        // Sub-block address for port B

// Sub-block interface arrays
wire [NUM_BLOCKS-1:0] sub_ena;               // Enable signals for sub-blocks port A
wire [NUM_BLOCKS-1:0] sub_wea;               // Write enable signals for sub-blocks port A  
wire [NUM_BLOCKS-1:0] sub_enb;               // Enable signals for sub-blocks port B
wire [NUM_BLOCKS-1:0] sub_web;               // Write enable signals for sub-blocks port B
wire [WIDTH-1:0] sub_douta [NUM_BLOCKS-1:0]; // Read data from sub-blocks port A
wire [WIDTH-1:0] sub_doutb [NUM_BLOCKS-1:0]; // Read data from sub-blocks port B

// Output multiplexing signals
reg [1:0] block_sel_a_reg;                   // Registered block selection port A
reg [1:0] block_sel_b_reg;                   // Registered block selection port B
wire [WIDTH-1:0] mux_douta;                  // Multiplexed output port A
wire [WIDTH-1:0] mux_doutb;                  // Multiplexed output port B

// Pipeline registers
reg [WIDTH-1:0] douta_reg;                   // Pipeline register for port A
reg [WIDTH-1:0] doutb_reg;                   // Pipeline register for port B

// Conflict detection signals
reg addr_conflict;                           // Address conflict detection
reg conflict_a_reg;                          // Registered conflict for port A
reg conflict_b_reg;                          // Registered conflict for port B

//==============================================================================
// Address Decoding Logic
//==============================================================================

// Extract block selection bits (MSB 2 bits)
assign block_sel_a = addr_a[ADDR_WIDTH-1:ADDR_WIDTH-2];
assign block_sel_b = addr_b[ADDR_WIDTH-1:ADDR_WIDTH-2];

// Extract sub-block address (LSB 8 bits)
assign sub_addr_a = addr_a[SUB_ADDR_WIDTH-1:0];
assign sub_addr_b = addr_b[SUB_ADDR_WIDTH-1:0];

//==============================================================================
// Sub-block Enable Generation
//==============================================================================

generate
    genvar i;
    for (i = 0; i < NUM_BLOCKS; i = i + 1) begin : gen_enable
        // Port A enable generation
        assign sub_ena[i] = ena && (block_sel_a == i[1:0]);
        assign sub_wea[i] = sub_ena[i] && wea;
        
        // Port B enable generation
        assign sub_enb[i] = enb && (block_sel_b == i[1:0]);
        assign sub_web[i] = sub_enb[i] && web;
    end
endgenerate

//==============================================================================
// Sub-block RAM Instantiation
//==============================================================================

generate
    for (i = 0; i < NUM_BLOCKS; i = i + 1) begin : gen_ram_blocks
        
        dp_ram_256x32_sub #(
            .DEPTH(SUB_DEPTH),
            .WIDTH(WIDTH),
            .ADDR_WIDTH(SUB_ADDR_WIDTH),
            .WRITE_FIRST(WRITE_FIRST),
            .RAM_STYLE(RAM_STYLE),
            .RESET_VALUE(RESET_VALUE)
        ) u_ram_sub (
            .clk        (clk),
            .rst_n      (rst_n),
            
            // Port A interface
            .addr_a     (sub_addr_a),
            .ena        (sub_ena[i]),
            .wea        (sub_wea[i]),
            .bea        (bea),
            .dina       (dina),
            .douta      (sub_douta[i]),
            
            // Port B interface
            .addr_b     (sub_addr_b),
            .enb        (sub_enb[i]),
            .web        (sub_web[i]),
            .beb        (beb),
            .dinb       (dinb),
            .doutb      (sub_doutb[i])
        );
        
    end
endgenerate

//==============================================================================
// Block Selection Register (for output multiplexing)
//==============================================================================

always @(posedge clk) begin
    if (!rst_n) begin
        block_sel_a_reg <= 2'b00;
        block_sel_b_reg <= 2'b00;
    end else begin
        if (ena) block_sel_a_reg <= block_sel_a;
        if (enb) block_sel_b_reg <= block_sel_b;
    end
end

//==============================================================================
// Output Multiplexing Logic
//==============================================================================

assign mux_douta = sub_douta[block_sel_a_reg];
assign mux_doutb = sub_doutb[block_sel_b_reg];

//==============================================================================
// Optional Pipeline Stage
//==============================================================================

generate
    if (PIPELINE_STAGES > 0) begin : gen_pipeline
        
        always @(posedge clk) begin
            if (!rst_n) begin
                douta_reg <= RESET_VALUE;
                doutb_reg <= RESET_VALUE;
            end else begin
                douta_reg <= mux_douta;
                doutb_reg <= mux_doutb;
            end
        end
        
        assign douta = douta_reg;
        assign doutb = doutb_reg;
        
    end else begin : gen_no_pipeline
        
        assign douta = mux_douta;
        assign doutb = mux_doutb;
        
    end
endgenerate

//==============================================================================
// Address Conflict Detection (Optional)
//==============================================================================

generate
    if (CONFLICT_CHECK) begin : gen_conflict_check
        
        always @(posedge clk) begin
            if (!rst_n) begin
                addr_conflict <= 1'b0;
                conflict_a_reg <= 1'b0;
                conflict_b_reg <= 1'b0;
            end else begin
                // Detect address conflict when both ports access same address
                addr_conflict <= ena && enb && (addr_a == addr_b) && (wea || web);
                conflict_a_reg <= addr_conflict && wea;
                conflict_b_reg <= addr_conflict && web;
            end
        end
        
        assign conflict_a = conflict_a_reg;
        assign conflict_b = conflict_b_reg;
        
    end else begin : gen_no_conflict_check
        
        assign conflict_a = 1'b0;
        assign conflict_b = 1'b0;
        
    end
endgenerate

endmodule

//==============================================================================
// Sub-block RAM Module Definition
//==============================================================================

module dp_ram_256x32_sub #(
    parameter DEPTH = 256,
    parameter WIDTH = 32,
    parameter ADDR_WIDTH = 8,
    parameter WRITE_FIRST = 1,
    parameter RAM_STYLE = "block",
    parameter RESET_VALUE = 32'h00000000
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Port A interface
    input  wire [ADDR_WIDTH-1:0]    addr_a,
    input  wire                     ena,
    input  wire                     wea,
    input  wire [3:0]               bea,
    input  wire [WIDTH-1:0]         dina,
    output reg  [WIDTH-1:0]         douta,
    
    // Port B interface
    input  wire [ADDR_WIDTH-1:0]    addr_b,
    input  wire                     enb,
    input  wire                     web,
    input  wire [3:0]               beb,
    input  wire [WIDTH-1:0]         dinb,
    output reg  [WIDTH-1:0]         doutb
);

    // Memory array declaration with synthesis attribute
    (* ram_style = RAM_STYLE *) reg [WIDTH-1:0] mem [0:DEPTH-1];
    
    // Temporary variables for byte-wise writing
    reg [WIDTH-1:0] temp_data_a;
    reg [WIDTH-1:0] temp_data_b;
    
    //==========================================================================
    // Port A Logic (Write-First Mode)
    //==========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            douta <= RESET_VALUE;
        end else if (ena) begin
            // Prepare data for byte-wise writing
            temp_data_a = mem[addr_a];
            if (bea[0]) temp_data_a[7:0]   = dina[7:0];
            if (bea[1]) temp_data_a[15:8]  = dina[15:8];
            if (bea[2]) temp_data_a[23:16] = dina[23:16];
            if (bea[3]) temp_data_a[31:24] = dina[31:24];
            
            if (wea) begin
                // Write operation
                mem[addr_a] <= temp_data_a;
                if (WRITE_FIRST) begin
                    douta <= temp_data_a;  // Write-first: output new data
                end else begin
                    douta <= mem[addr_a];  // Read-first: output old data
                end
            end else begin
                // Read-only operation
                douta <= mem[addr_a];
            end
        end
    end
    
    //==========================================================================
    // Port B Logic (Write-First Mode)
    //==========================================================================
    
    always @(posedge clk) begin
        if (!rst_n) begin
            doutb <= RESET_VALUE;
        end else if (enb) begin
            // Prepare data for byte-wise writing
            temp_data_b = mem[addr_b];
            if (beb[0]) temp_data_b[7:0]   = dinb[7:0];
            if (beb[1]) temp_data_b[15:8]  = dinb[15:8];
            if (beb[2]) temp_data_b[23:16] = dinb[23:16];
            if (beb[3]) temp_data_b[31:24] = dinb[31:24];
            
            if (web) begin
                // Write operation
                mem[addr_b] <= temp_data_b;
                if (WRITE_FIRST) begin
                    doutb <= temp_data_b;  // Write-first: output new data
                end else begin
                    doutb <= mem[addr_b];  // Read-first: output old data
                end
            end else begin
                // Read-only operation
                doutb <= mem[addr_b];
            end
        end
    end

endmodule

//==============================================================================
// End of dp_ram_1kx32.v
//==============================================================================