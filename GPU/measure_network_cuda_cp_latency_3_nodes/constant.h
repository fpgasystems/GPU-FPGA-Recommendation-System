// Input: (INPUT_FEATURE_LEN, BATCH_SIZE)
// Layer1: W1 * INPUT + B1 
//  -> W1 (HIDDEN_SIZE1, INPUT_FEATURE_LEN)
//  -> B1 (HIDDEN_SIZE1)
//  -> Result1 (HIDDEN_SIZE1, BATCH_SIZE)
// Layer2: W2 * Result1 + B2
//  -> W2 (HIDDEN_SIZE2, HIDDEN_SIZE1)
//  -> B2 (HIDDEN_SIZE2)
//  -> Result2 (HIDDEN_SIZE2, BATCH_SIZE)
// Layer3: W3 * Result2 + B3
//  -> W3 (HIDDEN_SIZE3, HIDDEN_SIZE2)
//  -> B3 (HIDDEN_SIZE3)
//  -> Result3 (HIDDEN_SIZE3, BATCH_SIZE)
// Output Layer: W_OUT * Result3 + B_OUT
//  -> W3 (OUTPUT_FEATURE_LEN, HIDDEN_SIZE3)
//  -> B3 (OUTPUT_FEATURE_LEN)
//  -> Result3 (OUTPUT_FEATURE_LEN, BATCH_SIZE)

///////////// OPTION: small model 384 -> 512 /////////////
///////////// OPTION: large model 876 -> 1024 /////////////

/// TODO: CHANGE THIS
// #define INPUT_FEATURE_LEN 1024

#define INPUT_FEATURE_LEN_RECEIVER 3968
#define INPUT_FEATURE_LEN_FPGA_SENDER 1952
#define INPUT_FEATURE_LEN_CPU_SENDER 64

// BATCH_SIZE GIVEN IN constant.h, need to revisit the constant definition later 
#define HIDDEN_SIZE1 2048 // 1024
#define HIDDEN_SIZE2 512 
#define HIDDEN_SIZE3 256  
#define OUTPUT_FEATURE_LEN 1 

/* constraint: SHM_DATA_SIZE === 1 GB */
/* FLOAT_SIZE * BATCH_SIZE * INPUT_DIM * BATCH_NUM_PER_LOOP = 1024 **3 */
#define FLOAT_SIZE 4 
#define BATCH_SIZE 512 // 256
// maintain the same FIFO memory size
// if batch size = 256, using 1024 as FIFO size, if batch size = 512, FIFO size = 512, etc.
// #define BATCH_NUM_PER_LOOP ((1024) / (BATCH_SIZE / 256))  // -> should be renamed as FIFO_BATCH_NUM
#define BATCH_NUM_PER_LOOP 16//((1024 * 256) / (BATCH_SIZE))  // -> should be renamed as FIFO_BATCH_NUM

// LOOP = number of GBs to perform 
#define LOOP_NUM 1// 16

#define TOTAL_BATCH_NUM (BATCH_NUM_PER_LOOP * LOOP_NUM)
/* matrix (batch * input_dim): 1024 * 1024, float: 4 byte, 1024 batches in queue */
/* 4 GB in total, (1024 * 1024 * 4 * 1024) */

/// TODO: CHANGE THIS
// #define BLOCK_ENTRY_NUM (BATCH_SIZE * INPUT_FEATURE_LEN)
// #define BLOCK_SIZE (BLOCK_ENTRY_NUM * FLOAT_SIZE)

#define BLOCK_ENTRY_NUM_RECEIVER (BATCH_SIZE * INPUT_FEATURE_LEN_RECEIVER)
#define BLOCK_SIZE_RECEIVER (BLOCK_ENTRY_NUM_RECEIVER * FLOAT_SIZE)

#define BLOCK_ENTRY_NUM_FPGA_SENDER (BATCH_SIZE * INPUT_FEATURE_LEN_FPGA_SENDER)
#define BLOCK_SIZE_FPGA_SENDER (BLOCK_ENTRY_NUM_FPGA_SENDER * FLOAT_SIZE)

#define BLOCK_ENTRY_NUM_CPU_SENDER (BATCH_SIZE * INPUT_FEATURE_LEN_CPU_SENDER)
#define BLOCK_SIZE_CPU_SENDER (BLOCK_ENTRY_NUM_CPU_SENDER * FLOAT_SIZE)

// maximum shared memory size: 1 GB
/// TODO: CHANGE THIS
// #define SHM_DATA_SIZE (BLOCK_SIZE * BATCH_NUM_PER_LOOP)  // TODO: may need to use smaller BATCH_NUM_PER_LOOP

#define SHM_DATA_SIZE_FPGA (BLOCK_SIZE_FPGA_SENDER * BATCH_NUM_PER_LOOP)  // TODO: may need to use smaller BATCH_NUM_PER_LOOP
#define SHM_DATA_SIZE_CPU (BLOCK_SIZE_CPU_SENDER * BATCH_NUM_PER_LOOP)  // TODO: may need to use smaller BATCH_NUM_PER_LOOP

#define SHM_CONTROL_SIZE 1024

#define THREAD_NUM 1 

/// TODO: CHANGE THIS
// #define PORT 8080

// Stream 0 port: PORT, Stream 1 port: PORT + 1, ...
#define PORT_FPGA_SENDER_0 8080
#define PORT_FPGA_SENDER_1 8084
#define PORT_CPU_SENDER_0 8088

#define PORT_RECEIVER 8080