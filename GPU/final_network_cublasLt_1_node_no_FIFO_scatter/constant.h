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

///////////// OPTION: small model 352 -> 512 /////////////
///////////// OPTION: large model 880 -> 1024 /////////////
#define INPUT_FEATURE_LEN 880

// BATCH_SIZE GIVEN IN constant.h, need to revisit the constant definition later 
#define HIDDEN_SIZE1 1024
#define HIDDEN_SIZE2 512 
#define HIDDEN_SIZE3 256  
#define OUTPUT_FEATURE_LEN 1 

/* constraint: SHM_DATA_SIZE === 1 GB */
/* FLOAT_SIZE * BATCH_SIZE * INPUT_DIM * BATCH_NUM_PER_LOOP = 1024 **3 */
#define FLOAT_SIZE 4 
#define BATCH_SIZE 128 // 1024
#define TOTAL_BATCH_NUM (2 * 1024 * 1024 / BATCH_SIZE)

/* matrix (batch * input_dim): 1024 * 1024, float: 4 byte, 1024 batches in queue */
/* 4 GB in total, (1024 * 1024 * 4 * 1024) */
#define BLOCK_ENTRY_NUM (BATCH_SIZE * INPUT_FEATURE_LEN)
#define BLOCK_SIZE (BLOCK_ENTRY_NUM * FLOAT_SIZE)

#define PORT 8080 // 8080

#define THREAD_NUM 4  // the number of sender / receiver threads
