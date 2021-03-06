#pragma once
#include <ap_int.h>

typedef ap_uint<128> axi_t;
typedef axi_t hbm_t;
typedef axi_t ddr_t;
typedef axi_t plram_t;

typedef ap_uint<512> network_t;

#define FLOATS_PER_AXI 4 // 128 bit = 4 floats

#define INPUT_SIZE 352
#define HIDDEN_SIZE1 1024
#define HIDDEN_SIZE2 512
#define HIDDEN_SIZE3 256
#define OUTPUT_SIZE 1

// 352 * 4 * 8 / 512 = 22
#define INPUT_SIZE_AXI_512 22

#define FIFO_BATCH_SIZE 8

#define BATCH_SIZE 32
#define BATCH_NUM 5000


//////////////////////////////   TEMPLATE START  //////////////////////////////

#define PLRAM_BANK_NUM 17
#define HBM_BANK_NUM 28
#define DDR_BANK 2

#define TABLE_NUM 47

#define TABLE_NUM_HBM 28
#define TABLE_NUM_DDR 2
#define TABLE_NUM_PLRAM 17

/////////////////////////   HBM   ///////////////////////// 
// alignment of tables to HBM: 
// table 0 ~ 31 -> HBM 0 ~ 31
// table 32 ~ 63 -> HBM 0 ~ 31
// alignment of tables to HBM: 
// table 0 ~ 27 -> HBM 0 ~ 27
// table 28 ~ 55 -> HBM 0 ~ 27

#define DATA_SIZE_HBM_0 8
#define PADDED_SIZE_HBM_0 8
#define AXI_PADDED_SIZE_HBM_0 2
#define TABLE_SIZE_HBM_0 10000
#define DATA_SIZE_HBM_1 8
#define PADDED_SIZE_HBM_1 8
#define AXI_PADDED_SIZE_HBM_1 2
#define TABLE_SIZE_HBM_1 10000
#define DATA_SIZE_HBM_2 8
#define PADDED_SIZE_HBM_2 8
#define AXI_PADDED_SIZE_HBM_2 2
#define TABLE_SIZE_HBM_2 10000
#define DATA_SIZE_HBM_3 8
#define PADDED_SIZE_HBM_3 8
#define AXI_PADDED_SIZE_HBM_3 2
#define TABLE_SIZE_HBM_3 10000
#define DATA_SIZE_HBM_4 8
#define PADDED_SIZE_HBM_4 8
#define AXI_PADDED_SIZE_HBM_4 2
#define TABLE_SIZE_HBM_4 10000
#define DATA_SIZE_HBM_5 8
#define PADDED_SIZE_HBM_5 8
#define AXI_PADDED_SIZE_HBM_5 2
#define TABLE_SIZE_HBM_5 10000
#define DATA_SIZE_HBM_6 8
#define PADDED_SIZE_HBM_6 8
#define AXI_PADDED_SIZE_HBM_6 2
#define TABLE_SIZE_HBM_6 10000
#define DATA_SIZE_HBM_7 8
#define PADDED_SIZE_HBM_7 8
#define AXI_PADDED_SIZE_HBM_7 2
#define TABLE_SIZE_HBM_7 10000
#define DATA_SIZE_HBM_8 8
#define PADDED_SIZE_HBM_8 8
#define AXI_PADDED_SIZE_HBM_8 2
#define TABLE_SIZE_HBM_8 10000
#define DATA_SIZE_HBM_9 8
#define PADDED_SIZE_HBM_9 8
#define AXI_PADDED_SIZE_HBM_9 2
#define TABLE_SIZE_HBM_9 10000
#define DATA_SIZE_HBM_10 8
#define PADDED_SIZE_HBM_10 8
#define AXI_PADDED_SIZE_HBM_10 2
#define TABLE_SIZE_HBM_10 10000
#define DATA_SIZE_HBM_11 8
#define PADDED_SIZE_HBM_11 8
#define AXI_PADDED_SIZE_HBM_11 2
#define TABLE_SIZE_HBM_11 10000
#define DATA_SIZE_HBM_12 8
#define PADDED_SIZE_HBM_12 8
#define AXI_PADDED_SIZE_HBM_12 2
#define TABLE_SIZE_HBM_12 10000
#define DATA_SIZE_HBM_13 8
#define PADDED_SIZE_HBM_13 8
#define AXI_PADDED_SIZE_HBM_13 2
#define TABLE_SIZE_HBM_13 10000
#define DATA_SIZE_HBM_14 8
#define PADDED_SIZE_HBM_14 8
#define AXI_PADDED_SIZE_HBM_14 2
#define TABLE_SIZE_HBM_14 10000
#define DATA_SIZE_HBM_15 8
#define PADDED_SIZE_HBM_15 8
#define AXI_PADDED_SIZE_HBM_15 2
#define TABLE_SIZE_HBM_15 20000
#define DATA_SIZE_HBM_16 8
#define PADDED_SIZE_HBM_16 8
#define AXI_PADDED_SIZE_HBM_16 2
#define TABLE_SIZE_HBM_16 30000
#define DATA_SIZE_HBM_17 8
#define PADDED_SIZE_HBM_17 8
#define AXI_PADDED_SIZE_HBM_17 2
#define TABLE_SIZE_HBM_17 100000
#define DATA_SIZE_HBM_18 8
#define PADDED_SIZE_HBM_18 8
#define AXI_PADDED_SIZE_HBM_18 2
#define TABLE_SIZE_HBM_18 100000
#define DATA_SIZE_HBM_19 8
#define PADDED_SIZE_HBM_19 8
#define AXI_PADDED_SIZE_HBM_19 2
#define TABLE_SIZE_HBM_19 100000
#define DATA_SIZE_HBM_20 8
#define PADDED_SIZE_HBM_20 8
#define AXI_PADDED_SIZE_HBM_20 2
#define TABLE_SIZE_HBM_20 100000
#define DATA_SIZE_HBM_21 8
#define PADDED_SIZE_HBM_21 8
#define AXI_PADDED_SIZE_HBM_21 2
#define TABLE_SIZE_HBM_21 100000
#define DATA_SIZE_HBM_22 8
#define PADDED_SIZE_HBM_22 8
#define AXI_PADDED_SIZE_HBM_22 2
#define TABLE_SIZE_HBM_22 100000
#define DATA_SIZE_HBM_23 8
#define PADDED_SIZE_HBM_23 8
#define AXI_PADDED_SIZE_HBM_23 2
#define TABLE_SIZE_HBM_23 100000
#define DATA_SIZE_HBM_24 8
#define PADDED_SIZE_HBM_24 8
#define AXI_PADDED_SIZE_HBM_24 2
#define TABLE_SIZE_HBM_24 100000
#define DATA_SIZE_HBM_25 8
#define PADDED_SIZE_HBM_25 8
#define AXI_PADDED_SIZE_HBM_25 2
#define TABLE_SIZE_HBM_25 100000
#define DATA_SIZE_HBM_26 8
#define PADDED_SIZE_HBM_26 8
#define AXI_PADDED_SIZE_HBM_26 2
#define TABLE_SIZE_HBM_26 100000
#define DATA_SIZE_HBM_27 16
#define PADDED_SIZE_HBM_27 16
#define AXI_PADDED_SIZE_HBM_27 4
#define TABLE_SIZE_HBM_27 500000

#define BURST_SIZE_HBM 16

#define ADDR_AXI_HBM_0 0
#define ADDR_AXI_HBM_1 0
#define ADDR_AXI_HBM_2 0
#define ADDR_AXI_HBM_3 0
#define ADDR_AXI_HBM_4 0
#define ADDR_AXI_HBM_5 0
#define ADDR_AXI_HBM_6 0
#define ADDR_AXI_HBM_7 0
#define ADDR_AXI_HBM_8 0
#define ADDR_AXI_HBM_9 0
#define ADDR_AXI_HBM_10 0
#define ADDR_AXI_HBM_11 0
#define ADDR_AXI_HBM_12 0
#define ADDR_AXI_HBM_13 0
#define ADDR_AXI_HBM_14 0
#define ADDR_AXI_HBM_15 0
#define ADDR_AXI_HBM_16 0
#define ADDR_AXI_HBM_17 0
#define ADDR_AXI_HBM_18 0
#define ADDR_AXI_HBM_19 0
#define ADDR_AXI_HBM_20 0
#define ADDR_AXI_HBM_21 0
#define ADDR_AXI_HBM_22 0
#define ADDR_AXI_HBM_23 0
#define ADDR_AXI_HBM_24 0
#define ADDR_AXI_HBM_25 0
#define ADDR_AXI_HBM_26 0
#define ADDR_AXI_HBM_27 0

#define HBM_BANK0_ROUND 1
#define HBM_BANK1_ROUND 1
#define HBM_BANK2_ROUND 1
#define HBM_BANK3_ROUND 1
#define HBM_BANK4_ROUND 1
#define HBM_BANK5_ROUND 1
#define HBM_BANK6_ROUND 1
#define HBM_BANK7_ROUND 1
#define HBM_BANK8_ROUND 1
#define HBM_BANK9_ROUND 1
#define HBM_BANK10_ROUND 1
#define HBM_BANK11_ROUND 1
#define HBM_BANK12_ROUND 1
#define HBM_BANK13_ROUND 1
#define HBM_BANK14_ROUND 1
#define HBM_BANK15_ROUND 1
#define HBM_BANK16_ROUND 1
#define HBM_BANK17_ROUND 1
#define HBM_BANK18_ROUND 1
#define HBM_BANK19_ROUND 1
#define HBM_BANK20_ROUND 1
#define HBM_BANK21_ROUND 1
#define HBM_BANK22_ROUND 1
#define HBM_BANK23_ROUND 1
#define HBM_BANK24_ROUND 1
#define HBM_BANK25_ROUND 1
#define HBM_BANK26_ROUND 1
#define HBM_BANK27_ROUND 1

#define HBM_BANK0_SIZE 20000
#define HBM_BANK1_SIZE 20000
#define HBM_BANK2_SIZE 20000
#define HBM_BANK3_SIZE 20000
#define HBM_BANK4_SIZE 20000
#define HBM_BANK5_SIZE 20000
#define HBM_BANK6_SIZE 20000
#define HBM_BANK7_SIZE 20000
#define HBM_BANK8_SIZE 20000
#define HBM_BANK9_SIZE 20000
#define HBM_BANK10_SIZE 20000
#define HBM_BANK11_SIZE 20000
#define HBM_BANK12_SIZE 20000
#define HBM_BANK13_SIZE 20000
#define HBM_BANK14_SIZE 20000
#define HBM_BANK15_SIZE 40000
#define HBM_BANK16_SIZE 60000
#define HBM_BANK17_SIZE 200000
#define HBM_BANK18_SIZE 200000
#define HBM_BANK19_SIZE 200000
#define HBM_BANK20_SIZE 200000
#define HBM_BANK21_SIZE 200000
#define HBM_BANK22_SIZE 200000
#define HBM_BANK23_SIZE 200000
#define HBM_BANK24_SIZE 200000
#define HBM_BANK25_SIZE 200000
#define HBM_BANK26_SIZE 200000
#define HBM_BANK27_SIZE 2000000

#define VECTOR_SIZE_HBM_BANK_0 8
#define VECTOR_SIZE_HBM_BANK_1 8
#define VECTOR_SIZE_HBM_BANK_2 8
#define VECTOR_SIZE_HBM_BANK_3 8
#define VECTOR_SIZE_HBM_BANK_4 8
#define VECTOR_SIZE_HBM_BANK_5 8
#define VECTOR_SIZE_HBM_BANK_6 8
#define VECTOR_SIZE_HBM_BANK_7 8
#define VECTOR_SIZE_HBM_BANK_8 8
#define VECTOR_SIZE_HBM_BANK_9 8
#define VECTOR_SIZE_HBM_BANK_10 8
#define VECTOR_SIZE_HBM_BANK_11 8
#define VECTOR_SIZE_HBM_BANK_12 8
#define VECTOR_SIZE_HBM_BANK_13 8
#define VECTOR_SIZE_HBM_BANK_14 8
#define VECTOR_SIZE_HBM_BANK_15 8
#define VECTOR_SIZE_HBM_BANK_16 8
#define VECTOR_SIZE_HBM_BANK_17 8
#define VECTOR_SIZE_HBM_BANK_18 8
#define VECTOR_SIZE_HBM_BANK_19 8
#define VECTOR_SIZE_HBM_BANK_20 8
#define VECTOR_SIZE_HBM_BANK_21 8
#define VECTOR_SIZE_HBM_BANK_22 8
#define VECTOR_SIZE_HBM_BANK_23 8
#define VECTOR_SIZE_HBM_BANK_24 8
#define VECTOR_SIZE_HBM_BANK_25 8
#define VECTOR_SIZE_HBM_BANK_26 8
#define VECTOR_SIZE_HBM_BANK_27 16

#define VECTOR_START_IDX_HBM_BANK_0 0
#define VECTOR_START_IDX_HBM_BANK_1 8
#define VECTOR_START_IDX_HBM_BANK_2 16
#define VECTOR_START_IDX_HBM_BANK_3 24
#define VECTOR_START_IDX_HBM_BANK_4 32
#define VECTOR_START_IDX_HBM_BANK_5 40
#define VECTOR_START_IDX_HBM_BANK_6 48
#define VECTOR_START_IDX_HBM_BANK_7 56
#define VECTOR_START_IDX_HBM_BANK_8 64
#define VECTOR_START_IDX_HBM_BANK_9 72
#define VECTOR_START_IDX_HBM_BANK_10 80
#define VECTOR_START_IDX_HBM_BANK_11 88
#define VECTOR_START_IDX_HBM_BANK_12 96
#define VECTOR_START_IDX_HBM_BANK_13 104
#define VECTOR_START_IDX_HBM_BANK_14 112
#define VECTOR_START_IDX_HBM_BANK_15 120
#define VECTOR_START_IDX_HBM_BANK_16 128
#define VECTOR_START_IDX_HBM_BANK_17 136
#define VECTOR_START_IDX_HBM_BANK_18 144
#define VECTOR_START_IDX_HBM_BANK_19 152
#define VECTOR_START_IDX_HBM_BANK_20 160
#define VECTOR_START_IDX_HBM_BANK_21 168
#define VECTOR_START_IDX_HBM_BANK_22 176
#define VECTOR_START_IDX_HBM_BANK_23 184
#define VECTOR_START_IDX_HBM_BANK_24 192
#define VECTOR_START_IDX_HBM_BANK_25 200
#define VECTOR_START_IDX_HBM_BANK_26 208
#define VECTOR_START_IDX_HBM_BANK_27 216

/////////////////////////   DDR   ///////////////////////// 
// alignment of tables to DDR: 
// table 0 ~ 1 -> DDR 0 ~ 1
// table 2 ~ 3 -> DDR 0 ~ 1

#define DATA_SIZE_DDR_0 16
#define PADDED_SIZE_DDR_0 16
#define AXI_PADDED_SIZE_DDR_0 4
#define TABLE_SIZE_DDR_0 1000000
#define DATA_SIZE_DDR_1 32
#define PADDED_SIZE_DDR_1 32
#define AXI_PADDED_SIZE_DDR_1 8
#define TABLE_SIZE_DDR_1 10000000

#define BURST_SIZE_DDR 32

#define ADDR_AXI_DDR_0 0
#define ADDR_AXI_DDR_1 0

#define DDR_BANK0_ROUND 1
#define DDR_BANK1_ROUND 1

#define DDR_BANK0_SIZE 4000000
#define DDR_BANK1_SIZE 80000000

#define VECTOR_SIZE_DDR_BANK_0 16
#define VECTOR_SIZE_DDR_BANK_1 32

#define VECTOR_START_IDX_DDR_BANK_0 232
#define VECTOR_START_IDX_DDR_BANK_1 248

/////////////////////////   PLRAM   ///////////////////////// 
// alignment of tables to PLRAM: 
// table 0 ~ 16 -> PLRAM 0 ~ 16
// table 17 ~ 33 -> PLRAM 0 ~ 16

#define DATA_SIZE_PLRAM_0 4
#define PADDED_SIZE_PLRAM_0 4
#define AXI_PADDED_SIZE_PLRAM_0 1
#define TABLE_SIZE_PLRAM_0 100
#define DATA_SIZE_PLRAM_1 4
#define PADDED_SIZE_PLRAM_1 4
#define AXI_PADDED_SIZE_PLRAM_1 1
#define TABLE_SIZE_PLRAM_1 100
#define DATA_SIZE_PLRAM_2 4
#define PADDED_SIZE_PLRAM_2 4
#define AXI_PADDED_SIZE_PLRAM_2 1
#define TABLE_SIZE_PLRAM_2 100
#define DATA_SIZE_PLRAM_3 4
#define PADDED_SIZE_PLRAM_3 4
#define AXI_PADDED_SIZE_PLRAM_3 1
#define TABLE_SIZE_PLRAM_3 100
#define DATA_SIZE_PLRAM_4 4
#define PADDED_SIZE_PLRAM_4 4
#define AXI_PADDED_SIZE_PLRAM_4 1
#define TABLE_SIZE_PLRAM_4 500
#define DATA_SIZE_PLRAM_5 4
#define PADDED_SIZE_PLRAM_5 4
#define AXI_PADDED_SIZE_PLRAM_5 1
#define TABLE_SIZE_PLRAM_5 500
#define DATA_SIZE_PLRAM_6 4
#define PADDED_SIZE_PLRAM_6 4
#define AXI_PADDED_SIZE_PLRAM_6 1
#define TABLE_SIZE_PLRAM_6 1000
#define DATA_SIZE_PLRAM_7 4
#define PADDED_SIZE_PLRAM_7 4
#define AXI_PADDED_SIZE_PLRAM_7 1
#define TABLE_SIZE_PLRAM_7 1000
#define DATA_SIZE_PLRAM_8 4
#define PADDED_SIZE_PLRAM_8 4
#define AXI_PADDED_SIZE_PLRAM_8 1
#define TABLE_SIZE_PLRAM_8 1000
#define DATA_SIZE_PLRAM_9 4
#define PADDED_SIZE_PLRAM_9 4
#define AXI_PADDED_SIZE_PLRAM_9 1
#define TABLE_SIZE_PLRAM_9 1000
#define DATA_SIZE_PLRAM_10 4
#define PADDED_SIZE_PLRAM_10 4
#define AXI_PADDED_SIZE_PLRAM_10 1
#define TABLE_SIZE_PLRAM_10 1000
#define DATA_SIZE_PLRAM_11 4
#define PADDED_SIZE_PLRAM_11 4
#define AXI_PADDED_SIZE_PLRAM_11 1
#define TABLE_SIZE_PLRAM_11 1000
#define DATA_SIZE_PLRAM_12 4
#define PADDED_SIZE_PLRAM_12 4
#define AXI_PADDED_SIZE_PLRAM_12 1
#define TABLE_SIZE_PLRAM_12 1000
#define DATA_SIZE_PLRAM_13 4
#define PADDED_SIZE_PLRAM_13 4
#define AXI_PADDED_SIZE_PLRAM_13 1
#define TABLE_SIZE_PLRAM_13 1000
#define DATA_SIZE_PLRAM_14 4
#define PADDED_SIZE_PLRAM_14 4
#define AXI_PADDED_SIZE_PLRAM_14 1
#define TABLE_SIZE_PLRAM_14 1000
#define DATA_SIZE_PLRAM_15 4
#define PADDED_SIZE_PLRAM_15 4
#define AXI_PADDED_SIZE_PLRAM_15 1
#define TABLE_SIZE_PLRAM_15 1000
#define DATA_SIZE_PLRAM_16 8
#define PADDED_SIZE_PLRAM_16 8
#define AXI_PADDED_SIZE_PLRAM_16 2
#define TABLE_SIZE_PLRAM_16 3000

#define BURST_SIZE_PLRAM 8

#define ADDR_AXI_PLRAM_0 0
#define ADDR_AXI_PLRAM_1 0
#define ADDR_AXI_PLRAM_2 0
#define ADDR_AXI_PLRAM_3 0
#define ADDR_AXI_PLRAM_4 0
#define ADDR_AXI_PLRAM_5 0
#define ADDR_AXI_PLRAM_6 0
#define ADDR_AXI_PLRAM_7 0
#define ADDR_AXI_PLRAM_8 0
#define ADDR_AXI_PLRAM_9 0
#define ADDR_AXI_PLRAM_10 0
#define ADDR_AXI_PLRAM_11 0
#define ADDR_AXI_PLRAM_12 0
#define ADDR_AXI_PLRAM_13 0
#define ADDR_AXI_PLRAM_14 0
#define ADDR_AXI_PLRAM_15 0
#define ADDR_AXI_PLRAM_16 0

#define PLRAM_BANK0_ROUND 1
#define PLRAM_BANK1_ROUND 1
#define PLRAM_BANK2_ROUND 1
#define PLRAM_BANK3_ROUND 1
#define PLRAM_BANK4_ROUND 1
#define PLRAM_BANK5_ROUND 1
#define PLRAM_BANK6_ROUND 1
#define PLRAM_BANK7_ROUND 1
#define PLRAM_BANK8_ROUND 1
#define PLRAM_BANK9_ROUND 1
#define PLRAM_BANK10_ROUND 1
#define PLRAM_BANK11_ROUND 1
#define PLRAM_BANK12_ROUND 1
#define PLRAM_BANK13_ROUND 1
#define PLRAM_BANK14_ROUND 1
#define PLRAM_BANK15_ROUND 1
#define PLRAM_BANK16_ROUND 1

#define PLRAM_BANK0_SIZE 100
#define PLRAM_BANK1_SIZE 100
#define PLRAM_BANK2_SIZE 100
#define PLRAM_BANK3_SIZE 100
#define PLRAM_BANK4_SIZE 500
#define PLRAM_BANK5_SIZE 500
#define PLRAM_BANK6_SIZE 1000
#define PLRAM_BANK7_SIZE 1000
#define PLRAM_BANK8_SIZE 1000
#define PLRAM_BANK9_SIZE 1000
#define PLRAM_BANK10_SIZE 1000
#define PLRAM_BANK11_SIZE 1000
#define PLRAM_BANK12_SIZE 1000
#define PLRAM_BANK13_SIZE 1000
#define PLRAM_BANK14_SIZE 1000
#define PLRAM_BANK15_SIZE 1000
#define PLRAM_BANK16_SIZE 6000

#define VECTOR_SIZE_PLRAM_BANK_0 4
#define VECTOR_SIZE_PLRAM_BANK_1 4
#define VECTOR_SIZE_PLRAM_BANK_2 4
#define VECTOR_SIZE_PLRAM_BANK_3 4
#define VECTOR_SIZE_PLRAM_BANK_4 4
#define VECTOR_SIZE_PLRAM_BANK_5 4
#define VECTOR_SIZE_PLRAM_BANK_6 4
#define VECTOR_SIZE_PLRAM_BANK_7 4
#define VECTOR_SIZE_PLRAM_BANK_8 4
#define VECTOR_SIZE_PLRAM_BANK_9 4
#define VECTOR_SIZE_PLRAM_BANK_10 4
#define VECTOR_SIZE_PLRAM_BANK_11 4
#define VECTOR_SIZE_PLRAM_BANK_12 4
#define VECTOR_SIZE_PLRAM_BANK_13 4
#define VECTOR_SIZE_PLRAM_BANK_14 4
#define VECTOR_SIZE_PLRAM_BANK_15 4
#define VECTOR_SIZE_PLRAM_BANK_16 8

#define VECTOR_START_IDX_PLRAM_BANK_0 280
#define VECTOR_START_IDX_PLRAM_BANK_1 284
#define VECTOR_START_IDX_PLRAM_BANK_2 288
#define VECTOR_START_IDX_PLRAM_BANK_3 292
#define VECTOR_START_IDX_PLRAM_BANK_4 296
#define VECTOR_START_IDX_PLRAM_BANK_5 300
#define VECTOR_START_IDX_PLRAM_BANK_6 304
#define VECTOR_START_IDX_PLRAM_BANK_7 308
#define VECTOR_START_IDX_PLRAM_BANK_8 312
#define VECTOR_START_IDX_PLRAM_BANK_9 316
#define VECTOR_START_IDX_PLRAM_BANK_10 320
#define VECTOR_START_IDX_PLRAM_BANK_11 324
#define VECTOR_START_IDX_PLRAM_BANK_12 328
#define VECTOR_START_IDX_PLRAM_BANK_13 332
#define VECTOR_START_IDX_PLRAM_BANK_14 336
#define VECTOR_START_IDX_PLRAM_BANK_15 340
#define VECTOR_START_IDX_PLRAM_BANK_16 344

//////////////////////////////   TEMPLATE END  //////////////////////////////

