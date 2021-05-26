#pragma once
#include <ap_int.h>

typedef ap_uint<128> axi_t;
typedef axi_t hbm_t;
typedef axi_t ddr_t;
typedef axi_t plram_t;

typedef ap_uint<512> network_t;

#define FLOATS_PER_AXI 4 // 128 bit = 4 floats

#define INPUT_SIZE 1952 // 1952 * 32 / 512 = 122
// #define HIDDEN_SIZE1 1024
// #define HIDDEN_SIZE2 512
// #define HIDDEN_SIZE3 256
// #define OUTPUT_SIZE 1

// 1952 * 32 / 512 = 122
#define INPUT_SIZE_AXI_512 122

#define FIFO_BATCH_SIZE 8

#define BATCH_SIZE 32
// 5 rounds x 400 ns = 2 us
// #define BATCH_NUM 1000000 // 5000


//////////////////////////////   TEMPLATE START  //////////////////////////////

#define PLRAM_BANK_NUM 11
#define HBM_BANK_NUM 28
#define DDR_BANK 2

#define TABLE_NUM 188

// #define ACCESS_IDX_SIZE 188
// #define PADDED_ACCESS_IDX_SIZE 188
// #define AXI_PADDED_ACCESS_IDX_SIZE 47

#define TABLE_NUM_HBM 140
#define TABLE_NUM_DDR 4
#define TABLE_NUM_PLRAM 44

/////////////////////////   HBM   ///////////////////////// 
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
#define TABLE_SIZE_HBM_12 30000
#define DATA_SIZE_HBM_13 8
#define PADDED_SIZE_HBM_13 8
#define AXI_PADDED_SIZE_HBM_13 2
#define TABLE_SIZE_HBM_13 30000
#define DATA_SIZE_HBM_14 8
#define PADDED_SIZE_HBM_14 8
#define AXI_PADDED_SIZE_HBM_14 2
#define TABLE_SIZE_HBM_14 30000
#define DATA_SIZE_HBM_15 8
#define PADDED_SIZE_HBM_15 8
#define AXI_PADDED_SIZE_HBM_15 2
#define TABLE_SIZE_HBM_15 30000
#define DATA_SIZE_HBM_16 8
#define PADDED_SIZE_HBM_16 8
#define AXI_PADDED_SIZE_HBM_16 2
#define TABLE_SIZE_HBM_16 100000
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
#define DATA_SIZE_HBM_27 8
#define PADDED_SIZE_HBM_27 8
#define AXI_PADDED_SIZE_HBM_27 2
#define TABLE_SIZE_HBM_27 100000
#define DATA_SIZE_HBM_28 8
#define PADDED_SIZE_HBM_28 8
#define AXI_PADDED_SIZE_HBM_28 2
#define TABLE_SIZE_HBM_28 100000
#define DATA_SIZE_HBM_29 8
#define PADDED_SIZE_HBM_29 8
#define AXI_PADDED_SIZE_HBM_29 2
#define TABLE_SIZE_HBM_29 100000
#define DATA_SIZE_HBM_30 8
#define PADDED_SIZE_HBM_30 8
#define AXI_PADDED_SIZE_HBM_30 2
#define TABLE_SIZE_HBM_30 100000
#define DATA_SIZE_HBM_31 8
#define PADDED_SIZE_HBM_31 8
#define AXI_PADDED_SIZE_HBM_31 2
#define TABLE_SIZE_HBM_31 100000
#define DATA_SIZE_HBM_32 8
#define PADDED_SIZE_HBM_32 8
#define AXI_PADDED_SIZE_HBM_32 2
#define TABLE_SIZE_HBM_32 100000
#define DATA_SIZE_HBM_33 8
#define PADDED_SIZE_HBM_33 8
#define AXI_PADDED_SIZE_HBM_33 2
#define TABLE_SIZE_HBM_33 100000
#define DATA_SIZE_HBM_34 8
#define PADDED_SIZE_HBM_34 8
#define AXI_PADDED_SIZE_HBM_34 2
#define TABLE_SIZE_HBM_34 100000
#define DATA_SIZE_HBM_35 8
#define PADDED_SIZE_HBM_35 8
#define AXI_PADDED_SIZE_HBM_35 2
#define TABLE_SIZE_HBM_35 100000
#define DATA_SIZE_HBM_36 8
#define PADDED_SIZE_HBM_36 8
#define AXI_PADDED_SIZE_HBM_36 2
#define TABLE_SIZE_HBM_36 100000
#define DATA_SIZE_HBM_37 8
#define PADDED_SIZE_HBM_37 8
#define AXI_PADDED_SIZE_HBM_37 2
#define TABLE_SIZE_HBM_37 100000
#define DATA_SIZE_HBM_38 8
#define PADDED_SIZE_HBM_38 8
#define AXI_PADDED_SIZE_HBM_38 2
#define TABLE_SIZE_HBM_38 100000
#define DATA_SIZE_HBM_39 8
#define PADDED_SIZE_HBM_39 8
#define AXI_PADDED_SIZE_HBM_39 2
#define TABLE_SIZE_HBM_39 100000
#define DATA_SIZE_HBM_40 8
#define PADDED_SIZE_HBM_40 8
#define AXI_PADDED_SIZE_HBM_40 2
#define TABLE_SIZE_HBM_40 100000
#define DATA_SIZE_HBM_41 8
#define PADDED_SIZE_HBM_41 8
#define AXI_PADDED_SIZE_HBM_41 2
#define TABLE_SIZE_HBM_41 100000
#define DATA_SIZE_HBM_42 8
#define PADDED_SIZE_HBM_42 8
#define AXI_PADDED_SIZE_HBM_42 2
#define TABLE_SIZE_HBM_42 100000
#define DATA_SIZE_HBM_43 8
#define PADDED_SIZE_HBM_43 8
#define AXI_PADDED_SIZE_HBM_43 2
#define TABLE_SIZE_HBM_43 100000
#define DATA_SIZE_HBM_44 8
#define PADDED_SIZE_HBM_44 8
#define AXI_PADDED_SIZE_HBM_44 2
#define TABLE_SIZE_HBM_44 100000
#define DATA_SIZE_HBM_45 8
#define PADDED_SIZE_HBM_45 8
#define AXI_PADDED_SIZE_HBM_45 2
#define TABLE_SIZE_HBM_45 100000
#define DATA_SIZE_HBM_46 8
#define PADDED_SIZE_HBM_46 8
#define AXI_PADDED_SIZE_HBM_46 2
#define TABLE_SIZE_HBM_46 100000
#define DATA_SIZE_HBM_47 8
#define PADDED_SIZE_HBM_47 8
#define AXI_PADDED_SIZE_HBM_47 2
#define TABLE_SIZE_HBM_47 100000
#define DATA_SIZE_HBM_48 8
#define PADDED_SIZE_HBM_48 8
#define AXI_PADDED_SIZE_HBM_48 2
#define TABLE_SIZE_HBM_48 100000
#define DATA_SIZE_HBM_49 8
#define PADDED_SIZE_HBM_49 8
#define AXI_PADDED_SIZE_HBM_49 2
#define TABLE_SIZE_HBM_49 100000
#define DATA_SIZE_HBM_50 8
#define PADDED_SIZE_HBM_50 8
#define AXI_PADDED_SIZE_HBM_50 2
#define TABLE_SIZE_HBM_50 100000
#define DATA_SIZE_HBM_51 8
#define PADDED_SIZE_HBM_51 8
#define AXI_PADDED_SIZE_HBM_51 2
#define TABLE_SIZE_HBM_51 100000
#define DATA_SIZE_HBM_52 8
#define PADDED_SIZE_HBM_52 8
#define AXI_PADDED_SIZE_HBM_52 2
#define TABLE_SIZE_HBM_52 100000
#define DATA_SIZE_HBM_53 8
#define PADDED_SIZE_HBM_53 8
#define AXI_PADDED_SIZE_HBM_53 2
#define TABLE_SIZE_HBM_53 100000
#define DATA_SIZE_HBM_54 8
#define PADDED_SIZE_HBM_54 8
#define AXI_PADDED_SIZE_HBM_54 2
#define TABLE_SIZE_HBM_54 100000
#define DATA_SIZE_HBM_55 8
#define PADDED_SIZE_HBM_55 8
#define AXI_PADDED_SIZE_HBM_55 2
#define TABLE_SIZE_HBM_55 100000
#define DATA_SIZE_HBM_56 8
#define PADDED_SIZE_HBM_56 8
#define AXI_PADDED_SIZE_HBM_56 2
#define TABLE_SIZE_HBM_56 100000
#define DATA_SIZE_HBM_57 8
#define PADDED_SIZE_HBM_57 8
#define AXI_PADDED_SIZE_HBM_57 2
#define TABLE_SIZE_HBM_57 100000
#define DATA_SIZE_HBM_58 8
#define PADDED_SIZE_HBM_58 8
#define AXI_PADDED_SIZE_HBM_58 2
#define TABLE_SIZE_HBM_58 100000
#define DATA_SIZE_HBM_59 8
#define PADDED_SIZE_HBM_59 8
#define AXI_PADDED_SIZE_HBM_59 2
#define TABLE_SIZE_HBM_59 100000
#define DATA_SIZE_HBM_60 8
#define PADDED_SIZE_HBM_60 8
#define AXI_PADDED_SIZE_HBM_60 2
#define TABLE_SIZE_HBM_60 100000
#define DATA_SIZE_HBM_61 8
#define PADDED_SIZE_HBM_61 8
#define AXI_PADDED_SIZE_HBM_61 2
#define TABLE_SIZE_HBM_61 100000
#define DATA_SIZE_HBM_62 8
#define PADDED_SIZE_HBM_62 8
#define AXI_PADDED_SIZE_HBM_62 2
#define TABLE_SIZE_HBM_62 100000
#define DATA_SIZE_HBM_63 8
#define PADDED_SIZE_HBM_63 8
#define AXI_PADDED_SIZE_HBM_63 2
#define TABLE_SIZE_HBM_63 100000
#define DATA_SIZE_HBM_64 8
#define PADDED_SIZE_HBM_64 8
#define AXI_PADDED_SIZE_HBM_64 2
#define TABLE_SIZE_HBM_64 100000
#define DATA_SIZE_HBM_65 8
#define PADDED_SIZE_HBM_65 8
#define AXI_PADDED_SIZE_HBM_65 2
#define TABLE_SIZE_HBM_65 100000
#define DATA_SIZE_HBM_66 8
#define PADDED_SIZE_HBM_66 8
#define AXI_PADDED_SIZE_HBM_66 2
#define TABLE_SIZE_HBM_66 100000
#define DATA_SIZE_HBM_67 8
#define PADDED_SIZE_HBM_67 8
#define AXI_PADDED_SIZE_HBM_67 2
#define TABLE_SIZE_HBM_67 100000
#define DATA_SIZE_HBM_68 8
#define PADDED_SIZE_HBM_68 8
#define AXI_PADDED_SIZE_HBM_68 2
#define TABLE_SIZE_HBM_68 100000
#define DATA_SIZE_HBM_69 8
#define PADDED_SIZE_HBM_69 8
#define AXI_PADDED_SIZE_HBM_69 2
#define TABLE_SIZE_HBM_69 100000
#define DATA_SIZE_HBM_70 8
#define PADDED_SIZE_HBM_70 8
#define AXI_PADDED_SIZE_HBM_70 2
#define TABLE_SIZE_HBM_70 100000
#define DATA_SIZE_HBM_71 8
#define PADDED_SIZE_HBM_71 8
#define AXI_PADDED_SIZE_HBM_71 2
#define TABLE_SIZE_HBM_71 100000
#define DATA_SIZE_HBM_72 8
#define PADDED_SIZE_HBM_72 8
#define AXI_PADDED_SIZE_HBM_72 2
#define TABLE_SIZE_HBM_72 100000
#define DATA_SIZE_HBM_73 8
#define PADDED_SIZE_HBM_73 8
#define AXI_PADDED_SIZE_HBM_73 2
#define TABLE_SIZE_HBM_73 100000
#define DATA_SIZE_HBM_74 8
#define PADDED_SIZE_HBM_74 8
#define AXI_PADDED_SIZE_HBM_74 2
#define TABLE_SIZE_HBM_74 100000
#define DATA_SIZE_HBM_75 8
#define PADDED_SIZE_HBM_75 8
#define AXI_PADDED_SIZE_HBM_75 2
#define TABLE_SIZE_HBM_75 100000
#define DATA_SIZE_HBM_76 8
#define PADDED_SIZE_HBM_76 8
#define AXI_PADDED_SIZE_HBM_76 2
#define TABLE_SIZE_HBM_76 100000
#define DATA_SIZE_HBM_77 8
#define PADDED_SIZE_HBM_77 8
#define AXI_PADDED_SIZE_HBM_77 2
#define TABLE_SIZE_HBM_77 100000
#define DATA_SIZE_HBM_78 8
#define PADDED_SIZE_HBM_78 8
#define AXI_PADDED_SIZE_HBM_78 2
#define TABLE_SIZE_HBM_78 100000
#define DATA_SIZE_HBM_79 8
#define PADDED_SIZE_HBM_79 8
#define AXI_PADDED_SIZE_HBM_79 2
#define TABLE_SIZE_HBM_79 100000
#define DATA_SIZE_HBM_80 8
#define PADDED_SIZE_HBM_80 8
#define AXI_PADDED_SIZE_HBM_80 2
#define TABLE_SIZE_HBM_80 100000
#define DATA_SIZE_HBM_81 8
#define PADDED_SIZE_HBM_81 8
#define AXI_PADDED_SIZE_HBM_81 2
#define TABLE_SIZE_HBM_81 100000
#define DATA_SIZE_HBM_82 8
#define PADDED_SIZE_HBM_82 8
#define AXI_PADDED_SIZE_HBM_82 2
#define TABLE_SIZE_HBM_82 100000
#define DATA_SIZE_HBM_83 8
#define PADDED_SIZE_HBM_83 8
#define AXI_PADDED_SIZE_HBM_83 2
#define TABLE_SIZE_HBM_83 100000
#define DATA_SIZE_HBM_84 8
#define PADDED_SIZE_HBM_84 8
#define AXI_PADDED_SIZE_HBM_84 2
#define TABLE_SIZE_HBM_84 100000
#define DATA_SIZE_HBM_85 8
#define PADDED_SIZE_HBM_85 8
#define AXI_PADDED_SIZE_HBM_85 2
#define TABLE_SIZE_HBM_85 100000
#define DATA_SIZE_HBM_86 8
#define PADDED_SIZE_HBM_86 8
#define AXI_PADDED_SIZE_HBM_86 2
#define TABLE_SIZE_HBM_86 100000
#define DATA_SIZE_HBM_87 8
#define PADDED_SIZE_HBM_87 8
#define AXI_PADDED_SIZE_HBM_87 2
#define TABLE_SIZE_HBM_87 100000
#define DATA_SIZE_HBM_88 8
#define PADDED_SIZE_HBM_88 8
#define AXI_PADDED_SIZE_HBM_88 2
#define TABLE_SIZE_HBM_88 100000
#define DATA_SIZE_HBM_89 8
#define PADDED_SIZE_HBM_89 8
#define AXI_PADDED_SIZE_HBM_89 2
#define TABLE_SIZE_HBM_89 100000
#define DATA_SIZE_HBM_90 8
#define PADDED_SIZE_HBM_90 8
#define AXI_PADDED_SIZE_HBM_90 2
#define TABLE_SIZE_HBM_90 100000
#define DATA_SIZE_HBM_91 8
#define PADDED_SIZE_HBM_91 8
#define AXI_PADDED_SIZE_HBM_91 2
#define TABLE_SIZE_HBM_91 100000
#define DATA_SIZE_HBM_92 16
#define PADDED_SIZE_HBM_92 16
#define AXI_PADDED_SIZE_HBM_92 4
#define TABLE_SIZE_HBM_92 200000
#define DATA_SIZE_HBM_93 16
#define PADDED_SIZE_HBM_93 16
#define AXI_PADDED_SIZE_HBM_93 4
#define TABLE_SIZE_HBM_93 200000
#define DATA_SIZE_HBM_94 16
#define PADDED_SIZE_HBM_94 16
#define AXI_PADDED_SIZE_HBM_94 4
#define TABLE_SIZE_HBM_94 200000
#define DATA_SIZE_HBM_95 16
#define PADDED_SIZE_HBM_95 16
#define AXI_PADDED_SIZE_HBM_95 4
#define TABLE_SIZE_HBM_95 200000
#define DATA_SIZE_HBM_96 16
#define PADDED_SIZE_HBM_96 16
#define AXI_PADDED_SIZE_HBM_96 4
#define TABLE_SIZE_HBM_96 300000
#define DATA_SIZE_HBM_97 16
#define PADDED_SIZE_HBM_97 16
#define AXI_PADDED_SIZE_HBM_97 4
#define TABLE_SIZE_HBM_97 300000
#define DATA_SIZE_HBM_98 16
#define PADDED_SIZE_HBM_98 16
#define AXI_PADDED_SIZE_HBM_98 4
#define TABLE_SIZE_HBM_98 300000
#define DATA_SIZE_HBM_99 16
#define PADDED_SIZE_HBM_99 16
#define AXI_PADDED_SIZE_HBM_99 4
#define TABLE_SIZE_HBM_99 300000
#define DATA_SIZE_HBM_100 16
#define PADDED_SIZE_HBM_100 16
#define AXI_PADDED_SIZE_HBM_100 4
#define TABLE_SIZE_HBM_100 1000000
#define DATA_SIZE_HBM_101 16
#define PADDED_SIZE_HBM_101 16
#define AXI_PADDED_SIZE_HBM_101 4
#define TABLE_SIZE_HBM_101 1000000
#define DATA_SIZE_HBM_102 16
#define PADDED_SIZE_HBM_102 16
#define AXI_PADDED_SIZE_HBM_102 4
#define TABLE_SIZE_HBM_102 1000000
#define DATA_SIZE_HBM_103 16
#define PADDED_SIZE_HBM_103 16
#define AXI_PADDED_SIZE_HBM_103 4
#define TABLE_SIZE_HBM_103 1000000
#define DATA_SIZE_HBM_104 16
#define PADDED_SIZE_HBM_104 16
#define AXI_PADDED_SIZE_HBM_104 4
#define TABLE_SIZE_HBM_104 1000000
#define DATA_SIZE_HBM_105 16
#define PADDED_SIZE_HBM_105 16
#define AXI_PADDED_SIZE_HBM_105 4
#define TABLE_SIZE_HBM_105 1000000
#define DATA_SIZE_HBM_106 16
#define PADDED_SIZE_HBM_106 16
#define AXI_PADDED_SIZE_HBM_106 4
#define TABLE_SIZE_HBM_106 1000000
#define DATA_SIZE_HBM_107 16
#define PADDED_SIZE_HBM_107 16
#define AXI_PADDED_SIZE_HBM_107 4
#define TABLE_SIZE_HBM_107 1000000
#define DATA_SIZE_HBM_108 16
#define PADDED_SIZE_HBM_108 16
#define AXI_PADDED_SIZE_HBM_108 4
#define TABLE_SIZE_HBM_108 1000000
#define DATA_SIZE_HBM_109 16
#define PADDED_SIZE_HBM_109 16
#define AXI_PADDED_SIZE_HBM_109 4
#define TABLE_SIZE_HBM_109 1000000
#define DATA_SIZE_HBM_110 16
#define PADDED_SIZE_HBM_110 16
#define AXI_PADDED_SIZE_HBM_110 4
#define TABLE_SIZE_HBM_110 1000000
#define DATA_SIZE_HBM_111 16
#define PADDED_SIZE_HBM_111 16
#define AXI_PADDED_SIZE_HBM_111 4
#define TABLE_SIZE_HBM_111 1000000
#define DATA_SIZE_HBM_112 16
#define PADDED_SIZE_HBM_112 16
#define AXI_PADDED_SIZE_HBM_112 4
#define TABLE_SIZE_HBM_112 1000000
#define DATA_SIZE_HBM_113 16
#define PADDED_SIZE_HBM_113 16
#define AXI_PADDED_SIZE_HBM_113 4
#define TABLE_SIZE_HBM_113 1000000
#define DATA_SIZE_HBM_114 16
#define PADDED_SIZE_HBM_114 16
#define AXI_PADDED_SIZE_HBM_114 4
#define TABLE_SIZE_HBM_114 1000000
#define DATA_SIZE_HBM_115 16
#define PADDED_SIZE_HBM_115 16
#define AXI_PADDED_SIZE_HBM_115 4
#define TABLE_SIZE_HBM_115 1000000
#define DATA_SIZE_HBM_116 16
#define PADDED_SIZE_HBM_116 16
#define AXI_PADDED_SIZE_HBM_116 4
#define TABLE_SIZE_HBM_116 1000000
#define DATA_SIZE_HBM_117 16
#define PADDED_SIZE_HBM_117 16
#define AXI_PADDED_SIZE_HBM_117 4
#define TABLE_SIZE_HBM_117 1000000
#define DATA_SIZE_HBM_118 16
#define PADDED_SIZE_HBM_118 16
#define AXI_PADDED_SIZE_HBM_118 4
#define TABLE_SIZE_HBM_118 1000000
#define DATA_SIZE_HBM_119 16
#define PADDED_SIZE_HBM_119 16
#define AXI_PADDED_SIZE_HBM_119 4
#define TABLE_SIZE_HBM_119 1000000
#define DATA_SIZE_HBM_120 16
#define PADDED_SIZE_HBM_120 16
#define AXI_PADDED_SIZE_HBM_120 4
#define TABLE_SIZE_HBM_120 1000000
#define DATA_SIZE_HBM_121 16
#define PADDED_SIZE_HBM_121 16
#define AXI_PADDED_SIZE_HBM_121 4
#define TABLE_SIZE_HBM_121 1000000
#define DATA_SIZE_HBM_122 16
#define PADDED_SIZE_HBM_122 16
#define AXI_PADDED_SIZE_HBM_122 4
#define TABLE_SIZE_HBM_122 1000000
#define DATA_SIZE_HBM_123 16
#define PADDED_SIZE_HBM_123 16
#define AXI_PADDED_SIZE_HBM_123 4
#define TABLE_SIZE_HBM_123 1000000
#define DATA_SIZE_HBM_124 16
#define PADDED_SIZE_HBM_124 16
#define AXI_PADDED_SIZE_HBM_124 4
#define TABLE_SIZE_HBM_124 1000000
#define DATA_SIZE_HBM_125 16
#define PADDED_SIZE_HBM_125 16
#define AXI_PADDED_SIZE_HBM_125 4
#define TABLE_SIZE_HBM_125 1000000
#define DATA_SIZE_HBM_126 16
#define PADDED_SIZE_HBM_126 16
#define AXI_PADDED_SIZE_HBM_126 4
#define TABLE_SIZE_HBM_126 1000000
#define DATA_SIZE_HBM_127 16
#define PADDED_SIZE_HBM_127 16
#define AXI_PADDED_SIZE_HBM_127 4
#define TABLE_SIZE_HBM_127 1000000
#define DATA_SIZE_HBM_128 16
#define PADDED_SIZE_HBM_128 16
#define AXI_PADDED_SIZE_HBM_128 4
#define TABLE_SIZE_HBM_128 1000000
#define DATA_SIZE_HBM_129 16
#define PADDED_SIZE_HBM_129 16
#define AXI_PADDED_SIZE_HBM_129 4
#define TABLE_SIZE_HBM_129 1000000
#define DATA_SIZE_HBM_130 16
#define PADDED_SIZE_HBM_130 16
#define AXI_PADDED_SIZE_HBM_130 4
#define TABLE_SIZE_HBM_130 1000000
#define DATA_SIZE_HBM_131 16
#define PADDED_SIZE_HBM_131 16
#define AXI_PADDED_SIZE_HBM_131 4
#define TABLE_SIZE_HBM_131 1000000
#define DATA_SIZE_HBM_132 16
#define PADDED_SIZE_HBM_132 16
#define AXI_PADDED_SIZE_HBM_132 4
#define TABLE_SIZE_HBM_132 1000000
#define DATA_SIZE_HBM_133 16
#define PADDED_SIZE_HBM_133 16
#define AXI_PADDED_SIZE_HBM_133 4
#define TABLE_SIZE_HBM_133 1000000
#define DATA_SIZE_HBM_134 16
#define PADDED_SIZE_HBM_134 16
#define AXI_PADDED_SIZE_HBM_134 4
#define TABLE_SIZE_HBM_134 1000000
#define DATA_SIZE_HBM_135 16
#define PADDED_SIZE_HBM_135 16
#define AXI_PADDED_SIZE_HBM_135 4
#define TABLE_SIZE_HBM_135 1000000
#define DATA_SIZE_HBM_136 16
#define PADDED_SIZE_HBM_136 16
#define AXI_PADDED_SIZE_HBM_136 4
#define TABLE_SIZE_HBM_136 1000000
#define DATA_SIZE_HBM_137 16
#define PADDED_SIZE_HBM_137 16
#define AXI_PADDED_SIZE_HBM_137 4
#define TABLE_SIZE_HBM_137 1000000
#define DATA_SIZE_HBM_138 16
#define PADDED_SIZE_HBM_138 16
#define AXI_PADDED_SIZE_HBM_138 4
#define TABLE_SIZE_HBM_138 5000000
#define DATA_SIZE_HBM_139 16
#define PADDED_SIZE_HBM_139 16
#define AXI_PADDED_SIZE_HBM_139 4
#define TABLE_SIZE_HBM_139 5000000

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
#define ADDR_AXI_HBM_28 20000
#define ADDR_AXI_HBM_29 20000
#define ADDR_AXI_HBM_30 20000
#define ADDR_AXI_HBM_31 20000
#define ADDR_AXI_HBM_32 20000
#define ADDR_AXI_HBM_33 20000
#define ADDR_AXI_HBM_34 20000
#define ADDR_AXI_HBM_35 20000
#define ADDR_AXI_HBM_36 20000
#define ADDR_AXI_HBM_37 20000
#define ADDR_AXI_HBM_38 20000
#define ADDR_AXI_HBM_39 20000
#define ADDR_AXI_HBM_40 60000
#define ADDR_AXI_HBM_41 60000
#define ADDR_AXI_HBM_42 60000
#define ADDR_AXI_HBM_43 60000
#define ADDR_AXI_HBM_44 200000
#define ADDR_AXI_HBM_45 200000
#define ADDR_AXI_HBM_46 200000
#define ADDR_AXI_HBM_47 200000
#define ADDR_AXI_HBM_48 200000
#define ADDR_AXI_HBM_49 200000
#define ADDR_AXI_HBM_50 200000
#define ADDR_AXI_HBM_51 200000
#define ADDR_AXI_HBM_52 200000
#define ADDR_AXI_HBM_53 200000
#define ADDR_AXI_HBM_54 200000
#define ADDR_AXI_HBM_55 200000
#define ADDR_AXI_HBM_56 220000
#define ADDR_AXI_HBM_57 220000
#define ADDR_AXI_HBM_58 220000
#define ADDR_AXI_HBM_59 220000
#define ADDR_AXI_HBM_60 220000
#define ADDR_AXI_HBM_61 220000
#define ADDR_AXI_HBM_62 220000
#define ADDR_AXI_HBM_63 220000
#define ADDR_AXI_HBM_64 220000
#define ADDR_AXI_HBM_65 220000
#define ADDR_AXI_HBM_66 220000
#define ADDR_AXI_HBM_67 220000
#define ADDR_AXI_HBM_68 260000
#define ADDR_AXI_HBM_69 260000
#define ADDR_AXI_HBM_70 260000
#define ADDR_AXI_HBM_71 260000
#define ADDR_AXI_HBM_72 400000
#define ADDR_AXI_HBM_73 400000
#define ADDR_AXI_HBM_74 400000
#define ADDR_AXI_HBM_75 400000
#define ADDR_AXI_HBM_76 400000
#define ADDR_AXI_HBM_77 400000
#define ADDR_AXI_HBM_78 400000
#define ADDR_AXI_HBM_79 400000
#define ADDR_AXI_HBM_80 400000
#define ADDR_AXI_HBM_81 400000
#define ADDR_AXI_HBM_82 400000
#define ADDR_AXI_HBM_83 400000
#define ADDR_AXI_HBM_84 420000
#define ADDR_AXI_HBM_85 420000
#define ADDR_AXI_HBM_86 420000
#define ADDR_AXI_HBM_87 420000
#define ADDR_AXI_HBM_88 420000
#define ADDR_AXI_HBM_89 420000
#define ADDR_AXI_HBM_90 420000
#define ADDR_AXI_HBM_91 420000
#define ADDR_AXI_HBM_92 420000
#define ADDR_AXI_HBM_93 420000
#define ADDR_AXI_HBM_94 420000
#define ADDR_AXI_HBM_95 420000
#define ADDR_AXI_HBM_96 460000
#define ADDR_AXI_HBM_97 460000
#define ADDR_AXI_HBM_98 460000
#define ADDR_AXI_HBM_99 460000
#define ADDR_AXI_HBM_100 600000
#define ADDR_AXI_HBM_101 600000
#define ADDR_AXI_HBM_102 600000
#define ADDR_AXI_HBM_103 600000
#define ADDR_AXI_HBM_104 600000
#define ADDR_AXI_HBM_105 600000
#define ADDR_AXI_HBM_106 600000
#define ADDR_AXI_HBM_107 600000
#define ADDR_AXI_HBM_108 600000
#define ADDR_AXI_HBM_109 600000
#define ADDR_AXI_HBM_110 600000
#define ADDR_AXI_HBM_111 600000
#define ADDR_AXI_HBM_112 620000
#define ADDR_AXI_HBM_113 620000
#define ADDR_AXI_HBM_114 620000
#define ADDR_AXI_HBM_115 620000
#define ADDR_AXI_HBM_116 620000
#define ADDR_AXI_HBM_117 620000
#define ADDR_AXI_HBM_118 620000
#define ADDR_AXI_HBM_119 620000
#define ADDR_AXI_HBM_120 1220000
#define ADDR_AXI_HBM_121 1220000
#define ADDR_AXI_HBM_122 1220000
#define ADDR_AXI_HBM_123 1220000
#define ADDR_AXI_HBM_124 1660000
#define ADDR_AXI_HBM_125 1660000
#define ADDR_AXI_HBM_126 1660000
#define ADDR_AXI_HBM_127 1660000
#define ADDR_AXI_HBM_128 4600000
#define ADDR_AXI_HBM_129 4600000
#define ADDR_AXI_HBM_130 4600000
#define ADDR_AXI_HBM_131 4600000
#define ADDR_AXI_HBM_132 4600000
#define ADDR_AXI_HBM_133 4600000
#define ADDR_AXI_HBM_134 4600000
#define ADDR_AXI_HBM_135 4600000
#define ADDR_AXI_HBM_136 4600000
#define ADDR_AXI_HBM_137 4600000
#define ADDR_AXI_HBM_138 4600000
#define ADDR_AXI_HBM_139 4600000

#define HBM_BANK0_ROUND 5
#define HBM_BANK1_ROUND 5
#define HBM_BANK2_ROUND 5
#define HBM_BANK3_ROUND 5
#define HBM_BANK4_ROUND 5
#define HBM_BANK5_ROUND 5
#define HBM_BANK6_ROUND 5
#define HBM_BANK7_ROUND 5
#define HBM_BANK8_ROUND 5
#define HBM_BANK9_ROUND 5
#define HBM_BANK10_ROUND 5
#define HBM_BANK11_ROUND 5
#define HBM_BANK12_ROUND 5
#define HBM_BANK13_ROUND 5
#define HBM_BANK14_ROUND 5
#define HBM_BANK15_ROUND 5
#define HBM_BANK16_ROUND 5
#define HBM_BANK17_ROUND 5
#define HBM_BANK18_ROUND 5
#define HBM_BANK19_ROUND 5
#define HBM_BANK20_ROUND 5
#define HBM_BANK21_ROUND 5
#define HBM_BANK22_ROUND 5
#define HBM_BANK23_ROUND 5
#define HBM_BANK24_ROUND 5
#define HBM_BANK25_ROUND 5
#define HBM_BANK26_ROUND 5
#define HBM_BANK27_ROUND 5

#define HBM_BANK0_SIZE 4620000
#define HBM_BANK1_SIZE 4620000
#define HBM_BANK2_SIZE 4620000
#define HBM_BANK3_SIZE 4620000
#define HBM_BANK4_SIZE 4620000
#define HBM_BANK5_SIZE 4620000
#define HBM_BANK6_SIZE 4620000
#define HBM_BANK7_SIZE 4620000
#define HBM_BANK8_SIZE 5220000
#define HBM_BANK9_SIZE 5220000
#define HBM_BANK10_SIZE 5220000
#define HBM_BANK11_SIZE 5220000
#define HBM_BANK12_SIZE 5660000
#define HBM_BANK13_SIZE 5660000
#define HBM_BANK14_SIZE 5660000
#define HBM_BANK15_SIZE 5660000
#define HBM_BANK16_SIZE 8600000
#define HBM_BANK17_SIZE 8600000
#define HBM_BANK18_SIZE 8600000
#define HBM_BANK19_SIZE 8600000
#define HBM_BANK20_SIZE 8600000
#define HBM_BANK21_SIZE 8600000
#define HBM_BANK22_SIZE 8600000
#define HBM_BANK23_SIZE 8600000
#define HBM_BANK24_SIZE 8600000
#define HBM_BANK25_SIZE 8600000
#define HBM_BANK26_SIZE 10000000 // 24600000 // Origin = 393.6 MB, cut the rest
#define HBM_BANK27_SIZE 10000000 // 24600000 // Origin = 393.6 MB, cut the rest

#define VECTOR_SIZE_HBM_BANK_0 48
#define VECTOR_SIZE_HBM_BANK_1 48
#define VECTOR_SIZE_HBM_BANK_2 48
#define VECTOR_SIZE_HBM_BANK_3 48
#define VECTOR_SIZE_HBM_BANK_4 48
#define VECTOR_SIZE_HBM_BANK_5 48
#define VECTOR_SIZE_HBM_BANK_6 48
#define VECTOR_SIZE_HBM_BANK_7 48
#define VECTOR_SIZE_HBM_BANK_8 56
#define VECTOR_SIZE_HBM_BANK_9 56
#define VECTOR_SIZE_HBM_BANK_10 56
#define VECTOR_SIZE_HBM_BANK_11 56
#define VECTOR_SIZE_HBM_BANK_12 56
#define VECTOR_SIZE_HBM_BANK_13 56
#define VECTOR_SIZE_HBM_BANK_14 56
#define VECTOR_SIZE_HBM_BANK_15 56
#define VECTOR_SIZE_HBM_BANK_16 56
#define VECTOR_SIZE_HBM_BANK_17 56
#define VECTOR_SIZE_HBM_BANK_18 56
#define VECTOR_SIZE_HBM_BANK_19 56
#define VECTOR_SIZE_HBM_BANK_20 56
#define VECTOR_SIZE_HBM_BANK_21 56
#define VECTOR_SIZE_HBM_BANK_22 56
#define VECTOR_SIZE_HBM_BANK_23 56
#define VECTOR_SIZE_HBM_BANK_24 56
#define VECTOR_SIZE_HBM_BANK_25 56
#define VECTOR_SIZE_HBM_BANK_26 56
#define VECTOR_SIZE_HBM_BANK_27 56

#define VECTOR_START_IDX_HBM_BANK_0 0
#define VECTOR_START_IDX_HBM_BANK_1 48
#define VECTOR_START_IDX_HBM_BANK_2 96
#define VECTOR_START_IDX_HBM_BANK_3 144
#define VECTOR_START_IDX_HBM_BANK_4 192
#define VECTOR_START_IDX_HBM_BANK_5 240
#define VECTOR_START_IDX_HBM_BANK_6 288
#define VECTOR_START_IDX_HBM_BANK_7 336
#define VECTOR_START_IDX_HBM_BANK_8 384
#define VECTOR_START_IDX_HBM_BANK_9 440
#define VECTOR_START_IDX_HBM_BANK_10 496
#define VECTOR_START_IDX_HBM_BANK_11 552
#define VECTOR_START_IDX_HBM_BANK_12 608
#define VECTOR_START_IDX_HBM_BANK_13 664
#define VECTOR_START_IDX_HBM_BANK_14 720
#define VECTOR_START_IDX_HBM_BANK_15 776
#define VECTOR_START_IDX_HBM_BANK_16 832
#define VECTOR_START_IDX_HBM_BANK_17 888
#define VECTOR_START_IDX_HBM_BANK_18 944
#define VECTOR_START_IDX_HBM_BANK_19 1000
#define VECTOR_START_IDX_HBM_BANK_20 1056
#define VECTOR_START_IDX_HBM_BANK_21 1112
#define VECTOR_START_IDX_HBM_BANK_22 1168
#define VECTOR_START_IDX_HBM_BANK_23 1224
#define VECTOR_START_IDX_HBM_BANK_24 1280
#define VECTOR_START_IDX_HBM_BANK_25 1336
#define VECTOR_START_IDX_HBM_BANK_26 1392
#define VECTOR_START_IDX_HBM_BANK_27 1448

/////////////////////////   DDR   ///////////////////////// 
// alignment of tables to DDR: 
// table 0 ~ 1 -> DDR 0 ~ 1
// table 2 ~ 3 -> DDR 0 ~ 1

#define DATA_SIZE_DDR_0 32
#define PADDED_SIZE_DDR_0 32
#define AXI_PADDED_SIZE_DDR_0 8
#define TABLE_SIZE_DDR_0 10000000
#define DATA_SIZE_DDR_1 32
#define PADDED_SIZE_DDR_1 32
#define AXI_PADDED_SIZE_DDR_1 8
#define TABLE_SIZE_DDR_1 10000000
#define DATA_SIZE_DDR_2 32
#define PADDED_SIZE_DDR_2 32
#define AXI_PADDED_SIZE_DDR_2 8
#define TABLE_SIZE_DDR_2 100000000
#define DATA_SIZE_DDR_3 32
#define PADDED_SIZE_DDR_3 32
#define AXI_PADDED_SIZE_DDR_3 8
#define TABLE_SIZE_DDR_3 100000000

#define BURST_SIZE_DDR 32

#define ADDR_AXI_DDR_0 0
#define ADDR_AXI_DDR_1 0
#define ADDR_AXI_DDR_2 80000000
#define ADDR_AXI_DDR_3 80000000

#define DDR_BANK0_ROUND 2
#define DDR_BANK1_ROUND 2

#define DDR_BANK0_SIZE 880000000
#define DDR_BANK1_SIZE 880000000

#define VECTOR_SIZE_DDR_BANK_0 64
#define VECTOR_SIZE_DDR_BANK_1 64

#define VECTOR_START_IDX_DDR_BANK_0 1504
#define VECTOR_START_IDX_DDR_BANK_1 1568

/////////////////////////   PLRAM   ///////////////////////// 
// alignment of tables to PLRAM: 
// table 0 ~ 10 -> PLRAM 0 ~ 10
// table 11 ~ 21 -> PLRAM 0 ~ 10

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
#define TABLE_SIZE_PLRAM_4 100
#define DATA_SIZE_PLRAM_5 4
#define PADDED_SIZE_PLRAM_5 4
#define AXI_PADDED_SIZE_PLRAM_5 1
#define TABLE_SIZE_PLRAM_5 100
#define DATA_SIZE_PLRAM_6 4
#define PADDED_SIZE_PLRAM_6 4
#define AXI_PADDED_SIZE_PLRAM_6 1
#define TABLE_SIZE_PLRAM_6 100
#define DATA_SIZE_PLRAM_7 4
#define PADDED_SIZE_PLRAM_7 4
#define AXI_PADDED_SIZE_PLRAM_7 1
#define TABLE_SIZE_PLRAM_7 100
#define DATA_SIZE_PLRAM_8 8
#define PADDED_SIZE_PLRAM_8 8
#define AXI_PADDED_SIZE_PLRAM_8 2
#define TABLE_SIZE_PLRAM_8 5000
#define DATA_SIZE_PLRAM_9 8
#define PADDED_SIZE_PLRAM_9 8
#define AXI_PADDED_SIZE_PLRAM_9 2
#define TABLE_SIZE_PLRAM_9 5000
#define DATA_SIZE_PLRAM_10 8
#define PADDED_SIZE_PLRAM_10 8
#define AXI_PADDED_SIZE_PLRAM_10 2
#define TABLE_SIZE_PLRAM_10 5000
#define DATA_SIZE_PLRAM_11 8
#define PADDED_SIZE_PLRAM_11 8
#define AXI_PADDED_SIZE_PLRAM_11 2
#define TABLE_SIZE_PLRAM_11 5000
#define DATA_SIZE_PLRAM_12 8
#define PADDED_SIZE_PLRAM_12 8
#define AXI_PADDED_SIZE_PLRAM_12 2
#define TABLE_SIZE_PLRAM_12 5000
#define DATA_SIZE_PLRAM_13 8
#define PADDED_SIZE_PLRAM_13 8
#define AXI_PADDED_SIZE_PLRAM_13 2
#define TABLE_SIZE_PLRAM_13 5000
#define DATA_SIZE_PLRAM_14 8
#define PADDED_SIZE_PLRAM_14 8
#define AXI_PADDED_SIZE_PLRAM_14 2
#define TABLE_SIZE_PLRAM_14 5000
#define DATA_SIZE_PLRAM_15 8
#define PADDED_SIZE_PLRAM_15 8
#define AXI_PADDED_SIZE_PLRAM_15 2
#define TABLE_SIZE_PLRAM_15 5000
#define DATA_SIZE_PLRAM_16 8
#define PADDED_SIZE_PLRAM_16 8
#define AXI_PADDED_SIZE_PLRAM_16 2
#define TABLE_SIZE_PLRAM_16 10000
#define DATA_SIZE_PLRAM_17 8
#define PADDED_SIZE_PLRAM_17 8
#define AXI_PADDED_SIZE_PLRAM_17 2
#define TABLE_SIZE_PLRAM_17 10000
#define DATA_SIZE_PLRAM_18 8
#define PADDED_SIZE_PLRAM_18 8
#define AXI_PADDED_SIZE_PLRAM_18 2
#define TABLE_SIZE_PLRAM_18 10000
#define DATA_SIZE_PLRAM_19 8
#define PADDED_SIZE_PLRAM_19 8
#define AXI_PADDED_SIZE_PLRAM_19 2
#define TABLE_SIZE_PLRAM_19 10000
#define DATA_SIZE_PLRAM_20 8
#define PADDED_SIZE_PLRAM_20 8
#define AXI_PADDED_SIZE_PLRAM_20 2
#define TABLE_SIZE_PLRAM_20 10000
#define DATA_SIZE_PLRAM_21 8
#define PADDED_SIZE_PLRAM_21 8
#define AXI_PADDED_SIZE_PLRAM_21 2
#define TABLE_SIZE_PLRAM_21 10000
#define DATA_SIZE_PLRAM_22 8
#define PADDED_SIZE_PLRAM_22 8
#define AXI_PADDED_SIZE_PLRAM_22 2
#define TABLE_SIZE_PLRAM_22 10000
#define DATA_SIZE_PLRAM_23 8
#define PADDED_SIZE_PLRAM_23 8
#define AXI_PADDED_SIZE_PLRAM_23 2
#define TABLE_SIZE_PLRAM_23 10000
#define DATA_SIZE_PLRAM_24 8
#define PADDED_SIZE_PLRAM_24 8
#define AXI_PADDED_SIZE_PLRAM_24 2
#define TABLE_SIZE_PLRAM_24 10000
#define DATA_SIZE_PLRAM_25 8
#define PADDED_SIZE_PLRAM_25 8
#define AXI_PADDED_SIZE_PLRAM_25 2
#define TABLE_SIZE_PLRAM_25 10000
#define DATA_SIZE_PLRAM_26 8
#define PADDED_SIZE_PLRAM_26 8
#define AXI_PADDED_SIZE_PLRAM_26 2
#define TABLE_SIZE_PLRAM_26 10000
#define DATA_SIZE_PLRAM_27 8
#define PADDED_SIZE_PLRAM_27 8
#define AXI_PADDED_SIZE_PLRAM_27 2
#define TABLE_SIZE_PLRAM_27 10000
#define DATA_SIZE_PLRAM_28 8
#define PADDED_SIZE_PLRAM_28 8
#define AXI_PADDED_SIZE_PLRAM_28 2
#define TABLE_SIZE_PLRAM_28 10000
#define DATA_SIZE_PLRAM_29 8
#define PADDED_SIZE_PLRAM_29 8
#define AXI_PADDED_SIZE_PLRAM_29 2
#define TABLE_SIZE_PLRAM_29 10000
#define DATA_SIZE_PLRAM_30 8
#define PADDED_SIZE_PLRAM_30 8
#define AXI_PADDED_SIZE_PLRAM_30 2
#define TABLE_SIZE_PLRAM_30 10000
#define DATA_SIZE_PLRAM_31 8
#define PADDED_SIZE_PLRAM_31 8
#define AXI_PADDED_SIZE_PLRAM_31 2
#define TABLE_SIZE_PLRAM_31 10000
#define DATA_SIZE_PLRAM_32 8
#define PADDED_SIZE_PLRAM_32 8
#define AXI_PADDED_SIZE_PLRAM_32 2
#define TABLE_SIZE_PLRAM_32 10000
#define DATA_SIZE_PLRAM_33 8
#define PADDED_SIZE_PLRAM_33 8
#define AXI_PADDED_SIZE_PLRAM_33 2
#define TABLE_SIZE_PLRAM_33 10000
#define DATA_SIZE_PLRAM_34 8
#define PADDED_SIZE_PLRAM_34 8
#define AXI_PADDED_SIZE_PLRAM_34 2
#define TABLE_SIZE_PLRAM_34 10000
#define DATA_SIZE_PLRAM_35 8
#define PADDED_SIZE_PLRAM_35 8
#define AXI_PADDED_SIZE_PLRAM_35 2
#define TABLE_SIZE_PLRAM_35 10000
#define DATA_SIZE_PLRAM_36 8
#define PADDED_SIZE_PLRAM_36 8
#define AXI_PADDED_SIZE_PLRAM_36 2
#define TABLE_SIZE_PLRAM_36 10000
#define DATA_SIZE_PLRAM_37 8
#define PADDED_SIZE_PLRAM_37 8
#define AXI_PADDED_SIZE_PLRAM_37 2
#define TABLE_SIZE_PLRAM_37 10000
#define DATA_SIZE_PLRAM_38 8
#define PADDED_SIZE_PLRAM_38 8
#define AXI_PADDED_SIZE_PLRAM_38 2
#define TABLE_SIZE_PLRAM_38 10000
#define DATA_SIZE_PLRAM_39 8
#define PADDED_SIZE_PLRAM_39 8
#define AXI_PADDED_SIZE_PLRAM_39 2
#define TABLE_SIZE_PLRAM_39 10000
#define DATA_SIZE_PLRAM_40 8
#define PADDED_SIZE_PLRAM_40 8
#define AXI_PADDED_SIZE_PLRAM_40 2
#define TABLE_SIZE_PLRAM_40 10000
#define DATA_SIZE_PLRAM_41 8
#define PADDED_SIZE_PLRAM_41 8
#define AXI_PADDED_SIZE_PLRAM_41 2
#define TABLE_SIZE_PLRAM_41 10000
#define DATA_SIZE_PLRAM_42 8
#define PADDED_SIZE_PLRAM_42 8
#define AXI_PADDED_SIZE_PLRAM_42 2
#define TABLE_SIZE_PLRAM_42 10000
#define DATA_SIZE_PLRAM_43 8
#define PADDED_SIZE_PLRAM_43 8
#define AXI_PADDED_SIZE_PLRAM_43 2
#define TABLE_SIZE_PLRAM_43 10000

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
#define ADDR_AXI_PLRAM_11 100
#define ADDR_AXI_PLRAM_12 100
#define ADDR_AXI_PLRAM_13 100
#define ADDR_AXI_PLRAM_14 100
#define ADDR_AXI_PLRAM_15 100
#define ADDR_AXI_PLRAM_16 100
#define ADDR_AXI_PLRAM_17 100
#define ADDR_AXI_PLRAM_18 100
#define ADDR_AXI_PLRAM_19 10000
#define ADDR_AXI_PLRAM_20 10000
#define ADDR_AXI_PLRAM_21 10000
#define ADDR_AXI_PLRAM_22 10100
#define ADDR_AXI_PLRAM_23 10100
#define ADDR_AXI_PLRAM_24 10100
#define ADDR_AXI_PLRAM_25 10100
#define ADDR_AXI_PLRAM_26 10100
#define ADDR_AXI_PLRAM_27 20100
#define ADDR_AXI_PLRAM_28 20100
#define ADDR_AXI_PLRAM_29 20100
#define ADDR_AXI_PLRAM_30 30000
#define ADDR_AXI_PLRAM_31 30000
#define ADDR_AXI_PLRAM_32 30000
#define ADDR_AXI_PLRAM_33 30100
#define ADDR_AXI_PLRAM_34 30100
#define ADDR_AXI_PLRAM_35 30100
#define ADDR_AXI_PLRAM_36 30100
#define ADDR_AXI_PLRAM_37 30100
#define ADDR_AXI_PLRAM_38 40100
#define ADDR_AXI_PLRAM_39 40100
#define ADDR_AXI_PLRAM_40 40100
#define ADDR_AXI_PLRAM_41 50000
#define ADDR_AXI_PLRAM_42 50000
#define ADDR_AXI_PLRAM_43 50000

#define PLRAM_BANK0_ROUND 4
#define PLRAM_BANK1_ROUND 4
#define PLRAM_BANK2_ROUND 4
#define PLRAM_BANK3_ROUND 4
#define PLRAM_BANK4_ROUND 4
#define PLRAM_BANK5_ROUND 4
#define PLRAM_BANK6_ROUND 4
#define PLRAM_BANK7_ROUND 4
#define PLRAM_BANK8_ROUND 4
#define PLRAM_BANK9_ROUND 4
#define PLRAM_BANK10_ROUND 4

#define PLRAM_BANK0_SIZE 50100
#define PLRAM_BANK1_SIZE 50100
#define PLRAM_BANK2_SIZE 50100
#define PLRAM_BANK3_SIZE 50100
#define PLRAM_BANK4_SIZE 50100
#define PLRAM_BANK5_SIZE 60100
#define PLRAM_BANK6_SIZE 60100
#define PLRAM_BANK7_SIZE 60100
#define PLRAM_BANK8_SIZE 70000
#define PLRAM_BANK9_SIZE 70000
#define PLRAM_BANK10_SIZE 70000

#define VECTOR_SIZE_PLRAM_BANK_0 28
#define VECTOR_SIZE_PLRAM_BANK_1 28
#define VECTOR_SIZE_PLRAM_BANK_2 28
#define VECTOR_SIZE_PLRAM_BANK_3 28
#define VECTOR_SIZE_PLRAM_BANK_4 28
#define VECTOR_SIZE_PLRAM_BANK_5 28
#define VECTOR_SIZE_PLRAM_BANK_6 28
#define VECTOR_SIZE_PLRAM_BANK_7 28
#define VECTOR_SIZE_PLRAM_BANK_8 32
#define VECTOR_SIZE_PLRAM_BANK_9 32
#define VECTOR_SIZE_PLRAM_BANK_10 32

#define VECTOR_START_IDX_PLRAM_BANK_0 1632
#define VECTOR_START_IDX_PLRAM_BANK_1 1660
#define VECTOR_START_IDX_PLRAM_BANK_2 1688
#define VECTOR_START_IDX_PLRAM_BANK_3 1716
#define VECTOR_START_IDX_PLRAM_BANK_4 1744
#define VECTOR_START_IDX_PLRAM_BANK_5 1772
#define VECTOR_START_IDX_PLRAM_BANK_6 1800
#define VECTOR_START_IDX_PLRAM_BANK_7 1828
#define VECTOR_START_IDX_PLRAM_BANK_8 1856
#define VECTOR_START_IDX_PLRAM_BANK_9 1888
#define VECTOR_START_IDX_PLRAM_BANK_10 1920