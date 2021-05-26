
#include "constants.hpp"
#include "ap_axi_sdata.h"
#include <ap_fixed.h>
#include "ap_int.h" 
#include "hls_stream.h"

#include "axi_utils.hpp"
#include "toe.hpp"

extern "C" {

#define DWIDTH512 512
#define DWIDTH256 256
#define DWIDTH128 128
#define DWIDTH64 64
#define DWIDTH32 32
#define DWIDTH16 16
#define DWIDTH8 8

const unsigned DATA_WIDTH = 64 * 8;


typedef ap_axiu<DWIDTH512, 0, 0, 0> pkt512;
typedef ap_axiu<DWIDTH256, 0, 0, 0> pkt256;
typedef ap_axiu<DWIDTH128, 0, 0, 0> pkt128;
typedef ap_axiu<DWIDTH64, 0, 0, 0> pkt64;
typedef ap_axiu<DWIDTH32, 0, 0, 0> pkt32;
typedef ap_axiu<DWIDTH16, 0, 0, 0> pkt16;
typedef ap_axiu<DWIDTH8, 0, 0, 0> pkt8;

void embedding_47_krnl(  
    const axi_t* table_HBM0, const axi_t* table_HBM1, 
    const axi_t* table_HBM2, const axi_t* table_HBM3, 
    const axi_t* table_HBM4, const axi_t* table_HBM5, 
    const axi_t* table_HBM6, const axi_t* table_HBM7, 
    const axi_t* table_HBM8, const axi_t* table_HBM9, 
    const axi_t* table_HBM10, const axi_t* table_HBM11, 
    const axi_t* table_HBM12, const axi_t* table_HBM13, 
    const axi_t* table_HBM14, const axi_t* table_HBM15, 
    const axi_t* table_HBM16, const axi_t* table_HBM17, 
    const axi_t* table_HBM18, const axi_t* table_HBM19, 
    const axi_t* table_HBM20, const axi_t* table_HBM21, 
    const axi_t* table_HBM22, const axi_t* table_HBM23, 
    const axi_t* table_HBM24, const axi_t* table_HBM25, 
    const axi_t* table_HBM26, const axi_t* table_HBM27, 
    const axi_t* table_DDR0, const axi_t* table_DDR1,
    // Internal Stream
    hls::stream<pkt512>& s_axis_udp_rx, 
    hls::stream<pkt512>& m_axis_udp_tx, 
    hls::stream<pkt256>& s_axis_udp_rx_meta, 
    hls::stream<pkt256>& m_axis_udp_tx_meta, 

    hls::stream<pkt16>& m_axis_tcp_listen_port, 
    hls::stream<pkt8>& s_axis_tcp_port_status, 
    hls::stream<pkt64>& m_axis_tcp_open_connection, 
    hls::stream<pkt32>& s_axis_tcp_open_status, 
    hls::stream<pkt16>& m_axis_tcp_close_connection, 
    hls::stream<pkt128>& s_axis_tcp_notification, 
    hls::stream<pkt32>& m_axis_tcp_read_pkg, 
    hls::stream<pkt16>& s_axis_tcp_rx_meta, 
    hls::stream<pkt512>& s_axis_tcp_rx_data, 
    hls::stream<pkt32>& m_axis_tcp_tx_meta, 
    hls::stream<pkt512>& m_axis_tcp_tx_data, 
    hls::stream<pkt64>& s_axis_tcp_tx_status,

    int useConn, 
    int pkgWordCount, 
    int basePort, 
    int baseIpAddress,
    int batch_num    );
}

template<
    const int start_addr_0, const int axi_padded_size_0, const int table_entry_num_0>
void init_plram_t_1_table(
    plram_t* table_PLRAM);

void load_access_idx(
    hls::stream<int>& s_idx_buffer_single_channel,
    int batch_num);

template<
    const long start_addr_0, const long AXI_padded_size_0>
void load_single_embedding_1_tables(
    hls::stream<int>& s_idx_buffer, const hbm_t* table_RAM, 
    hls::stream<hbm_t>& s_embedding_buffer,
    int batch_num);

template<
    const long start_addr_0, const long AXI_padded_size_0, 
    const long start_addr_1, const long AXI_padded_size_1>
void load_single_embedding_2_tables(
    hls::stream<int>& s_idx_buffer, const hbm_t* table_RAM, 
    hls::stream<hbm_t>& s_embedding_buffer,
    int batch_num);

void group_4_embeddings(
    hls::stream<axi_t>& s_embedding_0, hls::stream<axi_t>& s_embedding_1,
    hls::stream<axi_t>& s_embedding_2, hls::stream<axi_t>& s_embedding_3,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num);

void group_2_embeddings(
    hls::stream<axi_t>& s_embedding_0, hls::stream<axi_t>& s_embedding_1,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num);

void group_1_embeddings_16(
    hls::stream<axi_t>& s_embedding_0,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num);

void group_1_embeddings_32(
    hls::stream<axi_t>& s_embedding_0,
    hls::stream<network_t>& s_gathered_embedding_0,
    hls::stream<network_t>& s_gathered_embedding_1,
    int batch_num);

void gather_22_embedding_streams(
    hls::stream<network_t> (&s_embedding_gather_level_A)[22],
    hls::stream<network_t>& s_network, 
    int batch_num);

void gather_embeddings(
    hls::stream<hbm_t>& s_embedding_buffer_HBM0, hls::stream<hbm_t>& s_embedding_buffer_HBM1, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM2, hls::stream<hbm_t>& s_embedding_buffer_HBM3, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM4, hls::stream<hbm_t>& s_embedding_buffer_HBM5, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM6, hls::stream<hbm_t>& s_embedding_buffer_HBM7, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM8, hls::stream<hbm_t>& s_embedding_buffer_HBM9, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM10, hls::stream<hbm_t>& s_embedding_buffer_HBM11, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM12, hls::stream<hbm_t>& s_embedding_buffer_HBM13, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM14, hls::stream<hbm_t>& s_embedding_buffer_HBM15, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM16, hls::stream<hbm_t>& s_embedding_buffer_HBM17, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM18, hls::stream<hbm_t>& s_embedding_buffer_HBM19, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM20, hls::stream<hbm_t>& s_embedding_buffer_HBM21, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM22, hls::stream<hbm_t>& s_embedding_buffer_HBM23, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM24, hls::stream<hbm_t>& s_embedding_buffer_HBM25, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM26, hls::stream<hbm_t>& s_embedding_buffer_HBM27, 
    
    hls::stream<ddr_t>& s_embedding_buffer_DDR0, hls::stream<ddr_t>& s_embedding_buffer_DDR1,

    hls::stream<plram_t>& s_embedding_buffer_PLRAM0, hls::stream<plram_t>& s_embedding_buffer_PLRAM1, 
    hls::stream<plram_t>& s_embedding_buffer_PLRAM2, hls::stream<plram_t>& s_embedding_buffer_PLRAM3,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM4, hls::stream<plram_t>& s_embedding_buffer_PLRAM5,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM6, hls::stream<plram_t>& s_embedding_buffer_PLRAM7,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM8, hls::stream<plram_t>& s_embedding_buffer_PLRAM9,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM10, hls::stream<plram_t>& s_embedding_buffer_PLRAM11,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM12, hls::stream<plram_t>& s_embedding_buffer_PLRAM13,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM14, hls::stream<plram_t>& s_embedding_buffer_PLRAM15,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM16,
    
    hls::stream<network_t>& s_network,

    int batch_num);

void consume_and_write(
    // 22 = 352 (input feature len) * 4 * 8 / 512 bit
    hls::stream<network_t>& s_network, network_t* out_RAM,
    int batch_num); 

const int fifo_batch_size = FIFO_BATCH_SIZE;

const int depth_s_embedding_buffer_HBM0 = VECTOR_SIZE_HBM_BANK_0 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM1 = VECTOR_SIZE_HBM_BANK_1 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM2 = VECTOR_SIZE_HBM_BANK_2 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM3 = VECTOR_SIZE_HBM_BANK_3 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM4 = VECTOR_SIZE_HBM_BANK_4 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM5 = VECTOR_SIZE_HBM_BANK_5 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM6 = VECTOR_SIZE_HBM_BANK_6 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM7 = VECTOR_SIZE_HBM_BANK_7 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM8 = VECTOR_SIZE_HBM_BANK_8 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM9 = VECTOR_SIZE_HBM_BANK_9 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM10 = VECTOR_SIZE_HBM_BANK_10 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM11 = VECTOR_SIZE_HBM_BANK_11 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM12 = VECTOR_SIZE_HBM_BANK_12 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM13 = VECTOR_SIZE_HBM_BANK_13 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM14 = VECTOR_SIZE_HBM_BANK_14 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM15 = VECTOR_SIZE_HBM_BANK_15 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM16 = VECTOR_SIZE_HBM_BANK_16 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM17 = VECTOR_SIZE_HBM_BANK_17 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM18 = VECTOR_SIZE_HBM_BANK_18 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM19 = VECTOR_SIZE_HBM_BANK_19 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM20 = VECTOR_SIZE_HBM_BANK_20 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM21 = VECTOR_SIZE_HBM_BANK_21 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM22 = VECTOR_SIZE_HBM_BANK_22 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM23 = VECTOR_SIZE_HBM_BANK_23 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM24 = VECTOR_SIZE_HBM_BANK_24 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM25 = VECTOR_SIZE_HBM_BANK_25 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM26 = VECTOR_SIZE_HBM_BANK_26 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_HBM27 = VECTOR_SIZE_HBM_BANK_27 * FIFO_BATCH_SIZE;
// const int depth_s_embedding_buffer_HBM28 = VECTOR_SIZE_HBM_BANK_28 * FIFO_BATCH_SIZE;
// const int depth_s_embedding_buffer_HBM29 = VECTOR_SIZE_HBM_BANK_29 * FIFO_BATCH_SIZE;
// const int depth_s_embedding_buffer_HBM30 = VECTOR_SIZE_HBM_BANK_30 * FIFO_BATCH_SIZE;
// const int depth_s_embedding_buffer_HBM31 = VECTOR_SIZE_HBM_BANK_31 * FIFO_BATCH_SIZE;

const int depth_s_embedding_buffer_PLRAM0 = VECTOR_SIZE_PLRAM_BANK_0 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM1 = VECTOR_SIZE_PLRAM_BANK_1 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM2 = VECTOR_SIZE_PLRAM_BANK_2 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM3 = VECTOR_SIZE_PLRAM_BANK_3 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM4 = VECTOR_SIZE_PLRAM_BANK_4 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM5 = VECTOR_SIZE_PLRAM_BANK_5 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM6 = VECTOR_SIZE_PLRAM_BANK_6 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM7 = VECTOR_SIZE_PLRAM_BANK_7 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM8 = VECTOR_SIZE_PLRAM_BANK_8 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM9 = VECTOR_SIZE_PLRAM_BANK_9 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM10 = VECTOR_SIZE_PLRAM_BANK_10 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM11 = VECTOR_SIZE_PLRAM_BANK_11 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM12 = VECTOR_SIZE_PLRAM_BANK_12 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM13 = VECTOR_SIZE_PLRAM_BANK_13 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM14 = VECTOR_SIZE_PLRAM_BANK_14 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM15 = VECTOR_SIZE_PLRAM_BANK_15 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_PLRAM16 = VECTOR_SIZE_PLRAM_BANK_16 * FIFO_BATCH_SIZE;

const int depth_s_embedding_buffer_DDR0 = VECTOR_SIZE_DDR_BANK_0 * FIFO_BATCH_SIZE;
const int depth_s_embedding_buffer_DDR1 = VECTOR_SIZE_DDR_BANK_1 * FIFO_BATCH_SIZE;